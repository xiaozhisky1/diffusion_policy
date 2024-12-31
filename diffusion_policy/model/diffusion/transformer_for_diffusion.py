from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import copy
import numpy as np
logger = logging.getLogger(__name__)
import torch.nn.functional as F

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):  # d_model也为int
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)  # 创建一个形状为 (max_len, d_model) 的零张量，来存储位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1)  # torch.arange(0, max_len) 生成 [0, 1, ..., max_len-1] 的序列 .unsqueeze(1) 在索引为 1 的维度（第二维）插入一个新维度，形状为 (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)  # 生成偶数序列，如 [0, 2, 4, ..., d_model-2]
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引的维度使用正弦值sin, 奇数索引的维度使用余弦值cos
        pe[:, 1::2] = torch.cos(
            position * div_term)  # (max_len, 1) * (1, d_model//2), div_term的形状由(d_model//2,)广播为(1, d_model//2)
        pe = pe.unsqueeze(0).transpose(0, 1)  # 添加维度形状变为 (1, max_len, d_model)，再交换第0维和第1维，最终形状为 (max_len, 1, d_model)。
        self.register_buffer("pe", pe)  # 将位置编码张量 pe 注册为模块的缓冲区。缓冲区不会被视为模型的参数，但会随模型一起保存和加载。

    def forward(self, x):  # x：输入张量，形状为 (seq_len, batch_size, d_model)
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]  # 从预计算的位置编码中提取前 seq_len 个位置的编码-->(seq_len, 1, d_model)
        pe = pe.repeat((1, x.shape[1],
                        1))  # 通过 repeat 方法扩展其第二维（批次维度），将原始的单个位置编码复制 x.shape[1] 次，使其形状与输入 x 相匹配-->(seq_len, batch_size, d_model)
        return pe.detach().clone()  # 返回位置编码的副本，确保不会影响原始的缓冲区张量
        # detach()：分离张量 pe，防止其梯度被追踪或回传。clone()：生成一个新的张量对象


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):  # learnable_w：布尔值，表示频率缩放因子 w 是否可学习
        assert time_dim % 2 == 0, "time_dim must be even!"  # 确保 time_dim 是偶数
        half_dim = int(time_dim // 2)  # 将输入维度 time_dim 一分为二，每一部分用于计算正弦和余弦值。
        super().__init__()
        # w 是频率缩放因子，用于控制正弦和余弦函数的频率
        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()  # w 是形状为 (half_dim,) 的一维张量
        self.register_parameter("w", nn.Parameter(w,
                                                  requires_grad=learnable_w))  # 将 w 注册为模型参数,如果 learnable_w=True，这些缩放因子会在训练中更新

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )  # out_net 是一个两层的全连接网络（使用 SiLU 激活），用于将编码后的时间特征映射到指定的输出维度 out_dim

    def forward(self, x):  # 输入 x 是一维张量，形状为(num_timesteps, )
        assert len(x.shape) == 1, "assumes 1d input timestep array"  # 确保输入 x 是一维张量
        x = x[:, None] * self.w[
            None]  # x[:, None]：将 x 添加一个维度，变成形状为 (num_timesteps, 1);self.w[None]：将 w 添加一个维度，变成形状为 (1, half_dim);最终形状为(num_timesteps, half_dim)
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)  # 分别计算余弦和正弦编码，最后拼接为(num_timesteps, time_dim)
        return self.out_net(x)  # 将编码后的时间特征输入到 out_net 网络中，生成形状为 (num_timesteps, out_dim) 的输出


class _SelfAttnEncoder(nn.Module):
    def __init__(
            self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):  # d_model：输入特征的维度，表示每个时间步的特征向量大小；nhead：注意力机制的头数，多头注意力机制将输入分成 nhead 份并行计算。
        # dim_feedforward：前馈神经网络的隐藏层维度；dropout：Dropout 的概率，用于正则化，防止过拟合。
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model 两层全连接前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 第一层将输入从 d_model 维度投影到 dim_feedforward
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 第二层将其映射回 d_model
        # Layer Normalization（层归一化），用于稳定训练过程和模型性能
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 三个 Dropout 层，分别用于注意力模块和前馈网络
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # 使用用户选择的激活函数，默认为 gelu
        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):  # src: 输入序列张量，形状为 (seq_len, batch_size, d_model)；pos: 位置嵌入张量，形状与 src 相同，用于提供位置信息
        q = k = _with_pos_embed(src, pos)  # 将位置嵌入加到输入序列上（对应位置数值相加）
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)  # 使用多头自注意力模块计算输出，src2 表示自注意力机制后的特征表示，形状不变
        src = src + self.dropout1(src2)  # 残差连接 add
        src = self.norm1(src)  # norm
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(
            src))))  # 前馈神经网络：输入 src 经过第一层全连接层（linear1），激活函数（activation）和 Dropout，后通过第二层全连接层（linear2），输出 src2
        src = src + self.dropout3(src2)  # 再次通过残差连接 add
        src = self.norm2(src)  # norm
        return src  # 形状不变仍为 (seq_len, batch_size, d_model)

    def reset_parameters(self):  # 重新初始化模型参数
        for p in self.parameters():
            if p.dim() > 1:  # 对权重张量（维度大于 1）使用 Xavier 初始化方法，有助于提高模型训练的稳定性
                nn.init.xavier_uniform_(p)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):  # dim：表示输入特征向量的维度
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)  # 全连接层，用于生成缩放因子 γ
        self.shift = nn.Linear(dim, dim)  # 全连接层，用于生成偏移因子 β

    def forward(self, x, c):  # x：主输入张量，形状为 (seq_len, batch_size, d_model);c：条件变量，形状为 (batch_size, dim)
        c = self.act(c)  # 对条件变量 c 应用 SiLU 激活函数
        return x * self.scale(c)[None] + self.shift(c)[
            None]  # scale(c)[None]-->(1, batch_size, dim)；最终输出维度为(seq_len, batch_size, dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)  # 全连接层，用于生成缩放因子 α

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
            self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # create modulation layers
        self.attn_mod1 = _ShiftScaleMod(d_model)
        self.attn_mod2 = _ZeroScaleMod(d_model)
        self.mlp_mod1 = _ShiftScaleMod(d_model)
        self.mlp_mod2 = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        # x：输入序列，形状为 (seq_len, batch_size, d_model)。
        # t：时间步信息，形状为 (batch_size, d_model)。
        # cond：条件信息，形状为 (seq_len, batch_size, d_model)
        # process the conditioning vector first
        cond = torch.mean(cond, axis=0)  # 按时间步求均值，形状变为 (batch_size, d_model)
        cond = cond + t  # 将条件变量和时间步相加，为后续的缩放和偏移操作提供上下文信息（对应位置数值相加）

        x2 = self.attn_mod1(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), cond) + x

        x2 = self.mlp_mod1(self.norm2(x), cond)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), cond)
        return x + x2  # 输出维度不变，为 (seq_len, batch_size, d_model)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class _FinalLayer(nn.Module):  # 负责将输入特征处理为最终的输出，例如生成扩散模型中的噪声预测值或动作矢量。
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # 层归一化，用于稳定训练，减小特征分布的变化
        self.linear = nn.Linear(hidden_size, out_size, bias=True)  # 将输入特征从 hidden_size 投影到 out_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )  # SiLU 激活函数：引入非线性关系；线性层：输出维度为 2 * hidden_size，用于生成缩放（scale）和偏移（shift）因子

    def forward(self, x, t, cond):
        # x：输入序列，形状为 (seq_len, batch_size, hidden_size)。
        # t：时间步信息，形状为 (batch_size, hidden_size)。
        # cond：条件信息，形状为 (seq_len, batch_size, hidden_size)
        # process the conditioning vector first
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2,
                                                         dim=1)  # cond 通过 adaLN_modulation 模块生成 2 * hidden_size 大小的输出
        # 使用 chunk 方法将其分成两部分：shift：偏移量，形状为 (batch_size, hidden_size)。scale：缩放因子，形状为 (batch_size, hidden_size)
        x = x * scale[None] + shift[None]
        x = self.linear(x)  # （seq_len, batch_size, hidden_size）-->(seq_len, batch_size, out_size)
        return x.transpose(0, 1)  # -->(batch_size, seq_len, out_size)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)  # 使用 nn.init.zeros_ 方法将所有参数（包括权重和偏置）初始化为零


class _TransformerEncoder(nn.Module):  # _TransformerEncoder 是一个基于 Transformer 架构的多层编码器模块
    def __init__(self, base_module,
                 num_layers):  # 由多个子模块（base_module=_SelfAttnEncoder）堆叠而成,输入序列会依次通过这些子模块，每个子模块都会生成中间结果
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )
        # 初始化参数
        for l in self.layers:
            l.reset_parameters()

    def forward(self, src, pos):  # src:输入序列张量，(seq_len, batch_size, feature_dim)；pos：位置嵌入张量，形状通常与 src 相同
        x, outputs = src, []  # 初始化 x 为输入 src；定义一个空列表 outputs，用于存储每一层的输出
        for layer in self.layers:  # 逐层调用编码器模块
            x = layer(x, pos)  # 将当前层的输出 x 传递给下一层作为输入
            outputs.append(x)  # 将每一层的输出存入 outputs
        return outputs  # 返回所有层的输出，outputs 是一个列表，包含 num_layers 个张量，每个张量的形状为 (seq_len, batch_size, feature_dim)


class _TransformerDecoder(_TransformerEncoder):  # 继承自 _TransformerEncoder 的解码器类
    def forward(self, src, t, all_conds):  # src：输入序列张量，形状通常为 (seq_len, batch_size, d_model)；t：时间步信息，形状为 (batch_size, d_model)
        # all_conds：条件信息列表，包含每层所需的条件变量。每个条件的形状通常为 (seq_len, batch_size, d_model)
        x = src
        for layer, cond in zip(self.layers, all_conds):
            x = layer(x, t, cond)
        return x


class TransformerForDiffusion(nn.Module):
    def __init__(
            self,
            ac_dim,  # 动作的维度
            ac_chunk,  # 动作分块的数量
            time_dim=256,  # 时间嵌入的维度
            hidden_dim=512,  # 隐藏层维度
            num_blocks=6,  # 编码器和解码器的层数
            dropout=0.1,  # Dropout 的概率
            dim_feedforward=2048,  # 前馈网络隐藏层的大小
            nhead=8,  # 多头注意力机制的头数
            activation="gelu",
    ):
        super().__init__()

        # positional encoding blocks
        self.enc_pos = _PositionalEncoding(hidden_dim)  # enc_pos：用于为输入序列添加位置嵌入
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )  # dec_pos：解码器的初始位置嵌入，形状为 (ac_chunk, 1, hidden_dim)
        nn.init.xavier_uniform_(self.dec_pos.data)  # 初始化

        # input encoder mlps
        self.time_net = _TimeNetwork(time_dim, hidden_dim)  # 将时间步编码为隐藏向量 -->(num_timesteps, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )  # 负责将动作的输入特征投影到隐藏维度

        # encoder blocks 编码器模块
        encoder_module = _SelfAttnEncoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = _TransformerEncoder(encoder_module, num_blocks)

        # decoder blocks 解码器模块
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons 最终预测模块
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        print(
            "number of diffusion parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

        self.cond_pos_emb = None

    def forward(self, noise_actions, time, obs_enc, enc_cache=None):
        # noise_actions：噪声动作输入，形状为 (batch_size, ac_chunk, ac_dim);
        # time：时间步信息，形状为 (batch_size, time_dim);
        # obs_enc：观察信息的编码输入，形状为 (seq_len, batch_size, hidden_dim)

        if enc_cache is None:  # 如果 enc_cache 为 None，调用 forward_enc 方法处理观察信息生成编码缓存
            enc_cache = self.forward_enc(obs_enc)
        return enc_cache, self.forward_dec(noise_actions, time, enc_cache)  # 返回编码缓存和解码器的输出

    def forward_enc(self, obs_enc):  # 编码过程 输入：obs_enc：观察信息，形状为 (seq_len, batch_size, hidden_dim)
        obs_enc = obs_enc.transpose(0, 1)  # --> (batch_size, seq_len, hidden_dim)
        pos = self.enc_pos(obs_enc)  # pos维度为：(batch_size, seq_len, hidden_dim)
        # print("pos:", pos.shape)
        # print("obs_enc:", obs_enc.shape)
        enc_cache = self.encoder(obs_enc, pos)  # 调用_SelfAttnEncoder的forward
        return enc_cache

    def forward_dec(self, noise_actions, time, enc_cache):
        time_enc = self.time_net(time)

        ac_tokens = self.ac_proj(noise_actions)  # 通过 ac_proj 将噪声动作投影到隐藏维度-->(batch_size, ac_chunk, hidden_dim)
        ac_tokens = ac_tokens.transpose(0, 1)  # -->(ac_chunk, batch_size, hidden_dim)
        dec_in = ac_tokens + self.dec_pos  # 与解码器的位置嵌入相加，生成输入 (ac_chunk, batch_size, hidden_dim)

        # apply decoder 使用解码器
        dec_out = self.decoder(dec_in, time_enc, enc_cache)

        # apply final epsilon prediction layer 使用最终预测层
        return self.eps_out(dec_out, time_enc, enc_cache[-1])

# class TransformerForDiffusion(ModuleAttrMixin):
#     def __init__(self,
#             input_dim: int,
#             output_dim: int,
#             horizon: int,
#             n_obs_steps: int = None,
#             cond_dim: int = 0,
#             n_layer: int = 12,
#             n_head: int = 12,
#             n_emb: int = 256,
#             p_drop_emb: float = 0.1,
#             p_drop_attn: float = 0.1,
#             causal_attn: bool=False,
#             time_as_cond: bool=True,
#             obs_as_cond: bool=False,
#             n_cond_layers: int = 0
#         ) -> None:
#         super().__init__()
#
#         # compute number of tokens for main trunk and condition encoder
#         if n_obs_steps is None:
#             n_obs_steps = horizon
#
#         T = horizon
#         T_cond = 1
#         if not time_as_cond:
#             T += 1
#             T_cond -= 1
#         obs_as_cond = cond_dim > 0
#         if obs_as_cond:
#             assert time_as_cond
#             T_cond += n_obs_steps
#
#         # input embedding stem
#         self.input_emb = nn.Linear(input_dim, n_emb)
#         self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
#         self.drop = nn.Dropout(p_drop_emb)
#
#         # cond encoder
#         self.time_emb = SinusoidalPosEmb(n_emb)
#         self.cond_obs_emb = None
#
#         if obs_as_cond:
#             self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
#
#         self.cond_pos_emb = None
#         self.encoder = None
#         self.decoder = None
#         encoder_only = False


        # if T_cond > 0:
        #     self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
        #     if n_cond_layers > 0:
        #         encoder_layer = nn.TransformerEncoderLayer(
        #             d_model=n_emb,
        #             nhead=n_head,
        #             dim_feedforward=4*n_emb,
        #             dropout=p_drop_attn,
        #             activation='gelu',
        #             batch_first=True,
        #             norm_first=True
        #         )
        #         self.encoder = nn.TransformerEncoder(
        #             encoder_layer=encoder_layer,
        #             num_layers=n_cond_layers
        #         )
        #     else:
        #         self.encoder = nn.Sequential(
        #             nn.Linear(n_emb, 4 * n_emb),
        #             nn.Mish(),
        #             nn.Linear(4 * n_emb, n_emb)
        #         )
        #     # decoder
        #     decoder_layer = nn.TransformerDecoderLayer(
        #         d_model=n_emb,
        #         nhead=n_head,
        #         dim_feedforward=4*n_emb,
        #         dropout=p_drop_attn,
        #         activation='gelu',
        #         batch_first=True,
        #         norm_first=True # important for stability
        #     )
        #     self.decoder = nn.TransformerDecoder(
        #         decoder_layer=decoder_layer,
        #         num_layers=n_layer
        #     )
        # else:
        #     # encoder only BERT
        #     encoder_only = True
        #
        #     encoder_layer = nn.TransformerEncoderLayer(
        #         d_model=n_emb,
        #         nhead=n_head,
        #         dim_feedforward=4*n_emb,
        #         dropout=p_drop_attn,
        #         activation='gelu',
        #         batch_first=True,
        #         norm_first=True
        #     )
        #     self.encoder = nn.TransformerEncoder(
        #         encoder_layer=encoder_layer,
        #         num_layers=n_layer
        #     )

        # attention mask
        # if causal_attn:
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
        #     # therefore, the upper triangle should be -inf and others (including diag) should be 0.
        #     sz = T
        #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        #     self.register_buffer("mask", mask)
        #
        #     if time_as_cond and obs_as_cond:
        #         S = T_cond
        #         t, s = torch.meshgrid(
        #             torch.arange(T),
        #             torch.arange(S),
        #             indexing='ij'
        #         )
        #         mask = t >= (s-1) # add one dimension since time is the first token in cond
        #         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        #         self.register_buffer('memory_mask', mask)
        #     else:
        #         self.memory_mask = None
        # else:
        #     self.mask = None
        #     self.memory_mask = None
        #
        # # decoder head
        # self.ln_f = nn.LayerNorm(n_emb)
        # self.head = nn.Linear(n_emb, output_dim)
        #
        # # constants
        # self.T = T
        # self.T_cond = T_cond
        # self.horizon = horizon
        # self.time_as_cond = time_as_cond
        # self.obs_as_cond = obs_as_cond
        # self.encoder_only = encoder_only
        #
        # # init
        # self.apply(self._init_weights)
        # logger.info(
        #     "number of parameters: %e", sum(p.numel() for p in self.parameters())
        # )

    # def _init_weights(self, module):
    #     ignore_types = (nn.Dropout,
    #         SinusoidalPosEmb,
    #         nn.TransformerEncoderLayer,
    #         nn.TransformerDecoderLayer,
    #         nn.TransformerEncoder,
    #         nn.TransformerDecoder,
    #         nn.ModuleList,
    #         nn.Mish,
    #         nn.Sequential)
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.MultiheadAttention):
    #         weight_names = [
    #             'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
    #         for name in weight_names:
    #             weight = getattr(module, name)
    #             if weight is not None:
    #                 torch.nn.init.normal_(weight, mean=0.0, std=0.02)
    #
    #         bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
    #         for name in bias_names:
    #             bias = getattr(module, name)
    #             if bias is not None:
    #                 torch.nn.init.zeros_(bias)
    #     elif isinstance(module, nn.LayerNorm):
    #         torch.nn.init.zeros_(module.bias)
    #         torch.nn.init.ones_(module.weight)
    #     elif isinstance(module, TransformerForDiffusion):
    #         torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
    #         if module.cond_obs_emb is not None:
    #             torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
    #     elif isinstance(module, ignore_types):
    #         # no param
    #         pass
    #     else:
    #         raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention) # 需要权重衰减的模块
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding) # 不需要权重衰减的模块
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif "time_net.w" in fpn or "dec_pos" in fpn:  # Special handling for these parameters
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add("pos_emb")
        # no_decay.add("_dummy_variable")
        # if self.cond_pos_emb is not None:
        #     no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    # def forward(self,
    #     sample: torch.Tensor,
    #     timestep: Union[torch.Tensor, float, int],
    #     cond: Optional[torch.Tensor]=None, **kwargs):
    #     """
    #     x: (B,T,input_dim)
    #     timestep: (B,) or int, diffusion step
    #     cond: (B,T',cond_dim)
    #     output: (B,T,input_dim)
    #     """
    #     # 1. time
    #     timesteps = timestep
    #     if not torch.is_tensor(timesteps):
    #         # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
    #         timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
    #     elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
    #         timesteps = timesteps[None].to(sample.device)
    #     # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    #     timesteps = timesteps.expand(sample.shape[0])
    #     time_emb = self.time_emb(timesteps).unsqueeze(1)
    #     # (B,1,n_emb)
    #
    #     # process input
    #     input_emb = self.input_emb(sample)
    #
    #     if self.encoder_only:
    #         # BERT
    #         token_embeddings = torch.cat([time_emb, input_emb], dim=1)
    #         t = token_embeddings.shape[1]
    #         position_embeddings = self.pos_emb[
    #             :, :t, :
    #         ]  # each position maps to a (learnable) vector
    #         x = self.drop(token_embeddings + position_embeddings)
    #         # (B,T+1,n_emb)
    #         x = self.encoder(src=x, mask=self.mask)
    #         # (B,T+1,n_emb)
    #         x = x[:,1:,:]
    #         # (B,T,n_emb)
    #     else:
    #         # encoder
    #         cond_embeddings = time_emb
    #         if self.obs_as_cond:
    #             cond_obs_emb = self.cond_obs_emb(cond)
    #             # (B,To,n_emb)
    #             cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
    #         tc = cond_embeddings.shape[1]
    #         position_embeddings = self.cond_pos_emb[
    #             :, :tc, :
    #         ]  # each position maps to a (learnable) vector
    #         x = self.drop(cond_embeddings + position_embeddings)
    #         x = self.encoder(x)
    #         memory = x
    #         # (B,T_cond,n_emb)
    #
    #         # decoder
    #         token_embeddings = input_emb
    #         t = token_embeddings.shape[1]
    #         position_embeddings = self.pos_emb[
    #             :, :t, :
    #         ]  # each position maps to a (learnable) vector
    #         x = self.drop(token_embeddings + position_embeddings)
    #         # (B,T,n_emb)
    #         x = self.decoder(
    #             tgt=x,
    #             memory=memory,
    #             tgt_mask=self.mask,
    #             memory_mask=self.memory_mask
    #         )
    #         # (B,T,n_emb)
    #
    #     # head
    #     x = self.ln_f(x)
    #     x = self.head(x)
    #     # (B,T,n_out)
    #     return x



