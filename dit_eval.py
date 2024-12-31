import sys
import time
from calendar import EPOCH


# 设置标准输出和标准错误的缓冲方式为行缓冲，这样输出会在每行结束时刷新
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import h5py
import matplotlib.pyplot as plt
from einops import rearrange
import argparse  # 确保导入了 argparse
from RLBench_ACT.act.sim_env_rlbench import make_sim_env
from diffusion_policy.common.pytorch_util import dict_apply
import os
import pathlib
import click  # 用于命令行参数解析
import hydra  # 用于配置管理和依赖注入
import torch  # 用于加载和操作 PyTorch 模型
import dill  # 用于 pickle 操作（比标准 pickle 支持更多对象类型）
import wandb  # 用于日志和监控
import json  # 用于处理 JSON 数据
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # 导入基类，代表工作空间
import click
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict)
import numpy as np
from RLBench_ACT.act.constants import DT
import time

def get_image(ts):
    curr_images = []

    # 获取 wrist_rgb 图像

    wrist_rgb = ts.wrist_rgb
    curr_image = rearrange(wrist_rgb, 'h w c -> c h w')
    curr_images.append(curr_image)

    # 获取 head_rgb 图像

    head_rgb = ts.head_rgb
    curr_image = rearrange(head_rgb, 'h w c -> c h w')
    curr_images.append(curr_image)

    # 组合所有的图像
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    return curr_image


def main(checkpoint, ckpt_name0, device, task_name, robot_name, epochs=50, onscreen_render=False, variation=2, save_episode=True):

    # 检查输出目录是否存在，若存在则询问是否覆盖
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    #
    # # 创建输出目录
    # pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型检查点文件
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)  # 使用 dill 读取 pickle 文件
    cfg = payload['cfg']  # 从加载的 payload 中提取配置
    print(dir(cfg))
    cls = hydra.utils.get_class(cfg._target_)  # 获取配置中指定的类
    workspace = cls(cfg)  # 实例化工作空间对象
    workspace: BaseWorkspace  # 确保 workspace 是 BaseWorkspace 类型
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)  # 加载模型的 payload 数据到工作空间中

    policy = workspace.model
    print(policy)
    if cfg.training.use_ema:  # 如果使用了 EMA（Exponential Moving Average），则使用 EMA 模型
        policy = workspace.ema_model

    # 将模型移到指定的设备（GPU 或 CPU）
    device = torch.device(device)
    policy.eval().to(device)  # 设置模型为评估模式，并转移到 GPU

    policy.num_inference_steps = 100  # 设置 DDIM 推理步骤数
    policy.n_action_steps = 8  # 设置动作步骤数

    print("n_action_steps", policy.n_action_steps)
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps  # 获取观察步数
    print("n_obs_steps: ", n_obs_steps)  # 打印观察步数
    print("obs_res:", obs_res)

    env = make_sim_env(task_name, onscreen_render, robot_name)  # 创建模拟环境
    env_max_reward = 1  # 设置最大奖励（模拟环境）
    episode_returns = []  # 记录每回合的总奖励
    highest_rewards = []  # 记录每回合的最大奖励

    max_timesteps = 50

    # 定义预处理和后处理函数
    ckpt_dir = "/home/wxy/Diffusion/diffusion_policy/data/outputs/2024.12.30/20.42.37_train_diffusion_transformer_hybrid_reach_target/checkpoints"


    for epoch in range(epochs):
        num_rollouts = epochs
        rollout_id = epoch
        # 设置环境的任务变种
        # if variation >= 0:
        #     env.set_variation(variation)  # 使用指定的变种
        # else:
        #     random_variation = np.random.randint(3)  # 随机选择一个变种
        #     env.set_variation(random_variation)

        # descriptions, ts_obs = env.reset()  # 重置环境，获取初始观测
        try:
                    descriptions, ts_obs = env.reset()
                    time.sleep(1)  # 等待1000秒
        except Exception as e:
                    print(f"An error occurred: {e}")

        print(dir(ts_obs))
        print(epoch)

        # with h5py.File("/home/mar/RLBench_ACT/Data3/reach_target/variation0/merged_data.hdf5") as file:
        #     # count total steps
        #     demos = file['data']  # 打开 HDF5 文件并读取数据
        #     episode_ends = list()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        #     prev_end = 0
        #     demo = demos[f'demo_0']
        #     actions = demo['action']
        #     print("gripper_pose:"+str(ts_obs.gripper_pose))
        #     print(str(ts_obs.gripper_open))
        #
        #     # env.step(actions[0])
        #     for i in range(demo['action'].shape[0]):
        #         print(f"action{i}:" + str(actions[i]))
        #         # env.step(actions[i])

        ### 评估循环
        image_list = []  # 用于可视化图像的列表
        qpos_list = []  # 记录关节位置的列表
        target_qpos_list = []  # 记录目标关节位置的列表
        rewards = []  # 记录奖励的列表
        t = 0  # 时间步计数器

        with torch.no_grad():

            policy.reset()
            # t = 0
            terminate = False
            while not terminate and t < max_timesteps:
                obs = ts_obs  # 获取当前观察
                image_list.append({'front': obs.front_rgb, 'head': obs.head_rgb, 'wrist': obs.wrist_rgb})  # 保存图像
                import numpy as np

                qpos_list.append({'qpos': np.append(obs.joint_positions, obs.gripper_open)})

                obs.joint_positions = np.append(obs.joint_positions, obs.gripper_open)

                # 获取所有包含 'front' 键的字典数量


                # 假设你已经有了一个空字典来存储图像
                import numpy as np

                # 假设你已经有了一个空字典来存储图像
                # 在初始化时指定形状，确保拼接时的形状匹配
                observation = {
                    'wrist': np.empty((0, obs.wrist_rgb.shape[0], obs.wrist_rgb.shape[1], obs.wrist_rgb.shape[2]),
                                      dtype=np.uint8),
                    # 假设 wrist_rgb 是 (height, width, channels)，直接设定 dtype 为 uint8
                    'head': np.empty((0, obs.front_rgb.shape[0], obs.front_rgb.shape[1], obs.front_rgb.shape[2]),
                                     dtype=np.uint8),

                    # # 如果你想创建一个与 joint_positions 相同维度的空数组
                    'qpos': np.empty((0,) + obs.joint_positions.shape , dtype=np.float32),
                    # # 假设 front_rgb 也是 (height, width, channels)，直接设定 dtype 为 uint8
                }
                # print (t)
                if t == 0:
                    # 直接在拼接时将图像转换为 uint8 类型
                    observation["head"] = np.concatenate(
                        [observation["head"], obs.front_rgb[np.newaxis, ...].astype(np.uint8)], axis=0)
                    observation["wrist"] = np.concatenate(
                        [observation["wrist"], obs.wrist_rgb[np.newaxis, ...].astype(np.uint8)], axis=0)
                    observation["qpos"] = np.concatenate([observation["qpos"], obs.joint_positions[np.newaxis, ...]], axis=0)

                    # 直接在拼接时将图像转换为 uint8 类型
                    observation["head"] = np.concatenate(
                        [observation["head"], obs.front_rgb[np.newaxis, ...].astype(np.uint8)], axis=0)
                    observation["wrist"] = np.concatenate(
                        [observation["wrist"], obs.wrist_rgb[np.newaxis, ...].astype(np.uint8)], axis=0)
                    observation["qpos"] = np.concatenate([observation["qpos"], obs.joint_positions[np.newaxis, ...]], axis=0)



                    # # 打印图像的内容（可以用于调试）
                    # print(rgb_images["head"][0])
                    # print(rgb_images["wrist"][0])
                    #
                    # # 显示头部图像
                    # head_image = rgb_images["head"][0]
                    # image = Image.fromarray(head_image)
                    # plt.axis('off')  # 关闭坐标轴显示
                    # plt.imshow(image)
                    # plt.show()

                    # # 如果你也想显示手腕图像，可以按如下方式显示：
                    # wrist_image = rgb_images["wrist"][0]
                    # wrist_image_pil = Image.fromarray(wrist_image)
                    # plt.axis('off')
                    # plt.imshow(wrist_image_pil)
                    # plt.show()

                    # 设置终止条件
                    # terminate = True
                else:
                    last_image = image_list[-1]  # 倒数第一个元素
                    second_last_image = image_list[-2]  # 倒数第二个元素

                    last_qpos = qpos_list[-1]
                    second_last_qpos = qpos_list[-2]

                    observation['wrist'] = np.concatenate(
                        [observation['wrist'], second_last_image['wrist'][np.newaxis, ...]], axis=0)
                    observation['wrist'] = np.concatenate([observation['wrist'], last_image['wrist'][np.newaxis, ...]],
                                                         axis=0)


                    observation['head'] = np.concatenate(
                        [observation['head'], second_last_image['front'][np.newaxis, ...]], axis=0)
                    observation['head'] = np.concatenate([observation['head'], last_image['front'][np.newaxis, ...]],
                                                        axis=0)

                    observation['qpos'] = np.concatenate([observation['qpos'], second_last_qpos['qpos'][np.newaxis, ...]], axis=0)
                    observation['qpos'] = np.concatenate([observation['qpos'], last_qpos['qpos'][np.newaxis, ...]],axis=0)

                from PIL import Image

                # # 显示头部图像
                # head_image = observation["head"][0]
                # # head_image = observation["wrist"][0]
                # image = Image.fromarray(head_image)  # 正确
                # plt.axis('off')  # 关闭坐标轴显示
                # plt.imshow(image)
                # plt.show()

                # 打印图像的形状
                # print(observation["wrist"].shape)  # 输出 wrist 的形状
                # print(observation["head"].shape)  # 输出 head 的形状

                # file_path = '/home/wxy/diffusion_policy/data/dataset/reach_target/merged_data.hdf5'
                # with h5py.File(file_path, 'r') as f:
                #     # 获取 demo_0 组
                #     demo_0 = f['data']['demo_0']

                #     # 获取 action 数据集
                #     action = demo_0['action'][:3]  # 获取前 3 个 action (形状: (3, 7))

                #     # 获取 obs 数据集
                #     obs_1 = demo_0['obs']

                #     # 获取 head, wrist 和 qpos 数据
                #     head_images = obs_1['head'][:2]  # 获取前 2 张 head 图片 (形状: (2, 128, 128, 3))
                #     wrist_images = obs_1['wrist'][:2]  # 获取前 2 张 wrist 图片 (形状: (2, 128, 128, 3))
                #     qpos = obs_1['qpos'][:2]  # 获取前 2 个 qpos (形状: (2, 7))

                # # 打印提取的数据
                # print("前 3 个 action:", action)
                # print("前 2 张 head 图片:", head_images.shape)
                # print("前 2 张 wrist 图片:", wrist_images.shape)
                # print("前 2 个 qpos:", qpos)

                # # 可视化前 2 张 head 和 wrist 图片
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                #
                # # 显示 head 图片
                # axes[0].imshow(head_images[0])  # 显示第一张 head 图片
                # axes[0].set_title("Head Image 1")
                # axes[0].axis('off')
                #
                # axes[1].imshow(head_images[1])  # 显示第二张 head 图片
                # axes[1].set_title("Head Image 2")
                # axes[1].axis('off')

                # plt.show()
                #
                #
                # # 可视化 wrist 图片
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                #
                # # 显示 wrist 图片
                # axes[0].imshow(wrist_images[0])  # 显示第一张 wrist 图片
                # axes[0].set_title("Wrist Image 1")
                # axes[0].axis('off')
                #
                # axes[1].imshow(wrist_images[1])  # 显示第二张 wrist 图片
                # axes[1].set_title("Wrist Image 2")
                # axes[1].axis('off')
                #
                # plt.show()
                # new_observation = {
                #     'wrist': np.empty((0, obs.wrist_rgb.shape[0], obs.wrist_rgb.shape[1], obs.wrist_rgb.shape[2]),
                #                       dtype=np.uint8),
                #     # 假设 wrist_rgb 是 (height, width, channels)，直接设定 dtype 为 uint8
                #     'head': np.empty((0, obs.front_rgb.shape[0], obs.front_rgb.shape[1], obs.front_rgb.shape[2]),
                #                      dtype=np.uint8),

                #     # # 如果你想创建一个与 joint_positions 相同维度的空数组
                #     'qpos': np.empty((0,) + obs.joint_positions.shape , dtype=np.float32),
                #     # # 假设 front_rgb 也是 (height, width, channels)，直接设定 dtype 为 uint8
                # }


                # new_observation['wrist'] = np.concatenate(
                #     [observation['wrist'], wrist_images[0][np.newaxis, ...]], axis=0)
                # new_observation['wrist'] = np.concatenate(
                #     [observation['wrist'], wrist_images[1][np.newaxis, ...]], axis=0)

                # new_observation['head'] = np.concatenate(
                #     [new_observation['head'], head_images[0][np.newaxis, ...]], axis=0
                # )
                # new_observation['head'] = np.concatenate(
                #     [new_observation['head'], head_images[1][np.newaxis, ...]], axis=0
                # )

                # new_observation['qpos'] = np.concatenate(
                #     [new_observation['qpos'], qpos[0][np.newaxis, ...]], axis=0
                # )
                # new_observation['qpos'] = np.concatenate(
                #     [new_observation['qpos'], qpos[1][np.newaxis, ...]], axis=0
                # )

                # 将环境的观察数据（`obs`）转换为模型所需要的输入格式
                obs_dict_np = get_real_obs_dict(
                    env_obs=observation, shape_meta=cfg.task.shape_meta)




                # 将环境的原始观察数据（`obs`）转换为符合模型要求的字典格式，形状元数据从配置中提取。
                # 使用字典映射，将观察数据转换为 PyTorch 张量，并将其移到指定的设备（GPU 或 CPU）。
                obs_dict = dict_apply(obs_dict_np,
                                      lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                # `dict_apply` 函数用于将 numpy 数组转换为张量，并添加批量维度。



                # 使用策略模型推理动作
                result = policy.predict_action(obs_dict)
                # 使用当前的观察数据输入到策略模型中，返回动作结果。

                action = result['action_pred'][0].detach().to('cpu').numpy()
                # print("action", action)
                # 获取模型输出的动作数据，`.detach()` 用于移除计算图，`to('cpu')` 将张量转移到 CPU，
                # `numpy()` 用于转换为 NumPy 数组，方便后续处理。
                # print(action)
                for i in range(8):
                    action_first_six = action[i]
                # 发送动作到环境并获取新的状态和奖励
                    # if action_first_six[6] < 0.1:
                    #     action_first_six[6] = 1.0
                    ts_obs, reward, terminate = env.step(action_first_six)
                    obs = ts_obs
                    image_list.append(
                        {'front': obs.front_rgb, 'head': obs.head_rgb, 'wrist': obs.wrist_rgb})  # 保存图像
                    qpos_list.append({'qpos': np.append(obs.joint_positions, obs.gripper_open)})
                    rewards.append(reward)  # 记录奖励



                if reward == env_max_reward:
                    break  # 如果获得最大奖励，直接跳出循环（成功完成任务)

                t = t + 1  # 增加时间步计数器

                # plt.close()  # 关闭图形窗口（如果有可视化的话）
            if t >= 100:
                break

        # 记录每个回合的奖励信息
        rewards = np.array(rewards)  # 将奖励列表转换为NumPy数组
        episode_return = np.sum(rewards[rewards != None])  # 计算本回合的总奖励，忽略 None 值
        episode_returns.append(episode_return)  # 将本回合的总奖励添加到 episode_returns 列表中

        # 计算本回合的最大奖励
        episode_highest_reward = np.max(rewards)  # 获取本回合的最大奖励
        highest_rewards.append(episode_highest_reward)  # 将最大奖励添加到 highest_rewards 列表中

            # 打印当前回合的评估信息（这部分被注释掉了）
            # print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
            # print(f'{rollout_id} Rollout with {t} steps for [{descriptions[0]}]: {episode_highest_reward==env_max_reward}')

            # 如果是非首轮（rollout_id != 0），且需要保存视频
            # if (epoch):
            #     if save_episode:  # 如果设置了保存回合视频
            #         # 保存回合的视频文件，文件名包含模型检查点名、回合ID和是否成功完成任务的标记
            #         save_videos(image_list, DT, video_path=os.path.join(ckpt_dir,
            #                                                             f'video_{ckpt_name0}_{epoch}_{episode_highest_reward == env_max_reward}.mp4'))

        # 计算成功率：统计最大奖励等于环境的最大奖励的比例
        success_rate = np.mean(np.array(highest_rewards) == env_max_reward)

        # 计算平均回报：所有回合的总奖励的平均值
        avg_return = np.mean(episode_returns)

        # 构建摘要字符串，输出成功率和平均回报
        summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'

        # 针对每个奖励值，从0到env_max_reward，统计获得至少该奖励值的回合数和比例
        for r in range(env_max_reward + 1):
            more_or_equal_r = (np.array(highest_rewards) >= r).sum()  # 获得至少r奖励的回合数
            more_or_equal_r_rate = more_or_equal_r / num_rollouts  # 获得至少r奖励的回合比例
            summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n'

        # 打印回合的统计摘要
        print(summary_str)

        # 保存成功率和相关统计信息到文本文件
        result_file_name = 'result_' + ckpt_name0 + f'({more_or_equal_r_rate * 100}%).txt'
        with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
            f.write(summary_str)  # 写入统计摘要
            # f.write(repr(episode_returns))  # 如果需要，可以保存所有回合的奖励列表（这行代码被注释掉了）
            f.write('\n\n')
            f.write(repr(highest_rewards))  # 保存最大奖励列表
            print("t", t+1)
        # 返回成功率和平均回报
    return success_rate, avg_return


# %%
if __name__ == '__main__':
    ckpt_dir = "/home/rookie/桌面/diffusion_policy/data/outputs/2024.12.31/17.37.51_train_diffusion_transformer_hybrid_reach_target/checkpoints/latest.ckpt"
    checkpoint = ckpt_dir
    ckpt_name0 = "latest.ckpt"
    task_name = "reach_target"
    robot_name = "ur5"
    device = "cuda:0"
    main(checkpoint, ckpt_name0, device, task_name, robot_name, epochs=50, onscreen_render=True, variation=0)