#!/bin/sh
#
# Created by constructor 3.10.0
#
# NAME:  Mambaforge
# VER:   24.9.2-0
# PLAT:  linux-64
# MD5:   c7d793f612fa1d406af23ca87bc46d9a

set -eu

export OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
unset LD_LIBRARY_PATH
if ! echo "$0" | grep '\.sh$' > /dev/null; then
    printf 'Please run using "bash"/"dash"/"sh"/"zsh", but not "." or "source".\n' >&2
    return 1
fi

min_glibc_version="2.17"
system_glibc_version="${CONDA_OVERRIDE_GLIBC:-}"
if [ "${system_glibc_version}" = "" ]; then
    case "$(ldd --version 2>&1)" in
        *musl*)
            # musl ldd will report musl version; call libc.so directly
            # see https://github.com/conda/constructor/issues/850#issuecomment-2343756454
            libc_so="$(find /lib /usr/local/lib /usr/lib -name 'libc.so.*' -print -quit 2>/dev/null)"
            if [ -z "${libc_so}" ]; then
                libc_so="$(strings /etc/ld.so.cache | grep '^/.*/libc\.so.*' | head -1)"
            fi
            if [ -z "${libc_so}" ]; then
                echo "Warning: Couldn't find libc.so; won't be able to determine GLIBC version!" >&2
                echo "Override by setting CONDA_OVERRIDE_GLIBC" >&2
                system_glibc_version="0.0"
            else
                system_glibc_version=$("${libc_so}" --version | awk 'NR==1{ sub(/\.$/, ""); print $NF}')
            fi
        ;;
        *)
            # ldd reports glibc in the last field of the first line
            system_glibc_version=$(ldd --version | awk 'NR==1{print $NF}')
        ;;
    esac
fi
# shellcheck disable=SC2183 disable=SC2046
int_min_glibc_version="$(printf "%02d%02d%02d" $(echo "$min_glibc_version" | sed 's/\./ /g'))"
# shellcheck disable=SC2183 disable=SC2046
int_system_glibc_version="$(printf "%02d%02d%02d" $(echo "$system_glibc_version" | sed 's/\./ /g'))"
if [ "$int_system_glibc_version" -lt "$int_min_glibc_version" ]; then
    echo "Installer requires GLIBC >=${min_glibc_version}, but system has ${system_glibc_version}."
    exit 1
fi
# Export variables to make installer metadata available to pre/post install scripts
# NOTE: If more vars are added, make sure to update the examples/scripts tests too

  # Templated extra environment variable(s)
export INSTALLER_NAME='Mambaforge'
export INSTALLER_VER='24.9.2-0'
export INSTALLER_PLAT='linux-64'
export INSTALLER_TYPE="SH"

THIS_DIR=$(DIRNAME=$(dirname "$0"); cd "$DIRNAME"; pwd)
THIS_FILE=$(basename "$0")
THIS_PATH="$THIS_DIR/$THIS_FILE"
PREFIX="${HOME:-/opt}/mambaforge"
BATCH=0
FORCE=0
KEEP_PKGS=1
SKIP_SCRIPTS=0
TEST=0
REINSTALL=0
USAGE="
usage: $0 [options]

Installs ${INSTALLER_NAME} ${INSTALLER_VER}

-b           run install in batch mode (without manual intervention),
             it is expected the license terms (if any) are agreed upon
-f           no error if install prefix already exists
-h           print this help message and exit
-p PREFIX    install prefix, defaults to $PREFIX, must not contain spaces.
-s           skip running pre/post-link/install scripts
-u           update an existing installation
-t           run package tests after installation (may install conda-build)
"

# We used to have a getopt version here, falling back to getopts if needed
# However getopt is not standardized and the version on Mac has different
# behaviour. getopts is good enough for what we need :)
# More info: https://unix.stackexchange.com/questions/62950/
while getopts "bifhkp:sut" x; do
    case "$x" in
        h)
            printf "%s\\n" "$USAGE"
            exit 2
        ;;
        b)
            BATCH=1
            ;;
        i)
            BATCH=0
            ;;
        f)
            FORCE=1
            ;;
        k)
            KEEP_PKGS=1
            ;;
        p)
            PREFIX="$OPTARG"
            ;;
        s)
            SKIP_SCRIPTS=1
            ;;
        u)
            FORCE=1
            ;;
        t)
            TEST=1
            ;;
        ?)
            printf "ERROR: did not recognize option '%s', please try -h\\n" "$x"
            exit 1
            ;;
    esac
done

# For pre- and post-install scripts
export INSTALLER_UNATTENDED="$BATCH"

# For testing, keep the package cache around longer
CLEAR_AFTER_TEST=0
if [ "$TEST" = "1" ] && [ "$KEEP_PKGS" = "0" ]; then
    CLEAR_AFTER_TEST=1
    KEEP_PKGS=1
fi

if [ "$BATCH" = "0" ] # interactive mode
then
    if [ "$(uname -m)" != "x86_64" ]; then
        printf "WARNING:\\n"
        printf "    Your operating system appears not to be 64-bit, but you are trying to\\n"
        printf "    install a 64-bit version of %s.\\n" "${INSTALLER_NAME}"
        printf "    Are sure you want to continue the installation? [yes|no]\\n"
        printf "[no] >>> "
        read -r ans
        ans=$(echo "${ans}" | tr '[:lower:]' '[:upper:]')
        if [ "$ans" != "YES" ] && [ "$ans" != "Y" ]
        then
            printf "Aborting installation\\n"
            exit 2
        fi
    fi
    if [ "$(uname)" != "Linux" ]; then
        printf "WARNING:\\n"
        printf "    Your operating system does not appear to be Linux, \\n"
        printf "    but you are trying to install a Linux version of %s.\\n" "${INSTALLER_NAME}"
        printf "    Are sure you want to continue the installation? [yes|no]\\n"
        printf "[no] >>> "
        read -r ans
        ans=$(echo "${ans}" | tr '[:lower:]' '[:upper:]')
        if [ "$ans" != "YES" ] && [ "$ans" != "Y" ]
        then
            printf "Aborting installation\\n"
            exit 2
        fi
    fi
    printf "\\n"
    printf "Welcome to %s %s\\n" "${INSTALLER_NAME}" "${INSTALLER_VER}"
    printf "\\n"
    printf "In order to continue the installation process, please review the license\\n"
    printf "agreement.\\n"
    printf "Please, press ENTER to continue\\n"
    printf ">>> "
    read -r dummy
    pager="cat"
    if command -v "more" > /dev/null 2>&1; then
      pager="more"
    fi
    "$pager" <<'EOF'
!!!!!! Mambaforge is now deprecated !!!!!
Future Miniforge releases will NOT build Mambaforge installers.
We advise you switch to Miniforge at your earliest convenience.
More details at https://conda-forge.org/news/2024/07/29/sunsetting-mambaforge/.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Miniforge installer code uses BSD-3-Clause license as stated below.

Binary packages that come with it have their own licensing terms
and by installing miniforge you agree to the licensing terms of individual
packages as well. They include different OSI-approved licenses including
the GNU General Public License and can be found in pkgs/<pkg-name>/info/licenses
folders.

Miniforge installer comes with a bootstrapping executable that is used
when installing miniforge and is deleted after miniforge is installed.
The bootstrapping executable uses micromamba, cli11, cpp-filesystem,
curl, c-ares, krb5, libarchive, libev, lz4, nghttp2, openssl, libsolv,
nlohmann-json, reproc and zstd which are licensed under BSD-3-Clause,
MIT and OpenSSL licenses. Licenses and copyright notices of these
projects can be found at the following URL.
https://github.com/conda-forge/micromamba-feedstock/tree/master/recipe.

=============================================================================

Copyright (c) 2019-2022, conda-forge
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

EOF
    printf "\\n"
    printf "Do you accept the license terms? [yes|no]\\n"
    printf ">>> "
    read -r ans
    ans=$(echo "${ans}" | tr '[:lower:]' '[:upper:]')
    while [ "$ans" != "YES" ] && [ "$ans" != "NO" ]
    do
        printf "Please answer 'yes' or 'no':'\\n"
        printf ">>> "
        read -r ans
        ans=$(echo "${ans}" | tr '[:lower:]' '[:upper:]')
    done
    if [ "$ans" != "YES" ]
    then
        printf "The license agreement wasn't approved, aborting installation.\\n"
        exit 2
    fi
    printf "\\n"
    printf "%s will now be installed into this location:\\n" "${INSTALLER_NAME}"
    printf "%s\\n" "$PREFIX"
    printf "\\n"
    printf "  - Press ENTER to confirm the location\\n"
    printf "  - Press CTRL-C to abort the installation\\n"
    printf "  - Or specify a different location below\\n"
    printf "\\n"
    printf "[%s] >>> " "$PREFIX"
    read -r user_prefix
    if [ "$user_prefix" != "" ]; then
        case "$user_prefix" in
            *\ * )
                printf "ERROR: Cannot install into directories with spaces\\n" >&2
                exit 1
                ;;
            *)
                eval PREFIX="$user_prefix"
                ;;
        esac
    fi
fi # !BATCH

case "$PREFIX" in
    *\ * )
        printf "ERROR: Cannot install into directories with spaces\\n" >&2
        exit 1
        ;;
esac
if [ "$FORCE" = "0" ] && [ -e "$PREFIX" ]; then
    printf "ERROR: File or directory already exists: '%s'\\n" "$PREFIX" >&2
    printf "If you want to update an existing installation, use the -u option.\\n" >&2
    exit 1
elif [ "$FORCE" = "1" ] && [ -e "$PREFIX" ]; then
    REINSTALL=1
fi

total_installation_size_kb="624029"
total_installation_size_mb="$(( total_installation_size_kb / 1024 ))"
if ! mkdir -p "$PREFIX"; then
    printf "ERROR: Could not create directory: '%s'.\\n" "$PREFIX" >&2
    printf "Check permissions and available disk space (%s MB needed).\\n" "$total_installation_size_mb" >&2
    exit 1
fi

free_disk_space_kb="$(df -Pk "$PREFIX" | tail -n 1 | awk '{print $4}')"
free_disk_space_kb_with_buffer="$((free_disk_space_kb - 50 * 1024))"  # add 50MB of buffer
if [ "$free_disk_space_kb_with_buffer" -lt "$total_installation_size_kb" ]; then
    printf "ERROR: Not enough free disk space. Only %s MB are available, but %s MB are required (leaving a 50 MB buffer).\\n" \
        "$((free_disk_space_kb_with_buffer / 1024))" "$total_installation_size_mb" >&2
    exit 1
fi

# pwd does not convert two leading slashes to one
# https://github.com/conda/constructor/issues/284
PREFIX=$(cd "$PREFIX"; pwd | sed 's@//@/@')
export PREFIX

printf "PREFIX=%s\\n" "$PREFIX"

# 3-part dd from https://unix.stackexchange.com/a/121798/34459
# Using a larger block size greatly improves performance, but our payloads
# will not be aligned with block boundaries. The solution is to extract the
# bulk of the payload with a larger block size, and use a block size of 1
# only to extract the partial blocks at the beginning and the end.
extract_range () {
    # Usage: extract_range first_byte last_byte_plus_1
    blk_siz=16384
    dd1_beg=$1
    dd3_end=$2
    dd1_end=$(( ( dd1_beg / blk_siz + 1 ) * blk_siz ))
    dd1_cnt=$(( dd1_end - dd1_beg ))
    dd2_end=$(( dd3_end / blk_siz ))
    dd2_beg=$(( ( dd1_end - 1 ) / blk_siz + 1 ))
    dd2_cnt=$(( dd2_end - dd2_beg ))
    dd3_beg=$(( dd2_end * blk_siz ))
    dd3_cnt=$(( dd3_end - dd3_beg ))
    dd if="$THIS_PATH" bs=1 skip="${dd1_beg}" count="${dd1_cnt}" 2>/dev/null
    dd if="$THIS_PATH" bs="${blk_siz}" skip="${dd2_beg}" count="${dd2_cnt}" 2>/dev/null
    dd if="$THIS_PATH" bs=1 skip="${dd3_beg}" count="${dd3_cnt}" 2>/dev/null
}

# the line marking the end of the shell header and the beginning of the payload
last_line=$(grep -anm 1 '^@@END_HEADER@@' "$THIS_PATH" | sed 's/:.*//')
# the start of the first payload, in bytes, indexed from zero
boundary0=$(head -n "${last_line}" "${THIS_PATH}" | wc -c | sed 's/ //g')
# the start of the second payload / the end of the first payload, plus one
boundary1=$(( boundary0 + 14586008 ))
# the end of the second payload, plus one
boundary2=$(( boundary1 + 63528960 ))

# verify the MD5 sum of the tarball appended to this header
MD5=$(extract_range "${boundary0}" "${boundary2}" | md5sum -)
if ! echo "$MD5" | grep c7d793f612fa1d406af23ca87bc46d9a >/dev/null; then
    printf "WARNING: md5sum mismatch of tar archive\\n" >&2
    printf "expected: c7d793f612fa1d406af23ca87bc46d9a\\n" >&2
    printf "     got: %s\\n" "$MD5" >&2
fi

cd "$PREFIX"

# disable sysconfigdata overrides, since we want whatever was frozen to be used
unset PYTHON_SYSCONFIGDATA_NAME _CONDA_PYTHON_SYSCONFIGDATA_NAME

# the first binary payload: the standalone conda executable
CONDA_EXEC="$PREFIX/_conda"
extract_range "${boundary0}" "${boundary1}" > "$CONDA_EXEC"
chmod +x "$CONDA_EXEC"

export TMP_BACKUP="${TMP:-}"
export TMP="$PREFIX/install_tmp"
mkdir -p "$TMP"

# Check whether the virtual specs can be satisfied
# We need to specify CONDA_SOLVER=classic for conda-standalone
# to work around this bug in conda-libmamba-solver:
# https://github.com/conda/conda-libmamba-solver/issues/480
# micromamba needs an existing pkgs_dir to operate even offline,
# but we haven't created $PREFIX/pkgs yet... give it a temp location
# shellcheck disable=SC2050
if [ "" != "" ]; then
    echo "Checking virtual specs compatibility:" 
    CONDA_QUIET="$BATCH" \
    CONDA_SOLVER="classic" \
    CONDA_PKGS_DIRS="$(mktemp -d)" \
    "$CONDA_EXEC" create --dry-run --prefix "$PREFIX/envs/_virtual_specs_checks" --offline  --no-rc
fi

# Create $PREFIX/.nonadmin if the installation didn't require superuser permissions
if [ "$(id -u)" -ne 0 ]; then
    touch "$PREFIX/.nonadmin"
fi

# the second binary payload: the tarball of packages
printf "Unpacking payload ...\n"
extract_range $boundary1 $boundary2 | \
    CONDA_QUIET="$BATCH" "$CONDA_EXEC" constructor --extract-tarball --prefix "$PREFIX"

PRECONDA="$PREFIX/preconda.tar.bz2"
CONDA_QUIET="$BATCH" \
"$CONDA_EXEC" constructor --prefix "$PREFIX" --extract-tarball < "$PRECONDA" || exit 1
rm -f "$PRECONDA"

CONDA_QUIET="$BATCH" \
"$CONDA_EXEC" constructor --prefix "$PREFIX" --extract-conda-pkgs || exit 1

#The templating doesn't support nested if statements
if [ "$SKIP_SCRIPTS" = "1" ]; then
    export INST_OPT='--skip-scripts'
    printf "WARNING: skipping pre_install.sh by user request\\n" >&2
else
    export INST_OPT=''
    if ! "$PREFIX/pkgs/pre_install.sh"; then
        printf "ERROR: executing pre_install.sh failed\\n" >&2
        exit 1
    fi
fi
MSGS="$PREFIX/.messages.txt"
touch "$MSGS"
export FORCE

# original issue report:
# https://github.com/ContinuumIO/anaconda-issues/issues/11148
# First try to fix it (this apparently didn't work; QA reported the issue again)
# https://github.com/conda/conda/pull/9073
# Avoid silent errors when $HOME is not writable
# https://github.com/conda/constructor/pull/669
test -d ~/.conda || mkdir -p ~/.conda >/dev/null 2>/dev/null || test -d ~/.conda || mkdir ~/.conda

printf "\nInstalling base environment...\n\n"

shortcuts=""
# shellcheck disable=SC2086
CONDA_ROOT_PREFIX="$PREFIX" \
CONDA_REGISTER_ENVS="true" \
CONDA_SAFETY_CHECKS=disabled \
CONDA_EXTRA_SAFETY_CHECKS=no \
CONDA_CHANNELS="conda-forge" \
CONDA_PKGS_DIRS="$PREFIX/pkgs" \
CONDA_QUIET="$BATCH" \
"$CONDA_EXEC" install --offline --file "$PREFIX/pkgs/env.txt" -yp "$PREFIX" $shortcuts --no-rc || exit 1
rm -f "$PREFIX/pkgs/env.txt"

#The templating doesn't support nested if statements
mkdir -p "$PREFIX/envs"
for env_pkgs in "${PREFIX}"/pkgs/envs/*/; do
    env_name=$(basename "${env_pkgs}")
    if [ "$env_name" = "*" ]; then
        continue
    fi
    printf "\nInstalling %s environment...\n\n" "${env_name}"
    mkdir -p "$PREFIX/envs/$env_name"

    if [ -f "${env_pkgs}channels.txt" ]; then
        env_channels=$(cat "${env_pkgs}channels.txt")
        rm -f "${env_pkgs}channels.txt"
    else
        env_channels="conda-forge"
    fi
    env_shortcuts=""
    # shellcheck disable=SC2086
    CONDA_ROOT_PREFIX="$PREFIX" \
    CONDA_REGISTER_ENVS="true" \
    CONDA_SAFETY_CHECKS=disabled \
    CONDA_EXTRA_SAFETY_CHECKS=no \
    CONDA_CHANNELS="$env_channels" \
    CONDA_PKGS_DIRS="$PREFIX/pkgs" \
    CONDA_QUIET="$BATCH" \
    "$CONDA_EXEC" install --offline --file "${env_pkgs}env.txt" -yp "$PREFIX/envs/$env_name" $env_shortcuts --no-rc || exit 1
    rm -f "${env_pkgs}env.txt"
done
# ----- add condarc
cat <<EOF >"$PREFIX/.condarc"
channels:
  - conda-forge
EOF

POSTCONDA="$PREFIX/postconda.tar.bz2"
CONDA_QUIET="$BATCH" \
"$CONDA_EXEC" constructor --prefix "$PREFIX" --extract-tarball < "$POSTCONDA" || exit 1
rm -f "$POSTCONDA"
rm -rf "$PREFIX/install_tmp"
export TMP="$TMP_BACKUP"


#The templating doesn't support nested if statements
if [ -f "$MSGS" ]; then
  cat "$MSGS"
fi
rm -f "$MSGS"
if [ "$KEEP_PKGS" = "0" ]; then
    rm -rf "$PREFIX"/pkgs
else
    # Attempt to delete the empty temporary directories in the package cache
    # These are artifacts of the constructor --extract-conda-pkgs
    find "$PREFIX/pkgs" -type d -empty -exec rmdir {} \; 2>/dev/null || :
fi

cat <<'EOF'
installation finished.
EOF

if [ "${PYTHONPATH:-}" != "" ]; then
    printf "WARNING:\\n"
    printf "    You currently have a PYTHONPATH environment variable set. This may cause\\n"
    printf "    unexpected behavior when running the Python interpreter in %s.\\n" "${INSTALLER_NAME}"
    printf "    For best results, please verify that your PYTHONPATH only points to\\n"
    printf "    directories of packages that are compatible with the Python interpreter\\n"
    printf "    in %s: %s\\n" "${INSTALLER_NAME}" "$PREFIX"
fi

if [ "$BATCH" = "0" ]; then
    DEFAULT=no
    # Interactive mode.

    printf "Do you wish to update your shell profile to automatically initialize conda?\\n"
    printf "This will activate conda on startup and change the command prompt when activated.\\n"
    printf "If you'd prefer that conda's base environment not be activated on startup,\\n"
    printf "   run the following command when conda is activated:\\n"
    printf "\\n"
    printf "conda config --set auto_activate_base false\\n"
    printf "\\n"
    printf "You can undo this by running \`conda init --reverse \$SHELL\`? [yes|no]\\n"
    printf "[%s] >>> " "$DEFAULT"
    read -r ans
    if [ "$ans" = "" ]; then
        ans=$DEFAULT
    fi
    ans=$(echo "${ans}" | tr '[:lower:]' '[:upper:]')
    if [ "$ans" != "YES" ] && [ "$ans" != "Y" ]
    then
        printf "\\n"
        printf "You have chosen to not have conda modify your shell scripts at all.\\n"
        printf "To activate conda's base environment in your current shell session:\\n"
        printf "\\n"
        printf "eval \"\$(%s/bin/conda shell.YOUR_SHELL_NAME hook)\" \\n" "$PREFIX"
        printf "\\n"
        printf "To install conda's shell functions for easier access, first activate, then:\\n"
        printf "\\n"
        printf "conda init\\n"
        printf "\\n"
    else
        case $SHELL in
            # We call the module directly to avoid issues with spaces in shebang
            *zsh) "$PREFIX/bin/python" -m conda init zsh ;;
            *) "$PREFIX/bin/python" -m conda init ;;
        esac
        if [ -f "$PREFIX/bin/mamba" ]; then
            case $SHELL in
                # We call the module directly to avoid issues with spaces in shebang
                *zsh) "$PREFIX/bin/python" -m mamba.mamba init zsh ;;
                *) "$PREFIX/bin/python" -m mamba.mamba init ;;
            esac
        fi
    fi
    printf "Thank you for installing %s!\\n" "${INSTALLER_NAME}"
fi # !BATCH


if [ "$TEST" = "1" ]; then
    printf "INFO: Running package tests in a subshell\\n"
    NFAILS=0
    (# shellcheck disable=SC1091
     . "$PREFIX"/bin/activate
     which conda-build > /dev/null 2>&1 || conda install -y conda-build
     if [ ! -d "$PREFIX/conda-bld/${INSTALLER_PLAT}" ]; then
         mkdir -p "$PREFIX/conda-bld/${INSTALLER_PLAT}"
     fi
     cp -f "$PREFIX"/pkgs/*.tar.bz2 "$PREFIX/conda-bld/${INSTALLER_PLAT}/"
     cp -f "$PREFIX"/pkgs/*.conda "$PREFIX/conda-bld/${INSTALLER_PLAT}/"
     if [ "$CLEAR_AFTER_TEST" = "1" ]; then
         rm -rf "$PREFIX/pkgs"
     fi
     conda index "$PREFIX/conda-bld/${INSTALLER_PLAT}/"
     conda-build --override-channels --channel local --test --keep-going "$PREFIX/conda-bld/${INSTALLER_PLAT}/"*.tar.bz2
    ) || NFAILS=$?
    if [ "$NFAILS" != "0" ]; then
        if [ "$NFAILS" = "1" ]; then
            printf "ERROR: 1 test failed\\n" >&2
            printf "To re-run the tests for the above failed package, please enter:\\n"
            printf ". %s/bin/activate\\n" "$PREFIX"
            printf "conda-build --override-channels --channel local --test <full-path-to-failed.tar.bz2>\\n"
        else
            printf "ERROR: %s test failed\\n" $NFAILS >&2
            printf "To re-run the tests for the above failed packages, please enter:\\n"
            printf ". %s/bin/activate\\n" "$PREFIX"
            printf "conda-build --override-channels --channel local --test <full-path-to-failed.tar.bz2>\\n"
        fi
        exit $NFAILS
    fi
fi
exit 0
# shellcheck disable=SC2317
@@END_HEADER@@
ELF         >    �     @       ��         @ 8  @         @       @       @       �      �                   �      �      �                                                         0�     0�                                       		�     		�                   �      �      �     ��+     ��+                  ���     ���     ���     ��     x�	                  0��     0��     0��     p      p                   �      �      �                                   ���     ���     ���             �             P�td   P��     P��     P��     4�     4�            Q�td                                                  R�td   ���     ���     ���     @.     @.            /lib64/ld-linux-x86-64.so.2          GNU                     �      ;  >      `   \   !   �   w       b      f           �           F              *         [                       ,          R         �                   }           7                   �       /   �                   u                                   O          0   J           W  �               �                   3                          z                               �   �           �               &      _      �           �       $                 �   �   W   �   5  e   ^        \              �   Q     2      B              o           �           #                 '       �       �   [  y               i   L                          �                   e                              p                  b   �           �               $       �       -      �   <      l                   �                         �     �                               n  9  d       F               .  �                   �                �       �                   �   �                           6      S           E                       �   8                 �                   �           �                      �   (      `      �   �     �       �   �           �                      �   �              7                          �                                                       Q  g                                        ]           v       �   {      z                          j       _       N   O           �           r         �       >       V   k   -           �   A      	  y      5   �           c      M                   �         �   /      �   4       �           �          �                       )              �         �       x   �       �  %                      ?                  X  *   ^       �           �   �               S      ;                      �       m         �   
                         @  �                   ?   ,       4  3               �   �             q   x          �   �       T  �   �                  �               �   �       "      f  ~      �                               j                  �       |                      H       a      �   2   r               �                   Y   s       �       1          �         }      %           �   �   h          �   �         @               l        c  �               !  I       �       G  �   +      E  :          �           �   �                   �                  Y      8                  X       q  �   p   �   :                  H  �   �       �      �                                      �       (       �                          Z          �   �   �                       u                  U  '                      �     �   M  �   V  6   	       <           #           =       0                                                                       n            
                                                               �                   �           K                                      �                                                                           �   �                           �                                      �     ]      �      "               9           �                �           P   a   �           Z           T   1   m   C       �   R      �       �   �       �   �   {                           L           .                                                  |              �       C  t                      D  +                                                                                                                                                                           �   :     	   �B��0  (H�`@d��  � ���� �6�$H���D � ������    :                  ;  <  =  >      @          A          B  C  E  F                  G  I      J      L                      M      N  Q              R          S  U      V      W  X      Y              Z      [      \      ]  _  `                  a          c  d  e      g              i                          j              k  l      m      n  o  p          q  r                  s          t          u  v  w      x  y  z  {      |      }  ~            �  �  �      �                  �  �v�U�qA��7��z@M�|�r�뫪�}������$o7����!�	�'"O�?��,]�}`�|����Q���K�ї��<�ʖ����-���t�
                     E                                          f
                     �                     	                     o
                     ~                     Q                     �                     �                     f                     �                     �                     �                     8                     �                     �
                     �                      �                     �                      �                     �                     �                                          �
                     ?                     �                     K
                     #                     t                     �                     O                     �                     s                     �                      �                     #                     �                     S                     �                     �                     
                     C                     �                     �                     i                                           �                     <                     $                     7
                                          �                     �                                          �                     �                     s                     3                     �	                     7                     �                     �                     7                     {                     �                     �                     g
                                                               �                     �                     �                     �
                     ^
                     �                     �                     �                     K
                     (
                     �                     >                     �                     1                     �                                            
                      w                                            �
                     �	                     �                     X  !                                         
                     �                     f                     �                     ]                     �                     Y                     V
                     V                                          �                      >                                                               �                      B                     �                     �
                     �                     s                     �
                     �                     �                     D	                     }                     ]
                     *                     �
                     �   "                   �                                            R                     �                     c                     5                     �	                     �	                     �	                     -
                     
                     �                     �                      		                     ~                     F                      ;                     U
                     D
                     G                     �                     b                     �                     t                     �                     ]                     l                     �                     �                     V                      �                     "	                                          H                     �
                     h                     �	                     �                     >	                     �                     6                     �                                          -                     m                     Q                     �	                     .                                          F                      _ITM_deregisterTMCloneTable __gmon_start__ _ITM_registerTMCloneTable clock_gettime __pthread_key_create dlclose dlsym dladdr dlopen dlerror ns_parserr ns_initparse ns_name_uncompress __res_nsearch pthread_cond_init pthread_mutex_destroy send pthread_equal pthread_self pthread_rwlock_wrlock recvfrom pthread_sigmask pthread_mutex_init pthread_cond_wait pthread_cond_destroy pthread_rwlock_unlock recv fsync __errno_location pthread_key_delete pthread_once pthread_kill pthread_mutex_lock pthread_setspecific pread64 pthread_cond_signal pthread_cond_timedwait accept lseek64 fcntl pthread_cancel pthread_rwlock_init pthread_create pthread_join sigaction pthread_getspecific fork pthread_attr_init pthread_rwlock_rdlock connect pthread_cond_broadcast pthread_detach pthread_attr_destroy sendto pthread_rwlock_destroy pthread_mutex_unlock nanosleep pthread_attr_setdetachstate sigwait sendmsg lseek waitpid floor round __wcscoll_l strcasestr socket __xpg_basename mkdtemp wmemset fflush strcpy shmget __rawmemchr wmemcmp gmtime_r fchmod fnmatch readdir sprintf _IO_putc setlocale mbrtowc fopen strncmp ftruncate dcngettext posix_spawn_file_actions_addclose strrchr isalpha regexec pipe readlinkat __strdup __res_ninit perror shmat __isoc99_sscanf dcgettext wcrtomb _longjmp in6addr_any lutimes getpwuid dl_iterate_phdr sendfile ungetwc wmemcpy mbsrtowcs unlinkat __fdelt_chk closedir __freelocale fchdir fopencookie ftell inet_ntop strncpy __res_nclose __newlocale regfree sigfillset readdir_r __stack_chk_fail __lxstat unlink listen select mkdir shmdt realloc fstatfs btowc strtold abort stdin strtoll _exit __strtod_l strpbrk tolower lsetxattr getpid getcontext strspn bind_textdomain_codeset inet_pton strftime __assert_fail mkstemp flock rewind localtime_r isspace strtod __wctype_l gmtime __uselocale __ctype_get_mb_cur_max strtol isatty lchown getgrnam_r fstatvfs feof symlink fgetxattr isprint makecontext fgets getppid calloc futimens strlen wcschr isxdigit ungetc __towupper_l __cxa_atexit posix_spawn_file_actions_init setcontext sigemptyset getaddrinfo localeconv writev isalnum glob strstr strcspn __strtof_l utimensat rmdir tcsetattr bind fseek confstr getnameinfo dgettext putwc __strcoll_l mbsnrtowcs toupper getsockopt fchmodat clearerr unsetenv wctob vsnprintf _setjmp poll __fprintf_chk sigaddset getpwuid_r fchown stdout fputc __towlower_l fseeko64 fputs posix_spawn_file_actions_destroy fpathconf fclose wcscmp __strtol_internal __vsnprintf_chk __wcsftime_l strtoul setsockopt malloc umask strcasecmp ftello64 fdopendir realpath timegm remove __openat_2 getpeername __nl_langinfo_l nl_langinfo __ctype_b_loc __open_2 wcsnrtombs regcomp isgraph getservbyname stderr ioctl munmap __snprintf_chk posix_spawn_file_actions_adddup2 __memset_chk getuid readlink strtold_l execvp freeifaddrs __duplocale getpwnam_r freopen strncasecmp __xmknod getifaddrs if_nametoindex __fxstat strtoull qsort_r fileno gethostname getcwd lgetxattr fwrite fread iconv_close gettimeofday rename geteuid getresgid __memcpy_chk localtime difftime strchr flistxattr secure_getenv getsockname __poll_chk __vfprintf_chk mktime __strtok_r iconv fdopen qsort iconv_open getresuid tcgetattr __ctype_toupper_loc __strcpy_chk __ctype_tolower_loc llistxattr __cxa_finalize syscall freeaddrinfo __vasprintf_chk get_nprocs setvbuf __xpg_strerror_r __wcsxfrm_l setsid __iswctype_l __sprintf_chk fsetxattr __strftime_l openat futimesat __xstat getrlimit isdigit uname posix_spawnp fopen64 bindtextdomain access _IO_getc __fxstatat mkfifo setbuf strcmp strerror __asprintf_chk __libc_start_main dirfd ferror stpcpy __strxfrm_l wcslen fseeko vfprintf globfree __strtoul_internal wmemmove sysconf __environ __fxstat64 __tls_get_addr librt.so.1 libdl.so.2 libresolv.so.2 libpthread.so.0 libm.so.6 libc.so.6 ld-linux-x86-64.so.2 GLIBC_2.3 GLIBC_2.2.5 GLIBC_2.9 GLIBC_2.3.2 GLIBC_2.17 GLIBC_2.16 GLIBC_2.15 GLIBC_2.8 GLIBC_2.14 GLIBC_2.6 GLIBC_2.7 GLIBC_2.3.4 GLIBC_2.4 $ORIGIN/../lib XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ZSTD_trace_decompress_end ZSTD_trace_compress_end ZSTD_trace_decompress_begin getentropy ZSTD_trace_compress_begin                                                             	                   
                           
                    
                                                                         
                                           	                    
                                                                                                           �         ii
 �        �         ui	  	 �        �     0   ui	   �     ri	   �        �         ���        ���        ���        ii
G     ���            ��     ���            @	G     ���            G     ���            �	G      ��            �c�     ��            pfG     ���            r��     ���            昱     ���            �     ���            ���     ���            ��     ���            >E�     ���            ��     ���            ��     ���            )��     ���            3��     ���            C��     ���            T��      ��            _��     ��            f��      ��            �E�     (��            ��G     0��            0�G     8��            ��G     H��            ��G     ���            `�G     ���            ���     ���            *Ŷ     ���            9f�     ���            ��G     ���             �G     ���            ��G      ��            �G     ��            ��G     ��            ИG     ��            ��G      ��            ��G     (��            ��G     0��             �G     @��            ��G     ���            �r�     ���            ��G     ���             �G     ���            ��G     ���            �G     ���            ��G     ���            ИG     ���            ��G     ���            ��G     ���            ��G     ���             �G     ���            ��G      ��            W��     (��            \��     0��            �ֱ     8��            -�     @��            d��     H��            i��     P��            n��     X��            I��     `��            �`�     h��            I��     p��            ���     x��            s��     ���            ���     ���            {��     ���            ���     ���            ���     ���            ���     ���            ���     ���            ���     ���            ���     ���            ���     ���            ���     ���             �     ���            ۠�     ���            ̠�     ���            ֠�     ���            ���     ���            ࠱      ��            ��     ��            ꠱     ��            ��     ��            �,�      ��                 (��            ���     0��            <��     @��            Ef�     P��            PH     h��            @H     p��             H     ���            >f�     ���            PH     ���            ���     ���            _��     ���            C��     ���            ��     ���            =��     ���            ��     ���            Y[�     ���            �AH     ���            PxH     ���            �DH     ��            �DH     (��            �DH     H��            ��H     P��            `�H     ���            }[�     ���            �AH     ���            PxH     ���            �DH     ���            �DH     ���            �DH     ���            ��H     ���            `�H      ��            ��     0��            �&I     8��            @-I     @��            �&I     `��            �c�     h��             [I     p��            �RI     x��            pPI     ���            �OI     ���            Mf�     ���            pLI     ���            `I     ���            �PI     ���            `]I     ���            �NI     ���            �_I     ���            �NI     ���            �NI      ��            @iI     @��            �c�     H��            pLI     P��            `I     X��            �PI     h��            `]I     p��            �NI     x��            �_I     ���            �NI     ���            �NI     ���            @iI     ���            �     ���            Ȥ�     ��            S۱     ��            )��      ��            �ڱ     0��            puI     8��            �xI     @��            wI     H��            �uI     P��            @vI     X��            �uI     `��             �I     h��            �I     p��             �I     ���             �I     ���            �I     ���             �I     ���            Sf�     ���            0�I     ���            P�I     ���             �I     ���            МI     ��            ��I     `��            YZ�     h��             �I     p��            ��I     x��            p�I     ���            мI     ���            Xf�     ���             �I     ���             �I     ���            `�I     ���            @�I     ���            ��I     ���            ��I     ���            ��I     ���            ��I      ��            P�I     @��            d�     H��             �I     P��             �I     X��            `�I     h��            @�I     p��            ��I     x��            ��I     ���            ��I     ���            ��I     ���            P�I     ���            ^f�     ���            ��I     ���            ��I     ���            ��I     ��            p�I     (��            @�I     @��            P�I     H��            ��I     X��             �I     ���             �I     ���            p�I     ���            ��I     ���            kf�     ���            ��I     ���            ��I     ���            ��I     ���            ��I     ���            `�I      ��            ��I     ��            ��I      ��            @�I     `��            gf�     h��            ��I     p��            ��I     ���            ��I     ���            ��I     ���            `�I     ���            ��I     ���            ��I     ���            @�I      ��            �c�     ��            �J     ��            �J     ��            �J      ��            �J     @��            pf�     H��            �J     P��            �J     X��            �J     h��            PJ     p��            �J     x��            `J     ���            �J     ���            �J     ���            �J     ���            �c�     ���            �J     ���            �J     ���            �J     ��            PJ     ��            �J     ��            `J      ��            �J     (��            �J     @��            �J     ���            vf�     ���            dJ     ���            RJ      ��            ���     (��            ��     0��            $��     8��            *��     @��            ��     H��            .��     P��            2��     X��            8��     `��            ��     h��            <��     p��            ?��     x��            �.�     ���            ��     ���            >��     ���            C��     ���            0��     ���            5��     ���            ؕ�     ���            :��     ���            F��     ���            ���     ���            J��     ���            O��     ���            S��     ���            %��     ���            �t�     ���            e��     ���            q��      ��            v��     ��            {��     ��            ���     ��            ���      ��            ���     (��            ���     0��            ���     8��            ���     @��            ���     H��            ���     P��            �ױ     X��            ���     `��            ���     h��            ���     p��            ���     x��            ���     ���            ���     ���            ��     ���            ��     ���            ��     ���            *��     ���            1��     ���            =��     ���            D��     ���            I��     ���            T��     ���            Z��     ���            ���     ���            c��     ���            o��     ���            �#�     ���            ~��      ��            }f�     ��            ��J     ��            ��J     ��            ЅJ     (��            ��J     0��            `�J     8��             �J     @��            0xJ     H��            0xJ     `��            �wJ     ���            ױ     ���            �#�     ���            $�     ���            �#�      ��            �ڱ     ��            �#�     0��            ��     H��            �#�     `��            �#�     x��            $�     ���            $�     ���            P<�     ���            �LL     ���             LL     ���            `KL     ���            0KL     ���             KL      ��            �JL      ��            5s�     0��            �LL     8��             LL     @��            `KL     H��            0KL     P��             KL     `��            �JL     ���            x<�     ���            �PL     ���            �OL     ���            �ML     ���            �ML     ���            �<�     ���            �PL      ��            �OL     ��            �ML      ��            �ML     @��            �<�     X��            �PL     `��            �OL     p��            �ML     ���            �ML     ���            As�     ���            0RL     ���            �PL     ���            �OL     ���             QL     ���            �ML     ���            �ML      ��            Us�     ��            0RL     ��            �PL      ��            �OL     (��             QL     0��            �ML     @��            �ML     `��            is�     p��            0RL     x��            �PL     ���            �OL     ���             QL     ���            �ML     ���            �ML     ���            �s�     ���            @CL     ���            �BL     ���             BL     ���            PTL     ���             AL      ��             AL      ��            -<�     8��             FL     @��            �EL     P��            �DL     `��            �DL     ���            �&�     ���            �FL     ���             FL     ���            �EL     ���            �IL     ���             JL     ���            �DL     ���            '�     ���            �FL     ���             FL      ��            �EL     ��            �IL     ��            �HL      ��            �DL     @��            }s�     P��            �FL     X��             FL     `��            �EL     h��            �FL     p��            �DL     ���            �DL     ���            �?�     ���            �=�     ���            �~L     ���            �=�     ���            �~L     ���            '=�     ���            �{L     ��            =�     ��            �{L     0��            =�     8��            �{L     P��            �G�     X��            0�L     p��             H�     x��            �L     ���            �?�     ���            ��L     ���            �?�     ���            ��L     ���             @�     ���            ��L     ���            @�     ���             �L     ��            :@�     ��            p�L     @��            WT�     ���            jT�     ���            ��L     ���            ��L     ���            ��L     ���            �T�     ���            ��L     ���            ��L     ���            ��L      ��            �T�     ��            ��L     ��            ��L      ��            ��L     @��            ��     P��            ��L     X��            ��L     `��            ��L     ���            �
ǲ     X �            &ǲ     h �            ɲ     x �            0ɲ     � �            Xɲ     � �            �ɲ     � �            �ɲ     � �            Bǲ     � �            �ɲ     � �            Wǲ     � �            ʲ     � �            oǲ     �            8ʲ     �            ���     (�            �#�     8�            �ǲ     H�            �ǲ     P�            `�     h�            ��     p�            ��     ��            �#�     ��            ��     ��            в     ��            8в     ��            `в     ��            �Ҳ     ��            �в     ��            �Ҳ     ��            �Ҳ     ��            �в      �            �в     �             Ѳ     �            (Ѳ     �            �Ҳ      �            hѲ     (�            �Ѳ     0�            Ӳ     8�            ,Ӳ     @�            �Ѳ     H�            �Ѳ     P�            0Ҳ     X�            xҲ     `�            w��     ��            ���     ��            ���     ��            �     ��            ��     ��            x��     ��            ���     ��            ���     ��            С�     ��            ���     ��            �ײ     ��            4ٲ     ��            ���     ��            Qٲ     ��            eٲ     ��            �ײ      �            ~ٲ     �            �ٲ     �            ز     �            Pز      �            �ز     H�            ��     P�            �ز     X�            ���     h�            p�     p�            ���     ��            Q�     ��             Q�     ��            0Q�     ��            @Q�     �             �      �            ���     @�            PQ�     `�            `Q�     ��            pQ�     ��            �Q�     ��            ��     ��            ���     ��            �Q�     �            �Q�     0�            �Q�     P�            �Q�     p�            �Q�     ��            �Q�     ��            ��     ��            ��     ��            �Q�      �            0R�      �            pR�     @�            P�     P�            ��     p�            �R�     ��            �R�     ��            �R�     ��            �R�     ��             �      �             ��      �            �R�     @�            �R�     `�            p�     p�            0��     ��            �R�     ��             S�     ��            ��     ��            @��      �            S�      �             S�     @�            0S�     `�            @S�     ��            PS�     ��            `S�     ��            pS�     ��            �S�      	�            �S�      	�            �S�     @	�            �S�     `	�            T�     �	�            �	�     �	�            `��     �	�            0T�     �	�            @T�     �	�            PT�     
�            `T�     0
�            pT�     P
�            �T�     p
�            �T�     �
�            �T�     �
�            �
�     �
�            ���     �
�            �T�      �            �T�      �            �T�     @�            �T�     `�            �T�     ��             U�     ��            ��     ��            ���     ��            U�     ��             �      �            ���      �             U�     @�            0U�     `�            @U�     ��            PU�     ��            `U�     ��            pU�     ��            �U�      
�     (��            rK�     x��            +I�     ȗ�            �&�     ��            �H�     h��            L%�     ���            %&�     ��            �&�     X��            � �     ���            TO�     ���            � �     H��            [O�     ���            V��     ��            �H�     8��            �H�     ���             I�     ؛�            I�     (��            I�     ���            �kW     ���            �kW     ���            �kW     ���            pkW     ���            `kW     Ȝ�            PkW      ��            o�     @��            %o�     h��            <o�     ���            So�     ���            9s�     ���            jo�     X��            po�     ���            �o�     ���            �o�     О�            �o�     ���            �o�      ��            �o�     H��            p�     p��            p�     ���            )p�     ���            <p�     ��            Mp�     ��            ^p�     8��            op�     `��            �p�     0��             ��     P��            ��     p��            ���     ���            x¶     ���            ' �     ���            � �     Т�            � �     ��            TO�      ��            � �     ��            [O�     0��            � �     H��            SO�     `��            ZO�     ���            va�     ���            � �     ���            TO�     ���            � �     ���            [O�     У�            � �     ��            SO�     ��            ZO�      ��            Ķ     ��            Ķ     0��            Ķ     H��            dŶ     `��            $Ķ     x��            5Ķ     ���            SĶ     ���            QĶ     ���            \Ķ     ؤ�            vĶ     ��            �Ķ     ��            �Ķ      ��            �Ķ     8��            �Ķ     P��            �Ķ     h��            �Ķ     ���            �Ķ     ���            Ŷ     ���            Ŷ     ȥ�            !Ŷ     ��            )Ŷ     ���            .Ŷ     ��            >Ŷ     (��            WŶ     @��            pŶ     X��            �Ŷ     ���            �Ŷ     ���            ZP�     ���            �Ŷ     Ȧ�            �Ŷ     ��            �Ŷ     ���            �Ŷ     0��            �Ŷ     P��            �Ŷ     p��            �Ŷ     ���            �Ŷ     ���            �Ŷ     Ч�            ƶ     ��            
ƶ     ��            
ֶ     ��            ֶ     ��            'ֶ     (��            5�     8��            :ֶ     H��            Eֶ     X��            Wֶ     h��            sֶ     x��            {ֶ     ���            �ֶ     ���            �ֶ     ���            �ֶ     ���            �ֶ     Ȱ�            �ֶ     ذ�            �ֶ     ��            �˶     ���            �ֶ     ��            �ݻ     ��            ׶     (��            ׶     8��            =�     H��            )׶     X��            C׶     h��            QG�     x��            J׶     ���            _׶     ���            u׶     ���            �׶     ���            �׶     ȱ�            �׶     ر�            �׶     ��            �׶     ���            �׶     ��            �˶     ��            �׶     (��            ض     8��            'ض     H��            Bض     X��            n��     h��            ̶     x��            Wض     ���            qض     ���            �ض     ���            �ض     ���            �ض     Ȳ�            �ض     ز�            �ض     ��            8̶     ���            �ض     ��            `̶     ��            ٶ     (��            ٶ     8��            �̶     H��            3ٶ     X��            Mٶ     h��            iٶ     x��            �ٶ     ���            �ٶ     ���            �ٶ     ���            �ٶ     ���            �ٶ     ȳ�            �     س�            �̶     ��            �ٶ     ���            �̶     ��            
�     X��            Ҷ     h��            "�     x��            @�     ���            X�     ���            u�     ���            ��     ���            ��     ȿ�            ��     ؿ�            ��     ��            ��     ���            0Ҷ     ��            XҶ     ��            ��     (��            ��     8��            
�     H��            (�     X��            xҶ     h��            �Ҷ     x��            �Ҷ     ���            B�     ���            �Ҷ     ���            ]�     ���            z�     ���            ��     ���            ��     ���            Ӷ     ���            ��     ��            8Ӷ     ��            `Ӷ     (��            �Ӷ     8��            ��     H��            ��     X��            �     h��            �Ӷ     x��            4�     ���            I�     ���            ^�     ���            �Ӷ     ���            �Ӷ     ���            Զ     ���            8Զ     ���            r�     ���            ��     ��            ��     ��            ��     (��            ��     8��            ^��     H��            ��     X��            ��     h��            �     x��            ,�     ���            @�     ���            Q�     ���            a�     ���            p�     ���            ��     ���            ��     ���            ��     ���            ��     ��            7`�     ��            `Զ     (��            ��     8��            �Զ     H��            ��     X��            �Զ     h��            �     x��            (�     ���            C�     ���            X�     ���            p�     ���            ��     ���            ��     ���            ��     ���            ��     ���            ��     ��            ��     ��            �     (��            �     8��             �     H��            7�     X��            L�     h��            a�     x��            s�     ���            ��     ���            �Զ     ���            v]     ���            �u]     ���            0l]     ���            �u]     ���             n]     ���            @l]     ���            @q]     ���            �t]      ��            �o]     ��            �n]     ��            Pl]     ��            po]      ��            �t]     (��            `l]     0��            �u]     8��            pl]     H��            �t]     P��            �l]     X��            �l]     `��            �u]     h��            �m]     p��             m]     x��            �l]     ���            �u]     ���            P^     ���            0^     ���            ��]     ���            `^     ���             ^     ���            � ^     ���            ��]     ���            ��]     ���            ^     ���            ��]     ���            p^     ���            �^      ��            p^     ��            �^     P��            .�     `��            �5�     h��            ��     p��            ��     x��            �     ���            �     ���            t?�     ���            ��     ���            �     ���            �#�     ���            0�     ���            ��     ���            5�     ���            �$�     ���            ?�     ���            I�     ���            S�     ���            $�     ���            �p�     ���            ]�     ���            �$�      ��            �$�     ��            k�     ��            �$�     ��            �$�      ��            w$�     (��            z�     0��            d$�     8��            �$�     @��            O$�     H��            ��     P��            @$�     p��            ���     ���            ���     ���            [�     ���            ��^     ���            ���     ��            �#�     0��            ���     P��            �#�     ���            �#�     ���            �#�     ��            �#�     P��            �#�     ���            �#�     ���            �#�     ��            �#�     P��            $�     ���            $�     ���            %$�     ��            1$�     P��            ;$�     ���            J$�     ���            _$�     ��            r$�     P��            �$�     ���            �$�     ���            �$�     ��            �$�     P��            �$�     ���            �$�     ���            �$�     ��            �$�     P��            
%�     ���            %�     ���             ��     ���            �%�     ���            @��     ��            �%�     @��            �E�     P��            �%�     ���            �E�     ���            �%�     ���             F�     ���            �%�      ��             F�     ��            &�     @��            �E�     P��            �%�     ���            �E�     ���            �%�     ���             F�     ���            �%�      ��             F�     ��            �%�     (��            M�     0��            �%_     @��            �%_     P��            P%_     X��             %_     `��            �#_     h��            �#_     p��            0#_     x��             #_     ���            �'�     ���            �'�     ���            �'�     ���            (�     ���            "(�     ���            �E�     ��            .(�     ��            �*�     (��            >(�     8��            P(�     H��            n(�     X��            -��     h��            (�     x��            s7�     ���            �ٶ     ���            �(�     ���            �(�     ���            �(�     ���            �(�     ���            �*�     ���            +�     ���            �(�     ��            �(�     ��            �(�     (��            
�            ��c     H
�            0�c     P
�            Џc     p
�             �c     @�            �c     H�            0�c     P�            Џc     p�             �c     @�            ��c     H�            0�c     P�            Џc     p�             �c     @
�     Hr�            2w�     Xr�            �ڸ     hr�            ���     pr�            �	�     �r�            �ڸ     �r�            -C�     �r�            -C�     �r�            ۸     �r�            ;C�     �r�            CC�     �r�            ۸     �r�            �	�     �r�            XC�     �r�            ۸     s�            mC�     s�            yC�      s�             ۸     0s�            �C�     8s�            �C�     Hs�            )۸     Xs�            �C�     `s�            �C�     ps�            2۸     �s�            �C�     �s�            �C�     �s�            3۸     �s�            ��     �s�            �C�     �s�            5۸     �s�            H��     �s�            �C�     �s�            8۸     �s�            3�      t�            �C�     t�            ;۸      t�            �v�     (t�            �C�     8t�            >۸     Ht�            =��     Pt�            D�     `t�            A۸     pt�            D�     xt�             D�     �t�            D۸     �t�            �4�     �t�            U��     �t�            G۸     �t�            7D�     �t�            7D�     �t�            K۸     �t�            =D�     �t�            =D�      u�            S۸     u�            HD�     u�            HD�     (u�            \۸     8u�            YD�     @u�            YD�     Pu�            e۸     `u�            mD�     hu�            mD�     xu�            n۸     �u�            �D�     �u�            �D�     �u�            w۸     �u�            �D�     �u�            �D�     �u�            �۸     �u�            �D�     �u�            �D�     �u�            �۸      v�            �D�     v�            �D�     v�            �۸     (v�            �D�     0v�            �D�     @v�            �۸     Pv�            �D�     Xv�            �D�     hv�            �۸     xv�            5P�     �v�            V��     �v�            �۸     �v�            �D�     �v�            �D�     �v�            �۸     �v�            ���     �v�            �D�     �v�            �D�     �v�            E�     w�            �۸     w�            E�      w�            E�     @w�            !E�     Hw�            *E�     hw�            �G�     pw�            3E�     �w�            �۸     �w�            ;E�     �w�            CE�     �w�            KE�     �w�            SE�     �w�            [E�     �w�            cE�     x�            +I�     x�            �	�      x�            �۸     0x�            kE�     8x�            sE�     Hx�            �۸     Xx�            �E�     `x�            �E�     �x�            h��     �x�            �E�     �x�            �۸     �x�            �E�     �x�            �E�     �x�            �۸     �x�            �E�     �x�            �E�     �x�            �E�      y�            �E�     y�            �۸      y�            �E�     (y�            �E�     8y�            �۸     Hy�            �E�     Py�            �E�     `y�            �۸     py�            �E�     xy�            �E�     �y�            �۸     �y�            �E�     �y�            �E�     �y�            �۸     �y�            
T�     ���            
T�     ��            ��     ��            (��      ��            (��     0��            ��     @��            H��     H��            H��     X��            ��     h��            &T�     p��            &T�     ���            ��     ���            AT�     ���            AT�     ���            �     ���            h��     ���            h��     Е�            �     ���            ���     ��            ���     ���            �     ��            ���     ��            ���      ��            &�     0��            ^T�     8��            ^T�     H��            1�     X��            ���     `��            ���     p��            <�     ���            yT�     ���            yT�     ���            G�     ���            ���     ���            ���     ���            R�     Ж�             ��     ؖ�             ��     ��            ]�     ���            �T�      ��            �T�     ��            h�      ��            �T�     (��            �T�     8��            s�     H��            �T�     P��            �T�     `��            ~�     p��            �T�     x��            �T�     ���            ��     ���            �T�     ���            �T�     ���            ��     ���            U�     ȗ�            U�     ؗ�            ��     ��            #U�     ��            #U�      ��            ��     ��            <�     ��            <�     (��            ��     8��            5U�     @��            5U�     P��            ��     `��            MU�     h��            MU�     x��            ��     ���            ^U�     ���            ^U�     ���            ��     ���            wU�     ���            wU�     Ș�            ��     ؘ�            H��     ���            H��     ��            ��      ��            h��     ��            h��     ��            ��     (��            ���     0��            ���     @��            �     P��            ���     X��            ���     h��            
X�     `��            
X�     p��            s�     ���            !X�     ���            !X�     ���            {�     ���            7X�     ���            7X�     ���            ��     Р�            NX�     ؠ�            NX�     ��            ��     ���            dX�      ��            dX�     ��            ��      ��            uX�     (��            uX�     8��            ��     H��            �X�     P��            �X�     `��            ��     p��            �X�     x��            �X�     ���            ��     ���            �X�     ���            �X�     ���            ��     ���            �X�     ȡ�            �X�     ء�            ��     ��            �X�     ��            �X�      ��            ��     ��             Y�     ��             Y�     (��            ��     8��            Y�     @��            Y�     P��            ��     `��            ,Y�     h��            ,Y�     x��            ��     ���            AY�     ���            AY�     ���            ��     ���            LY�     ���            LY�     Ȣ�            ��     آ�            WY�     ��            WY�     ��            ��      ��            kY�     ��            kY�     ��            ��     (��            �Y�     0��            �Y�     @��            �     P��            �Y�     X��            �Y�     h��            �     x��            �Y�     ���            �Y�     ���            �     ���            �Y�     ���            �Y�     ���             �     ȣ�            �Y�     У�            �Y�     ��            )�     ��            Z�     ���            Z�     ��            2�     ��            Z�      ��            Z�     0��            ;�     @��            $Z�     H��            $Z�     X��            C�     h��            7Z�     p��            7Z�     ���            K�     ���            OZ�     ���            OZ�     ���            S�     ���            ]Z�     ���            ]Z�     Ф�            [�     ��            oZ�     ��            oZ�     ���            c�     ��            �Z�     ��            �Z�      ��            k�     0��            �Z�     8��            �Z�     H��            s�     X��            �Z�     `��            �Z�     p��            {�     ���            �Z�     ���            �Z�     ���            ��     ���            �Z�     ���            �Z�     ���            ��     Х�            �Z�     إ�            �Z�     ��            ��     ���            �Z�      ��            �Z�     ��            ��      ��            [�     (��            [�     8��            ��     H��            '[�     P��            '[�     `��            ��     p��            <[�     x��            <[�     ���            ��     ���            K[�     ���            K[�     ���            ��     ���            Y[�     Ȧ�            Y[�     ئ�            ��     ��            n[�     ��            n[�      ��            ��     ��            }[�     ��            }[�     (��            ��     8��            �[�     @��            �[�     P��            ��     `��            �[�     h��            �[�     x��            ��     ���            �[�     ���            �[�     ���            ��     ���            �[�     ���            �[�     ȧ�            ��     ا�            �[�     ��            �[�     ��            ��      ��             \�     ��             \�     ��            �     (��            \�     0��            \�     @��            �     x��            '\�     ���            '\�     ���            �     ���            5\�     ���            5\�     ���            �     Ȩ�            Q\�     Ш�            Q\�     ��            #�     ��            k\�     ���            k\�     ��            +�     ��            �\�      ��            �\�     0��            3�     @��            �\�     H��            �\�     X��            ;�     h��            �\�     p��            �\�     ���            C�     ���            �\�     ���            �\�     ���            K�     ���            �\�     ���            �\�     Щ�            S�     ��            �\�     ��            �\�     ���            [�     ��            �\�     ��            �\�      ��            c�     0��            �\�     8��            �\�     H��            k�     X��            ]�     `��             ]�     p��            s�     ���            1]�     ���            9]�     ���            {�     ���            A]�     ���            S]�     ���            ��     Ъ�            l]�     ت�            g]�     ��            ��     ���            r]�      ��            x]�     ��            ��      ��            �]�     (��            �]�     8��            ��     H��            �]�     P��            �]�     `��            ��     p��            �]�     x��            �]�     ���            ��     ���            �]�     ���            �]�     ���            ��     ���            ^�     ȫ�            ^�     ث�            ��     ��            �e�     ��            �e�      ��            ��     ��            �I�     ��            �I�     (��            ��     8��            2^�     @��            <^�     P��            ��     `��            F�     h��            F�     x��            ��     ���            G^�     ���            G^�     ���            ��     ���            T^�     ���              �     Ȭ�            ��     ج�            c^�     ��            �=�     ��            ��      ��            g^�     ��            k^�     ��            ��     (��            o^�     0��            t^�     @��            ��     P��            9r�     X��            y^�     h��            ��     x��            �^�     ���            �^�     ���            ��     ���            �^�     ���            �^�     ���            ��     ȭ�            )��     Э�            �^�     ��            �     ��            (b�     ���            �^�     ��            �     ��            �^�      ��            �^�     0��            �     @��            �^�     H��            �^�     X��            �     h��            �^�     p��            �^�     ���            �     ���            �^�     ���            �^�     ���            �     ���            ?�     ���            �^�     Ю�            !�     ��            ���     ��            �b�     ���            +�     ��            t?�     ��            t?�     0��            _�     8��            $_�     H��            5�     X��            =_�     `��            =_�     p��            8�     ���            �	�     ���            G_�     ���            <�     ���            \_�     ���            \_�     ���            E�     Я�            h_�     د�            z_�     ��            M�     ���            �_�      ��            �_�     ��            U�      ��            �\�     (��            �\�     8��            ]�     H��            �_�     P��            �_�     `��            `�     p��            �_�     x��            �_�     ���            c�     ���            �_�     ���            @ �     ���            f�     ���            t?�     Ȱ�            t?�     ��            `�     ��            
�     0��            ���     8��            �`�     H��            �     X��            ͔�     `��            �`�     p��            �     ���            �`�     ���            a�     ���            %�     ���            a�     ���            a�     ���            .�     д�            'a�     ش�            ;a�     ��            7�     ���            Qa�      ��            ea�     ��            :�      ��            {a�     (��            �a�     8��            A�     H��            �a�     P��            �a�     `��            H�     p��            ���     x��            ���     ���            O�     ���            �a�     ���            �a�     ���            P�     ���            �a�     ȵ�            �a�     ص�            S�     ��            �a�     ��            �a�      ��            Z�     ��            �a�     ��            �a�     (��            b�     8��            b�     @��            b�     P��            k�     `��            b�     h��            b�     x��            t�     ���            'b�     ���            'b�     ���            }�     ���            3b�     ���            3b�     ȶ�            ��     ض�            Cb�     ��            Cb�     ��            ��      ��            ]b�     ��            ]b�     ��            ��     (��            ib�     0��            ib�     @��            ��     P��            ub�     X��            ub�     h��            ��     x��            }b�     ���            }b�     ���            ��     ���            �b�     ���            �b�     ���            ��     ȷ�            �b�     з�            �b�     ��            ��     ��            �b�     ���            �b�     ��            ��     ��            �b�      ��            �b�     0��            ��     @��            �b�     H��            �b�     X��            ��     h��            �b�     p��            �b�     ���            ��     ���            �b�     ���            �b�     ���            ��     ���            �b�     ���            �b�     и�            �     ��            �b�     ��            �b�     ���            �     ��            c�     ��            c�      ��            �     0��            ��     8��            c�     H��            &�     X��            #c�     `��            #c�     p��            0�     ���            
6�     ���            8c�     ���            :�     ���            ���     ���            ���     ���            D�     й�            Fc�     ع�            Fc�     ��            N�     ���            Uc�      ��            Uc�     ��            X�      ��            `c�     (��            `c�     8��            b�     H��            fc�     P��            fc�     `��            l�     p��            ��     x��            ��     ���            v�     ���            pc�     ���            pc�     ���            ��     ���            xc�     Ⱥ�            xc�     غ�            ��     ��            �c�     ��            �c�      ��            ��     ��            �c�     ��            �c�     (��            ��     8��            �c�     @��            �c�     P��            ��     `��            �c�     h��            �c�     x��            ��     ���            �c�     ���            �c�     ���            ��     ���            �c�     ���            �c�     Ȼ�            ��     ػ�            �c�     ��            �c�     ��            ��      ��            �c�     ��            �c�     ��            ��     (��            d�     0��            d�     @��            ��     P��            d�     X��            d�     h��            ��     x��            d�     ���            d�     ���            ��     ���            1d�     ���            1d�     ���            �     ȼ�            :d�     м�            :d�     ��            �     ��            Cd�     ���            Cd�     ��            �     ��            Md�      ��            Md�     0��             �     @��            Yd�     H��            Yd�     X��            *�     h��            jd�     p��            jd�     ���            4�     ���            yd�     ���            yd�     ���            >�     ���            �d�     ���            �d�     н�            H�     ��            �d�     ��            �d�     ���            R�     ��            �d�     ��            �d�      ��            \�     0��            �d�     8��            �d�     H��            f�     X��            �d�     `��            �d�     p��            p�     ���            �d�     ���            �d�     ���            z�     ���            �d�     ���            �d�     ���            ��     о�            e�     ؾ�            e�     ��            ��     ���            e�      ��            e�     ��            ��      ��            'e�     (��            'e�     8��            ��     H��            :e�     P��            :e�     `��            ��     p��            Pe�     x��            Pe�     ���            ��     ���            fe�     ���            fe�     ���            ��     ���            xe�     ȿ�            xe�     ؿ�            ��     ��            �e�     ��            �e�      ��            ��     ��            �e�     ��            �e�     (��            ��     8��            �e�     @��            �e�     P��            ��     `��            �e�     h��            �e�     x��            ��     ���            �e�     ���            �e�     ���            ��     ���            �e�     ���            �e�     ���            ��     ���            �e�     ���            �e�     ���            ��      ��            �e�     ��            �e�     ��            �     (��            f�     0��            f�     @��            
�     P��            )f�     X��            )f�     h��            
k�      ��            
k�     0��            �     @��            k�     H��            k�     X��            �     h��            /k�     p��            /k�     ���            �     ���            @k�     ���            @k�     ���             �     ���            Pk�     ���            Pk�     ���            $�     ���            ck�     ���            ck�     ���            (�     ��            wk�     ��            wk�      ��            ,�     0��            �k�     8��            �k�     H��            0�     X��            �k�     `��            �k�     p��            4�     ���            �k�     ���            �k�     ���            8�     ���            �k�     ���            �k�     ���            <�     ���            �k�     ���            �k�     ���            @�     ���            �k�      ��            �k�     ��            D�      ��            �k�     (��            �k�     8��            H�     H��            l�     P��            l�     `��            L�     p��            )l�     x��            )l�     ���            P�     ���            =l�     ���            =l�     ���            T�     ���            Nl�     ���            Nl�     ���            X�     ���            `l�     ���            `l�      ��            \�     ��            ql�     ��            ql�     (��            `�     8��            �l�     @��            �l�     P��            d�     `��            �l�     h��            �l�     x��            h�     ���            �l�     ���            �l�     ���            l�     ���            �l�     ���            �l�     ���            p�     ���            m�     ���            m�     ���            t�      ��            m�     ��            m�     ��            x�     (��            'm�     0��            'm�     @��            |�     P��            5m�     X��            ?m�     h��            ��     x��            Wm�     ���            Wm�     ���            ��     ���            gm�     ���            gm�     ���            ��     ���            zm�     ���            zm�     ���            ��     ���            �m�     ���            �m�     ��            ��     ��            �m�      ��            �m�     0��            ��     @��            �m�     H��            �m�     X��            ��     h��            �m�     p��            �m�     ���            ��     ���            �m�     ���            �m�     ���            ��     ���            �m�     ���            �m�     ���            ��     ���            �m�     ���            �m�     ���            ��     ��            n�     ��            n�      ��            ��     0��            )n�     8��            )n�     H��            ��     X��            ;n�     `��            ;n�     p��            ��     ���            Vn�     ���            Vn�     ���            ��     ���            cn�     ���            sn�     ���            ��     ���            �n�     ���            �n�     ���            ��     ���            �n�      ��            �n�     ��            ��      ��            �n�     (��            �n�     8��            ��     H��            �n�     P��            �n�     `��            ��     p��            �n�     x��            �n�     ���            ��     ���            �n�     ���            �n�     ���            ��     ���            o�     ���            o�     ���            ��     ���            o�     ���            o�      ��            ��     ��            1o�     ��            1o�     (��            ��     8��            Do�     @��            Xo�     P��            ��     `��            lo�     h��            zo�     x��            ��     ���            �o�     ���            �o�     ���            ��     ���            �o�     ���            �o�     ���            ��     ���            �o�     ���            �o�     ���            �      ��            p�     ��            p�     ��            	�     (��            p�     0��            p�     @��            