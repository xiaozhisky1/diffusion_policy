#!/bin/sh
#
# Created by constructor 3.9.3
#
# NAME:  Mambaforge
# VER:   24.9.2-0
# PLAT:  osx-64
# MD5:   6d5ce6f094730345c1560ff8d6a7af6b

set -eu

unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH
if ! echo "$0" | grep '\.sh$' > /dev/null; then
    printf 'Please run using "bash"/"dash"/"sh"/"zsh", but not "." or "source".\n' >&2
    return 1
fi

min_osx_version="10.13"
system_osx_version=$(SYSTEM_VERSION_COMPAT=0 sw_vers -productVersion)
# shellcheck disable=SC2183 disable=SC2046
int_min_osx_version="$(printf "%02d%02d%02d" $(echo "$min_osx_version" | sed 's/\./ /g'))"
# shellcheck disable=SC2183 disable=SC2046
int_system_osx_version="$(printf "%02d%02d%02d" $(echo "$system_osx_version" | sed 's/\./ /g'))"
if [  "$int_system_osx_version" -lt "$int_min_osx_version" ]; then
    echo "Installer requires macOS >=${min_osx_version}, but system has ${system_osx_version}."
    exit 1
fi
# Export variables to make installer metadata available to pre/post install scripts
# NOTE: If more vars are added, make sure to update the examples/scripts tests too

  # Templated extra environment variable(s)
export INSTALLER_NAME='Mambaforge'
export INSTALLER_VER='24.9.2-0'
export INSTALLER_PLAT='osx-64'
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
    if [ "$(uname)" != "Darwin" ]; then
        printf "WARNING:\\n"
        printf "    Your operating system does not appear to be macOS, \\n"
        printf "    but you are trying to install a macOS version of %s.\\n" "${INSTALLER_NAME}"
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

if ! mkdir -p "$PREFIX"; then
    printf "ERROR: Could not create directory: '%s'\\n" "$PREFIX" >&2
    exit 1
fi

total_installation_size_kb="467435"
free_disk_space_bytes="$(df -Pk "$PREFIX" | tail -n 1 | awk '{print $4}')"
free_disk_space_kb="$((free_disk_space_bytes / 1024))"
free_disk_space_kb_with_buffer="$((free_disk_space_bytes - 100 * 1024))"  # add 100MB of buffer
if [ "$free_disk_space_kb_with_buffer" -lt "$total_installation_size_kb" ]; then
    printf "ERROR: Not enough free disk space: %s < %s\\n" "$free_disk_space_kb_with_buffer" "$total_installation_size_kb" >&2
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
boundary1=$(( boundary0 + 13550184 ))
# the end of the second payload, plus one
boundary2=$(( boundary1 + 49643520 ))

# verify the MD5 sum of the tarball appended to this header
MD5=$(extract_range "${boundary0}" "${boundary2}" | md5)
if ! echo "$MD5" | grep 6d5ce6f094730345c1560ff8d6a7af6b >/dev/null; then
    printf "WARNING: md5sum mismatch of tar archive\\n" >&2
    printf "expected: 6d5ce6f094730345c1560ff8d6a7af6b\\n" >&2
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
if [ "'__osx >=10.13'" != "" ]; then
    echo 'Checking virtual specs compatibility: '__osx >=10.13''
    CONDA_QUIET="$BATCH" \
    CONDA_SOLVER="classic" \
    CONDA_PKGS_DIRS="$(mktemp -d)" \
    "$CONDA_EXEC" create --dry-run --prefix "$PREFIX/envs/_virtual_specs_checks" --offline '__osx >=10.13'
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
"$CONDA_EXEC" install --offline --file "$PREFIX/pkgs/env.txt" -yp "$PREFIX" $shortcuts || exit 1
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
    "$CONDA_EXEC" install --offline --file "${env_pkgs}env.txt" -yp "$PREFIX/envs/$env_name" $env_shortcuts || exit 1
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
����           �
  ���        H   __PAGEZERO                                                          __TEXT                  �              �           	       __text          __TEXT           0     ���      0               �            __stubs         __TEXT          ���    Z      ���             �           __stub_helper   __TEXT          X̛    �      X̛              �            __gcc_except_tab__TEXT           �    ��      �                            __const         __TEXT           ߞ     �      ߞ                            __cstring       __TEXT           f�    �      f�                           __ustring       __TEXT          0��    N       0��                            __unwind_info   __TEXT          ���    D�     ���                            __eh_frame      __TEXT          �|�    ��      �|�                               �  __DATA           �      	      �      P                  __nl_symbol_ptr __DATA           �            �               �          __got           __DATA          �    P      �               �          __la_symbol_ptr __DATA          X"�    x      X"�               �          __mod_init_func __DATA          �6�           �6�            	               __const         __DATA          �8�    h�     �8�                            __cfstring      __DATA          X��    @       X��                            __data          __DATA          ���    ��      ���                            __thread_vars   __DATA          x\�    x       x\�                           __thread_bss    __DATA          �\�                                         __bss           __DATA          �r�    H�                                    __common        __DATA          @�    �                                       H   __LINKEDIT       0�    hb      `�     hb                   "  �0    `� �P  �� 
    *              (  �   `|                h          j�  � /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation      `         )j�   /System/Library/Frameworks/Security.framework/Versions/A/Security          `               /System/Library/Frameworks/Kerberos.framework/Versions/A/Kerberos          p         	(U   /System/Library/Frameworks/SystemConfiguration.framework/Versions/A/SystemConfiguration    8          �   /usr/lib/libc++abi.dylib           8              /usr/lib/libSystem.B.dylib      &      @� x�  )      �� �    �(      @loader_path/../lib/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                UH��AWAVATSI��H��H��諚� H���shI��H��sC�6�H��M��u5�AL��H��H�HH��A�   LE�L����� H�CI��L�#L�sH��H��L��L���K�� B�3 [A\A^A_]�H���
� H��p���L��x���L��p���L�e�H��P���L�}�H��0���H��L��E1�M���fN  H��H�M�H��p���H9�tH��tH�H��(H���H��p���H�� ���0���tH��@������ H�M�L9�tH��tH�H��(I���H�E�H�� L�����P���tH��`����ԉ� L��(���ƅ���ǅ���TEXTƅ��� H�����H����Y  �����tH��(���葉� (*�� )�`  fǃ"   H�� H� H;E�uH��H���   [A\A^A_]���� H��H�M�L9�tG�|H���������   �   H��H�M�L9�u
H�E�H�� �+H��u��`���u+H�M�H9�u7H�E�H�� �<H�H��(I��L�����`���t�H��p����u�� H�M�H9�t�H��tH�H��(H��H���L����� UH��AWAVSPI��H���? t)L�{pH���   Hǃ�       L9�t:H��tAH�H��(�6L���   H���   Hǃ�       L9�tDH��tKH�H��(�@I�H�� L���H���   I�V H��tDI�F L9�u6L�9H�8H�L���P�4I�H�� L���H���   I�V H��tI�F L9�tH��H��H�     H��H��[A^A_]�L�9H�8H�L���P��H���d  H���\  fff.�     UH��SPH���G`t	H�{p�(�� H�{0H�KPH9�tH��tH�H��(H���H�H�� �H�K H9�tH��tH�H��(H���H�H�� H��H��[]� H��[]�fffff.�     UH��H�=)� �    UH��AVSI���   �̄� H��H��L���.   H�5�ֹ H��ҹ H������ I��H���Ǆ� L����� D  UH��SPH���bϒ H�3� H��H�H��[]�ffff.�     UH��SP�   �P�� H��H���i�� H�5�ֹ H�Gҹ H��胄� ffffff.�     UH���)�� �փ� f�UH��H�u�� H��Gu]�H�]���� UH��SPH��H�P�� H��Gt	H�{螃� H��H��[]鐃� UH��AVSI���    舃� H��H��� H�H��H��A�FuI��I�FH�GA�
Թ HD�]�D  UH��H�Mѹ ]� UH��]�f.�     UH��]�6�� fD  UH��   �.�� H�
� @H�E�)   ��  �@) A�I�v����IEvIEVH�}���t� H�HH�K W� H�@    �E������H�}��y�����w� H���E�tHH�}��)�=���7H���E�tH�}���v� �H���E�tH�}���v� H�}���  H���|s� H��H�}���  H���hs� UH��AWAVAUATSH��8  H��H�2չ H� H�E�W�A��I��H�G    ����  I�FH����  H��������  H�����L���4t� H��������  W�)�����Hǅ����    E��L������L�������1ffffff.�     H������I�F(�����AI��L�sL�sL��L��D���  H�H�I��D uFL�sL;ss(������t�H������H������L����a� ��    H��L���5  I��H�C�������tH�������gu� L�5 ɹ I�I�N@H������H�@�H������I�FHH������H�|й H��H�����������tH�� ����u� H�������ft� I��H������L���Gt� H��8����t� H��ӹ H� H;E�u<H��H��8  [A\A]A^A_]���H���Z���H���a  H�CH�Fӹ H� H;E�t��Xu� � I��H���   L���;q� I���!I��L�s�I��������tH�������]t� H�������  H���i   L����p� �UH��SPD�H�BD����A��HEBH��HEJH��H��H���q� H�HH�K W� H�@    H��H��[]��     UH��AWAVSPL�7M��t9H��L�L��M9�u�6ffff.�     I���M9�tA�G�t�I���s� ��H��[A^A_]�H�;L�sH��[A^A_]�ts� @ UH��AWAVAUATSH��H��L�-]Ϲ I�EhH�E�H���   M�}@L�H�wL�5�ƹ I�FI�NH�H�@�H�H�G    H�L�`�I�L��H�u�貌� IǄ$�       AǄ$�   ����I�FI�N H�KI�V(H�I�H�TH�I�N0H�@�H�I��L�+H�E�H���   L�{H�}���q� H��͹ H��H�CW�CXCh�Cx   H��[A\A]A^A_]�I��I��H��L���q� �I��H��H���r� L��� o� �     UH��AWAVATSH��A��I��H��H��й H� H�E�H�}�H�޺   �p� �}� ��   A�ufA�  �I�G�  I�G    E1�ffffff.�     H�H�@�H�|(H�GH;G tH�HH�O� �ffff.�     H��PP���t+D8�t"��L���]o� I��A�t�I��u��   �1��1�M�����4�   H�H�@�H�<t �7�� H��Ϲ H� H;E�uH��H��[A\A^A_]���q� � H���Zq� H�H�H�L H�@��D$u�Pq� �   ��hq� H���:q� H���m� H�������fD  UH��AWAVSPH��L�5|Ĺ I�I�N@H�H�@�H�I�FHH�GL�H��˹ H��H�G�GXt	H�{h�p� L����o� I��H��L���o� H��H��H��[A^A_]�p�  UH��AWAVAUATSPI��H��������
L�'H�_I��M)�I��H���������L��M�oI9��'  I�VL)�H��H��H�L9�LG�H�UUUUUUUH9�LC�M��tI9���   J��    H�<@��o� �1�K�4H��J�m    L�W��H��H�D�    L�<�I��L9���    H�C�H�B�K�J�C�H���H�C�    H�C�H��L9�u�M�&I�^I�M�~I�NL9�u�(ff.�     H���L9�t�C�t�H�{��o� ��L��H��tH���o� L��H��[A\A]A^A_]�I�M�~I�NH��u���L���j   ����D  UH��AVSH��L�wH�GL9�u!H�;H��t6[A^]�n� �     H��L9�t�H�H�H�K�@�t�H�x��n� H�K��[A^]�fD  UH��H�=�� ����UH��AWAVSPH��H�GX    �G@L�wAA��A��LEwPLEH�O`��tK�>H�SXL�sL�sH�S ����   H�{@K�>H�S@H���H�ʨH�KX�   HE�1��k� �C@�tH�CH���L�L�s0L�s(H�C8�C`t^L��H��tFI��   �H�      H��H��H)�H��H�H��H��H��H)�I�I�����I)�I��  �L�s0M��t
D��I�L�s0H��[A^A_]�D  UH��AWAVAUATSH��8I��H��˹ H� H�E�H��������
H�L�oM��I)�I��H���������L��M�|$I9���  I�VH�:H)�H��H��H�?L9�LG�H�UUUUUUUH9�LC�H�U�M��t$H�u�I9��R  J��    H�<@�l� H�u��1�H�E�K�dL�$�L�e�L�e�K�H��H�M��u_H�FI�D$A$M�|$I9�tkW��    I�E�I�D$�AM�AL$�AE�I���I�E�    I�E�I��H9�u�M�.I�^�*I��H�VH�vL���SX� I�M�nL��M�|$I9�u�I��M�&L�m�M�~I�FI�NH�E�L�m�L9�t/H����fD  H�C�L9�H��tH�]��t�H�{�k� ��L��H��tH���k� H�)ʹ H� H;E�uL��H��8[A\A]A^A_]�L��������!l� �,���H��H�}�����H��� h� �     UH��AWAVAUATSH��HH��I��H��ɹ H� H�E���tI�FH��u	�C��H��t<H�E�    �k� �     M�nA�L��tI�~H�u�1��s� I���gk� �8"u)E1�H�\ɹ H� H;E��V  D��H��H[A\A]A^A_]�D�#A�����I�VI�vH��ID�HE�L�A�L9E�u
I��L9�u��(L9�t#I���t��  I�FH�E�A)E���   H�
f�H��I9�t ���'t��H����H�u�H�M��U�H�}��D�u�H�}�H�M�D���A���HD�LE�I�I)�H)�H�}�H��L����d� H�}�H���4���A���E������H�}��^g� �����h� H���\���fff.�     UH��AVSI��H��H���o� ���tH�SH9�t1�[A^]É���H9�u�H���t&��tH�[�H��H��L��H���@l� ����[A^]�H���   H�������fff.�     UH��H�=� �    UH��AVSI���   ��f� H��H��L���.   H�5��� H���� H����f� I��H����f� L���c� D  UH��SPH���b�� H�;Ĺ H��H�H��[]�ffff.�     UH��]�f.�     UH��]�f� fD  UH��   �f� H�
��tI�FH��s�\����H��rR��tI�N��rރ�>v�<I�N��rރ�>w-H�!      @H��s:T�uL���"   �'   ��  ��   L���\   1���`� H�����   E�>A��tM�~�A��H�
\� ��t:A�uI�NH�E�H�HA �  I�VI�vH�]�H���K� H���  W�H�}�H�G    L��L)�L)��f\� M)�M9��c  M�n�/�     A�ID�B�4 �   H�}��=\� I�M9��.  I�FA�H��ID�B�<!\u�A�H��ID�A�|xtA�H��ID�B�|!Xu�A�H��ID�A�L�q�@��	w���A�T�r�@��	w?�   �q�@��w���A�T�r�@��	w�g�q�@��rL�����A�T�r�@��	vJ�r�@��w��ɉ�	΃������Q�r�@��r8�������	΃�������4���A�T�r�@��	w���Љ�	΃��������©��	΃��������ʻ   ��H�}��[� �����H�E�H��[A\A]A^A_]��� I��H�E�� t
sU����   A�����I�FID�IENH�L)�H��
�u  A�E�HЀ�
s����01��X����H���w�����H���w<����F�H�����  �������  A�M�QЀ�
s����S�Q���s<����F�H����x  ������l  A�M�QЀ�
s����U�Q���s>����H�Q�����  ������  A�U�r�@��
s����W�r�@��s>����I�Q����  �������  A�U�r�@��
s����X�r�@��s?����J�r�@���  �©���  A�}�w�@��
s����X�w�@��s?����J�r�@����  �©����  A�u�~�@��
s����\�~�@��sC����N�w�@����  �ǩ����  E�EA�p�@��
sA����^A�p�@��sBA����N�~�@���  �Ʃ���  ���������H����  I��M�������A�p�@���)  A���A���  E�MA�q�@��
sA����"A�q�@��sA����A�q�@����   A���A����   E�UA�r�@��
sA����"A�r�@��sA����A�r�@����   A�©A����   A�u	D�^�A��
s����D�^�A��s����
D������]A����  wD��% ���= �  tmD������&A���� wUD�������H����Q� D����$?���H����Q� D����$?���H���Q� A��?A�΀A��H��[A^]�Q� [A^]ÿ   ��S� H��H�5��� H���6   H�5'�� H�С� H���T� I��H����S� L���5P� ffff.�     UH��SPH���r�� H�c�� H��H�H��[]�ffff.�     UH��AVSH��`H��H�˱� H� H�E��uH�FH�E�)E��H�VH�vH�}��M?� H�6ι H�E�L�u�L�u�H�u�H��L���   H�M�H�}�H9�tH��tH�H��(H���H�E�H�� ��E�t	H�}��R� H�=�� H� H;E�u	H��`[A^]��FS� H��H�M�L9�u
H�E�H�� �H��u�E�uH���O� H�H��(I��L����E�t�H�}��?R� H����N� �    UH��AVSH�� I��H��H���� H� H�E��uH�FH�E�)E��H�VH�vH�}��:>� H�C     �    ��Q� H�
�� H�}��M� H�HH�M� )E�W� H�@    H�5��� H�}���L� H�HH�K W� H�@    �E�u(�E�u1�E�u:H���� H� H;E�uCH��H��p[A^]�H�}�� O� �E�t�H�}���N� �E�t�H�}���N� H�{�� H� H;E�t��O� H���E�u�E�u(�E�u<H���hK� H�}��N� �E�t��	H���E�t�H�}��N� �E�t��	H���E�t�H�}��sN� H���#K� ff.�     UH��AWAVAUATSH��HI��I��H�⬹ H� H�E���tI�NH��u�r����H��tiI�^H�E�    H�ߨtI�~H�u��V� A�$A�����I�VI�vHE�HE�H�<H9}�t&H��t1��D�;A��'t<A��_t6H��H9�u�E1��A�H�H�� H� H;E���  D��H��H[A\A]A^A_]�E1�H9�t�H���t̨L�e�uI�FH�E�A)E��	H�}��9� �]�A��A���L�}�H�E�L�}�LD�L�e�L�e�MD�O�,'L���_   L���mR� H��ID�H��L)�L9�t0H��H��L9�t%L�e���     H��I9�t���_t��H����L�e�H�M�H�U��D�u�H�U�H�M�D���A���ID�LE�I�I)�H)�H�}�H��L���J� �]�A��A���L�}�L�}�MD�L�e�L�e�MD�O�,'L���'   L���Q� H��ID�H��L)�L9�tGH��H��L9�L�}�t<H�u�� H��I9�t���'t��H����D�u�H�U�H�M�D���A���L�}�H�u�H�M�H�U���HD�LE�I�I)�H)�H�}�H��L���MI� H�}�L���c���A���E�����H�}��K� ������cL� �� H���E�t	H�}��K� H���=H� D  UH��H�5ǹ H��Gu]�H�]�`K� UH��SPH��H�ǹ H��Gt	H�{�>K� H��H��[]�0K� UH��AVSI���    �(K� H��H��ƹ H�H��H��A�FuI��I�FH�GA�
  I��������tH�������]9� M���[  I��M9�u�L������L������M9�uM��   H������H;������?  I��$�   ��   H������H���  L������L������M9���   L��P���@ L�+A�u'I�GH��`���A)�P����ffffff.�     I�WI�wL����$� L��L����	  I����P���tH��`����}8� M����  I��M9�u�H������L���   L;�   sI�    I��D������L�������H��������_  D������L������I��L���   �0  �8� I��H�������uH�AH��0���)� ���H�������H�QH�qH�� ����$� H�������uH�AH�����)� ����H�QH�qH�� �����#� I�} H��t
F�h  L�s L�k(M9���   H������H������L�������     A�uI�FH������A)������f�I�VI�vL��� � H��x���L����<  ��������tH�������(� ����   I��M9�u�H������L�`8L�h@M9���  H������H������L������fff.�     A�$u)I�D$H������A$)������!ffffff.�     I�T$I�t$L���n� H��x���L����:  ��������tH��������'� ����   I��M9�u��  M���  A���   ����I���   HE�H���\���M���   �udI�GH��@���A)�0����_���   ����H���   HE�H���0���L���   �ukI�GH�� ���A)������fM���  I���   H��0����� H��0���H���G;  A�����   A�ujI�GH�� ���A)�����lH���   H�������;� H������L����:  A�����   A���   I�GH������A)������   I���   I���   H�������� H�����H���s9  ��uH���   L����;  A�������tH�� ����X&� ��0���tH��@����C&� E��������rH���   H���   H�������n� H������L����8  ��uI���   L���L;  A��������tH��������%� ������tH�� �����%� E�������H�_�� H� H;E�uiL��H��h  [A\A]A^A_]�H�=2&� ��%� �������W�&� H�&�     H�5�%� H�=�s� H�c���%� H�=�%� �%� �����&� H��������t+H�������0%� �H�������t1H�� ����%� �#H����������   H�� ���H�������mH����0���tpH��@���H�������QH��������uE�RH��������u7�DH����P���H��`����H����p���u�#H���E�H�E�H������tH������H�8�x$� H���(!� H���E�u���D  UH��AWAVAUATSH���   H��H�₹ H� H�E���GOpti�Gions�G �G   W�GG,G<GLG\GlG|��   ��   Ǉ�       H�BH���   
��   H�B    L��   ��   ��   ��   H�E}� H��H��   L��   HǇP      (
I�E H�� �H��tH�H��(I��L���H���  H�������&  H���  H�������&  H�� �����  H��P  H;�0���uH��0���H� H�� �H��tH�H��(H��0���H��0����H��   L9�u
� ��P����tL��X���M��u�  A��A��M����  I��u+H��Q����tH��`����9-��   �  ff.�     �L��`���L��H��Q���HDʀ9-uI�U�H��R���HD��2@��-�G  I��rQƅ�����	��--  u`I���H��S���LD�I���I�����  I���[  C�$������L�������  H��Q����tH��`����	��--  �_  ��p�����tH��x����f�     ��H���  H��Q����tH��`����<-��  <"��  �   ff.�     I9�t;���:��  ��=��  ��{��  H���� wԀ�	t��  f.�     H��p���H��P������ ��P��������H��`������ �����I���N  @��"�x  ǅ����   H�E�H;E�s�
� �H�@ H��H�E��H�}�H�������   H�E��H��x���L��H��H�XH���   HD�H���o� I��H�����H��H������L�� ���H��x���L��L��L��� � C�& ��P���tH��`����� H����������)�P���H��`�����P�������H��X���HE�H��L��p�����  �H��`���H��L��Q���ID�D�A��-��  A��"��  A�   ffffff.�     L9�t7F�A��:�{  A��=�q  A��{�g  I��A�� w�A��	t��S  L�e�L;e�s)�uAH��`���I�D$(�P���A$I��L�e��$���H�}�H��P��������I��H�E�����L���M� I��L�e������H�u�H�U��1�1�W�H�C    H��H)�H��I��H���������H��L���  M�~W�AFI�F(    H�u�H�U�H��H)�H��H��L���  I�~0��p���uM��H�E�H�G(�p����%H��x���H�u��� M����p���t	H�}��6� H�]�H��tEL�u�H��I9�u�.ffffff.�     I���I9�tA�F�t�I�~���� ��H�}�H�]���� H�]�H��t7L�u�H��I9�u� �I���I9�tA�F�t�I�~��� ��H�}�H�]��� H�Aw� H� H;E��u  L��H��h  [A\A]A^A_]ÿ0   �� H����P���uH��`���H������(�P���)������H��X���H��`���H�������� A�H������H���  E1�H�5�g� H��  H���� �  �0   �'� H����P���uH��`���H������(�P���)������H��X���H��`���H�������� A�H������H����  E1�H�5g� H�  H���� �  �0   �� H����P���uRH��`���H������(�P���)������N�0   �u� H����P���uhH��`���H������(�P���)������dH��X���H��`���H�������Q� A�H������H���_  E1�H�5ef� H�^
  H���Z� �T  H��X���H��`���H�������� A�H������H���  E1�H�5f� H�
  H���� �
  �0   �� H����P���uhH��`���H��@���(�P���)�0����d�0   �y� H����P���u{H��`���H�� ���(�P���)�����w��� H�������N����   H��X���H��`���H��0����?� A�H��0���H����  E1�H�5Se� H�L	  H���H� �EH��X���H��`���H�������� A�H�����H���V	  E1�H�5e� H�	  H���� I��L�e���P����,  �d  I���K  I���C  �   L��I��L���_  ��   �   L��I��H���  ��p�����   ��   �rI���������   H�� ����   I����0�����   H��@����   I����   I���   I��������ttH�������fI��������t_H�������QI����p���ua�hI����P���tJ�   I���lI��������t&H�������I���RI��������tH�������I� E��u5��P���u=��p���t	H�}��)� H�}��@���H�}��7���L����� H���a� ��P���t�H��`������ ��p���u���    UH��AWAVAUATSH��xI��H�er� H� H�E�W�H��p���H�G    H���,   1��� H����+  I��I�GH��`���L��h����5f�     H�E�I�GE�AL���,   1��Q� I��H�����  E�'A��t
M�gM�w�
A��L��`���M9�MC�I����'  I��s_C�$��x���H��y���M����   D  B�# ��x�������   H�E�H�E��x���)E�L��p���L�e��   �     L��H��L�xH���   LD�L���� H��H�E�I��L��x���L�e�L��h���H��L��L����� B�# ��x������u���H�U�H�u�L�e�L��L��p������ L���m  H���U  � u H�HH�M� )E�� ffffff.�     H�PH�pH�}��?�� �E�t	H�}���� I�FI;FsVH�M�H�H(E� W�)E�H�E�    H��I�F��x���tJH�}��� A���uA�����?ffffff.�     L��H�u��  �E�I�F��   ��x���u�A���t�I�GL9��r  L��`�����tM�wI�MI��I)�I����C  I��s-C�?�U�L�e�H9�uPC�< L��h���A�������`D  L��H��H�XH���   HD�H����� I��H�E�H��H�]�L�}�K�4.H��L��L���� C�< L��h���A��%���I��� �����    H�}��w� ��x����
� H��L���  �E�t	H�}���
� H�mi� H� H;E�uH��H�� [A^]��s� H���E�t	H�}��
� H���Q� f�     UH��AVSH�� H��H��H�i� H� H�E�H�5f�� L�u�L���4
� H��L���#  �E�t	H�}��D
� H��h� H� H;E�uH��H�� [A^]���
� H���E�t	H�}��
� H����� f�     UH��AVSH�� H��H��H��h� H� H�E�H�5�� L�u�L���	� H��L���  �E�t	H�}��	� H�Mh� H� H;E�uH��H�� [A^]��S
� H���E�t	H�}��	� H���1� f�     UH��AWAVAUATSH��8I��H��H��g� H� H�E�H��������
H�L�I)�I��H���������L��M�oI9���  I��H�sH�>H)�H��H��H�?L9�LG�H�UUUUUUUH9�LC�H�u�M��tI9���  J��    H�<@��� �1�H�E�K�L�<�L�}�L�}�K�Lm H��H�E�Mc$$I����.  E�.A��sC�$A�I��E��u5�BL��H��H�HH��A�   LE�L���a� I�GI��M�7M�gI��A��L��L���
)M�H��H�B    H�u�H�U��   �E�t	H�}��� �E�t	H�}��� H��Y� H��H�H�/d� H� H;E�uH��H[]��:� H���E�u�E�uH���� H�}��Z� �E�t�H�}��K� H�����  UH��SH��HH��c� H� H�E�H�FH�E�)E�W�H�F    H�BH�E�
)M�H��H�B    H�u�H�U���   �E�t	H�}���� �E�t	H�}���� H�%Y� H��H�H�Oc� H� H;E�uH��H[]��Z� H���E�u�E�uH���;� H�}��z� �E�t�H�}��k� H����  UH��SPH��H��X� H��H��Gt	H�{(�:� H���� H��H��[]�$� @ UH��AWAVSH��HA��I��H��H��b� H� H�E��u(H�FH�E�)E�A�t*I�WI�wH�}��!� �'H�VH�vH�}��� A�u�I�GH�E�A)E�H�u�H����Q� H�X� H��H�D�sH�E�H�C((E�CW�)E�H�E�    �E�tH�}��\� �E�t	H�}��M� H��a� H� H;E�uH��H[A^A_]���� H���E�t�&H���E�u�E�uH������ H�}��� �E�t�H�}���� H����� ff.�     UH��SPH��H�XW� H��H��Gt	H�{(�� H��H��[]���� fff.�     UH��SPH��H�W� H��H��Gt	H�{(�z� H������ H��H��[]�d� @ UH��SPH��H��V� H��H��Gt	H�{(�:� H��H��[]�z�� fff.�     UH��SPH��H��V� H��H��Gt	H�{(��� H���@�� H��H��[]��� @ UH��AWAVSPL�7M��t9H��L�L��M9�u�6ffff.�     I���M9�tA�G�t�I���� ��H��[A^A_]�H�;L�sH��[A^A_]�t� @ UH��AWAVSPL�7M��t9H��L�L��M9�u�6ffff.�     I���M9�tA�G�t�I���'� ��H��[A^A_]�H�;L�sH��[A^A_]�� @ UH��AWAVAUATSH��H��_� H� H�E�H�}��E� H����   I��H��H��������
H9���   I��I��J��    H�<@� � I��H�H�CJ�m    L�I��H�CM9�tPE1��,f�     K�,H�HH�O K�,H��I��L9�tK�<.C�,t�K�T,K�t,�� ��M�L�sH��^� H� H;E�uH��[A\A]A^A_]��� � H��莑��I��H�}��P   L����� I��M��u&L�sH�}��3   L����� ff.�     I���t�C�D.�t�K�|.���� ��fD  UH��AWAVATS� t	[A\A^A_]�H��L�7M�>M��t�M�fL��M9�u�-f.�     I���M9�tA�D$�t�I�|$��E�� ��H�H�8M�~[A\A^A_]�,�� fff.�     UH��SH��(H��H��]� H� H�E��uH�FH�E�)E��H�VH�vH�}��/� H�}��  H����  � uH�HH�K �H�PH�pH����� �E�t	H�}���� H�3]� H� H;E�u
H��H��([]��;�� H���E�t	H�}��i�� H����� �UH��AWAVATSH��H��������
H�L�I)�I��H���������L��M�gI9��8  I��H�sH)�H��H��H�6L9�LG�H�UUUUUUUH9�LC�M��tI9��  J��    H�<@���� �1�K�4H��K�dH��I�~H�|�A�W�AI�F    L�4�I��H�L�{I9���   fD  I�w�H�r�AO�J�AG�H���I�G�    I�w�I��H9�u�L�#L�{H�L�sH�KM9�u�'f�     I���M9�tA�G�t�I����� ��M��M��tL����� L��[A\A^A_]�H�L�sH�KM��u���H���o����x��f.�     UH��AWAVAUATSH��(H��H�U[� H� H�E��H�OA��A���L�wLD�LEoM��tqH�M�K�.H�E�L�}�ffff.�     E�&L����c� L��H�5*a� �%n� E��x'H�@F�$�L����� A�� @  tI��I��u�L�u��	H�}���� H�M��tH�KI)�H��1�L����� H��Z� H� H;E�uH��H��([A\A]A^A_]���� H��H�}��p� H����� �     UH��AWAVAUATSH��(H�HZ� H� H�E��H�OA��A���H�GH�M�HD�H�}�LEoH�E�N�4(L�}�L�%:`� f.�     M��t>A�^�L����b� L��L���m� ��x(I��H�@��L����� I���� @  u�I���L�u��	H�}��� H�]������H�E�HECHESH�L)�I)�H��L���]�� H�xY� H� H;E�uH��H��([A\A]A^A_]��x�� H��H�}��L� H���\�� @ UH��AWAVAUATSPH��I��L�?M�'M��tGM�oL��M9�u�! I���M9�tA�E�t�I�}��W�� ��I�?M�g�I�� W�AI�G    AH�CI�GW�H�C    L�{M�fM�,$M��t]L�u�M�t$L��M9�u�(f�     I���M9�tA�F�t�I�~����� ��I�<$M�l$���� W�A$I�D$    L�u�W�CA$H�C(I�D$AI�G    M�vA�t	I�~�~�� H�C@I�FC0Af�C0  H��[A\A]A^A_]�f�     H��t-UH��AVSH��I��H�6�����H�sL�������H��[A^]��� �ffff.�     UH��AWAVAUATSH��(  I��I��H��W� H� H�E���tM�~I��sI����   I�N�UA��A��I��r�M�f�I�NLE�A�<$-u4A�|$-u,I���I�����
  I��I���^  C�?�E�L�u��  M�f�LE�A�<$-u*I��I����_
  I��I���  C�?�E�L�u��0  A���   ������I���   HE�H����  ��uqI���   H�QH�U�)�p����toI�VI�vH��P����.� A�} L������L������tq��p�����   H�E�H��@���(�p���)�0�����   I���   H��p������ A��u�I�FH��`���A)�P���A�} L������L������u�A�} ��p�����  �~  L��H��H�HH��L��A�   LE�L���)�� I��H�E�I��L�m�I��L�}�L��L��L���]�� C�> H�u�L���  A���E��!  H�}����� �  H��x���H�u�H��0����� D��0���E��A��A��L��@���H��1���L������LD�L��8���L������MD�K�,L��_   L����� H��HD�H��L)�L9�t:H��H��H9�t/H��1����ffffff.�     H��H9�t&���_t��H����H��1���H������H������� D��0���H��8���H��@���E��A��A��E��HD�LE�I�I)�H)�H��0���H��L���:� ��0���D��1���H��2���H�E�H��8���L��@���W�)�0���H�E�Hǅ@���    ��p���u1��H�}��e�� ��0�����p���D��q���H�M�H�U�H��r���H��x���L�}��tH��@����'�� ��P���uH��`���H�� ���(�P���)�����H��X���H��`���H������6� D�����E��A��A��L�� ���H�����L������LD�L�����L������MD�K�,L��_   L������ H��HD�H��L)�L9���   H��H��H9�L��������   H������H��H9�t���_t��H����D�����H�����H�� ���E��A��A���   L��H��H�HH��L��A�   LE�L����� I��H�E�I��L�m�I��L�}�L��L��L���@�� C�> H�u�L���  A���E��  H�}��� ��  L������H�����H������H������E��HD�LE�I�I)�H)�H�����H��L����� �����D�����H�����H�E�H�����L�� ���W�)����H�E�Hǅ ���    ��P���u1��H��`����� �������P���D��Q���H�M�H�U�H��R���H��X���L��`����tH�� ������ L������A�} ��p�����  @��uH�E�H�� ���(�p���)������H��x���H�u�H��������ޒ D������D����A��L�� ���L������ME�HE�����H��thE1�L�}�ff.�     G�,&L���Y� L��H�5�V� ��c� H�A��H���Q(A��L���� G�,&I��L9�u�D������L�� ���L������������H������H�E�H������W�)�����H�E�Hǅ ���    ��p���u1��H�}��� ������D��p�����q���H�M�H�U�H��r���H��x���L�e��tH�� ����d� ��P���uH��`���H������(�P���)������H��X���H��`���H�������sݒ D������D����A��L������L������ME�HE�����H��tiE1�L�}�fff.�     G�,&L���SX� L��H�5�U� �b� H�A��H���Q(A��L���L� G�,&I��L9�u�D������L������L������������H������H�E�H������W�)�����H�E�Hǅ����    ��P���u3D��P�����Q���H�E�H�M�H��R���H��X���L��`���L�������RH��`����� ������D��P�����Q���H�E�H�M�H��R���H��X���L��`���L������tH��������� ��p�����P���������H��X���I��LDǉ�$tH��x���I9�t1���tOL��`����5@��@��I9�u����   H�u�����   L��`���L����� ����L���H� ��p���$��t	H�}��2� A���uqA���   ��tI���   H��u�S����H��tJA�6����@��I�VI��LD�I9�u.����   M���   @����   I�~L����� ��A���E1�H�SM� H� H;E���   D��H��(  [A\A]A^A_]�H��q�������������7���H��1ҐD��Q���D:�������L�BH9�L��u�����I�ř   @���g���A�@���q���H��1�fD  A�TA:T
   �R��D  UH��H�=�� �pQ��UH��AWAVSH��(I��H��H��4� H� H�E�H���7  ��t)H��L����Ӛ H��4� H� H;E�uUH��H��([A^A_]ÿ0   ��՚ H��H�5~� H�}��MC��A�H�u�H���^  E1�H�5l%� H�]  H���	֚ �>֚ I���E�u
К H��H��[]�J͚ fff.�     UH��SH��HH��.� H� H�E�H�FH�E�)E�W�H�F    H�BH�E�
)M�H��H�B    H�u�H�U��R����E�t	H�}��Ϛ �E�t	H�}��tϚ H��#� H��H�H��-� H� H;E�uH��H[]��
К H���E�u�E�uH����˚ H�}��*Ϛ �E�t�H�}��Ϛ H����˚  UH��SPH��H��#� H��H��Gt	H�{(��Κ H���0̚ H��H��[]��Κ @ UH��AWAVSPH��L��  M��tEL��  L��M9�u�(fD  I���M9�tA�G�t�I���Κ ��H��  L��  �rΚ L���  M��tDL���  L��M9�u�'D  I���M9�tA�G�t�I���7Κ ��H���  L���  �"Κ H���  H���  H9�tH��tH�H��(H���H�H�� �H���  H���  ����H���  H���  ����L��p  M����   L��x  L��M9���   I�ǀ�fD  I�H�� L���I�G�M9�I��tXA�G`t	I�p�x͚ I�0I�OPH9�tH��tH�H��(H���f.�     H�H�� �I� I9�t�H��t�H�H��(�H��p  L��x  �͚ H��0  H��P  H9�tH��tH�H��(H���H�H�� �H��   H��   H9�tH��tH�H��(H���H�H�� ����   u|���   ��   ���   ��   ���   ��   ���   ��   L�shM����   L�sPM����   L�s8M���T  L�s M����  ���  H��[A^A_]�H���   �2̚ ���   �x���H���   �̚ ���   �l���H���   � ̚ ���   �`���H���   ��˚ ���   �T���H���   ��˚ L�shM���H���L�{pL��M9�u�" I���M9�tA�G�t�I���˚ ��H�{hL�sp�˚ L�sPM������L�{XL��M9�u�Lffff.�     I���M9�t2A�G�uA�G�t���    I���7˚ A�G�t�I���'˚ ��H�{PL�sX�˚ L�s8M�������L�{@L��M9�u�,ffff.�     I���M9�tA�G�t�I����ʚ ��H�{8L�s@��ʚ L�s M���i���L�{(L��M9�u�,ffff.�     I���M9�tA�G�t�I���ʚ ��H�{ L�s(�xʚ ��&���H�{H��[A^A_]�\ʚ fff.�     UH��]�f.�     UH��]�6ʚ fD  UH��SPH���   �)ʚ H�
H��H9�u��uH9�tpH���tjH������L����  L���6  I�D$H������A$)�����W�A$I�D$    I�G H����  L9���  I�� H�E�I�    �  I�D$H������A$)�����W�A$I�D$    I�G H��tL9�t$I�� H��`���I�    �'H��`���H�     �H��@���H��`���I�L���PI�FH������A)�p���W�AI�F    Hǅ0���    H������L��@���H��p���L�����H��L��E1�M������I��H��0���H�����H9�tH��tH�H��(H���H�����H�� ���p���tH�������Ě H��`���H��@���H9�tH��tH�H��(H���H��@���H�� �������tH�������eĚ A���   ��;  I���   �1  H�E�H�     �H�u�H�u�I�L���PI�FH������A)�����W�AI�F    H�E�    H������L�}�H������L��p���H��L��E1�M������I��H�M�H��p���H9�tH��tH�H��(H���H��p���H�� �������tH�������Ú H�M�H�}�H9�tH��tH�H��(H���H�E�H�� �H��H���������tH�������JÚ L������H������I9�twM�fhM�np�3ffff.�     I�GI�EAAE I��M�npM�npI��0I9�t:M;nxs$A�t�I�WI�wL���.�� ��fff.�     L��L���uU��I���M�~PM��toL��H��PM�fXL��M9�u
�� D�e�E��A��A��L�u�H�]�IE�LEm�M��t^E1�L�u�fffff.�     F�$;L����"� L��H�59 � �4-� H�A��H���Q(A��L����ݓ F�$;I��M9�u�D�e�L�u��]�H�E�H�E�H�E�H�E�W�)E�H�E�    �L�}�A�tI����� �E���E�'A�_H�M�H�U�I�OI�WM�w��L��u
A�EeA�E fals�E� A�H����q  H����M�f������MEfIEFA�|�}�J  H9���  H�NI��I)�I�����  I��sjB�m    ��h���L��i���H9���   C�/ �E�tH�}�I��H������ H��L���h���)E��M���L��P���L�m���   H�E��   H��X���L��H��H�XH���   HD�H��貴� I��H��x���H��H��h���L��p���H��X���H��L�H��L��L���ӹ� H��H��`���C�/ �E��H����X�������H��H��x���H�U���u� �M�L���H�M�H�E�� L��H������蒱� A�I�F������IEF��IENH��t31��ffffff.�     H��H9�t�4��-t��!t���     H������L��1��-�� L�cL;cs%A�uOI�FI�D$AA$�Kff.�     H��L��H�U��q  I��L�c�E�������afffff.�     I�VI�vL��萟� I�|$�E�uH�E�H�G(E��@ H�U�H�u��c�� I��0L�cL�c�E��=���H�}����� �/���L�u�M��t=H�]�L��L9�u�&�     H���L9�t�C�t�H�{�踲� ��H�}�L�u�該� H�B� H� H;E�u5H��`���H�Ĉ   [A\A]A^A_]�H��h����K���H��h����-����� I���[I���bI���E�tbH�}��C�� L���� �(I��A�$tI�|$�%�� �I��H��`���L�`�� I���E�t	H�}����� H��`���胣��H�}��
>��L��蚮� f�UH��AWAVAUATSH��(H���   �{   轮� H�����   L�{�'ffff.�     H�߾{   L��萮� H�����   L�p�������H�KID�HESL9�v�H��H)�H�~H��   @ D�D�A��,t$A��}tL�I��H��I��u��fff.�     H�H���s���H��H��H9��d����|
�}�Y���H��H���l�� �I����H�CA��A����M�L�{L�}�H�E�LD�L�cL�e�MD�O�,'L���!   L������ H��ID�H��L)�L9�tH��H��L9�u!H�M��U�H�u��8fff.�     H��I9�t���!t��H����D�3H�sH�KD���A���HDM�LE�I�I)�H)�H��H��L��H��([A\A]A^A_]钭� @ UH��AWAVAUATSH��  I��H������H��� H� H�E���tII�FH��tG��tqW�)E�H�E�    ���  A���   ��  I���   H���  ��   ��H��u�W�H������ H�@    H�
  L9��e  L�{�%�     I�GH�CAfA�  H��I��M9��1  A��tI�GH��u���    ��H��t�L���{   1���� A�H���t0I�W�����I�GIEwHEЀ|2�}u�)ffffff.�     I�G��I�W��HEЀ:!�o�����M���H�{贠� �?���ffffff.�     H�C��H�S��HEЀ:!�����H��L9�ta��tH�CH��u������     ��H�������H�߾{   1��F�� �H���t�H�S�����H�CHEsHEЀ|2�}u��L��H��H��[A^A_]�ff.�     UH��AWAVATSH�H�GH)�H��H���������H��H9��"  I��H�VUUUUUUH9��  M�fH��L�<vL��貟� I�L��H)�H���   W�H��ffff.�     I�T$�H�Q�AL$�I�AD$�I�D$�    I�T$�H�Q�AL$�I�AD$�H���I�D$�    I�T$�I��H9�u�M�&I�^I�I�FM�~L9�u
H��H�=�� �S�� H��裌� �,H��H�}������H���E���   H�}��ŏ� H���u�� H��L����� H���b�� H����P���tOH��`����AH��H���������H����0���t6H��@����l�� H����� H����p���t	H�}��O�� E��tL��蜏� H���� f.�     UH��AWAVAUATSH��(  I��H��H������H���� H� H�E�L�=� M�w@L��8���H������H�
)M�H��H�B    H�u�H�U��   �E�t	H�}�蓉� �E�t	H�}�脉� H��ݸ H��H�H�� H� H;E�uH��H[]���� H���E�u�E�uH������ H�}��:�� �E�t�H�}��+�� H���ۅ�  UH��SH��HH��� H� H�E�H�FH�E�)E�W�H�F    H�BH�E�
)M�H��H�B    H�u�H�U�袄���E�t	H�}�賈� �E�t	H�}�褈� H��ܸ H��H�H�/� H� H;E�uH��H[]��:�� H���E�u�E�uH����� H�}��Z�� �E�t�H�}��K�� H������  UH��SPH��H��ܸ H��H��Gt	H�{(��� H���`�� H��H��[]��� @ UH��SPH��H�xܸ H��H��Gt	H�{(�ڇ� H��H��[]��� fff.�     UH��SPH��H�8ܸ H��H��Gt	H�{(蚇� H������ H��H��[]鄇� @ UH��AWAVSPH��L�54۸ I�I�NH�H�@�H�L�H��� H��H�G�GHt	H�{X�8�� L��莆� I��H��L���1�� H��pH��H��[A^A_]�Ɇ� ff.�     UH��SH�H��H��tH�FH��H��t@ H��H� H��u�H�JH��t1H�BH�AE1�L�JI�H9�t+I�ID�JH9�uDE����   []�A�1�L�JI�H9�u�I�	H9�tI�AD�JH9�u��1�H��D�JH9�t�L�VL�RL�^1�I93��I��L�L�I�RL�VL�RM��tI�RD�VD�RH9�HD�E���y���H���p���E��u�A[]�f�H�H1�H9��H��H�H�PH9t��t/H�H��uz�   f�����   H�H����   ��   �    �@�A H�QH�2H�qH��tH�NH�qH�rH�qE1�H9A��J��H�
H�QH�H9�HD�H�AH�H��t
�y ��   H�PH��t
�z ��   �@ H�@H9���   �x �%����{�@�A H�PH�H��tH�JH�QH�PH�Q1�H9
@��H��H�HH�AH9�HD�H�H�H��t
�z ��   H�HH��t�y t|�@ H�@�x t	H9�������@[]�H���G[]�H�PH��t
�z ��   �A�@ H�QH�H��tH�BH�PH�QH�P1�H9@��H��H�AH�HH���   H��t
�z tAH�H�A�@ H�H�PH��tH�BH�PH�QH�P1�H9@��H��H�H�HH���H��H�A�p@�q�@�BH�H�QH�H��tH�BH�PH�QH�P1�H9@��H��H�AH�H[]�H��H�A�p@�q�@�BH�HH�H�PH��tH�BH�PH�QH�P1�H9@��H��H�H�H[]�ff.�     UH��]�f.�     UH��]�6�� fD  UH��   �.�� H�
��   H�powersheI3E A�MH��ll  H	�up�P   �(�� H�������1� @0Hǅx���Q   m1� @ Hǅ����H   G1� @,1�  H�pressionH�H@�@H �Q�H   �  L������L������Hǅ����    H�^ָ H��H������H������H������Hǅ�����  H�5	1� H������L�������*   �
  H�Hظ H� H;E���  H��L��H�Ĉ   [A\A]A^A_]�/  H�ظ H� H;E���  H��L��H�Ĉ   [A\A]A^A_]�?  H�
%� H�%� HJ�H��p���D����1�E����H���E��fIn�M��yf���� 
H��rH�<
L)�H�� ��   H��H��L��H��H��t6E1� F�F�I��L9�u�H��L)�L�L�H��H���V����@ H��H��H���@���1�fffff.�     D�9D�:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:H��H9�u������H�� sE1��;I��I���H�H��1ҐAALD�H�� I9�u�L9�������t5I��I����ǃ�J�K�f�O�N�I��M9�u�L9��V��������K�LƉǃ�H������H��[A\A^A_]ÐUH��H�� H��H�
L��H�� L�m���   H�U��   M��M��ffffff.�     �c   L9�    L�sT��  L9�    L�sGI��'  L��H�� r9�'  L��L��1��xi� ����� L9�    L�I��I��r�����������L�m�H�U�H��H��H��?Hc�H�I�UI�MH�H9�sH���   I�EM�}M���{   I�H��yA�-I��I�I��dL��H�� H�� �]  �     M�o��d   L��L��1���h� Hk�dL��H)��sfA�O��'  L9�    L�M��I��I��r��  H��y&H�pH9�sI�E L���I�EH�pI�MI�u�-W�)E�)E�H�E�    L�,+I�ŠI��dL��H�� L�m�rX�'  M���     M�o��d   L��L��1��h� Hk�dL��H)�H�
H��H�� H�}���   �   M��M��fD  �c   L9�    L�sT��  L9�    L�sGI��'  L��H�� r9�'  L��L��1��f� ����� L9�    L�I��I��r�����������H�}�Hc�H�OH�H9WryH�WL�gM��tlI�I�I��dL��H�� H�v� �'   M�l$��d   L��L��1��7f� Hk�dL��H)��sfA�L$��'  L9��    L�M��I��I��r���   W�)E�)E�H�E�    L�,(I�ŠI��dL��H�� L�m�rPA�'  L��L�k��d   L��L��1��e� Hk�dL��H)�H�
H���ʕ ��Tʕ H�5հ� H�}��ӕ L�}��t8H�5��� H�}��|ԕ H�H�IA(A(O)L$)$H��L��H���щ��hH�}�H�u���
  H�E�H�@A(A(O)L$)$H�}�L��H���Љ�H� �� H��H�E��E�uG�E�uP�E�t	H�}��%b� H�}��}� H�}�賄� H���� H� H;E�u0��H�Ĉ   [A^A_]�H�}���a� �E�t�H�}���a� �E�u���b� H��H�}���	  H�}��Y�� H���i^� � H��H�}��C�� H���S^� H������ UH��AWAVAUATSH��8H��� H� H�E�D�bI���	  I��I��H��I��I�� H�	  Jc�H���D������H�B�� D��L�4�I�A�A�s�F	���  M��I�� H�    ����L!�H	�E��A��C�A�{	��f����  9���  H��H)�H����  A�C	��y)1�A������
� H��A��d��  f�     D��Hi���QH��%k�dE��A)�B�<@f�y�H���A��'  A��wЃ�	�{  ��0�Q��&  fff.�     H�KH�sD�,D����A���   A���W���H�CH�pH9ss�H�H���H�CH�p�E��t	A����� unH�CL�H9C�V  H�CH�KH���E  H�H��D��D  D������0���H��A��A��s��  f�H�KH�sD�,D����A���   A��v�H�CH�pH9ss�H�H���H�CH�p��E��t
  �������������������������     UH��SPH���G@u�C(u�Cu%H��[]�H�{P�uS� �C(t�H�{8�fS� �Ct�H�{ H��[]�QS� �UH��AWAVAUATSH��xA��I��H�ұ� H� H�E�H�_��xP�C	����H�
  �E�u6�E�u?��`���uH�E�uTH���� H� H;E�u]�H��x[A\A]A^A_]�H�}���Q� �E�t�H�}���Q� ��`���t�H��p�����Q� �E�t�H�}��Q� H�Q�� H� H;E�t��cR� H���E�t�5H��H�}��,  ��`���u�E�uH���-N� H��p����iQ� �E�t�H�}��ZQ� H���
N� f�UH��AWAVAUATSH��xA��I��H�ү� H� H�E�L�7H�_�C	����H�
H��rH�<
H)�H�� ��   H��H��H��H��H��t:E1��    F�F�I��L9�u�H��L)�L�L�H��H���V����@ H��H��H���@���1�fffff.�     D�9D�:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:H��H9�u������H�� sE1��9I��I���H�H��1ҐLD�H�� I9�u�L9�������t7I��I����ǃ�J�J�@ N�N�I��M9�u�L9��V�������J�LƉǃ�H������L��H��[A\A]A^A_]�ff.�     UH��AWAVSH��(A��H��H��� H� H�E�D�u�H�GH�pH9wsH�H���H�CH�pH�KH�s�'E���A�� rA��"tA��\tA��t
D����  4A��"��A��'t> �u:H�CH�pH9ssH�H���H�CH�pH�KH�sD�4H�CH�pH9sr3�AH�E�H�E�H�E�H�E�D�}�H�u�H���^   H��H�CH�pH9ssH�H���H�CH�pH�KH�s�'H��� H� H;E�uH��H��([A^A_]��<� fffff.�     UH��AWAVATSH��D�vA�F���w>H�
�L  I��M9�r�I��I��I��)��  F�IF�TIE��O�D9���  u�E��t�ff.�     B8<
��   I��M9�r��i������� ��   �ǉ���E1�H�c�� H�5l�� E1�B�BF�\BE��O�9�rguE��t@ B8<��   I��M9�r�O� I��B�F�\E��O�9�r,uE��tf�     B8<t`I��M9�r�I��I��&������1�H�5� f�     �<@��y����D�DH��D	�)�x��H��H���  r҄���[]�1�[]�fn�fp� f�4T� �����=� �������=�  ������������
 �������� @��f��S� fo
@��y����D�D
H��D	�)�x4H��H��5  rӄ���[]�ff.�     UH��AWAVATSH��A��A��H��H��� H� H�E�H�GH�pH9wsH�H���H�CH�pH�KH�s�\H�CH�pH9ssH�H���H�CH�pH�KH�sD�<f�E�00H�E�H�
�� H� H;E�uH��H��[A\A^A_]��6� fff.�     UH��AWAVATSH��A��A��H��H���� H� H�E�H�GH�pH9wsH�H���H�CH�pH�KH�s�\H�CH�pH9ssH�H���H�CH�pH�KH�sD�<�E�0000H�E�H�
�H��E��~�I��A��I�@H������D��H������E1�L�������& H�KH�sD�<I��L;�����L�������q  D��D)�H������Ic�9��%  L������A�@A��A���M�`(LD�����MEh M����  M�H�K�$ffff.�     H�KH�H�KI�M9���  M��M)�I�4H�CH9�sH�H���H�KH�CH)�L9�IC�H��t�H�SH�4
H��rH�<
L)�H�� ��   H��H��L��H��H��t6E1� F�F�I��L9�u�H��L)�L�L�H��H���V����@ H��H��H���@���1�fffff.�     D�9D�:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:H��H9�u������H�� sE1��;I��I���H�H��1ҐAALD�H�� I9�u�L9�������t5I��I����ǃ�J�K�f�O�N�I��M9�u�L9��V��������K�LƉǃ�H������ffff.�     L������A��L������H������F�<0H�CH�pH9s�����H�H���H�CH�p�l���H������H������H9������������.� �� H��H������H������H9�t�-� H���\*� @ UH��AWAVATSH��I��������?H�GI��I��I�I9�rM9�vL9�IF�I��L9��_  L�sJ�<�    �Y-� H�{H���  H��rjH��L)�H�� r^I���I!�����J��K�4�E1��    C�CL�B�BL�I��M9�u�L9���   H�y�I��I��u!H��sD�   H��H��L��H�y�I��I��t�E1��    D�D�H��H��I��M9�u�L)�H��rh1�ffffff.�     D��D��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�H��H9�u�H�CL�{H�� I9�tL��[A\A^A_]�,� [A\A^A_]��ʧ��f.�     UH��AWAVAUATSH���  L�����H�����H�g�� H� H�EЉ�$���Hǅ8���    H�p�� H��H��(���L��H���L��0���Hǅ@����  L�����A�@H���`  A��H��I��H�
H��H�� �Z  W�)�P���)�@���Hǅ_���    L��A���A�   D��L��H������  H������A	�t,1�<����
E�A�A�����  H��E�E9�|'��  fffff.�     D�I�H��E�E9���  ��H9�u���I��H�����H�GH�pH9wsH��H�����H�GH�pH�OH�wD�,L9�uH��H��0���L9���  �  L)�H��L�������o  A����A��W�)�P���)�@���Hǅ_���    Ic�L�4(I��@���I��dH��H�� L�� ���rQM���M�w��d   L��H��1��8%� Hk�dL��H)�H�
H��rH�<
L)�H�� ��   H��H��L��H��H��t?E1�fff.�     F�F�I��L9�u�H��L)�L�L�H��H���F����@ H��H��H���0���1�fffff.�     D�9D�:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:H��H9�u�����H�� sE1��<I��I���H�H��1ҐAD ALD�H�� I9�u�L9��y����t5I��I����ǃ�J�K�)�O�T N�I��M9�u�L9��E�������K�(LƉǃ�H������H�u�L9�uL��H��[A\A]A^A_]�L)�L��H�U�H��[A\A]A^A_]�����@ f~�A��A���� �����   t��j���E����  A��  � ��k���E���:  UH��AVSi�� ���   )��    )�H�5�D� L��i�O� ��։���L��H��G�	B�M   ����H�� I��H��H�� Li���QI��%Ek�A�A9�s E��uTA��tN��H��uGA��A�d   �   ��   A�C�I���@@(���1�H���N ��H��1Ʌ���A��A!�A	�t^��Ai�)\����1�=(\�w&1�ffff.�     A��i�)\����=)\�r�Ai�������1�=�����AC�	���u  C����A)�Ai֚  ���  �����B���  �J  D��L���@@(���I����A��A��A0��!  �� �މ�I��E������ ���)��  i�	�	 �������i��l����ָ   )�H�
H��rL�
M)�I)�I�� ��   I��H��H��L��H��t;1�f�     D�:D�9H��H9�u�L��H)�H�H�I��I���F����@ L��I��I���0���1�fffff.�     D�:D�9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9D�D:D�D9H��H9�u�����H�� sE1��KI��I���L�E�I�H�H��1�f.�     AD�AD�H�� I9�u�L9��j����t5I��I���A��A��J�J�N�N�I��M9�u�L9��6�������L�L�A��A������L9u�uL��H��([A\A]A^A_]�L��H�u�H�U�H��([A\A]A^A_]�	���f�     UH��AWAVAUATSH���   I��H��0���I��H��p� H� H�E�H�M�D�D�؃���H�
L���y� ���x� H�5�v� H��8������ H�H���Q��H��8�����3� L�����H�����H��(���L�����L�� ����]��FF�0A��tWA�ͅ�uE��   AO�A���|=A9�8D�M����X  H�E)�D�e�A��   H��(�����  D��I��M����  D���A��   u1�A��u�E� 1�1��E)�1�E��AO�H�A��   ������ED��8���D��<���D��@�����D�����H���ƅL���0��M�����P���L��0���A�$���  ���   H�� �   D)�E��N�1�=�  @��H����d�   HM�H�H�1�H)�HC�A�D$	��H�
A��H9�u���D��H�D��H�H��H�E�H��8���H�E�H��@���H�E�H��H���H�E�H��P���H�E�H��X���H�E�H��`���H�E�H��h���H�E�H��p���L��8���H����  I���E�t	H�}��� �E�t	H�}���� H��j� H� H;E���   L��H���   [A\A]A^A_]�A��   D�����M��   E��t�   �H�H�E�H��8���H�E�H��@���H�E�H��H���H�E�H��P���H�E�H��X���H�E�H��`���H�E�H��h���L��0���A�$1�H)�HC�A�D$	��H�
D(�I��I��I�����z  I���L����  ��H���a�w̫L��H���������   H9���   H��i�)\�����   ��(\���   �ʃ�i�)\������)\�r��n1�1��y  ��I���a�w̫H��I���������   L9���   H��i�)\�����   ��(\�w#fff.�     �ʃ�i�)\������)\�r�i�������1Ɂ������C�	����   H�)\���(\�H�\���(\�L��H��L��1�H9�w%1�ffffff.�     I����H��L��H9�v�H���������I��L��H��������1�H9���IC�	���xH���sM9�I�� L���gH�)\���(\�I�\���(\�I��L��I��1�M9�w1� L�Ƀ�L��I��M9�v�H��H��H��������E1�H9�A��HC�A	�D�H����[A\A^A_]��    UH��AWAVAUATSH���   I��H��(���I��H��Q� H� H�E�H�M�L�L��x���L��H��H��H�
L����Y� ���Y� H�5W� H�}��d� H�H���Q��H�}���� L��0���H�����H�����L�����L�� �����q����NE�A��t]A�Ņ�uE��   AO�A���|CA9�>D��l������D  H�E)�D��h���A��   ��  H��0���H��E��M����  A���A��   u1�A��uƅq��� 1�1��E)�1�E��AO�H�A��   ������ED�}�L�]�D�u��]��M��E�0�U��E�L��(���A�$���  ���   H�� �   D)�E��N�1�=�  @��H����d�   HM�H�H�1�H)�HC�A�D$	��H�
����2  A��E9�D��AL�E��AH�E��AE���8���A	��^  ƅh����o  ��H��(���L��H��D��0���tq��8���������H��9���HE�H���HE�@���H�E1�1�H9�t%D  D�E�H�A���r,H��D�D9�|�D  D�@�H��D�D9�}
A��H9�u���D��H�D��H�H��H��r���H�E�H��x���H�E�H��t���H�E�H��l���H�E�H��q���H�E�H��8���H�E�H��h���H�E�H��s���H�E�L�E�H���  I����P���tH��`������ ��8���tH��H������ H�_K� H� H;E���   L��H���   [A\A]A^A_]�A��   D������h����   E��t�   �H�H��r���H�E�H��s���H�E�H��h���H�E�H��q���H�E�H��8���H�E�H��x���H�E�H��t���H�E�L��(���A�$1�H)�HC�A�D$	��H�
� D�<H�CH�pH9ssH�H���H�CH�pH�KH�sD�<M�~H�CH�pH9ssH�H���H�CH�pA�H�SH�s�I�F�8 �0  M�~H�CH�pH9ssH�H���H�CH�pA�H�SH�s�I�F D�8E��~FM�f� ff.�     A�$H�SH�s�A��t H�CH�pH9ss�H�H���H�CH�p��I�F(I�N0L� HcW�)E��E�    H�4(H���H�
H��rH�<
L)�H�� ��   H��H��L��H��H��t6E1� F�F�