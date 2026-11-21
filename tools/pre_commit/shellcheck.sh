#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm
set -e

scversion="v0.11.0"
rcfile="$PWD/tools/pre_commit/.shellcheckrc"
shellcheck_bin="$(command -v shellcheck || true)"

# ShellCheck 0.10.0 and newer support an explicit rcfile.
if [ -z "$shellcheck_bin" ] || ! "$shellcheck_bin" --rcfile="$rcfile" --version >/dev/null 2>&1; then
    shellcheck_bin="$PWD/shellcheck-${scversion}/shellcheck"
fi

if [ ! -x "$shellcheck_bin" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install ShellCheck 0.10.0 or newer: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing" >&2
        exit 1
    fi

    # automatic local install if linux x86_64
    tarball="shellcheck-${scversion}.linux.x86_64.tar.xz"
    wget -qO "$tarball" "https://github.com/koalaman/shellcheck/releases/download/${scversion}/$tarball"
    echo "8c3be12b05d5c177a04c29e3c78ce89ac86f1595681cab149b65b97c4e227198  $tarball" | sha256sum -c -
    tar -xJvf "$tarball"
    rm "$tarball"
fi

status=0
while IFS= read -r -d '' script; do
    if ! git check-ignore -q "$script"; then
        "$shellcheck_bin" -s bash --rcfile="$rcfile" "$script" || status=1
    fi
done < <(find . -name "*.sh" -not -path "./trc/*" -print0)
exit "$status"
