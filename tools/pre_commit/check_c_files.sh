#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

if [[ $# -eq 0 ]]; then
    exit 0
fi

py_inc="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [[ -n "${py_inc}" && -d "${py_inc}" ]]; then
    inc_flags=("-I${py_inc}")
elif command -v python3-config >/dev/null 2>&1; then
    # shellcheck disable=SC2207
    inc_flags=($(python3-config --includes))
else
    echo "Could not locate Python C header directory." >&2
    exit 1
fi

for file in "$@"; do
    cc -fsyntax-only -Wall -Wextra -Werror "${inc_flags[@]}" "${file}"
done
