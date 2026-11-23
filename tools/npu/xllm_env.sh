# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# shellcheck shell=bash

# Source into the shell that will import the xLLM Python binding.
load_xllm_env() {
    local toolkit=${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}
    local root="$toolkit/opp/vendors/custom_xllm_math"
    if [[ -z $toolkit || ! -d $root ]]; then
        if [[ ${1:-} == --if-installed ]]; then
            return 0
        fi
        echo "Cannot find the custom_xllm_math vendor directory under '$toolkit'" >&2
        return 1
    fi
    local env_file="$root/bin/set_env.bash" workdir="$PWD"
    if [[ ! -f $env_file ]]; then
        local -a candidates=()
        mapfile -d '' -t candidates < <(find "$root" -name set_env.bash -print0)
        if (( ${#candidates[@]} != 1 )) || [[ ! -f ${candidates[0]:-} ]]; then
            echo "Expected one vendor set_env.bash under $root, found ${#candidates[@]}" >&2
            return 1
        fi
        env_file=${candidates[0]}
    fi
    echo "Loading xLLM environment from $env_file"
    export LD_LIBRARY_PATH="$root/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # shellcheck disable=SC1090
    source "$env_file" || return 1
    cd -- "$workdir"
}
