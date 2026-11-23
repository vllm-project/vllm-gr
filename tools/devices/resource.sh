# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


# This script is meant to be sourced.
# It provides utilities to allocate and manage local GPU or NPU resources.

_vllm_gr_device_tools_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Detect if we are in an NPU environment
if command -v npu-smi &> /dev/null; then
    npu_state=true
else
    npu_state=false
fi

if [ "$npu_state" = true ]; then
    # shellcheck source=tools/devices/resource-npu.sh
    source "$_vllm_gr_device_tools_dir/resource-npu.sh"
fi

function is_available() {
    if [ "$npu_state" = true ]; then
        is_available_npu "$@"
        return $?
    fi
    local i=$1
    local mem_gpu=$2

    # Query utilization, memory used, and pstate (to detect errors) in one call.
    local stats
    stats=$(nvidia-smi -i "$i" --query-gpu=utilization.gpu,memory.used,pstate --format=csv,noheader,nounits 2>/dev/null) || return 1

    # Only a complete idle record with numeric memory and a valid pstate is usable.
    local idle_record='^[[:blank:]]*0+[[:blank:]]*,[[:blank:]]*([0-9]+)[[:blank:]]*,[[:blank:]]*P[0-9]+[[:blank:]]*$'
    [[ $stats =~ $idle_record ]] && [ "${BASH_REMATCH[1]}" -lt "$mem_gpu" ]
}

# _cuda_device_indices <gpu_count>
# Prints the physical GPU indices to consider, one per line. Honors
# CUDA_VISIBLE_DEVICES (a comma-separated list of physical indices, matching
# nvidia-smi under CUDA_DEVICE_ORDER=PCI_BUS_ID) so selection stays inside the
# devices the CUDA runtime can see. Falls back to all indices when unset, and
# exposes none when the variable is set but empty, matching the CUDA runtime.
function _cuda_device_indices() {
    local gpu_count=$1
    if ! [[ -v CUDA_VISIBLE_DEVICES ]]; then
        # Unset: consider every physical GPU.
        local i
        for (( i=0; i<gpu_count; i++ )); do
            echo "$i"
        done
        return 0
    fi
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # Set but empty: CUDA exposes no devices.
        echo "CUDA_VISIBLE_DEVICES is set but empty: no devices visible" >&2
        return 1
    fi
    if [[ ! $CUDA_VISIBLE_DEVICES =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Invalid CUDA_VISIBLE_DEVICES: expected comma-separated non-negative integers" >&2
        return 1
    fi
    local idx found=0
    local -a indices
    IFS=',' read -ra indices <<< "$CUDA_VISIBLE_DEVICES"
    for idx in "${indices[@]}"; do
        if [ "$idx" -lt "$gpu_count" ]; then
            echo "$((10#$idx))"
            found=$((found + 1))
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "No CUDA_VISIBLE_DEVICES id is within [0, $gpu_count)" >&2
        return 1
    fi
    return 0
}

function get_gpu() {
    if [ "$npu_state" = true ]; then
        get_npu "$@"
        return $?
    fi

    local mem_gpu=${1:-16}
    local timeout_secs=${2:-15} # seconds to wait; 0 waits indefinitely

    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | tail -n 1) || return 1
    if [ -z "$gpu_count" ]; then echo "No GPUs available" >&2; return 1; fi

    local indices
    indices=$(_cuda_device_indices "$gpu_count") || return 1
    if [ -z "$indices" ]; then echo "No GPUs visible to CUDA" >&2; return 1; fi

    local start_time=$(date +%s)

    while true; do
        # Loop through each visible GPU index
        for i in $indices; do
            if is_available "$i" "$mem_gpu"; then
                echo "$i"
                return 0
            fi
        done

        if [ "$timeout_secs" -gt 0 ]; then
            local current_time=$(date +%s)
            if [ $((current_time - start_time)) -ge "$timeout_secs" ]; then
                echo "Timeout reached" >&2
                return 1
            fi
        fi
        sleep 5
    done
}

function get_gpus() {
    if [ "$npu_state" = true ]; then
        get_npus "$@"
        return $?
    fi

    local num_gpus=${1:-1}
    local mem_gpu=${2:-16}
    local timeout_secs=${3:-15} # seconds to wait; 0 waits indefinitely

    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | tail -n 1) || return 1
    if [ -z "$gpu_count" ]; then return 1; fi

    local indices
    indices=$(_cuda_device_indices "$gpu_count") || return 1
    if [ -z "$indices" ]; then return 1; fi

    local start_time=$(date +%s)

    while true; do
        local idle_gpus=""
        local found=0

        # Loop through each visible GPU index
        for i in $indices; do
            # Skip ids already selected (CUDA_VISIBLE_DEVICES may repeat them).
            if [[ ",$idle_gpus," == *",$i,"* ]]; then
                continue
            fi
            if is_available "$i" "$mem_gpu"; then
                if [ -z "$idle_gpus" ]; then
                    idle_gpus="$i"
                else
                    idle_gpus="${idle_gpus},$i"
                fi
                found=$((found + 1))
                if [ "$found" -eq "$num_gpus" ]; then
                    echo "$idle_gpus"
                    return 0
                fi
            fi
        done

        if [ "$timeout_secs" -gt 0 ]; then
            local current_time=$(date +%s)
            if [ $((current_time - start_time)) -ge "$timeout_secs" ]; then
                echo "Timeout reached" >&2
                return 1
            fi
        fi
        sleep 5
    done
}

# lock_gpu <gpu_id>
# Allocates a small amount of memory on the given GPU to "lock" it.
# This prevents other scripts using get_gpu from selecting it.
# It starts a background process and prints its PID.
# The caller is responsible for killing this PID to unlock the GPU.
# Requires Python on CUDA.
function lock_gpu() {
    local gpu_id=$1
    local lock_pid=0
    if [ -z "$gpu_id" ]; then
        echo "Invalid GPU ID: argument is empty" >&2
        return 1
    fi
    local ready_file
    ready_file=$(mktemp) || return 1
    if [ "$npu_state" = true ]; then
        if ! lock_pid=$(lock_npu "$@"); then
            rm -f -- "$ready_file"
            return 1
        fi
        echo "READY" > "$ready_file"
    else
        local interpreter
        if ! interpreter=$(command -v python || command -v python3); then
            rm -f -- "$ready_file"
            echo "Cannot lock GPU: install python or python3" >&2
            return 1
        fi
        "$interpreter" "$_vllm_gr_device_tools_dir/lock_gpu.py" "$gpu_id" > "$ready_file" 2>&1 &
        lock_pid=$!
    fi
    # Poll until we see 'READY' or the process dies (max ~10s).
    local ready=0
    for (( i=0; i<40; i++ )); do
        if ! ps -p "$lock_pid" >/dev/null 2>&1; then
            break
        fi
        if grep -qx "READY" "$ready_file" 2>/dev/null; then
            ready=1
            break
        fi
        sleep 0.25
    done
    if [ "$ready" -eq 1 ]; then
        rm -f -- "$ready_file"
        echo "$lock_pid"
        return 0
    else
        kill -9 "$lock_pid" 2>/dev/null || true
        cat "$ready_file" >&2
        rm -f -- "$ready_file"
        echo ""
        return 1
    fi
}

# unlock_gpu <pid>
# Kills the background process that is holding the GPU memory.
function unlock_gpu() {
    if [ "$npu_state" = true ]; then
        unlock_npu "$@"
        return $?
    fi

    local pid=$1
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
}

function lock_gpus() {
    local gpus_list=$1
    if [ "$npu_state" = true ]; then
        lock_npus "$@"
        return $?
    fi

    local pids=""
    IFS=',' read -ra gpus <<< "$gpus_list"
    for gpu in "${gpus[@]}"; do
        local pid
        if ! pid=$(lock_gpu "$gpu") || [ -z "$pid" ]; then
            # Lock failed — roll back any GPUs locked so far
            if [ -n "$pids" ]; then
                unlock_gpus "$pids"
            fi
            echo "" >&2
            echo "Failed to lock GPU $gpu, rolled back previously locked GPUs." >&2
            return 1
        fi
        if [ -z "$pids" ]; then
            pids="$pid"
        else
            pids="${pids},${pid}"
        fi
    done
    echo "$pids"
}

function unlock_gpus() {
    local pids_list=$1
    if [ "$npu_state" = true ]; then
        unlock_npus "$@"
        return $?
    fi

    local -a lock_pids
    local pid status=0
    IFS=',' read -ra lock_pids <<< "$pids_list"
    for pid in "${lock_pids[@]}"; do
        unlock_gpu "$pid" || status=1
    done
    return "$status"
}
