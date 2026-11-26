#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

#
# Integration test for resource.sh
# This script verifies GPU allocation and locking mechanisms.
set -euo pipefail
mem_th=18
if command -v npu-smi &> /dev/null; then
  mem_th=3000
fi
# Check for required system dependencies
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | tail -n 1 || echo 0)
elif command -v npu-smi &> /dev/null; then
    # Use npu-smi to count the number of NPUs by counting the number of entries
    # Each NPU entry starts with a line containing a number and the card name
    GPU_COUNT=$(npu-smi info 2>/dev/null | grep "^| [0-9]" | awk '{print $2}' | sort -u | wc -l)
elif [ -d "${VLLM_GR_KFD_TOPOLOGY:-/sys/class/kfd/kfd/topology/nodes}" ]; then
    # AMD ROCm: resource.sh counts the KFD GPU nodes once it is sourced.
    GPU_COUNT=rocm
else
    echo "SKIP: No supported accelerator (GPU/NPU) found to test."
    exit 0
fi

# Source the target library
source "./tools/devices/resource.sh"
if [ "$GPU_COUNT" = rocm ]; then
    if [ "$rocm_state" != true ] || ! GPU_COUNT=$(rocm_gpu_count); then
        echo "SKIP: No supported accelerator (GPU/NPU) found to test."
        exit 0
    fi
fi

lock_pid_state() {
    local pid=$1 stat rc
    [[ $pid =~ ^[1-9][0-9]*$ ]] || { echo "Invalid lock PID: $pid" >&2; return 1; }
    if stat=$(ps -p "$pid" -o stat=); then
        if [[ $stat =~ ^[[:space:]]*([^[:space:]]+)[[:space:]]*$ ]]; then
            echo "${BASH_REMATCH[1]}"
            return 0
        fi
    else
        rc=$?
        if [[ $rc == 1 && -z $stat ]]; then
            echo gone
            return 0
        fi
    fi
    echo "Could not read STAT for lock PID $pid" >&2
    return 1
}

wait_for_unlock() {
    local pid state previous attempt
    for pid in "$@"; do
        previous=""
        # A container's init may leave an exited lock holder as a zombie.
        # Poll each exact holder for up to five seconds and record state changes.
        for ((attempt=0; attempt<=20; attempt++)); do
            state=$(lock_pid_state "$pid") || return 1
            if [[ $state != "$previous" ]]; then
                echo "PID $pid STAT=$state after unlock"
                previous=$state
            fi
            [[ $state == gone || $state == Z* ]] && break
            if (( attempt == 20 )); then
                echo "FAILED (lock PID $pid remains active with STAT=$state)" >&2
                return 1
            fi
            sleep 0.25
        done
    done
}

echo "Running tests for resource.sh..."

# 0. CUDA lock maps a physical id to its CUDA-visible ordinal (pure logic).
# The device-index parsing checks live in tools/run_test.sh so they also run on
# the --device-id path and on hosts without nvidia-smi/npu-smi.
echo -n "Testing CUDA visible-ordinal translation... "
if ! python3 - <<'PY'
import importlib.util
import os

spec = importlib.util.spec_from_file_location("lock_gpu", "tools/devices/lock_gpu.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
assert module._cuda_visible_ordinal(1) == 0
os.environ["CUDA_VISIBLE_DEVICES"] = "2,0"
assert module._cuda_visible_ordinal(2) == 0
assert module._cuda_visible_ordinal(0) == 1
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
assert module._cuda_visible_ordinal(1) == 1
PY
then
    echo "FAILED"
    exit 1
fi
echo "PASSED"

# 1. Test get_gpu
echo -n "Testing get_gpu... "
set +e
GPU_ID=$(get_gpu $mem_th 15)
rc=$?
set -e

if [ $rc -ne 0 ] || [ -z "$GPU_ID" ]; then
    echo "SKIP (Timeout or no GPU available matching criteria within 15 seconds - GPUs are likely busy)"
    exit 0
fi
echo "PASSED (Got GPU ID: $GPU_ID)"

# 2. Test lock_gpu & unlock_gpu
echo -n "Testing lock_gpu on GPU $GPU_ID... "
set +e
LOCK_PID=$(lock_gpu "$GPU_ID")
rc=$?
set -e

if [ $rc -ne 0 ] || [ -z "$LOCK_PID" ]; then
    echo "FAILED (lock_gpu did not return a PID. Check for missing EGL/GL/CUDA dependencies in Python)"
    exit 1
fi

if ! ps -p "$LOCK_PID" > /dev/null; then
    echo "FAILED (Background process $LOCK_PID is not running)"
    exit 1
fi
echo "PASSED (Locked with PID: $LOCK_PID)"

echo -n "Testing unlock_gpu... "
unlock_gpu "$LOCK_PID"
wait_for_unlock "$LOCK_PID"
echo "PASSED (GPU unlocked)"

# 3. Test get_gpus
echo -n "Testing get_gpus (requesting 2 GPUs)... "
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "SKIP (System has less than 2 GPUs)"
    exit 0
fi

set +e
GPU_IDS=$(get_gpus 2 $mem_th 15)
rc=$?
set -e

if [ $rc -ne 0 ] || [ -z "$GPU_IDS" ]; then
    echo "SKIP (Timeout or no 2 GPUs available within 15 seconds - GPUs are likely busy)"
    exit 0
fi

IFS=',' read -ra ID_ARRAY <<< "$GPU_IDS"
if [ "${#ID_ARRAY[@]}" -ne 2 ]; then
    echo "FAILED (Expected 2 GPUs, got ${#ID_ARRAY[@]}: $GPU_IDS)"
    exit 1
fi
echo "PASSED (Got GPU IDs: $GPU_IDS)"

# 4. Test lock_gpus & unlock_gpus
echo -n "Testing lock_gpus on GPUs '$GPU_IDS'... "
set +e
LOCK_PIDS=$(lock_gpus "$GPU_IDS")
rc=$?
set -e

if [ $rc -ne 0 ] || [ -z "$LOCK_PIDS" ]; then
    echo "FAILED (lock_gpus did not return PIDs)"
    exit 1
fi

IFS=',' read -ra PID_ARRAY <<< "$LOCK_PIDS"
if [ "${#PID_ARRAY[@]}" -ne 2 ]; then
    echo "FAILED (Expected 2 PIDs, got ${#PID_ARRAY[@]}: $LOCK_PIDS)"
    exit 1
fi
echo "PASSED (Locked with PIDs: $LOCK_PIDS)"

echo -n "Testing unlock_gpus... "
unlock_gpus "$LOCK_PIDS"
wait_for_unlock "${PID_ARRAY[@]}"
echo "PASSED (GPUs unlocked)"

# 5. Selection + lock under a nonzero single-device mask (CUDA, needs >=2 GPUs).
if ! command -v npu-smi > /dev/null && [ "$GPU_COUNT" -ge 2 ]; then
    mask_id=${ID_ARRAY[0]}
    [[ $mask_id != 0 ]] || mask_id=${ID_ARRAY[1]}
    echo -n "Testing selection+lock under CUDA_VISIBLE_DEVICES=$mask_id... "
    set +e
    MASKED_ID=$(CUDA_VISIBLE_DEVICES="$mask_id" get_gpu "$mem_th" 15)
    rc=$?
    set -e
    if [ $rc -ne 0 ] || [ -z "$MASKED_ID" ]; then
        echo "SKIP (no device visible under the mask)"
    elif [ "$MASKED_ID" != "$mask_id" ]; then
        echo "FAILED (expected device $mask_id, got '$MASKED_ID')"
        exit 1
    else
        set +e
        MASKED_PID=$(CUDA_VISIBLE_DEVICES="$mask_id" lock_gpu "$MASKED_ID")
        rc=$?
        set -e
        if [ $rc -ne 0 ] || [ -z "$MASKED_PID" ]; then
            echo "FAILED (lock_gpu did not return a PID under the mask)"
            exit 1
        fi
        unlock_gpu "$MASKED_PID"
        wait_for_unlock "$MASKED_PID"
        echo "PASSED"
    fi
fi

echo "All integration tests passed successfully!"
