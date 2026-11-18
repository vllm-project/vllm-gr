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
else
    echo "SKIP: No supported accelerator (GPU/NPU) found to test."
    exit 0
fi

# Source the target library
source "./tools/resource.sh"

echo "Running tests for resource.sh..."

# 0. CUDA lock maps a physical id to its CUDA-visible ordinal (pure logic).
# The device-index parsing checks live in tools/run_test.sh so they also run on
# the --device-id path and on hosts without nvidia-smi/npu-smi.
echo -n "Testing CUDA visible-ordinal translation... "
if ! python3 - <<'PY'
import importlib.util
import os

spec = importlib.util.spec_from_file_location("lock_gpu", "tools/lock_gpu.py")
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
sleep 1
if ps -p "$LOCK_PID" > /dev/null; then
    echo "FAILED (Background process $LOCK_PID is still running)"
    kill -9 "$LOCK_PID" 2>/dev/null || true # Cleanup
    exit 1
fi
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
sleep 1
if pgrep -P "$$" -f "import ctypes" > /dev/null; then
    echo "WARNING (Some Python lock processes might still be lingering)"
fi
echo "PASSED (GPUs unlocked)"

# 5. Selection + lock under a nonzero single-device mask (CUDA, needs >=2 GPUs).
if ! command -v npu-smi > /dev/null && [ "$GPU_COUNT" -ge 2 ]; then
    echo -n "Testing selection+lock under CUDA_VISIBLE_DEVICES=1... "
    set +e
    MASKED_ID=$(CUDA_VISIBLE_DEVICES=1 get_gpu "$mem_th" 15)
    rc=$?
    set -e
    if [ $rc -ne 0 ] || [ -z "$MASKED_ID" ]; then
        echo "SKIP (no device visible under the mask)"
    elif [ "$MASKED_ID" != "1" ]; then
        echo "FAILED (expected device 1, got '$MASKED_ID')"
        exit 1
    else
        set +e
        MASKED_PID=$(CUDA_VISIBLE_DEVICES=1 lock_gpu "$MASKED_ID")
        rc=$?
        set -e
        if [ $rc -ne 0 ] || [ -z "$MASKED_PID" ]; then
            echo "FAILED (lock_gpu did not return a PID under the mask)"
            exit 1
        fi
        unlock_gpu "$MASKED_PID"
        echo "PASSED"
    fi
fi

echo "All integration tests passed successfully!"
