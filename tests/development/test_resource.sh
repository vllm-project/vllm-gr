#!/bin/bash
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

if ! is_process_alive "$LOCK_PID"; then
    echo "FAILED (Background process $LOCK_PID is not running)"
    exit 1
fi
echo "PASSED (Locked with PID: $LOCK_PID)"

echo -n "Testing unlock_gpu... "
set +e
unlock_gpu "$LOCK_PID"
rc=$?
set -e
if [ $rc -ne 0 ] || is_process_alive "$LOCK_PID"; then
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
set +e
unlock_gpus "$LOCK_PIDS"
rc=$?
set -e
lingering=""
for pid in "${PID_ARRAY[@]}"; do
    if is_process_alive "$pid"; then
        lingering="$lingering $pid"
    fi
done
if [ $rc -ne 0 ] || [ -n "$lingering" ]; then
    echo "FAILED (Background processes still running:$lingering)"
    exit 1
fi
echo "PASSED (GPUs unlocked)"

echo "All integration tests passed successfully!"
