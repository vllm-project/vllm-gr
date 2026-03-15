#!/bin/bash
set -e

# Logging setup
REPO=https://github.com/ehaleva/vllm-gr.git
BRANCH=master
LOG_DIR="/home/yoni/xGr/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEMP_LOG="$LOG_DIR/temp_${TIMESTAMP}.log"
NIGHTLY_LOG="$LOG_DIR/nightly.log"

#Redirect all output to temp log
exec > "$TEMP_LOG" 2>&1

finish() {
    local exit_code=$?
    local commit_id="unknown"
    if [ -d "/home/yoni/xGr/nightly" ] && git -C /home/yoni/xGr/nightly rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        commit_id=$(git -C /home/yoni/xGr/nightly rev-parse --short HEAD)
    fi
    local status_str="fail"
    if [ $exit_code -eq 0 ]; then
        status_str="success"
    fi
    echo "${TIMESTAMP},${commit_id},${status_str}" >> "$NIGHTLY_LOG"
    mv "$TEMP_LOG" "$LOG_DIR/execution_${TIMESTAMP}_${commit_id}.log"
}
trap finish EXIT

if ! . "$(dirname "$0")/get_index.sh"; then
    echo "Error: Failed to source get_index.sh"
    exit 1
fi
idle_gpu=$(get_gpu 4)
if [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 4MiB memory, and error-free."
    exit 1
fi
cuda_dev=$idle_gpu
rm -rf /home/yoni/xGr/nightly
if ! git clone -b $BRANCH $REPO /home/yoni/xGr/nightly; then
    echo "Error: Failed to clone repository"
    exit 1
fi
if ! cd /home/yoni/xGr/nightly; then
    echo "Error: Failed to change directory"
    exit 1
fi
if ! act --container-options "--runtime=nvidia --gpus device=$cuda_dev" --artifact-server-path /home/yoni/xGr/artifacts \
-P self-hosted=ghcr.io/catthehacker/ubuntu:act-latest; then
    echo "Error: act failed"
    exit 1
fi
echo "Your repo is perfect 👍"
