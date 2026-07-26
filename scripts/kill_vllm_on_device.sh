#!/usr/bin/env bash
#
# Kills any VLLM::EngineCore (or VLLM:EngineCore) process running on GPU device(s)
# specified by $CUDA_VISIBLE_DEVICES. If $CUDA_VISIBLE_DEVICES is not set, checks all GPUs.

set -euo pipefail
VLLM_ENGINE_CORE_PROCESS_NAME="VLLM::EngineCore"
# Ensure nvidia-smi is installed
if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: nvidia-smi command not found." >&2
    exit 1
fi

CUDA_DEVS="${CUDA_VISIBLE_DEVICES:-}"
if [ "$CUDA_DEVS" == "" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set."
  exit 1
fi

SMI_ARGS=("--query-compute-apps=pid,process_name" "--format=csv,noheader,nounits")

if [[ -n "$CUDA_DEVS" ]]; then
    # Strip whitespace (e.g., "0, 1" -> "0,1")
    CUDA_DEVS_CLEANED="$(echo "$CUDA_DEVS" | tr -d '[:space:]')"
    SMI_ARGS=("-i" "$CUDA_DEVS_CLEANED" "${SMI_ARGS[@]}")
fi

# Fetch compute applications on the selected GPU(s)
APPS_OUTPUT="$(nvidia-smi "${SMI_ARGS[@]}" 2>/dev/null || true)"

if [[ -z "$APPS_OUTPUT" ]]; then
    echo "No compute processes running on GPU device(s) [${CUDA_DEVS:-all}]."
    exit 0
fi

# Filter PIDs for processes matching VLLM::EngineCore or VLLM:EngineCore
MATCHING_APPS="$(echo "$APPS_OUTPUT" | grep -E "$VLLM_ENGINE_CORE_PROCESS_NAME" || true)"

if [[ -n "$MATCHING_APPS" ]]; then
    echo "Found matching $VLLM_ENGINE_CORE_PROCESS_NAME process(es) on GPU device(s) [${CUDA_DEVS:-all}]:"
    echo "$MATCHING_APPS"

    # Extract PID column (first comma-separated value)
    PIDS="$(echo "$MATCHING_APPS" | awk -F',' '{print $1}' | tr -d '[:space:]')"

    for pid in $PIDS; do
        if [[ -n "$pid" ]]; then
            echo "Killing process PID: $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    echo "Done."
else
    echo "No VLLM EngineCore process found on GPU device(s) [${CUDA_DEVS:-all}]."
fi
