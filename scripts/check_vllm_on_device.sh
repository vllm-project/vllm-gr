#!/usr/bin/env bash
#
# Check if a VLLM::EngineCore process exists on GPU device(s) in $CUDA_VISIBLE_DEVICES.
# Exit code:
#   0 - VLLM::EngineCore process found
#   1 - VLLM::EngineCore process NOT found
#   2 - Error (e.g., nvidia-smi not available)

set -euo pipefail
VLLM_ENGINE_CORE_PROCESS_NAME="VLLM::EngineCore"

# Ensure nvidia-smi is installed
if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: nvidia-smi command not found." >&2
    exit 1
fi

# Retrieve target GPUs from CUDA_VISIBLE_DEVICES (if set)
CUDA_DEVS="${CUDA_VISIBLE_DEVICES:-}"
if [ "$CUDA_DEVS" == "" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set." >&2
  exit 1
fi

SMI_ARGS=("--query-compute-apps=gpu_bus_id,pid,process_name,used_memory" "--format=csv,noheader")

if [[ -n "$CUDA_DEVS" ]]; then
    # Strip any whitespace from the device list (e.g. "0, 7" -> "0,7")
    CUDA_DEVS_CLEANED="$(echo "$CUDA_DEVS" | tr -d '[:space:]')"
    SMI_ARGS=("-i" "$CUDA_DEVS_CLEANED" "${SMI_ARGS[@]}")
fi

# Query active compute processes on the selected GPU(s)
APPS_OUTPUT="$(nvidia-smi "${SMI_ARGS[@]}" 2>/dev/null || true)"
MATCHES="$(echo "$APPS_OUTPUT" | grep "$VLLM_ENGINE_CORE_PROCESS_NAME" || true)"

if [[ -n "$MATCHES" ]]; then
    echo "FAIL: Found $VLLM_ENGINE_CORE_PROCESS_NAME process(es) on CUDA_VISIBLE_DEVICES=\"${CUDA_DEVS:-all}\":"
    echo "$MATCHES" | while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        echo "  - $line"
    done
    exit 1
else
    echo "SUCCESS: No $VLLM_ENGINE_CORE_PROCESS_NAME process found on CUDA_VISIBLE_DEVICES=\"${CUDA_DEVS:-all}\"."
    exit 0
fi
