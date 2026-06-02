#!/bin/bash
set -euo pipefail
# test development suite
./tests/development/test_resource.sh
# test pre-commit
source "./tools/resource.sh"
idle_gpu=$(get_gpu 18)
rc=$?
if [ $rc -ne 0 ] || [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 16MiB memory, and error-free. (exit code: $rc)"
    exit 1
fi
echo "running on device $idle_gpu"
if [ "${npu_state:-false}" == "true" ]; then
    PATH="${CONDA_PREFIX:-}/lib:$PATH" \
    PYTORCH_NPU_ALLOW_FORK=true \
    VLLM_WORKER_MULTIPROC_METHOD="spawn" \
    TORCH_MULTIPROCESSING_METHOD="spawn" \
    LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}" \
    ASCEND_RT_VISIBLE_DEVICES=$idle_gpu \
    python3 -m pytest --run-slow --durations=10 -s tests
else
    NODEENV_SSL_NO_VERIFY=1 \
    CUDA_VISIBLE_DEVICES=$idle_gpu \
    pre-commit run --all-files
fi
