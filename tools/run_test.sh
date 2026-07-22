#!/bin/bash
set -euo pipefail

RUN_WHEEL=false

usage() {
    echo "Usage: $0 [--wheel]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --wheel)
            RUN_WHEEL=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# test development suite
./tests/development/test_resource.sh
# test pre-commit
source "./tools/resource.sh"
if [ "${npu_state:-}" == "true" ]; then
    mem_th=3000
else
    mem_th=18
fi

idle_gpu=$(get_gpu $mem_th)
rc=$?
if [ $rc -ne 0 ] || [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 16MiB memory, and error-free. (exit code: $rc)"
    exit 1
fi

if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
    export PATH=${CONDA_PREFIX}/bin:${PATH:-}
fi

echo "running on device $idle_gpu"
if [ "${npu_state:-false}" == "true" ]; then
    PYTORCH_NPU_ALLOW_FORK=true \
    VLLM_WORKER_MULTIPROC_METHOD="spawn" \
    TORCH_MULTIPROCESSING_METHOD="spawn" \
    ASCEND_RT_VISIBLE_DEVICES=$idle_gpu \
    bash tools/pre_commit/run_tests.sh
else
    NODEENV_SSL_NO_VERIFY=1 \
    CUDA_VISIBLE_DEVICES=$idle_gpu \
    pre-commit run --all-files
fi

if [[ "$RUN_WHEEL" == "true" ]]; then
    if [[ "${npu_state:-false}" == "true" ]]; then
        VLLM_GR_TARGET_DEVICE=npu \
        ASCEND_RT_VISIBLE_DEVICES=$idle_gpu \
        ./tools/run_wheel_smoke_test.sh --python python3
    else
        VLLM_GR_TARGET_DEVICE=cuda \
        CUDA_VISIBLE_DEVICES=$idle_gpu \
        ./tools/run_wheel_smoke_test.sh --python python3
    fi
fi
