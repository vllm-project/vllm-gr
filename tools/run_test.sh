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

if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
    export PATH=${CONDA_PREFIX}/bin:${PATH:-}
fi

source "./tools/resource.sh"
if [ "${npu_state:-}" == "true" ]; then
    export VLLM_GR_TARGET_DEVICE=npu
    # Jenkins must validate the same vendor Python that will run pytest.
    # Installing the NPU GR extra does not upgrade the vendor runtime.
    npu_runtime_spec=$(mktemp "${TMPDIR:-/tmp}/vllm-gr-npu-runtime.XXXXXX")
    if ! python3 tools/ci/resolve_runtime.py resolve --device npu --output "$npu_runtime_spec"; then
        rm -f -- "$npu_runtime_spec"
        echo "NPU test runtime is incompatible with this checkout. Select the matching vendor environment before running this gate; see docs/quickstart-ascend.md." >&2
        exit 1
    fi
    rm -f -- "$npu_runtime_spec"
    mem_th=3000
else
    mem_th=18
fi

# test development suite only after checking the NPU vendor runtime
./tests/development/test_resource.sh
# test pre-commit
idle_gpu=$(get_gpu $mem_th)
rc=$?
if [ $rc -ne 0 ] || [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 16MiB memory, and error-free. (exit code: $rc)"
    exit 1
fi

echo "running on device $idle_gpu"
if [ "${npu_state:-false}" == "true" ]; then
    # Beam-related NPU tests depend on xllm-ops. Ensure the custom operator
    # library is available before entering the pre-commit/test gate. The
    # installer is a no-op on non-NPU hosts and skips work when already ready.
    echo "ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-}"
    run_test_workdir="$(pwd)"
    ASCEND_RT_VISIBLE_DEVICES=$idle_gpu \
    bash tools/install_xllm_ops.sh

    CUSTOM_XLLM_VENDORS_ROOT="${ASCEND_TOOLKIT_HOME}/opp/vendors"
    CUSTOM_XLLM_MATH_ROOT="${CUSTOM_XLLM_VENDORS_ROOT}/custom_xllm_math"
    CUSTOM_XLLM_MATH_SET_ENV=$(find "${CUSTOM_XLLM_MATH_ROOT}" -name set_env.bash -print -quit)
    test -n "${CUSTOM_XLLM_MATH_SET_ENV}" || {
        echo "set_env.bash not found under ${CUSTOM_XLLM_MATH_ROOT}"
        exit 1
    }
    echo "Found set_env.bash at ${CUSTOM_XLLM_MATH_SET_ENV}"
    CUSTOM_XLLM_MATH_LIB="${CUSTOM_XLLM_MATH_ROOT}/op_api/lib"
    export LD_LIBRARY_PATH="${CUSTOM_XLLM_MATH_LIB}:${LD_LIBRARY_PATH}"
    # shellcheck source=/dev/null
    source "${CUSTOM_XLLM_MATH_ROOT}/bin/set_env.bash"
    cd "${run_test_workdir}"

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
