#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

set -euo pipefail

RUN_WHEEL=false

CLI_DEVICE_ID=""

usage() {
    echo "Usage: $0 [--wheel] [--device-id N]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --wheel)
            RUN_WHEEL=true
            shift
            ;;
        --device-id)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo "--device-id requires a non-empty value" >&2
                usage >&2
                exit 1
            fi
            CLI_DEVICE_ID="$2"
            shift 2
            ;;
        --device-id=*)
            CLI_DEVICE_ID="${1#*=}"
            if [ -z "$CLI_DEVICE_ID" ]; then
                echo "--device-id requires a non-empty value" >&2
                usage >&2
                exit 1
            fi
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

# Verify device-index parsing honors the visibility variables. This is pure
# logic, so it runs before auto-detection on both the explicit-device and
# auto-detect paths, and does not need nvidia-smi/npu-smi to be present.
set +e
_parsed_ids=$(CUDA_VISIBLE_DEVICES="0,2" _cuda_device_indices 4)
rc=$?
set -e
if [ $rc -ne 0 ] || [ "$_parsed_ids" != "$(printf '0\n2')" ]; then
    echo "CUDA_VISIBLE_DEVICES is not honored by _cuda_device_indices" >&2
    exit 1
fi
if [ "$(unset CUDA_VISIBLE_DEVICES; _cuda_device_indices 3)" != "$(printf '0\n1\n2')" ]; then
    echo "unset CUDA_VISIBLE_DEVICES fallback is wrong" >&2
    exit 1
fi
if CUDA_VISIBLE_DEVICES='' _cuda_device_indices 3 >/dev/null 2>&1; then
    echo "empty CUDA_VISIBLE_DEVICES should expose no devices" >&2
    exit 1
fi
if [ "$(CUDA_VISIBLE_DEVICES="0,99" _cuda_device_indices 4)" != "0" ]; then
    echo "out-of-range CUDA_VISIBLE_DEVICES id is not intersected out" >&2
    exit 1
fi
if CUDA_VISIBLE_DEVICES="GPU-0" _cuda_device_indices 4 >/dev/null 2>&1; then
    echo "invalid CUDA_VISIBLE_DEVICES token was not rejected" >&2
    exit 1
fi

DEVICE_ID="$CLI_DEVICE_ID"

if [ -n "$DEVICE_ID" ]; then
    if ! [[ "$DEVICE_ID" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Invalid device id '$DEVICE_ID': must be a comma-separated list of non-negative integers" >&2
        exit 1
    fi
    idle_gpu="$DEVICE_ID"
    echo "using externally provided device $idle_gpu (skip resource auto-detection)"
else
    # test development suite only after checking the NPU vendor runtime
    ./tests/development/test_resource.sh
    # test pre-commit
    set +e
    idle_gpu=$(get_gpu "$mem_th")
    rc=$?
    set -e
    if [ $rc -ne 0 ] || [ -z "$idle_gpu" ]; then
        echo "No GPUs found that are 0% utilized, using < ${mem_th}MiB memory, and error-free. (exit code: $rc)"
        exit 1
    fi
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
    pre-commit run --hook-stage manual --all-files
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
