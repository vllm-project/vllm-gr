#!/bin/bash
set -e

# Build this checkout's optional FFN package before the real NPU regression.
# This entry point is shared by pre-commit CI and tools/run_test.sh. CUDA and
# hosts without torch-npu retain the existing pytest path. An installed but
# broken NPU runtime must fail the probe instead of silently skipping it.
ffn_device_name=$(python3 - <<'PY'
import importlib.util

if importlib.util.find_spec("torch_npu") is not None:
    import torch
    import torch_npu

    if torch.npu.is_available():
        print(torch.npu.get_device_name(0))
PY
)
if [[ "$ffn_device_name" == "Ascend910B4" ]]; then
    ffn_test_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
    export GR_FFN_PYTHON
    GR_FFN_PYTHON=$(command -v python3)
    export GR_FFN_BUILD_DIR="${ffn_test_root}/csrc/npu/fused_ffn/build"
    unset GR_FFN_VENDOR_DIR
    bash "${ffn_test_root}/csrc/npu/fused_ffn/build.sh"
    # shellcheck source=/dev/null
    source "${ffn_test_root}/csrc/npu/fused_ffn/set_env.sh"
    export VLLM_GR_TEST_REQUIRE_FFN=1
fi

# Run all tests recursively under tests/
python3 -m pytest -x --run-slow --durations=10 -s tests
