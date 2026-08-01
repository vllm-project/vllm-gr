#!/bin/bash
set -e

# NPU is intentionally provided through distribution entry-point metadata
# rather than Core built-ins. Refresh the editable install before tests so
# this also works when the hook is invoked directly after a source update.
if [ "${VLLM_GR_TARGET_DEVICE:-}" = "npu" ] || \
    [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ] || \
    python3 -c 'import torch, torch_npu; raise SystemExit(0 if torch.npu.is_available() else 1)' \
        >/dev/null 2>&1; then
    VLLM_GR_TARGET_DEVICE=npu \
        python3 -m pip install --no-build-isolation --no-deps -e .
    python3 -c \
        'from importlib.metadata import entry_points; assert any(ep.name == "npu" for ep in entry_points(group="vllm_gr.platform_backends")), "NPU Provider entry point is not installed"'
fi

# Run all tests recursively under tests/
python3 -m pytest -x --run-slow --durations=10 -s tests
