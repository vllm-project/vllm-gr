# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Check the vendor runtime used by Ascend."""

from importlib import import_module

from ..compat import (
    SUPPORTED_TRITON_ASCEND_VERSION,
    SUPPORTED_VLLM_ASCEND_VERSION,
    check_vllm_runtime,
    installed_version,
    require_runtime_version,
)


def check_npu_environment(*, require_npu_platform: bool = False) -> dict[str, str]:
    """Validate the complete Ascend runtime used by vllm-gr.

    Checking the ``triton-ascend`` distribution alone is insufficient because
    the standard Triton wheel installs the same top-level ``triton`` package
    and can overwrite its files. Import the Ascend backend as the authoritative
    check of the module that Python will actually use.
    """
    report = check_vllm_runtime()
    report["vllm-ascend"] = require_runtime_version("vllm-ascend", SUPPORTED_VLLM_ASCEND_VERSION)

    report["triton-ascend"] = require_runtime_version(
        "triton-ascend", SUPPORTED_TRITON_ASCEND_VERSION
    )

    triton_module = None
    try:
        triton_module = import_module("triton")
        import_module("triton._C.libtriton.ascend")
    except Exception as exc:
        triton_origin = getattr(triton_module, "__file__", "unavailable")
        raise RuntimeError(
            "Triton-Ascend distribution is installed, but the active triton module "
            "does not provide triton._C.libtriton.ascend. This usually means the "
            "standard Triton wheel overwrote the shared triton package. "
            f"Active module: {triton_origin}. Reinstall triton-ascend "
            f"{SUPPORTED_TRITON_ASCEND_VERSION} after vllm and vllm-ascend."
        ) from exc

    report["triton-module"] = str(getattr(triton_module, "__file__", "unknown"))
    report["triton-ascend-backend"] = "available"

    report["triton"] = installed_version("triton") or "not installed"

    if require_npu_platform:
        from vllm.platforms import current_platform

        device_type = str(current_platform.device_type)
        report["platform"] = device_type
        if device_type != "npu":
            raise RuntimeError(
                "Ascend environment check requires the vLLM NPU platform, "
                f"but the detected platform device_type is {device_type!r}."
            )

    return report
