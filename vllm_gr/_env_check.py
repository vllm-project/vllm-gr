# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional environment sanity checks for common deployment pitfalls.

Call ``check_opencv_env()`` **before** importing vllm to catch opencv/numpy
ABI conflicts early. Some model/rendering dependency paths load cv2 at module
level, so a broken cv2 can manifest as an opaque C-level crash rather than a
clear error message.
"""

import importlib
import importlib.metadata
import logging
import os

logger = logging.getLogger("vllm_gr.env_check")

_SUPPORTED_VLLM_VERSION = "0.22.1"
_SUPPORTED_VLLM_ASCEND_VERSION = "0.22.1rc1"
_SUPPORTED_TRITON_ASCEND_VERSION = "3.2.1"

_OPENCV_PKGS = [
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
    "opencv",
]


def configure_model_runner_default() -> None:
    """Default vllm-gr processes to the V1 model runner.

    The GR beam-search worker patches currently target the V1 GPU/NPU model
    runner lifecycle.  vLLM 0.22 enables ModelRunner V2 automatically for
    supported models such as Qwen3, which bypasses those patches and can
    silently produce incorrect beam-search results.  Set the environment
    default before vLLM constructs its configuration, while preserving an
    explicit user override for V2 development and validation.
    """
    if "VLLM_USE_V2_MODEL_RUNNER" not in os.environ:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        logger.info(
            "VLLM_USE_V2_MODEL_RUNNER was not set; defaulting to ModelRunner V1 "
            "for vllm-gr compatibility."
        )


def _distribution_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"vllm-gr requires {distribution} to be installed before runtime initialization"
        ) from exc


def _public_version(version: str) -> str:
    """Drop a local build suffix (for example ``+npu``) before comparison."""
    return version.partition("+")[0]


def check_runtime_versions(*, require_ascend: bool = False) -> dict[str, str]:
    """Validate the single runtime pair supported by this vllm-gr release."""
    vllm_version = _distribution_version("vllm")
    if _public_version(vllm_version) != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            "Unsupported vLLM version: "
            f"found {vllm_version}, required {_SUPPORTED_VLLM_VERSION}. "
            f"This vllm-gr build only supports the vLLM {_SUPPORTED_VLLM_VERSION} API."
        )

    versions = {"vllm": vllm_version}
    if not require_ascend:
        return versions

    ascend_version = _distribution_version("vllm-ascend")
    if _public_version(ascend_version) != _SUPPORTED_VLLM_ASCEND_VERSION:
        raise RuntimeError(
            "Unsupported vLLM-Ascend version: "
            f"found {ascend_version}, required {_SUPPORTED_VLLM_ASCEND_VERSION}. "
            "This vllm-gr build relies on the native 0.22 YuanRong and hybrid KV-pool APIs."
        )
    versions["vllm-ascend"] = ascend_version
    return versions


def check_ascend_environment(*, require_npu_platform: bool = False) -> dict[str, str]:
    """Validate the complete Ascend runtime used by vllm-gr.

    Checking the ``triton-ascend`` distribution alone is insufficient because
    the standard Triton wheel installs the same top-level ``triton`` package
    and can overwrite its files. Import the Ascend backend as the authoritative
    check of the module that Python will actually use.
    """
    report = check_runtime_versions(require_ascend=True)

    triton_ascend_version = _distribution_version("triton-ascend")
    if _public_version(triton_ascend_version) != _SUPPORTED_TRITON_ASCEND_VERSION:
        raise RuntimeError(
            "Unsupported Triton-Ascend version: "
            f"found {triton_ascend_version}, required {_SUPPORTED_TRITON_ASCEND_VERSION}."
        )
    report["triton-ascend"] = triton_ascend_version

    triton_module = None
    try:
        triton_module = importlib.import_module("triton")
        importlib.import_module("triton._C.libtriton.ascend")
    except Exception as exc:
        triton_origin = getattr(triton_module, "__file__", "unavailable")
        raise RuntimeError(
            "Triton-Ascend distribution is installed, but the active triton module "
            "does not provide triton._C.libtriton.ascend. This usually means the "
            "standard Triton wheel overwrote the shared triton package. "
            f"Active module: {triton_origin}. Reinstall triton-ascend "
            f"{_SUPPORTED_TRITON_ASCEND_VERSION} after vllm and vllm-ascend."
        ) from exc

    report["triton-module"] = str(getattr(triton_module, "__file__", "unknown"))
    report["triton-ascend-backend"] = "available"

    try:
        report["triton"] = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        report["triton"] = "not installed"

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


def check_environment() -> dict[str, str]:
    """Validate the runtime selected by vLLM's current platform.

    Keep platform-specific dispatch at the environment-check boundary so
    runtime entry points do not need to know which accelerator dependencies
    are required.  In particular, non-NPU platforms must not import or
    validate the optional Ascend runtime.
    """
    from vllm.platforms import current_platform

    if str(current_platform.device_type) == "npu":
        return check_ascend_environment()
    return check_runtime_versions()


def check_opencv_env() -> None:
    """Raise early if the opencv / numpy environment is broken."""

    # Multiple opencv packages overwrite each other's cv2.so
    installed = {}
    for pkg in _OPENCV_PKGS:
        try:
            installed[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass

    if len(installed) > 1:
        pkg_list = ", ".join(f"{k}=={v}" for k, v in installed.items())
        others = [k for k in installed if k != "opencv-python-headless"]
        raise RuntimeError(
            f"Multiple opencv packages detected: {pkg_list}\n"
            "They all provide cv2 and conflict with each other.\n"
            f"Fix: pip uninstall {' '.join(others)} -y && "
            "pip install --no-cache-dir opencv-python-headless>=4.13.0"
        )

    # numpy 2.x ABI vs cv2 compiled against numpy 1.x
    import numpy as np

    if np.lib.NumpyVersion(np.__version__) < "2.0.0":
        return

    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        msg = str(exc).lower()
        if "numpy" in msg or "multiarray" in msg:
            raise ImportError(
                f"cv2 was compiled against NumPy 1.x but NumPy {np.__version__} "
                "is installed (ABI incompatible).\n"
                "Fix: pip install --force-reinstall --no-cache-dir "
                "opencv-python-headless>=4.13.0"
            ) from exc
    except Exception:
        pass
