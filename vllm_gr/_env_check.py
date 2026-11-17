# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

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

_SUPPORTED_VLLM_VERSION = "0.30.0"
_SUPPORTED_VLLM_ASCEND_VERSION = "0.30.0rc2"
# The rc2 vendor image ships triton-ascend 3.6.0+dev...; _public_version()
# strips the "+dev..." local suffix, so pinning the public 3.6.0 matches it.
_SUPPORTED_TRITON_ASCEND_VERSION = "3.6.0"

_OPENCV_PKGS = [
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
    "opencv",
]


def configure_model_runner_default() -> None:
    """Default vllm-gr processes to the V1 model runner.

    The GR beam-search worker patches target the V1 GPU/NPU model runner
    lifecycle by default. vLLM can enable ModelRunner V2 automatically for
    supported models such as Qwen3, so set the environment default before
    vLLM constructs its configuration.

    An explicit ``VLLM_USE_V2_MODEL_RUNNER=1`` (or any truthy spelling) is
    preserved as an opt-in experimental path: the V2 runner is
    beam_search_v1-only (persistent GR stage path).  The
    LEGACY ``beam_data`` frame stack is not supported under V2 and fails fast.
    """
    value = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if value is None:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        logger.info(
            "VLLM_USE_V2_MODEL_RUNNER was not set; defaulting to ModelRunner V1 "
            "for vllm-gr compatibility."
        )
        return

    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        # Normalize truthy spellings to "1": vLLM's env parser converts the
        # value with int(), which rejects "true"/"yes"/"on" (ValueError).
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        logger.warning(
            "VLLM_USE_V2_MODEL_RUNNER=%s is an experimental vllm-gr path; "
            "the V2 runner is beam_search_v1-only (legacy beam_data frames "
            "fail fast).",
            value,
        )
    elif normalized in ("0", "false", "no", "off", ""):
        # Likewise normalize falsy spellings (and empty string) to "0".
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    # Any other value is left untouched so vLLM surfaces its own parse error.


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
            "The Ascend runtime must match the version declared by this GR build."
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
    """Activate and validate the provider selected by native vLLM.

    Providers call the leaf version/environment checks above, not this
    dispatcher. Unknown platforms fail through the registry before patches
    are applied; unselected vendor runtimes are never validated here.
    """
    from vllm_gr.platforms.loader import activate_current_backend

    report: dict[str, str] = activate_current_backend().check_environment()
    return report


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
            f"Fix: pip uninstall {' '.join(others)} -y, then keep exactly one "
            "opencv distribution. On CUDA/CPU reinstall "
            "opencv-python-headless>=4.13.0; on Ascend restore the "
            "opencv-python-headless supplied by the vendor image instead: "
            "current opencv releases declare numpy>=2 and conflict with the "
            "Ascend numpy pin."
        )

    # numpy 2.x ABI vs cv2 compiled against numpy 1.x
    import numpy as np

    numpy_major_2 = np.lib.NumpyVersion(np.__version__) >= "2.0.0"

    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        msg = str(exc).lower()
        if "numpy" in msg or "multiarray" in msg:
            if numpy_major_2:
                raise ImportError(
                    f"cv2 was compiled against NumPy 1.x but NumPy {np.__version__} "
                    "is installed (ABI incompatible).\n"
                    "Fix: pip install --force-reinstall --no-cache-dir "
                    "opencv-python-headless>=4.13.0"
                ) from exc
            raise ImportError(
                f"cv2 failed to import against NumPy {np.__version__} "
                "(numpy ABI mismatch).\n"
                "Fix: reinstall the cv2 provider supplied by the platform image."
            ) from exc
    except Exception:
        pass
