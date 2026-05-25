# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional environment sanity checks for common deployment pitfalls.

Call ``check_opencv_env()`` **before** importing vllm to catch opencv/numpy
ABI conflicts early.  vllm 0.14.1 transitively loads cv2 (via
mistral_common) at module level, so a broken cv2 manifests as an opaque
C-level crash rather than a clear error message.
"""

import importlib.metadata
import logging

logger = logging.getLogger("vllm_gr.env_check")

_OPENCV_PKGS = [
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
    "opencv",
]


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
