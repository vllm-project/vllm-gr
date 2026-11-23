# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""OpenCV and NumPy ABI preflight before importing vLLM."""

import importlib.metadata

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
