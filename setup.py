# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Setup script for vllm-gr with hardware-dependent installation.

This setup.py implements platform-aware dependency routing so users can run
`pip install vllm-gr` and automatically receive the correct source-specific
dependencies without requiring extras like `[gpu]`.
"""

import logging
import os
import sys
import warnings
from pathlib import Path

from setuptools import setup
from setuptools_scm import get_version

logger = logging.getLogger("vllm_gr.setup")

_VALID_DEVICES = frozenset(("cuda", "npu", "cpu"))
VLLM_GR_BASE_VERSION = "0.22.1"
REQUIREMENTS_DIR = Path(__file__).parent / "requirements"

# Module-level cache so that detect_target_device() is evaluated exactly once
# per setup.py invocation, avoiding duplicate log output and potential
# inconsistency if env vars change between calls.
_cached_device: str | None = None


def warn_deprecated_extra_install() -> None:
    """Warn when installing via the deprecated GPU extra."""
    argv = " ".join(sys.argv).lower()
    for extra in ("[gpu]", ".[gpu"):
        if extra in argv:
            warnings.warn(
                f"Installing vllm-gr via the deprecated extra {extra!r} is "
                "not the supported installation path. Use `pip install -e .` "
                "instead — platform dependencies are "
                "auto-selected by setup.py.",
                DeprecationWarning,
                stacklevel=2,
            )
            return


def detect_target_device() -> str:
    """
    Detect the target device for installation.

    Priority order:
    1. VLLM_GR_TARGET_DEVICE environment variable (highest priority)
    2. READTHEDOCS=1 -> cpu
    3. Torch accelerator detection
    4. No torch -> default cuda (consistent with vllm-omni)
    5. Torch present but no supported accelerator -> cpu

    The result is cached after the first call.
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    target_device = os.environ.get("VLLM_GR_TARGET_DEVICE")
    if target_device:
        if target_device.lower() in _VALID_DEVICES:
            logger.info("Using target device from VLLM_GR_TARGET_DEVICE: %s", target_device.lower())
            _cached_device = target_device.lower()
            return _cached_device
        logger.warning(
            "Invalid VLLM_GR_TARGET_DEVICE '%s', falling back to auto-detection",
            target_device,
        )

    if os.environ.get("READTHEDOCS"):
        logger.info("Detected READTHEDOCS=1; using cpu requirements for docs build")
        _cached_device = "cpu"
        return _cached_device

    try:
        import torch

        if torch.version.cuda is not None:
            logger.info("Detected CUDA backend from torch")
            _cached_device = "cuda"
            return _cached_device

        if hasattr(torch, "npu"):
            try:
                if torch.npu.is_available():
                    logger.info("Detected alternate accelerator backend from torch")
                    _cached_device = "npu"
                    return _cached_device
            except Exception as exc:  # noqa: BLE001 - backend probing must not abort setup
                logger.debug("Alternate accelerator probe failed: %s", exc)

        logger.info("No supported accelerator detected in torch, defaulting to CPU")
        _cached_device = "cpu"
        return _cached_device

    except ImportError:
        logger.info("PyTorch not found, defaulting to CUDA installation")
        _cached_device = "cuda"
        return _cached_device


def get_vllm_gr_version() -> str:
    """
    Get the vllm-gr version with optional device-specific suffix.

    Version format examples:
        - 0.22.1 (cuda, no suffix)
        - 0.22.1+cpu

    Environment variables:
        VLLM_GR_VERSION_OVERRIDE: Override version completely
        VLLM_GR_TARGET_DEVICE: Override device detection
    """
    if env_version := os.getenv("VLLM_GR_VERSION_OVERRIDE"):
        logger.info("Overriding version with %s from VLLM_GR_VERSION_OVERRIDE", env_version)
        version = env_version
    else:
        # vllm-gr releases follow the matching vLLM public version. Do not let
        # an untagged source checkout produce an unrelated 0.1.devN version.
        version = VLLM_GR_BASE_VERSION
        device = detect_target_device()

        if device == "cuda":
            pass
        elif device in ("npu", "cpu"):
            sep = "+" if "+" not in version else "."
            version += f"{sep}{device}"
        else:
            raise RuntimeError(f"Unknown target device: {device}")

    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = version
    get_version(write_to="vllm_gr/_version.py")
    logger.info("Generated version: %s", version)
    return version


def load_requirements(file_path: Path) -> list[str]:
    """Load requirements from a file, supporting -r directive for recursive loading."""
    requirements: list[str] = []

    if not file_path.exists():
        raise RuntimeError(f"Requirements file not found: {file_path}")

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                nested_path = file_path.parent / line[3:].strip()
                requirements.extend(load_requirements(nested_path))
            else:
                requirements.append(line)

    return requirements


def get_install_requires() -> list[str]:
    """Get dependencies based on detected platform."""
    warn_deprecated_extra_install()

    device = detect_target_device()
    requirements_file = REQUIREMENTS_DIR / f"{device}.txt"

    if not requirements_file.exists():
        raise RuntimeError(
            f"Detected target device {device!r}, but this source "
            "distribution does not include that backend."
        )

    logger.info("Loading requirements from: %s", requirements_file)
    requirements = load_requirements(requirements_file)

    if not requirements:
        raise RuntimeError(
            f"Detected target device {device!r}, but its requirements file "
            "does not define any install dependencies."
        )
    logger.info("Loaded %d requirements for %s", len(requirements), device)

    return requirements


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    setup(
        version=get_vllm_gr_version(),
        install_requires=get_install_requires(),
    )
