# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Setup script for vLLM-gr with hardware-dependent installation.

This setup.py implements platform-aware dependency routing so users can run
`python setup.py install` and automatically receive the correct platform-specific
dependencies (CUDA/NPU) without requiring extras like `[gpu]`.
"""

import os
from pathlib import Path

from setuptools import setup


def detect_target_device() -> str:
    """
    Detect the target device for installation following RFC priority rules.

    Priority order:
    1. VLLM_GR_TARGET_DEVICE environment variable (highest priority)
    2. Torch backend detection (cuda, npu)
    3. GPU fallback (default)

    Returns:
        str: Device name ('cuda', 'npu')
    """
    # Priority 1: Explicit override via environment variable
    target_device = os.environ.get("VLLM_GR_TARGET_DEVICE")
    if target_device:
        valid_devices = ["cuda", "npu"]
        if target_device.lower() in valid_devices:
            print(f"Using target device from VLLM_GR_TARGET_DEVICE: {target_device.lower()}")
            return target_device.lower()
        else:
            print(
                f"Warning: Invalid VLLM_GR_TARGET_DEVICE '{target_device}', falling back to auto-detection"
            )

    # Priority 2: backend detection
    import shutil

    if shutil.which("npu-smi") is not None:
        print("Detected NPU backend from torch")
        return "npu"

    print("Using default CUDA backend")
    return "cuda"


def load_requirements(file_path: Path) -> list[str]:
    """
    Load requirements from a file, supporting -r directive for recursive loading.

    Args:
        file_path: Path to the requirements file

    Returns:
        List of requirement strings
    """
    requirements = []

    if not file_path.exists():
        print(f"Warning: Requirements file not found: {file_path}")
        return requirements

    with open(file_path) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Handle -r directive for recursive loading
            if line.startswith("-r "):
                nested_file = line[3:].strip()
                nested_path = file_path.parent / nested_file
                requirements.extend(load_requirements(nested_path))
            else:
                requirements.append(line)

    return requirements


def get_install_requires() -> list[str]:
    """
    Get the list of dependencies based on detected platform.

    Returns:
        List of requirement strings for the detected platform
    """
    device = detect_target_device()
    requirements_dir = Path(__file__).parent / "requirements"
    requirements_file = requirements_dir / f"{device}.txt"

    print(f"Loading requirements from: {requirements_file}")
    requirements = load_requirements(requirements_file)

    if not requirements:
        print(f"Warning: No requirements loaded for device '{device}'")
    else:
        print(f"Loaded {len(requirements)} requirements for {device}")

    return requirements


if __name__ == "__main__":
    # Get platform-specific dependencies
    install_requires = get_install_requires()

    # Setup configuration
    setup(version="0.3.0", install_requires=install_requires)
