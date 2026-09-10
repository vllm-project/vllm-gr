# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Setup script for vllm-gr with hardware-dependent installation.

This setup.py implements platform-aware dependency routing so users can run
`pip install vllm-gr` and automatically receive the correct platform-specific
dependencies (CUDA/NPU/CPU) without requiring extras like `[gpu]`.
"""

import logging
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py
from setuptools_scm import get_version

logger = logging.getLogger("vllm_gr.setup")

_VALID_DEVICES = frozenset(("cuda", "npu", "cpu"))
VLLM_GR_BASE_VERSION = "0.22.1"
REPO_ROOT = Path(__file__).resolve().parent
RUST_MANIFEST = REPO_ROOT / "rust" / "Cargo.toml"
RUST_BINARY_NAME = "beam-proc.exe" if os.name == "nt" else "beam-proc"
SKIP_RUST_BUILD_ENV = "VLLM_GR_SKIP_RUST_BUILD"

# Module-level cache so that detect_target_device() is evaluated exactly once
# per setup.py invocation, avoiding duplicate log output and potential
# inconsistency if env vars change between calls.
_cached_device: str | None = None


class BinaryDistribution(Distribution):
    """Install the bundled BeamProc executable into the platform library."""

    def has_ext_modules(self) -> bool:
        return True


def build_rust_sidecar(build_lib: Path, target_dir: Path) -> Path | None:
    """Compile BeamProc and copy it into the wheel's Python package."""
    if os.getenv(SKIP_RUST_BUILD_ENV, "").lower() in {"1", "true", "yes", "on"}:
        logger.warning("Skipping BeamProc compilation because %s is enabled", SKIP_RUST_BUILD_ENV)
        return None

    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError(
            "Building vllm-gr requires Cargo so the BeamProc sidecar can be compiled. "
            f"Install Rust/Cargo or set {SKIP_RUST_BUILD_ENV}=1 for a Python-only build."
        )

    target_dir = target_dir.resolve()
    subprocess.run(
        [
            cargo,
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(RUST_MANIFEST),
            "--bin",
            "beam-proc",
            "--target-dir",
            str(target_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    compiled_binary = target_dir / "release" / RUST_BINARY_NAME
    if not compiled_binary.is_file():
        raise RuntimeError(f"Cargo completed without producing {compiled_binary}")

    packaged_binary = Path(build_lib) / "vllm_gr" / "bin" / RUST_BINARY_NAME
    packaged_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(compiled_binary, packaged_binary)
    packaged_binary.chmod(packaged_binary.stat().st_mode | 0o111)
    return packaged_binary


class BuildPyWithRust(build_py):
    """Build Python modules and the bundled BeamProc executable."""

    def run(self) -> None:
        super().run()
        # Keep editable installs aligned with beam_proc_manager's source-tree
        # fallback at rust/target/release/beam-proc. The compiled executable is
        # still copied into build_lib for regular wheel builds.
        target_dir = REPO_ROOT / "rust" / "target"
        build_rust_sidecar(Path(self.build_lib), target_dir)


class BdistWheelWithRust(bdist_wheel):
    """Tag the bundled executable as Python-independent native code.

    This produces the normal host-specific build tag. Release artifacts are
    built on the supported manylinux baseline and tagged only after auditwheel
    has validated and repaired their native dependencies.
    """

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


def warn_deprecated_extra_install() -> None:
    """Warn when installing via deprecated [gpu] or [npu] extras."""
    argv = " ".join(sys.argv).lower()
    for extra in ("[gpu]", "[npu]", ".[gpu", ".[npu"):
        if extra in argv:
            warnings.warn(
                f"Installing vllm-gr via the deprecated extra {extra!r} is "
                "supported for one transition release only. "
                "Use `pip install -e .` instead — platform dependencies are "
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
    3. Torch backend detection (cuda, npu)
    4. No torch -> default cuda (consistent with vllm-omni)
    5. Torch present but no GPU/NPU backend -> cpu

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
                    logger.info("Detected NPU backend from torch")
                    _cached_device = "npu"
                    return _cached_device
            except Exception:
                pass

        logger.info("No GPU/NPU backend detected in torch, defaulting to CPU")
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
        - 0.22.1+npu
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
        logger.warning("Requirements file not found: %s", file_path)
        return requirements

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
    requirements_dir = Path(__file__).parent / "requirements"
    requirements_file = requirements_dir / f"{device}.txt"

    logger.info("Loading requirements from: %s", requirements_file)
    requirements = load_requirements(requirements_file)

    if not requirements:
        logger.warning("No requirements loaded for device '%s'", device)
    else:
        logger.info("Loaded %d requirements for %s", len(requirements), device)

    return requirements


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    setup(
        version=get_vllm_gr_version(),
        install_requires=get_install_requires(),
        distclass=BinaryDistribution,
        cmdclass={"build_py": BuildPyWithRust, "bdist_wheel": BdistWheelWithRust},
    )
