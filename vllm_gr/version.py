# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Version information for vllm-gr."""

import warnings
from importlib import metadata

import regex as re

try:
    from vllm_gr._version import __version__, __version_tuple__
except ImportError as exc:
    warnings.warn(
        f"Failed to import version from _version.py: {exc}\n"
        "This typically happens in development mode before building.\n"
        "Using source-tree fallback version '0.30.0'.",
        RuntimeWarning,
        stacklevel=2,
    )
    __version__ = "0.30.0"
    __version_tuple__ = (0, 30, 0)


def _is_dev_version(version: str, version_tuple: tuple[str | int, ...]) -> bool:
    return "dev" in version or any(str(part).startswith("dev") for part in version_tuple)


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def warn_if_misaligned_vllm_version() -> None:
    """Warn if vLLM and vllm-gr versions do not match (major.minor)."""
    try:
        vllm_version = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        return

    if _is_dev_version(__version__, __version_tuple__) or "dev" in vllm_version:
        return

    gr_ver = _major_minor(__version__)
    vllm_ver = _major_minor(vllm_version)
    if gr_ver is None or vllm_ver is None or gr_ver == (0, 0) or vllm_ver == (0, 0):
        return

    if gr_ver != vllm_ver:
        warnings.warn(
            "vLLM and vllm-gr appear to have mismatched major/minor versions:\n"
            f" --> vllm-gr version {__version__}\n"
            f" --> vLLM version {vllm_version}\n"
            "This will likely cause compatibility issues.",
            RuntimeWarning,
            stacklevel=2,
        )


__all__ = ["__version__", "__version_tuple__"]

try:
    warn_if_misaligned_vllm_version()
except metadata.PackageNotFoundError:
    pass
