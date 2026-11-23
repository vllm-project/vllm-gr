# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Version information for vllm-gr."""

import warnings

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


__all__ = ["__version__", "__version_tuple__"]
