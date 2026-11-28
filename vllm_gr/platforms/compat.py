# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Version checks for the runtimes supported by vLLM-GR."""

from __future__ import annotations

from importlib import metadata

# Exact versions supported by the in-tree integrations. The CI resolver reads
# these literals without importing the target checkout.
SUPPORTED_VLLM_VERSION = "0.30.0"
SUPPORTED_VLLM_ASCEND_VERSION = "0.30.0rc2"
SUPPORTED_TRITON_ASCEND_VERSION = "3.6.0"


def installed_version(distribution: str) -> str | None:
    """Read distribution metadata without importing its runtime modules."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def require_runtime_version(distribution: str, required: str) -> str:
    """Require an exact public runtime version and retain its local suffix."""
    installed = installed_version(distribution)
    if installed is None:
        raise RuntimeError(
            f"vllm-gr requires {distribution} to be installed before runtime initialization"
        )
    # Match the CI resolver's public version comparison for built-in pins.
    if installed.partition("+")[0] != required:
        raise RuntimeError(
            f"Unsupported {distribution} version: found {installed}, required {required}."
        )
    return installed


def check_vllm_runtime() -> dict[str, str]:
    """Check the vLLM runtime supported by vllm-gr."""
    return {"vllm": require_runtime_version("vllm", SUPPORTED_VLLM_VERSION)}
