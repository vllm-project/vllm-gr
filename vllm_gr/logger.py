# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""vLLM-compatible logger helpers for vllm-gr."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger as _init_vllm_logger

if TYPE_CHECKING:
    from vllm.logger import _VllmLogger


def init_logger(name: str) -> "_VllmLogger":
    """Attach vllm-gr loggers under the `vllm` namespace.

    vLLM only configures the `vllm` logger tree by default. Mapping local
    module names into `vllm.*` ensures DEBUG/INFO behavior is controlled
    uniformly by vLLM logging configuration.
    """
    if name == "vllm" or name.startswith("vllm."):
        resolved_name = name
    else:
        resolved_name = f"vllm.{name}"
    return _init_vllm_logger(resolved_name)
