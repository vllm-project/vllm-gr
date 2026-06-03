# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Configuration module for vllm-GR.

Provides:
- HSTUConfig: HuggingFace-compatible PretrainedConfig subclass (model arch)
- GRConfig: Typed runtime / serving configuration dataclass
"""

from vllm_gr.config.fuxi_config import FuXiConfig
from vllm_gr.config.gr_config import GRConfig
from vllm_gr.config.hstu_config import HSTUConfig

__all__ = [
    "HSTUConfig",
    "FuXiConfig",
    "GRConfig",
]
