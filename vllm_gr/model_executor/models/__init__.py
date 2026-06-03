# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Model wrappers and registry for vllm-GR.

This module provides:
- HSTUForRecommendation: Unified HSTU model (vLLM compatible)
- register_gr_model_architectures(): Registers GR models with vLLM's
  ModelRegistry
"""

from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation
from vllm_gr.model_executor.models.registry import (
    get_gr_architectures,
    is_gr_model,
    is_registered,
    register_gr_model_architectures,
    register_gr_models_to_vllm,
)

__all__ = [
    # Unified model
    "HSTUForRecommendation",
    # Model registry
    "register_gr_model_architectures",
    "register_gr_models_to_vllm",
    "is_registered",
    "get_gr_architectures",
    "is_gr_model",
]
