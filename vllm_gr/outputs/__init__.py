# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Output data types for generative recommendation.

Re-exports all public types from submodules:
- request.py: RecommendedItem, GRRequestOutput
- base.py: GRModelOutput, ScoringOutput, EmbeddingOutput, TextOutput
"""

from vllm_gr.outputs.request import RecommendedItem, GRRequestOutput
from vllm_gr.outputs.base import (
    GRModelOutput,
    ScoringOutput,
    EmbeddingOutput,
    TextOutput,
)

__all__ = [
    "RecommendedItem",
    "GRRequestOutput",
    "GRModelOutput",
    "ScoringOutput",
    "EmbeddingOutput",
    "TextOutput",
]
