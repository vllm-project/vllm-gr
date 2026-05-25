# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
HSTU Model Operators Package

Contains attention, compute, jagged tensor operations, and normalization operators.
Currently provides PyTorch implementations, extensible to Triton/CUDA optimized versions.
"""

from vllm_gr.ops.hstu_attention import hstu_mha, delta_hstu_mha
from vllm_gr.ops.hstu_compute import (
    hstu_compute_uqvk,
    hstu_compute_output,
    hstu_preprocess_and_attention,
)
from vllm_gr.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from vllm_gr.ops.layer_norm import LayerNorm, SwishLayerNorm, layer_norm

__all__ = [
    "hstu_mha",
    "delta_hstu_mha",
    "hstu_compute_uqvk",
    "hstu_compute_output",
    "hstu_preprocess_and_attention",
    "concat_2D_jagged",
    "split_2D_jagged",
    "LayerNorm",
    "SwishLayerNorm",
    "layer_norm",
]
