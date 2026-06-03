# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

from typing import List

import torch


def pytorch_swish_layer_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    PyTorch native Swish + LayerNorm implementation.

    Swish activation function: input * sigmoid(input)
    Performs LayerNorm first, then applies Swish activation.

    Args:
        input: Input tensor
        normalized_shape: Normalized shape
        weight: Scale weight parameter
        bias: Bias parameter
        eps: Numerical stability parameter

    Returns:
        Tensor after LayerNorm and Swish activation
    """
    input_dtype = input.dtype
    input = input.to(torch.float32)
    return (
        input
        * torch.sigmoid(
            torch.nn.functional.layer_norm(
                input,
                normalized_shape,
                weight.to(torch.float32),
                bias.to(torch.float32),
                eps,
            )
        )
    ).to(input_dtype)


def pytorch_layer_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    PyTorch native LayerNorm implementation.

    Args:
        input: Input tensor
        normalized_shape: Normalized shape
        weight: Scale weight parameter
        bias: Bias parameter
        eps: Numerical stability parameter

    Returns:
        Normalized tensor
    """
    input_dtype = input.dtype
    return torch.nn.functional.layer_norm(
        input.to(torch.float32),
        normalized_shape,
        weight.to(torch.float32),
        bias.to(torch.float32),
        eps,
    ).to(input_dtype)
