# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch implementation of HSTU linear operations"""

import torch
import torch.nn.functional as F


def pytorch_norm_mul_dropout(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
) -> torch.Tensor:
    """
    Apply normalization, gating, and optional concatenation.
    
    This matches the original generative-recommenders implementation.
    """
    dtype = x.dtype
    if silu_u:
        u = F.silu(u)
    x = x.to(torch.float32)
    u = u.to(torch.float32)
    if group_norm:
        y = u * F.group_norm(
            x.view(-1, num_heads, linear_dim),
            num_groups=num_heads,
            weight=weight.to(torch.float32),
            bias=bias.to(torch.float32),
            eps=eps,
        ).view(-1, num_heads * linear_dim)
        if concat_u and concat_x:
            y = torch.cat([u, x, y], dim=1)
    else:
        y = u * F.layer_norm(
            x,
            normalized_shape=(x.shape[-1],),
            weight=weight.to(torch.float32),
            bias=bias.to(torch.float32),
            eps=eps,
        )
        if concat_u and concat_x:
            y = torch.cat([u, x, y], dim=1)
        elif concat_u:
            y = torch.cat([u, y], dim=1)
        elif concat_x:
            y = torch.cat([x, y], dim=1)
    y = F.dropout(
        y,
        p=dropout_ratio,
        training=training,
    )
    return y.to(dtype)


def pytorch_hstu_compute_output(
    attn: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    output_weight: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    concat_u: bool,
    concat_x: bool,
    group_norm: bool,
    num_heads: int,
    linear_dim: int,
    silu_u: bool = False,
) -> torch.Tensor:
    """
    Compute HSTU output with layer normalization and dropout.
    
    This matches the original generative-recommenders implementation exactly.
    
    Args:
        attn: Attention output (L, H*D)
        u: Gate values (L, H*D)
        x: Original input (L, E)
        norm_weight, norm_bias: Normalization parameters
        output_weight: Output projection weight, shape (y_dim, E) where
                       y_dim depends on concat_u and concat_x settings
        eps: Normalization epsilon
        dropout_ratio: Dropout ratio
        training: Training mode
        concat_u: Concatenate gate values
        concat_x: Concatenate original input
        group_norm: Use group normalization
        num_heads: Number of attention heads
        linear_dim: Linear dimension (hidden_dim)
        silu_u: Apply SiLU activation to u
    
    Returns:
        Output tensor (L, E)
    """
    dtype = x.dtype
    y = pytorch_norm_mul_dropout(
        x=attn,
        u=u,
        weight=norm_weight,
        bias=norm_bias,
        eps=eps,
        dropout_ratio=dropout_ratio,
        training=training,
        silu_u=silu_u,
        concat_u=concat_u,
        concat_x=concat_x,
        group_norm=group_norm,
        num_heads=num_heads,
        linear_dim=linear_dim,
    )
    # Use torch.addmm for: x + y @ output_weight
    # output_weight shape: (y_dim, embedding_dim)
    return torch.addmm(x, y, output_weight.to(x.dtype)).to(dtype)

