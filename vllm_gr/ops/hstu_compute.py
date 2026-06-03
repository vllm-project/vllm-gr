# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

"""
HSTU Compute Operators

Contains UVQK projection, attention computation, and output projection operations.
This module keeps the upstream kernel-shaped API, but the current fork still
executes the linear, normalization, and output-projection pieces through
PyTorch implementations.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from vllm_gr.models.dlrm_v3.common import HammerKernel
from vllm_gr.ops.hstu_attention import hstu_mha
from vllm_gr.ops.layer_norm import layer_norm
from vllm_gr.ops.pytorch.pt_hstu_linear import pytorch_hstu_compute_output


def addmm(
    bias: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """Matrix multiplication with bias, using torch.addmm only."""
    del kernel
    return torch.addmm(bias, input, weight)


def hstu_compute_uqvk(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute U, Q, V, K for HSTU layer.

    Args:
        x: Input tensor (L, E)
        norm_weight, norm_bias: LayerNorm parameters
        norm_eps: LayerNorm epsilon
        num_heads: Number of attention heads
        attn_dim: Attention dimension
        hidden_dim: Hidden dimension
        uvqk_weight, uvqk_bias: Projection parameters
        kernel: Requested kernel type. Specialized paths are not implemented
            here beyond the downstream PyTorch-backed attention entry point.

    Returns:
        (u, q, k, v) tensors
    """
    # Layer normalization
    normed_x = layer_norm(x, weight=norm_weight, bias=norm_bias, eps=norm_eps, kernel=kernel)

    # Project to U, V, Q, K
    uvqk = addmm(uvqk_bias, normed_x, uvqk_weight, kernel)

    # Split into components
    u, v, q, k = torch.split(
        uvqk,
        [
            hidden_dim * num_heads,
            hidden_dim * num_heads,
            attn_dim * num_heads,
            attn_dim * num_heads,
        ],
        dim=1,
    )

    # Apply SiLU activation to gate
    u = F.silu(u)

    # Reshape Q, K, V
    q = q.view(-1, num_heads, attn_dim)
    k = k.view(-1, num_heads, attn_dim)
    v = v.view(-1, num_heads, hidden_dim)

    return u, q, k, v


def hstu_compute_output(
    attn: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    output_weight: torch.Tensor,
    num_heads: int,
    linear_dim: int,
    dropout_ratio: float,
    training: bool,
    concat_u: bool,
    concat_x: bool,
    group_norm: bool,
    recompute_y_in_backward: bool,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """
    Compute HSTU output

    Includes normalization, gating, dropout, and residual connection.
    The current fork always uses the PyTorch output-projection helper.
    """
    del recompute_y_in_backward, kernel
    return pytorch_hstu_compute_output(
        attn=attn,
        u=u,
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        output_weight=output_weight,
        eps=norm_eps,
        dropout_ratio=dropout_ratio,
        training=training,
        concat_u=concat_u,
        concat_x=concat_x,
        group_norm=group_norm,
        num_heads=num_heads,
        linear_dim=linear_dim,
    )


def hstu_preprocess_and_attention(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    attn_alpha: float,
    causal: bool,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    recompute_uvqk_in_backward: bool,
    recompute_normed_x_in_backward: bool,
    sort_by_length: bool,
    prefill: bool = False,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Combined HSTU preprocessing and attention computation

    Includes UVQK projection and multi-head attention computation.
    The projection and output-side pieces in this helper stay on PyTorch in the
    current fork; only the downstream attention entry point interprets the
    requested kernel value, and it remains conservatively PyTorch-backed today.
    """
    del recompute_uvqk_in_backward, recompute_normed_x_in_backward, prefill
    # Compute U, Q, K, V
    u, q, k, v = hstu_compute_uqvk(
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        norm_eps=norm_eps,
        num_heads=num_heads,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        uvqk_weight=uvqk_weight,
        uvqk_bias=uvqk_bias,
        kernel=kernel,
    )

    # Compute attention
    attn_output = hstu_mha(
        max_seq_len=max_seq_len,
        alpha=attn_alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        dropout_pr=0.0,
        training=False,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        sort_by_length=sort_by_length,
        kernel=kernel,
    ).view(-1, hidden_dim * num_heads)

    return u, attn_output, k, v
