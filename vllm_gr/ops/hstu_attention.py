# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HSTU Attention Operator Entry Point

Current attention execution is still PyTorch-backed in this fork.
``HammerKernel.NPU`` is accepted as a distinct request value, but it currently
routes into the same conservative PyTorch attention implementation rather than a
dedicated NPU custom kernel.
"""

from typing import Optional

import torch

from vllm_gr.models.dlrm_v3.common import HammerKernel
from vllm_gr.ops.pytorch.pt_hstu_attention import (
    pytorch_cached_hstu_mha,
    pytorch_hstu_mha,
)


def hstu_mha_npu(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    """NPU-labelled entry point that still executes the PyTorch attention path."""
    return pytorch_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        dropout_pr=dropout_pr,
        training=training,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=min_full_attn_seq_len,
    )


def hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
    sort_by_length: bool = False,
    kernel: HammerKernel = HammerKernel.PYTORCH,
    enable_tma: bool = False,
) -> torch.Tensor:
    """
    HSTU Multi-Head Attention

    Current dispatch is intentionally conservative:
    - ``HammerKernel.NPU`` calls a separate wrapper for clarity, but that
      wrapper still executes the PyTorch implementation.
    - All other kernel values also fall back to the PyTorch implementation in
      this fork.
    """
    del sort_by_length, enable_tma
    if kernel == HammerKernel.NPU:
        return hstu_mha_npu(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            dropout_pr=dropout_pr,
            training=training,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            min_full_attn_seq_len=min_full_attn_seq_len,
        )

    return pytorch_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        dropout_pr=dropout_pr,
        training=training,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=min_full_attn_seq_len,
    )


def delta_hstu_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    kernel: HammerKernel = HammerKernel.PYTORCH,
    enable_tma: bool = False,
) -> torch.Tensor:
    """
    Incremental HSTU Attention (for KV cache inference)

    The current fork does not expose a backend-specific delta attention kernel;
    this entry point always executes the PyTorch cached-attention path.
    """
    del kernel, enable_tma
    return pytorch_cached_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        delta_q=delta_q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
