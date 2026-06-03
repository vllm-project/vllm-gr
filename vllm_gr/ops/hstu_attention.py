# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

"""
HSTU Attention Operator Entry Point

This module preserves the existing PyTorch-backed implementation while adding a
unified dual-path dispatch scaffold for NPU fused kernels.
"""

from typing import Optional

import torch

from vllm_gr.logger import init_logger
from vllm_gr.models.dlrm_v3.common import HammerKernel
from vllm_gr.ops.npu import can_run_hstu_jagged, run_hstu_jagged
from vllm_gr.ops.pytorch.pt_hstu_attention import (
    pytorch_cached_hstu_mha,
    pytorch_hstu_mha,
)

logger = init_logger(__name__)


_HSTU_USE_NPU_FUSED_PREFILL: bool = False


def configure_hstu_npu_fused_dispatch(
    *,
    prefill_enabled: Optional[bool] = None,
) -> None:
    """Configure HSTU fused dispatch switches."""
    global _HSTU_USE_NPU_FUSED_PREFILL

    _HSTU_USE_NPU_FUSED_PREFILL = bool(prefill_enabled)


def hstu_npu_fused_dispatch_config() -> dict[str, bool]:
    return {
        "prefill_enabled": _HSTU_USE_NPU_FUSED_PREFILL,
    }


def should_use_hstu_npu_fused_prefill(
    *,
    kernel: HammerKernel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_seq_len: int,
    num_targets: Optional[torch.Tensor],
    contextual_seq_len: int,
) -> bool:
    if not _HSTU_USE_NPU_FUSED_PREFILL:
        return False
    if kernel != HammerKernel.NPU:
        return False
    return can_run_hstu_jagged(
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        max_seq_len=max_seq_len,
        num_targets=num_targets,
        contextual_seq_len=contextual_seq_len,
        target_group_size=1,
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
    """NPU-labelled prefill entry point backed by mxrec.hstu_jagged when eligible."""
    if dropout_pr != 0.0 or training or max_attn_len != 0 or min_full_attn_seq_len != 0:
        logger.debug(
            "[HSTU] Prefill selects PyTorch attention instead of mxrec.hstu_jagged "
            "because dropout/training/max_attn_len/min_full_attn_seq_len is unsupported: "
            "dropout_pr=%s training=%s max_attn_len=%s min_full_attn_seq_len=%s",
            dropout_pr,
            training,
            max_attn_len,
            min_full_attn_seq_len,
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

    try:
        logger.debug(
            "[HSTU] Prefill selects mxrec.hstu_jagged: tokens=%d heads_q=%d qk_dim=%d v_dim=%d batch=%d max_seq_len=%d",
            int(q.shape[0]),
            int(q.shape[1]),
            int(q.shape[2]),
            int(v.shape[2]),
            int(seq_offsets.numel()) - 1,
            int(max_seq_len),
        )
        return run_hstu_jagged(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            num_targets=num_targets,
            contextual_seq_len=contextual_seq_len,
        )
    except (RuntimeError, AttributeError, OSError) as exc:
        logger.debug(
            "[HSTU] Prefill falls back to PyTorch attention after mxrec.hstu_jagged failure: %s: %s",
            type(exc).__name__,
            exc,
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


def delta_hstu_mha_npu_paged(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    kv_cache: Optional[torch.Tensor] = None,
    block_table_tensor: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    num_kv_heads: Optional[int] = None,
    head_size: Optional[int] = None,
    head_size_v: Optional[int] = None,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Reserved decode fused branch. Decode fused dispatch is disabled in this PR."""
    del kv_cache, block_table_tensor, seq_lens, num_kv_heads, head_size, head_size_v, block_size
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


def hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = False,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
    sort_by_length: bool = False,
    kernel: HammerKernel = HammerKernel.PYTORCH,
    enable_tma: bool = False,
) -> torch.Tensor:
    """HSTU Multi-Head Attention with legacy/fused dispatch scaffold."""
    del sort_by_length, enable_tma
    if should_use_hstu_npu_fused_prefill(
        kernel=kernel,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        max_seq_len=max_seq_len,
        num_targets=num_targets,
        contextual_seq_len=contextual_seq_len,
    ):
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

    logger.debug(
        "[HSTU] Prefill selects PyTorch attention path: prefill_enabled=%s kernel=%s",
        _HSTU_USE_NPU_FUSED_PREFILL,
        kernel.value if isinstance(kernel, HammerKernel) else kernel,
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
    kv_cache: Optional[torch.Tensor] = None,
    block_table_tensor: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    num_kv_heads: Optional[int] = None,
    head_size: Optional[int] = None,
    head_size_v: Optional[int] = None,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Incremental HSTU attention with legacy/fused dispatch scaffold."""
    del enable_tma, kernel, kv_cache, block_table_tensor, seq_lens
    del num_kv_heads, head_size, head_size_v, block_size

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
