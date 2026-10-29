# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-independent attention states consumed by Beam suffix/merge kernels."""

from typing import NamedTuple

import torch


class AttentionState(NamedTuple):
    """Normalized attention output and its softmax mass.

    ``out`` is [tokens, query_heads, head_dim] in activation dtype; ``lse`` is
    [query_heads, tokens] in FP32, natural-log units and canonical head order.
    Strides need not be contiguous. An infinite LSE of either sign denotes an
    absent state; consumers must ignore its output (which may be uninitialized).
    Accepting both empty-state sentinels avoids a GPU copy solely to translate
    FA2's +inf to FA3/FA4's -inf. NaNs are not valid LSE values.
    """

    out: torch.Tensor
    lse: torch.Tensor


def normalize_flash_attn_state(
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    fa_version: int,
    num_kv_heads: int,
    max_query_len: int,
    paged_kv: bool = False,
) -> AttentionState:
    """Adapt vLLM's FA2/3/4 varlen results without reading device data on CPU.

    FA3/FA4 and FA2 paged KV already return canonical head-major LSE. FA2's
    non-paged single-query GQA specialization needs a layout conversion.
    Ordinary multi-query prefix calls reuse their tensors without GPU work.
    """
    if fa_version not in (2, 3, 4):
        raise ValueError(f"Unsupported FlashAttention state version: {fa_version}")
    if out.ndim != 3:
        raise ValueError("Attention output must have shape [tokens, heads, head_dim]")
    tokens, heads, _ = out.shape
    if num_kv_heads <= 0 or heads % num_kv_heads:
        raise ValueError("Attention state requires integral GQA groups")
    if lse.shape != (heads, tokens) or lse.dtype != torch.float32:
        raise ValueError("Attention LSE must be FP32 with shape [heads, tokens]")
    if lse.device != out.device:
        raise ValueError("Attention output and LSE must be on the same device")

    group_size = heads // num_kv_heads
    if fa_version == 2 and not paged_kv and max_query_len == 1 and group_size > 1:
        lse = (
            lse.reshape(num_kv_heads, group_size, tokens)
            .transpose(1, 2)
            .contiguous()
            .view(heads, tokens)
        )
    return AttentionState(out, lse)


def clear_empty_prefix_states(
    state: AttentionState,
    cu_query_lens: torch.Tensor,
    kv_lens: torch.Tensor,
) -> AttentionState:
    """Make empty paged prefixes explicit, including during graph replay.

    Some FA2 builds leave parts of empty-request LSE uninitialized in GQA.
    Infinity checks alone cannot identify those entries. Consult the device
    lengths and clear only empty requests; nonempty outputs stay untouched.
    """
    from vllm_gr.v1.attention.backends.beam_prefix_state_kernels import (
        clear_empty_prefix,
    )

    clear_empty_prefix(state.out, state.lse, cu_query_lens, kv_lens)
    return state
