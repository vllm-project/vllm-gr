# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU execution primitives for the Beam Attention backend."""

from __future__ import annotations

from typing import Any, NoReturn, cast

import torch
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version as _get_flash_attn_version,
)
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
)
from vllm_gr.v1.attention.backends.beam_attn_triton import extract_suffix_kv

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
else:

    def _flash_attn_varlen_unavailable(
        *args: Any,
        **kwargs: Any,
    ) -> NoReturn:
        raise RuntimeError("BeamAttentionBackend requires FlashAttention varlen support")

    flash_attn_varlen_func = _flash_attn_varlen_unavailable
    reshape_and_cache_flash = _flash_attn_varlen_unavailable
    merge_attn_states = _flash_attn_varlen_unavailable
    flash_attn_supports_sinks = _flash_attn_varlen_unavailable


def is_available() -> bool:
    return bool(is_flash_attn_varlen_func_available())


def get_flash_attn_version(
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
) -> int | None:
    return cast(
        int | None,
        _get_flash_attn_version(
            requires_alibi=requires_alibi,
            head_size=head_size,
        ),
    )


def supports_sinks() -> bool:
    return bool(is_available() and flash_attn_supports_sinks())


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: BeamAttentionMetadata,
    sliding_window_size: list[int] | tuple[int, int] | None,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
) -> None:
    """Dedicated pure prefill attention execution for the prefill graph path."""
    cu_seqlens_q = getattr(attn_metadata, "query_start_loc", None)
    max_q_len = getattr(attn_metadata, "max_query_len", 0)
    seqused_k = getattr(attn_metadata, "seq_lens", None)
    max_seq_len = getattr(attn_metadata, "max_seq_len", 0)
    block_table = (
        getattr(attn_metadata, "block_table_tensor", None)
        if getattr(attn_metadata, "block_table_tensor", None) is not None
        else getattr(attn_metadata, "block_table", None)
    )
    causal = getattr(attn_metadata, "causal", True)
    assert cu_seqlens_q is not None, "query_start_loc must not be None for prefill attention"
    assert seqused_k is not None, "seq_lens must not be None for prefill attention"

    # Compute only the real rows: FA2 silently ignores query rows beyond
    # cu_seqlens_q[-1], but slicing keeps the fast path aligned with the NPU
    # FIA convention (queryT == last(actual_seq_qlen)).
    num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
    query = query[:num_actual_tokens]

    std_out = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_q_len,
        seqused_k=seqused_k,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=causal,
        alibi_slopes=alibi_slopes,
        window_size=sliding_window_size,
        block_table=block_table,
        softcap=logits_soft_cap,
        fa_version=vllm_flash_attn_version,
        s_aux=s_aux,
    )

    output[: std_out.shape[0]] = std_out


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: BeamAttentionMetadata,
    sliding_window_size: list[int] | tuple[int, int] | None,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
) -> None:
    if attn_metadata.std_indices is not None:
        if attn_metadata.prefix_indices is None:
            std_query = query[: attn_metadata.num_std_indices]
        else:
            std_query = query[attn_metadata.std_indices]
        std_out = flash_attn_varlen_func(
            q=std_query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.std_cu_seqlens_q_device,
            max_seqlen_q=attn_metadata.std_max_q_len,
            seqused_k=attn_metadata.std_seqlens_kv_device,
            max_seqlen_k=attn_metadata.std_max_seq_len,
            softmax_scale=scale,
            causal=attn_metadata.causal,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=attn_metadata.std_block_table,
            softcap=logits_soft_cap,
            fa_version=vllm_flash_attn_version,
        )
        if attn_metadata.prefix_indices is None:
            output[: attn_metadata.num_std_indices] = std_out
        else:
            output[attn_metadata.std_indices] = std_out

    if attn_metadata.prefix_indices is not None:
        if attn_metadata.std_indices is None:
            beam_query = query[: attn_metadata.num_prefix_indices]
        else:
            beam_query = query[attn_metadata.prefix_indices]

        prefix_out, prefix_lse = flash_attn_varlen_func(
            q=beam_query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.prefix_cu_seqlens_q_device,
            seqused_k=attn_metadata.prefix_seqlens_kv_device,
            max_seqlen_q=attn_metadata.prefix_max_q_len,
            max_seqlen_k=attn_metadata.prefix_max_seq_len,
            softmax_scale=scale,
            causal=False,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=attn_metadata.prefix_block_table,
            softcap=logits_soft_cap,
            return_softmax_lse=True,
            fa_version=vllm_flash_attn_version,
            scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            num_splits=attn_metadata.max_num_splits,
            s_aux=s_aux,
        )

        # Standard beam decode path: extract_suffix_kv from paged cache
        contig_k, contig_v = extract_suffix_kv(
            key_cache,
            value_cache,
            attn_metadata.suffix_slot_mapping_out,
            attn_metadata.suffix_total_tokens,
            attn_metadata.block_size,
        )

        suffix_out, suffix_lse = flash_attn_varlen_func(
            q=beam_query,
            k=contig_k,
            v=contig_v,
            cu_seqlens_q=attn_metadata.suffix_cu_seqlens_q_device,
            cu_seqlens_k=attn_metadata.suffix_cu_seqlens_k_device,
            seqused_k=None,
            max_seqlen_q=attn_metadata.suffix_max_q_len,
            max_seqlen_k=attn_metadata.suffix_max_seq_len,
            softmax_scale=scale,
            causal=True,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=None,
            softcap=logits_soft_cap,
            return_softmax_lse=True,
            fa_version=vllm_flash_attn_version,
        )

        # FA2 GQA de-swizzle: with grouped-query attention (group_size > 1) and a
        # single query token, flash-attention returns softmax_lse laid out as
        # [num_kv_heads, group_size, num_tokens] instead of [num_heads, num_tokens].
        # Transpose back so merge_attn_states sees the per-head layout. FA3 already
        # returns the head layout, and the prefix call (max_q > 1) is unaffected.
        group_size = attn_metadata.num_heads // attn_metadata.num_kv_heads
        if attn_metadata.suffix_max_q_len == 1 and group_size > 1 and vllm_flash_attn_version != 3:
            suffix_lse = (
                suffix_lse.view(attn_metadata.num_kv_heads, group_size, -1)
                .transpose(1, 2)
                .contiguous()
                .view(attn_metadata.num_heads, -1)
            )

        out = torch.empty_like(beam_query)
        merge_attn_states(out, prefix_out, prefix_lse, suffix_out, suffix_lse)

        if attn_metadata.std_indices is None:
            output[: attn_metadata.num_prefix_indices] = out
        else:
            output[attn_metadata.prefix_indices] = out
