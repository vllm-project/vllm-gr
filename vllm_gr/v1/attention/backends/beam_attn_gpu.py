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


def cache_unshared_kv(
    unshared_key: Any,
    unshared_value: Any,
    curr_key: Any,
    curr_value: Any,
    slot: int,
    decode_step: int,
    beam_session: object = None,  # unused on GPU; xllm-only contract
) -> None:
    """Cascade-mode pool write: slot/0-based semantics, unchanged behavior."""
    from vllm_gr.v1.attention.backends.beam_kernels import (
        cache_unshared_kv as _cache_unshared_kv,
    )

    del beam_session
    _cache_unshared_kv(
        unshared_key,
        unshared_value,
        curr_key,
        curr_value,
        slot,
        decode_step,
    )


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
    beam_pool: object = None,  # Per-layer pool, passed explicitly by forward()
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

        # --- Incremental decode path: read suffix KV from pool slice ---
        beam_session = getattr(attn_metadata, "beam_session", None)
        is_beam_decode = getattr(attn_metadata, "is_beam_decode", False)

        if is_beam_decode and beam_session is not None and beam_session.is_active:
            # Incremental decode: read suffix KV from pool slice
            # instead of extract_suffix_kv from paged cache.
            # Prefer the explicitly-passed beam_pool; fall back to metadata
            # for backward compatibility with callers that still set
            # beam_layer_idx on the metadata.
            if beam_pool is not None:
                pool = beam_pool
            else:
                layer_idx = getattr(attn_metadata, "beam_layer_idx", 0)
                pool = beam_session.pools[layer_idx]
            slot = beam_session.slot
            H = pool.num_kv_heads
            D = pool.head_dim
            W = beam_session.beam_width
            max_steps = pool.max_decode_steps
            suffix_k_buf = beam_session.suffix_k_buf
            suffix_v_buf = beam_session.suffix_v_buf

            # Copy the pool's valid suffix KV into the packed suffix buffer.
            #
            # Graph-compatible path: a Triton kernel reads decode_step from
            # the device buffer and computes the packed destination offsets at
            # runtime.  The eager path below uses host-side Python slices
            # (`: step + 1`), which CUDA Graph capture would bake in — a
            # graph captured at step=1 would forever copy only 2 KV rows.
            decode_step_buf = getattr(beam_session.context, "decode_step_buf", None)
            if decode_step_buf is not None and decode_step_buf.is_cuda:
                from vllm_gr.v1.attention.backends.beam_graph_kernels import (
                    copy_suffix_kv_graph,
                )

                copy_suffix_kv_graph(
                    pool.unshared_key,
                    pool.unshared_value,
                    suffix_k_buf,
                    suffix_v_buf,
                    slot,
                    decode_step_buf,
                    beam_width=W,
                    active_mask=getattr(attn_metadata, "beam_active_mask", None),
                )
            else:
                # Eager fallback: host-int slice copy (packed layout)
                step = beam_session.decode_step  # host int
                pool_k = pool.unshared_key[slot, :W, :, : step + 1, :]
                pool_v = pool.unshared_value[slot, :W, :, : step + 1, :]

                # Copy to suffix_buf in packed layout: [W*(step+1), H, D]
                # Beam 0's KV rows first, then beam 1's, etc.
                total_kv = W * (step + 1)
                suffix_k_buf[:total_kv] = pool_k.permute(0, 2, 1, 3).reshape(-1, H, D)
                suffix_v_buf[:total_kv] = pool_v.permute(0, 2, 1, 3).reshape(-1, H, D)

            # CUDA Graph-compatible: pass the full suffix_buf (fixed shape)
            # instead of a variable-length view. FA2 uses cu_seqlens_k to
            # determine the actual KV boundaries — the padding zeros beyond
            # the packed prefix don't affect the output.
            # max_seqlen_k is set to max_steps (fixed upper bound).
            # Verified: FA2 produces identical output with over-set max_seqlen_k
            # when cu_seqlens_k correctly delimits each batch element's KV.
            suffix_out, suffix_lse = flash_attn_varlen_func(
                q=beam_query,
                k=suffix_k_buf,
                v=suffix_v_buf,
                cu_seqlens_q=attn_metadata.suffix_cu_seqlens_q_device,
                cu_seqlens_k=attn_metadata.suffix_cu_seqlens_k_device,
                seqused_k=None,
                max_seqlen_q=attn_metadata.suffix_max_q_len,
                max_seqlen_k=max_steps,
                softmax_scale=scale,
                causal=True,
                alibi_slopes=alibi_slopes,
                window_size=sliding_window_size,
                block_table=None,
                softcap=logits_soft_cap,
                return_softmax_lse=True,
                fa_version=vllm_flash_attn_version,
            )
        else:
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
