# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU execution primitives for the Beam Attention backend."""

from __future__ import annotations

from copy import copy
from typing import Any, NoReturn, cast

import torch
from vllm import envs
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


class GPUBeamAttentionBatch:
    """Zero-copy request-major attention view over contiguous Beam slots."""

    def __init__(self, sessions: tuple[Any, ...], *, width: int | None = None) -> None:
        if not sessions:
            raise ValueError("GPU Beam attention batch must not be empty")
        first = sessions[0]
        self.sessions = sessions
        self.context = first.context
        self.slot = 0
        self.batch_size = len(sessions)
        self.request_width = first.beam_width if width is None else width
        self.beam_width = self.batch_size * self.request_width
        self.session_id = first.session_id
        if tuple(session.slot for session in sessions) != tuple(range(self.batch_size)):
            raise RuntimeError("GPU Beam batch requires contiguous ordered KV slots")
        if any(session.context is not self.context for session in sessions):
            raise RuntimeError("GPU Beam batch sessions must share one context")
        geometry = (first.beam_width, first.decode_step)
        if any((session.beam_width, session.decode_step) != geometry for session in sessions):
            raise RuntimeError("GPU Beam batch members changed width or Decode step")
        if not 1 <= self.request_width <= self.context.max_beams:
            raise RuntimeError("GPU Beam batch requires the configured exact Beam width")

        self.pools = []
        from vllm_gr.v1.beam.dynamic_beam import get_stage_pools

        for source in get_stage_pools(self.context, self.request_width):
            pool = copy(source)
            storage_rows = self.batch_size * self.request_width
            pool.max_batch = 1
            pool.max_beams = storage_rows
            pool.unshared_key = source.unshared_key[: self.batch_size].reshape(
                1, storage_rows, *source.unshared_key.shape[2:]
            )
            pool.unshared_value = source.unshared_value[: self.batch_size].reshape(
                1, storage_rows, *source.unshared_value.shape[2:]
            )
            self.pools.append(pool)

    @property
    def is_active(self) -> bool:
        return all(session.is_active for session in self.sessions)

    @property
    def decode_step(self) -> int:
        return int(self.sessions[0].decode_step)

    @property
    def suffix_k_buf(self) -> torch.Tensor:
        rows = self.beam_width * self.context.max_decode_steps
        return self.context.suffix_k_buf[:rows]

    @property
    def suffix_v_buf(self) -> torch.Tensor:
        rows = self.beam_width * self.context.max_decode_steps
        return self.context.suffix_v_buf[:rows]


def _num_splits(default: int) -> int:
    """Use deterministic FlashAttention reductions when vLLM requests them."""
    return 1 if envs.VLLM_BATCH_INVARIANT else default


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


def resolve_prefill_flash_attn_version(
    requested: int | str | None,
    decode_version: int,
    *,
    head_size: int,
    kv_cache_dtype: str,
    requires_alibi: bool,
    has_sinks: bool,
    batch_invariant: bool,
) -> int:
    """Resolve once before graph capture, without mutating vLLM's decode config.

    Auto is deliberately limited to FA2 -> FA3 on eligible devices. FA4,
    quantized KV and special attention features need separate qualification.
    Explicit requests fail early instead of silently benchmarking a fallback.
    """
    from vllm_gr.config.validation import validate_prefill_flash_attn_version

    validate_prefill_flash_attn_version(requested)
    if requested is None or requested == decode_version:
        return decode_version
    if requested == "auto" and decode_version != 2:
        return decode_version

    target = 3 if requested == "auto" else int(requested)
    reason = None
    if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
        reason = "independent prefill selection requires unquantized KV"
    elif batch_invariant:
        reason = "independent prefill selection is not enabled with batch invariance"
    elif requires_alibi or has_sinks:
        reason = "independent prefill selection does not support ALiBi or sinks"
    elif head_size <= 0 or head_size > 256 or head_size % 8:
        reason = "prefill head size must be a multiple of 8 in [8, 256]"
    else:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        if not is_fa_version_supported(target):
            reason = fa_version_unsupported_reason(target) or "backend unavailable"
    if reason is not None:
        if requested == "auto":
            return decode_version
        raise ValueError(f"Cannot select prefill FlashAttention {target}: {reason}")
    return target


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
        num_splits=_num_splits(0),
        s_aux=s_aux,
    )

    output[: std_out.shape[0]] = std_out


def validate_short_suffix_attention(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    fa_version: int,
    sliding_window: list[int] | tuple[int, int] | None,
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    sinks: torch.Tensor | None,
) -> None:
    """Validate static layer semantics once, before graph capture/execution."""
    if num_kv_heads <= 0 or num_heads % num_kv_heads or head_size not in (64, 128, 256):
        raise ValueError(
            "Short suffix attention requires integral GQA groups and head size 64/128/256"
        )
    if fa_version != 2:
        raise ValueError("Short suffix attention currently requires FlashAttention 2 prefix states")
    if (
        sliding_window not in (None, (-1, -1), [-1, -1])
        or softcap
        or alibi_slopes is not None
        or sinks is not None
    ):
        raise ValueError(
            "Short suffix attention does not support sliding window, softcap, ALiBi or sinks"
        )


def _is_pure_beam_decode(attn_metadata: BeamAttentionMetadata, beam_width: int) -> bool:
    """Only fuse batches whose complete output is one query per beam.

    For a fixed-membership multi-request batch, beam_width is the session's
    flattened B*W width. Mixed standard/beam rows and prefix recomputation must
    retain the generic path's indexed output writes.
    """
    return (
        beam_width > 0
        and attn_metadata.std_indices is None
        and attn_metadata.prefix_indices is not None
        and attn_metadata.suffix_max_q_len == 1
        and attn_metadata.num_prefix_indices == beam_width
        and attn_metadata.num_actual_tokens == beam_width
    )


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
    prefill_flash_attn_version: int | None = None,
) -> None:
    if attn_metadata.std_indices is not None:
        if attn_metadata.prefix_indices is None:
            std_query = query[: attn_metadata.num_std_indices]
        else:
            std_query = query[attn_metadata.std_indices]
        # Standard rows may share a dispatch with Beam decode rows. Apply the
        # override only to the standard multi-token path, never prefix/suffix.
        std_fa_version = (
            prefill_flash_attn_version
            if prefill_flash_attn_version is not None and attn_metadata.std_max_q_len > 1
            else vllm_flash_attn_version
        )
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
            fa_version=std_fa_version,
            num_splits=_num_splits(0),
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
            num_splits=_num_splits(attn_metadata.max_num_splits),
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
            # Pure incremental short decode: consume the pool directly and
            # merge into the final output, avoiding packed KV and suffix FA2.
            decode_step_buf = getattr(beam_session.context, "decode_step_buf", None)
            if getattr(
                beam_session.context, "enable_short_suffix_attention", False
            ) and _is_pure_beam_decode(attn_metadata, W):
                from vllm_gr.v1.attention.backends.beam_short_suffix import (
                    merge_short_suffix_attention,
                )

                merge_short_suffix_attention(
                    output[:W],
                    beam_query,
                    prefix_out,
                    prefix_lse,
                    pool.unshared_key,
                    pool.unshared_value,
                    slot,
                    decode_step_buf,
                    scale,
                    getattr(attn_metadata, "beam_active_mask", None),
                )
                return

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
                num_splits=_num_splits(0),
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
                num_splits=_num_splits(0),
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
