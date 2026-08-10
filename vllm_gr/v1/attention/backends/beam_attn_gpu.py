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


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")


def _copy_pool_suffix_to_buffer(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    suffix_k_buf: torch.Tensor,
    suffix_v_buf: torch.Tensor,
    beam_width: int,
    suffix_kv_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack an active 5-D Pool region into fixed-address TND buffers.

    The Pool is laid out as ``[1, beam, head, step, dim]`` while FlashAttention
    consumes beam-major, step-minor ``[beam * step, head, dim]`` tensors. The
    strided source view is copied directly into preallocated contiguous storage;
    no temporary K/V storage is created by this function.
    """

    _validate_positive_int("beam_width", beam_width)
    _validate_positive_int("suffix_kv_len", suffix_kv_len)

    if unshared_key.ndim != 5:
        raise ValueError(
            "unshared_key must have shape [1, max_beams, num_heads, max_decode_steps, head_dim]"
        )
    if unshared_value.shape != unshared_key.shape:
        raise ValueError("unshared_value must have the same shape as unshared_key")
    if unshared_key.shape[0] != 1:
        raise ValueError("incremental Pool currently supports exactly one logical request")
    if unshared_key.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("incremental Pool tensors must use torch.float16 or torch.bfloat16")
    if suffix_k_buf.ndim != 3:
        raise ValueError("suffix_k_buf must have shape [capacity, num_heads, head_dim]")
    if suffix_v_buf.shape != suffix_k_buf.shape:
        raise ValueError("suffix_v_buf must have the same shape as suffix_k_buf")

    _, max_beams, num_heads, max_decode_steps, head_dim = unshared_key.shape
    if beam_width > max_beams:
        raise ValueError(f"beam_width={beam_width} exceeds Pool capacity {max_beams}")
    if suffix_kv_len > max_decode_steps:
        raise ValueError(f"suffix_kv_len={suffix_kv_len} exceeds Pool capacity {max_decode_steps}")
    if suffix_k_buf.shape[1:] != (num_heads, head_dim):
        raise ValueError("suffix buffers must match the Pool num_heads and head_dim dimensions")

    for name, tensor in (
        ("unshared_value", unshared_value),
        ("suffix_k_buf", suffix_k_buf),
        ("suffix_v_buf", suffix_v_buf),
    ):
        if tensor.dtype != unshared_key.dtype:
            raise TypeError(f"{name} must have the same dtype as unshared_key")
        if tensor.device != unshared_key.device:
            raise ValueError(f"{name} must be on {unshared_key.device}, got {tensor.device}")

    if not suffix_k_buf.is_contiguous() or not suffix_v_buf.is_contiguous():
        raise ValueError("suffix buffers must be contiguous")

    total_tokens = beam_width * suffix_kv_len
    if total_tokens > suffix_k_buf.shape[0]:
        raise ValueError(
            f"active suffix requires {total_tokens} tokens, "
            f"but buffer capacity is {suffix_k_buf.shape[0]}"
        )

    contig_k = suffix_k_buf[:total_tokens]
    contig_v = suffix_v_buf[:total_tokens]
    target_shape = (beam_width, suffix_kv_len, num_heads, head_dim)
    source_k = unshared_key[0, :beam_width, :, :suffix_kv_len, :]
    source_v = unshared_value[0, :beam_width, :, :suffix_kv_len, :]
    contig_k.view(target_shape).copy_(source_k.permute(0, 2, 1, 3))
    contig_v.view(target_shape).copy_(source_v.permute(0, 2, 1, 3))
    return contig_k, contig_v


def _fixup_single_token_gqa_lse(
    suffix_lse: torch.Tensor,
    attn_metadata: BeamAttentionMetadata,
    vllm_flash_attn_version: int,
) -> torch.Tensor:
    group_size = attn_metadata.num_heads // attn_metadata.num_kv_heads
    if (
        attn_metadata.suffix_max_q_len == 1
        and attn_metadata.suffix_max_seq_len == 1
        and group_size > 1
        and vllm_flash_attn_version != 3
    ):
        return (
            suffix_lse.view(attn_metadata.num_kv_heads, group_size, -1)
            .transpose(1, 2)
            .contiguous()
            .view(attn_metadata.num_heads, -1)
        )
    return suffix_lse


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
        suffix_lse = _fixup_single_token_gqa_lse(suffix_lse, attn_metadata, vllm_flash_attn_version)

        out = torch.empty_like(beam_query)
        merge_attn_states(out, prefix_out, prefix_lse, suffix_out, suffix_lse)
        if attn_metadata.std_indices is None:
            output[: attn_metadata.num_prefix_indices] = out
        else:
            output[attn_metadata.prefix_indices] = out


def cascade_attention_incremental(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: BeamAttentionMetadata,
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    suffix_k_buf: torch.Tensor,
    suffix_v_buf: torch.Tensor,
    sliding_window_size: list[int] | tuple[int, int] | None,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
) -> None:
    """Run prefix/suffix cascade attention from a fixed-address KV Pool."""

    if not attn_metadata.is_incremental_decode:
        raise ValueError("incremental cascade requires is_incremental_decode=True")

    beam_width = attn_metadata.incremental_beam_width
    suffix_kv_len = attn_metadata.incremental_suffix_kv_len
    _validate_positive_int("incremental_beam_width", beam_width)
    _validate_positive_int("incremental_suffix_kv_len", suffix_kv_len)
    if attn_metadata.num_prefix_indices != beam_width:
        raise ValueError("num_prefix_indices must equal incremental_beam_width")
    if attn_metadata.suffix_total_tokens != beam_width * suffix_kv_len:
        raise ValueError("suffix_total_tokens must equal beam_width * suffix_kv_len")
    if attn_metadata.suffix_max_q_len != 1:
        raise ValueError("incremental suffix query length must be one token per beam")
    if attn_metadata.suffix_max_seq_len != suffix_kv_len:
        raise ValueError("suffix_max_seq_len must equal incremental_suffix_kv_len")
    if query.shape[0] < beam_width or output.shape[0] < beam_width:
        raise ValueError("query and output must contain every active beam row")
    if attn_metadata.prefix_indices is None:
        raise ValueError("incremental cascade requires prefix_indices")
    if attn_metadata.prefix_indices.numel() != beam_width:
        raise ValueError("prefix_indices must contain every active beam row")

    beam_query = query[:beam_width]
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

    contig_k, contig_v = _copy_pool_suffix_to_buffer(
        unshared_key,
        unshared_value,
        suffix_k_buf,
        suffix_v_buf,
        beam_width,
        suffix_kv_len,
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
    suffix_lse = _fixup_single_token_gqa_lse(suffix_lse, attn_metadata, vllm_flash_attn_version)
    merge_attn_states(
        output[:beam_width],
        prefix_out,
        prefix_lse,
        suffix_out,
        suffix_lse,
    )
