# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Reference PyTorch Beam Attention for the eager CPU backend.

Follows the Ascend cascade contract (``beam_attn_npu``): the shared prompt is
read from the paged cache, each beam's private suffix from the session pool,
and the two partial results are merged by log-sum-exp. Every sequence is
handled with ``torch.nn.functional.scaled_dot_product_attention``-equivalent
math in FP32, so this module favors clarity over speed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vllm_gr.v1.beam.attention_pool import write_eager_step

cache_unshared_kv = write_eager_step


def is_available() -> bool:
    return True


def get_flash_attn_version(*, requires_alibi: bool = False, head_size: int | None = None) -> int:
    del requires_alibi, head_size
    return 0


def supports_sinks() -> bool:
    return False


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: Any,
    v_scale: Any,
) -> None:
    del kv_cache_dtype, k_scale, v_scale
    slots = slot_mapping[: key.shape[0]].long()
    valid = slots >= 0
    block_size = key_cache.shape[1]
    blocks, offsets = slots[valid] // block_size, slots[valid] % block_size
    key_cache[blocks, offsets] = key[: slots.shape[0]][valid].to(key_cache.dtype)
    value_cache[blocks, offsets] = value[: slots.shape[0]][valid].to(value_cache.dtype)


def _attend(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float, causal: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """[Q, Hq, D] x [K, Hkv, D] -> output [Q, Hq, D] and LSE [Q, Hq, 1]."""
    q_len, kv_len = query.shape[0], key.shape[0]
    group = query.shape[1] // key.shape[1]
    q = query.float().transpose(0, 1)
    k = key.float().repeat_interleave(group, dim=1).transpose(0, 1)
    v = value.float().repeat_interleave(group, dim=1).transpose(0, 1)
    scores = torch.matmul(q, k.transpose(1, 2)) * scale
    if causal and q_len > 1:
        # Bottom-right aligned: query i sees keys up to kv_len - q_len + i.
        rows = torch.arange(q_len).unsqueeze(1) + (kv_len - q_len)
        scores.masked_fill_(torch.arange(kv_len).unsqueeze(0) > rows, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    out = torch.matmul(torch.exp(scores - lse), v)
    return out.transpose(0, 1), lse.transpose(0, 1)


def _paged(cache: torch.Tensor, blocks: torch.Tensor, length: int) -> torch.Tensor:
    return cache[blocks.long()].reshape(-1, *cache.shape[2:])[:length]


def _paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_query: Sequence[int],
    kv_lengths: Sequence[int],
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence attention; ``cu_query`` is cumulative without leading zero."""
    outs, lses, start = [], [], 0
    for index, (end, kv_len) in enumerate(zip(cu_query, kv_lengths, strict=False)):
        blocks = block_table[index]
        out, lse = _attend(
            query[start:end],
            _paged(key_cache, blocks, kv_len),
            _paged(value_cache, blocks, kv_len),
            scale,
            causal,
        )
        outs.append(out)
        lses.append(lse)
        start = end
    return torch.cat(outs), torch.cat(lses)


def prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: Any,
    sliding_window_size: Any,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: Any = None,
    alibi_slopes: Any = None,
) -> None:
    del sliding_window_size, logits_soft_cap, vllm_flash_attn_version, s_aux, alibi_slopes
    num_reqs = attn_metadata.num_reqs
    if num_reqs <= 0:
        return
    tokens = attn_metadata.num_actual_tokens
    out, _ = _paged_attention(
        query[:tokens],
        key_cache,
        value_cache,
        attn_metadata.block_table,
        list(attn_metadata.query_start_loc_cpu[1 : num_reqs + 1]),
        list(attn_metadata.seq_lens_cpu[:num_reqs]),
        scale,
        attn_metadata.causal,
    )
    output[:tokens] = out.to(output.dtype)


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: Any,
    sliding_window_size: Any,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: Any = None,
    alibi_slopes: Any = None,
    beam_pool: Any = None,
) -> None:
    del sliding_window_size, logits_soft_cap, vllm_flash_attn_version, s_aux, alibi_slopes
    from vllm_gr.v1.attention.backends.beam_attn_metadata import (
        require_beam_decode_session,
        resolve_beam_pool,
    )

    beam = attn_metadata.prefix_indices is not None
    if attn_metadata.std_indices is not None:
        rows = attn_metadata.std_indices if beam else slice(attn_metadata.num_std_indices)
        out, _ = _paged_attention(
            query[rows],
            key_cache,
            value_cache,
            attn_metadata.std_block_table,
            attn_metadata.std_cu_seqlens_q_cpu,
            attn_metadata.std_seqlens_kv_cpu,
            scale,
            attn_metadata.causal,
        )
        output[rows] = out.to(output.dtype)
    if not beam:
        return

    session = require_beam_decode_session(attn_metadata)
    pool = resolve_beam_pool(attn_metadata, beam_pool)
    rows = (
        attn_metadata.prefix_indices
        if attn_metadata.std_indices is not None
        else slice(attn_metadata.num_prefix_indices)
    )
    beam_query = query[rows]
    prefix_out, prefix_lse = _paged_attention(
        beam_query,
        key_cache,
        value_cache,
        attn_metadata.prefix_block_table,
        attn_metadata.prefix_cu_seqlens_q_cpu,
        attn_metadata.prefix_seqlens_kv_cpu,
        scale,
        causal=False,
    )
    # Each beam row attends to its own private suffix of decode_step + 1 tokens.
    step = session.decode_step
    width = int(getattr(session, "attention_width", session.beam_width))
    suffix_k = pool.unshared_key[session.slot, :width, :, : step + 1, :].transpose(1, 2)
    suffix_v = pool.unshared_value[session.slot, :width, :, : step + 1, :].transpose(1, 2)
    per_row = beam_query.shape[0] // width
    suffix = [
        _attend(beam_query[i * per_row : (i + 1) * per_row], suffix_k[i], suffix_v[i], scale, True)
        for i in range(width)
    ]
    suffix_out = torch.cat([out for out, _ in suffix])
    suffix_lse = torch.cat([lse for _, lse in suffix])
    # Log-sum-exp merge of the shared-prefix and private-suffix partial results.
    total = torch.logaddexp(prefix_lse, suffix_lse)
    out = torch.exp(prefix_lse - total) * prefix_out + torch.exp(suffix_lse - total) * suffix_out
    output[rows] = out.to(output.dtype)
