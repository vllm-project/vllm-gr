# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton implementations of xllm-compatible OneRec decode operators."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm_gr.ops.onerec_decode import (
    _beam_search_group_torch,
    _stable_topk_indices,
    _validate_final_select,
    _validate_group,
    _validate_select_unshared_kv,
)

_MAX_TRITON_GROUP_K = 8192
_MAX_FUSED_SELECT_BEAM_SIZE = 128
_FUSED_SCORE_TOPK_WIDTH = 128
_SCORE_TOPK_CHUNK_SIZE = 256
_SCORE_TOPK_REDUCE_SIZE = 1024


@triton.jit
def _score_topk_chunk_kernel(
    log_probs,
    top_probs,
    out_keys,
    log_probs_stride_row,
    top_probs_stride_row,
    top_probs_stride_candidate,
    out_stride_batch,
    out_stride_candidate,
    beam_width: tl.constexpr,
    candidate_width: tl.constexpr,
    candidate_count: tl.constexpr,
    index_bits: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Select stable chunk-local top-k values as sortable int64 keys."""

    request = tl.program_id(0)
    chunk = tl.program_id(1)
    flat = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = flat < candidate_count
    parent = flat // candidate_width
    candidate = flat % candidate_width
    global_parent = request * beam_width + parent
    score = tl.load(
        log_probs + global_parent * log_probs_stride_row,
        mask=valid,
        other=-float("inf"),
    ) + tl.load(
        top_probs + global_parent * top_probs_stride_row + candidate * top_probs_stride_candidate,
        mask=valid,
        other=0.0,
    )

    # Map float32 to monotonically ordered unsigned bits. Canonicalizing signed
    # zero keeps PyTorch's stable-sort equality semantics for +0.0 and -0.0.
    score = tl.where(score == 0.0, 0.0, score)
    signed_bits = score.to(tl.int32, bitcast=True)
    bits = signed_bits.to(tl.int64) & 0xFFFFFFFF
    ordered = tl.where(signed_bits < 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)
    tie_break = candidate_count - 1 - flat
    key = (ordered << index_bits) | tie_break
    key = tl.where(valid, key, 0)
    selected = tl.topk(key, TOP_K)
    output = chunk * TOP_K + tl.arange(0, TOP_K)
    tl.store(
        out_keys + request * out_stride_batch + output * out_stride_candidate,
        selected,
    )


@triton.jit
def _key_topk_reduce_kernel(
    input_keys,
    output_keys,
    input_stride_batch,
    input_stride_candidate,
    output_stride_batch,
    output_stride_candidate,
    input_count: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    request = tl.program_id(0)
    chunk = tl.program_id(1)
    offset = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    key = tl.load(
        input_keys + request * input_stride_batch + offset * input_stride_candidate,
        mask=offset < input_count,
        other=0,
    )
    selected = tl.topk(key, TOP_K)
    output = chunk * TOP_K + tl.arange(0, TOP_K)
    tl.store(
        output_keys + request * output_stride_batch + output * output_stride_candidate,
        selected,
    )


def _stable_score_topk_keys_triton(
    log_probs: torch.Tensor,
    top_probs: torch.Tensor,
    batch_size: int,
    beam_width: int,
    candidate_width: int,
    result_width: int,
) -> tuple[torch.Tensor, int]:
    """Return stable descending top-k keys without sorting every candidate."""

    candidate_count = beam_width * candidate_width
    index_bits = max(1, (candidate_count - 1).bit_length())
    chunk_count = triton.cdiv(candidate_count, _SCORE_TOPK_CHUNK_SIZE)
    keys = torch.empty(
        (batch_size, chunk_count * result_width),
        dtype=torch.int64,
        device=log_probs.device,
    )
    _score_topk_chunk_kernel[(batch_size, chunk_count)](
        log_probs,
        top_probs,
        keys,
        log_probs.stride(0),
        top_probs.stride(0),
        top_probs.stride(1),
        keys.stride(0),
        keys.stride(1),
        beam_width=beam_width,
        candidate_width=candidate_width,
        candidate_count=candidate_count,
        index_bits=index_bits,
        TOP_K=result_width,
        BLOCK_SIZE=_SCORE_TOPK_CHUNK_SIZE,
        num_warps=8,
    )

    key_count = chunk_count * result_width
    while key_count > result_width:
        reduce_chunks = triton.cdiv(key_count, _SCORE_TOPK_REDUCE_SIZE)
        reduced = torch.empty(
            (batch_size, reduce_chunks * result_width),
            dtype=torch.int64,
            device=log_probs.device,
        )
        _key_topk_reduce_kernel[(batch_size, reduce_chunks)](
            keys,
            reduced,
            keys.stride(0),
            keys.stride(1),
            reduced.stride(0),
            reduced.stride(1),
            input_count=key_count,
            TOP_K=result_width,
            BLOCK_SIZE=_SCORE_TOPK_REDUCE_SIZE,
            num_warps=8,
        )
        keys = reduced
        key_count = reduce_chunks * result_width
    return keys, index_bits


@triton.jit
def _parents_from_prefix_kernel(
    group_token_num,
    parents,
    prefix_stride,
    parent_stride_batch,
    parent_stride_beam,
    beam_size: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    request = tl.program_id(0)
    new_beam = tl.program_id(1)
    prefix_base = request * beam_size
    last_prefix = tl.load(group_token_num + (prefix_base + beam_size - 1) * prefix_stride)
    global_offset = tl.where(last_prefix > beam_size, request * beam_size, 0)

    low = 0
    high = beam_size
    for _ in range(SEARCH_STEPS):
        middle = (low + high) // 2
        value = tl.load(group_token_num + (prefix_base + middle) * prefix_stride) - global_offset
        move_right = value <= new_beam
        low = tl.where(move_right, middle + 1, low)
        high = tl.where(move_right, high, middle)

    tl.store(
        parents + request * parent_stride_batch + new_beam * parent_stride_beam,
        low,
    )


@triton.jit
def _select_kv_fused_kernel(
    key_cache,
    value_cache,
    parents,
    block_table,
    key_stride_block,
    key_stride_beam,
    key_stride_head,
    key_stride_step,
    key_stride_dim,
    value_stride_block,
    value_stride_beam,
    value_stride_head,
    value_stride_step,
    value_stride_dim,
    parent_stride_batch,
    parent_stride_beam,
    block_table_stride,
    beam_size: tl.constexpr,
    max_decode_steps: tl.constexpr,
    head_dim,
    BLOCK_BEAM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather one dimension tile through registers before writing in place."""

    request = tl.program_id(0)
    head_step = tl.program_id(1)
    dim_block = tl.program_id(2)
    head = head_step // max_decode_steps
    step = head_step % max_decode_steps

    dst_beam = tl.arange(0, BLOCK_BEAM)
    dim = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    parent = tl.load(
        parents + request * parent_stride_batch + dst_beam * parent_stride_beam,
        mask=dst_beam < beam_size,
        other=0,
    )
    block = tl.load(block_table + request * block_table_stride)
    mask = (dst_beam[:, None] < beam_size) & (dim[None, :] < head_dim)

    key_src = (
        key_cache
        + block * key_stride_block
        + parent[:, None] * key_stride_beam
        + head * key_stride_head
        + step * key_stride_step
        + dim[None, :] * key_stride_dim
    )
    value_src = (
        value_cache
        + block * value_stride_block
        + parent[:, None] * value_stride_beam
        + head * value_stride_head
        + step * value_stride_step
        + dim[None, :] * value_stride_dim
    )
    key = tl.load(key_src, mask=mask)
    value = tl.load(value_src, mask=mask)

    # One program owns every beam for this (request, head, step, dim tile).
    # Keep all source values in registers before any warp overwrites a beam.
    tl.debug_barrier()

    key_dst = (
        key_cache
        + block * key_stride_block
        + dst_beam[:, None] * key_stride_beam
        + head * key_stride_head
        + step * key_stride_step
        + dim[None, :] * key_stride_dim
    )
    value_dst = (
        value_cache
        + block * value_stride_block
        + dst_beam[:, None] * value_stride_beam
        + head * value_stride_head
        + step * value_stride_step
        + dim[None, :] * value_stride_dim
    )
    tl.store(key_dst, key, mask=mask)
    tl.store(value_dst, value, mask=mask)


@triton.jit
def _snapshot_kv_kernel(
    key_cache,
    value_cache,
    block_table,
    source_key,
    source_value,
    key_stride_block,
    key_stride_beam,
    key_stride_head,
    key_stride_step,
    key_stride_dim,
    value_stride_block,
    value_stride_beam,
    value_stride_head,
    value_stride_step,
    value_stride_dim,
    block_table_stride,
    source_key_stride_batch,
    source_key_stride_beam,
    source_key_stride_head,
    source_key_stride_step,
    source_key_stride_dim,
    source_value_stride_batch,
    source_value_stride_beam,
    source_value_stride_head,
    source_value_stride_step,
    source_value_stride_dim,
    max_decode_steps: tl.constexpr,
    head_dim,
    BLOCK_SIZE_D: tl.constexpr,
):
    request = tl.program_id(0)
    beam = tl.program_id(1)
    head_step = tl.program_id(2)
    head = head_step // max_decode_steps
    step = head_step % max_decode_steps
    block = tl.load(block_table + request * block_table_stride)
    dim = tl.arange(0, BLOCK_SIZE_D)
    mask = dim < head_dim

    key_cache_ptrs = (
        key_cache
        + block * key_stride_block
        + beam * key_stride_beam
        + head * key_stride_head
        + step * key_stride_step
        + dim * key_stride_dim
    )
    value_cache_ptrs = (
        value_cache
        + block * value_stride_block
        + beam * value_stride_beam
        + head * value_stride_head
        + step * value_stride_step
        + dim * value_stride_dim
    )
    source_key_ptrs = (
        source_key
        + request * source_key_stride_batch
        + beam * source_key_stride_beam
        + head * source_key_stride_head
        + step * source_key_stride_step
        + dim * source_key_stride_dim
    )
    source_value_ptrs = (
        source_value
        + request * source_value_stride_batch
        + beam * source_value_stride_beam
        + head * source_value_stride_head
        + step * source_value_stride_step
        + dim * source_value_stride_dim
    )

    key = tl.load(key_cache_ptrs, mask=mask)
    value = tl.load(value_cache_ptrs, mask=mask)
    tl.store(source_key_ptrs, key, mask=mask)
    tl.store(source_value_ptrs, value, mask=mask)


@triton.jit
def _select_kv_kernel(
    source_key,
    source_value,
    parents,
    block_table,
    key_cache,
    value_cache,
    source_key_stride_batch,
    source_key_stride_beam,
    source_key_stride_head,
    source_key_stride_step,
    source_key_stride_dim,
    source_value_stride_batch,
    source_value_stride_beam,
    source_value_stride_head,
    source_value_stride_step,
    source_value_stride_dim,
    parent_stride_batch,
    parent_stride_beam,
    block_table_stride,
    key_stride_block,
    key_stride_beam,
    key_stride_head,
    key_stride_step,
    key_stride_dim,
    value_stride_block,
    value_stride_beam,
    value_stride_head,
    value_stride_step,
    value_stride_dim,
    max_decode_steps: tl.constexpr,
    head_dim,
    BLOCK_SIZE_D: tl.constexpr,
):
    request = tl.program_id(0)
    new_beam = tl.program_id(1)
    head_step = tl.program_id(2)
    head = head_step // max_decode_steps
    step = head_step % max_decode_steps
    old_beam = tl.load(parents + request * parent_stride_batch + new_beam * parent_stride_beam)
    block = tl.load(block_table + request * block_table_stride)
    dim = tl.arange(0, BLOCK_SIZE_D)
    mask = dim < head_dim

    source_key_ptrs = (
        source_key
        + request * source_key_stride_batch
        + old_beam * source_key_stride_beam
        + head * source_key_stride_head
        + step * source_key_stride_step
        + dim * source_key_stride_dim
    )
    source_value_ptrs = (
        source_value
        + request * source_value_stride_batch
        + old_beam * source_value_stride_beam
        + head * source_value_stride_head
        + step * source_value_stride_step
        + dim * source_value_stride_dim
    )
    key_cache_ptrs = (
        key_cache
        + block * key_stride_block
        + new_beam * key_stride_beam
        + head * key_stride_head
        + step * key_stride_step
        + dim * key_stride_dim
    )
    value_cache_ptrs = (
        value_cache
        + block * value_stride_block
        + new_beam * value_stride_beam
        + head * value_stride_head
        + step * value_stride_step
        + dim * value_stride_dim
    )

    key = tl.load(source_key_ptrs, mask=mask)
    value = tl.load(source_value_ptrs, mask=mask)
    tl.store(key_cache_ptrs, key, mask=mask)
    tl.store(value_cache_ptrs, value, mask=mask)


def select_unshared_kv_triton(
    beam_index: torch.Tensor,
    x_key_block: list[torch.Tensor],
    x_value_block: list[torch.Tensor],
    block_table: torch.Tensor,
    group_token_num: torch.Tensor,
    decode_step: int,
    beam_size: int,
    layer_num: int,
) -> None:
    """Select K/V histories with the same ABI as ``select_unshared_kv``."""

    batch_size, num_heads, max_decode_steps, head_dim = _validate_select_unshared_kv(
        beam_index,
        x_key_block,
        x_value_block,
        block_table,
        group_token_num,
        decode_step,
        beam_size,
        layer_num,
    )
    if x_key_block[0].device.type != "cuda":
        raise ValueError("Triton operators require CUDA tensors")
    if batch_size == 0:
        return

    parents = torch.empty(
        (batch_size, beam_size),
        dtype=torch.int32,
        device=group_token_num.device,
    )
    _parents_from_prefix_kernel[(batch_size, beam_size)](
        group_token_num,
        parents,
        group_token_num.stride(0),
        parents.stride(0),
        parents.stride(1),
        beam_size=beam_size,
        SEARCH_STEPS=beam_size.bit_length(),
    )

    block_size_d = triton.next_power_of_2(head_dim)
    if beam_size <= _MAX_FUSED_SELECT_BEAM_SIZE:
        fused_block_d = min(block_size_d, 32)
        fused_grid = (
            batch_size,
            num_heads * max_decode_steps,
            triton.cdiv(head_dim, fused_block_d),
        )
        fused_num_warps = 8 if beam_size >= 64 else 4
        block_beam = triton.next_power_of_2(beam_size)
        for key, value in zip(x_key_block, x_value_block):
            _select_kv_fused_kernel[fused_grid](
                key,
                value,
                parents,
                block_table,
                key.stride(0),
                key.stride(1),
                key.stride(2),
                key.stride(3),
                key.stride(4),
                value.stride(0),
                value.stride(1),
                value.stride(2),
                value.stride(3),
                value.stride(4),
                parents.stride(0),
                parents.stride(1),
                block_table.stride(0),
                beam_size=beam_size,
                max_decode_steps=max_decode_steps,
                head_dim=head_dim,
                BLOCK_BEAM=block_beam,
                BLOCK_D=fused_block_d,
                num_warps=fused_num_warps,
            )
        return

    grid = (batch_size, beam_size, num_heads * max_decode_steps)
    snapshot_shape = (
        2,
        batch_size,
        beam_size,
        num_heads,
        max_decode_steps,
        head_dim,
    )
    for key, value in zip(x_key_block, x_value_block):
        snapshot = torch.empty(snapshot_shape, dtype=key.dtype, device=key.device)
        source_key = snapshot[0]
        source_value = snapshot[1]
        _snapshot_kv_kernel[grid](
            key,
            value,
            block_table,
            source_key,
            source_value,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            key.stride(4),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            value.stride(4),
            block_table.stride(0),
            source_key.stride(0),
            source_key.stride(1),
            source_key.stride(2),
            source_key.stride(3),
            source_key.stride(4),
            source_value.stride(0),
            source_value.stride(1),
            source_value.stride(2),
            source_value.stride(3),
            source_value.stride(4),
            max_decode_steps=max_decode_steps,
            head_dim=head_dim,
            BLOCK_SIZE_D=block_size_d,
        )
        _select_kv_kernel[grid](
            source_key,
            source_value,
            parents,
            block_table,
            key,
            value,
            source_key.stride(0),
            source_key.stride(1),
            source_key.stride(2),
            source_key.stride(3),
            source_key.stride(4),
            source_value.stride(0),
            source_value.stride(1),
            source_value.stride(2),
            source_value.stride(3),
            source_value.stride(4),
            parents.stride(0),
            parents.stride(1),
            block_table.stride(0),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            key.stride(4),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            value.stride(4),
            max_decode_steps=max_decode_steps,
            head_dim=head_dim,
            BLOCK_SIZE_D=block_size_d,
        )


@triton.jit
def _final_select_output_kernel(
    flat_indices,
    scores,
    top_tokens,
    sequence,
    out_token_ids,
    out_token_index,
    out_log_probs,
    out_sequence,
    flat_stride_batch,
    flat_stride_result,
    scores_stride_batch,
    scores_stride_candidate,
    top_tokens_stride_row,
    top_tokens_stride_candidate,
    sequence_stride_batch,
    sequence_stride_beam,
    sequence_stride_step,
    out_token_stride_batch,
    out_token_stride_result,
    out_index_stride_batch,
    out_index_stride_result,
    out_score_stride_batch,
    out_score_stride_result,
    out_sequence_stride_batch,
    out_sequence_stride_result,
    out_sequence_stride_step,
    beam_width: tl.constexpr,
    candidate_width: tl.constexpr,
    current_step: tl.constexpr,
    total_rounds: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    request = tl.program_id(0)
    result = tl.program_id(1)
    flat = tl.load(flat_indices + request * flat_stride_batch + result * flat_stride_result)
    parent = flat // candidate_width
    candidate = flat % candidate_width
    global_parent = request * beam_width + parent
    token = tl.load(
        top_tokens + global_parent * top_tokens_stride_row + candidate * top_tokens_stride_candidate
    )
    score = tl.load(scores + request * scores_stride_batch + flat * scores_stride_candidate)

    tl.store(
        out_token_ids + request * out_token_stride_batch + result * out_token_stride_result,
        token,
    )
    tl.store(
        out_token_index + request * out_index_stride_batch + result * out_index_stride_result,
        global_parent,
    )
    tl.store(
        out_log_probs + request * out_score_stride_batch + result * out_score_stride_result,
        score,
    )

    step = tl.arange(0, BLOCK_SIZE_T)
    step_mask = step < total_rounds
    history_mask = step < current_step
    history = tl.load(
        sequence
        + request * sequence_stride_batch
        + parent * sequence_stride_beam
        + step * sequence_stride_step,
        mask=history_mask,
        other=0,
    )
    output = tl.where(step == current_step, token, history)
    tl.store(
        out_sequence
        + request * out_sequence_stride_batch
        + result * out_sequence_stride_result
        + step * out_sequence_stride_step,
        output,
        mask=step_mask,
    )


@triton.jit
def _final_select_keys_output_kernel(
    selection_keys,
    log_probs,
    top_probs,
    top_tokens,
    sequence,
    out_token_ids,
    out_token_index,
    out_log_probs,
    out_sequence,
    keys_stride_batch,
    keys_stride_result,
    log_probs_stride_row,
    top_probs_stride_row,
    top_probs_stride_candidate,
    top_tokens_stride_row,
    top_tokens_stride_candidate,
    sequence_stride_batch,
    sequence_stride_beam,
    sequence_stride_step,
    out_token_stride_batch,
    out_token_stride_result,
    out_index_stride_batch,
    out_index_stride_result,
    out_score_stride_batch,
    out_score_stride_result,
    out_sequence_stride_batch,
    out_sequence_stride_result,
    out_sequence_stride_step,
    beam_width: tl.constexpr,
    candidate_width: tl.constexpr,
    candidate_count: tl.constexpr,
    index_mask: tl.constexpr,
    current_step: tl.constexpr,
    total_rounds: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    request = tl.program_id(0)
    result = tl.program_id(1)
    key = tl.load(selection_keys + request * keys_stride_batch + result * keys_stride_result)
    flat = candidate_count - 1 - (key & index_mask)
    parent = flat // candidate_width
    candidate = flat % candidate_width
    global_parent = request * beam_width + parent
    token = tl.load(
        top_tokens + global_parent * top_tokens_stride_row + candidate * top_tokens_stride_candidate
    )
    score = tl.load(log_probs + global_parent * log_probs_stride_row) + tl.load(
        top_probs + global_parent * top_probs_stride_row + candidate * top_probs_stride_candidate
    )

    tl.store(
        out_token_ids + request * out_token_stride_batch + result * out_token_stride_result,
        token,
    )
    tl.store(
        out_token_index + request * out_index_stride_batch + result * out_index_stride_result,
        global_parent,
    )
    tl.store(
        out_log_probs + request * out_score_stride_batch + result * out_score_stride_result,
        score,
    )

    step = tl.arange(0, BLOCK_SIZE_T)
    step_mask = step < total_rounds
    history = tl.load(
        sequence
        + request * sequence_stride_batch
        + parent * sequence_stride_beam
        + step * sequence_stride_step,
        mask=step < current_step,
        other=0,
    )
    output = tl.where(step == current_step, token, history)
    tl.store(
        out_sequence
        + request * out_sequence_stride_batch
        + result * out_sequence_stride_result
        + step * out_sequence_stride_step,
        output,
        mask=step_mask,
    )


def beam_search_rec_final_select_triton(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable final selection with hierarchical top-k and fused output gather."""

    batch_size, beam_width, candidate_width, total_rounds = _validate_final_select(
        log_probs,
        top_tokens,
        top_probs,
        sequence,
        current_step,
        result_width,
    )
    if log_probs.device.type != "cuda":
        raise ValueError("Triton operators require CUDA tensors")

    out_token_ids = torch.empty(
        (batch_size, result_width), dtype=torch.int32, device=sequence.device
    )
    out_token_index = torch.empty_like(out_token_ids)
    out_log_probs = torch.empty(
        (batch_size, result_width), dtype=torch.float32, device=sequence.device
    )
    out_sequence = torch.empty(
        (batch_size, result_width, total_rounds),
        dtype=torch.int32,
        device=sequence.device,
    )
    block_size_t = triton.next_power_of_2(total_rounds)
    candidate_count = beam_width * candidate_width
    if result_width == _FUSED_SCORE_TOPK_WIDTH and result_width <= _SCORE_TOPK_CHUNK_SIZE:
        selection_keys, index_bits = _stable_score_topk_keys_triton(
            log_probs,
            top_probs,
            batch_size,
            beam_width,
            candidate_width,
            result_width,
        )
        _final_select_keys_output_kernel[(batch_size, result_width)](
            selection_keys,
            log_probs,
            top_probs,
            top_tokens,
            sequence,
            out_token_ids,
            out_token_index,
            out_log_probs,
            out_sequence,
            selection_keys.stride(0),
            selection_keys.stride(1),
            log_probs.stride(0),
            top_probs.stride(0),
            top_probs.stride(1),
            top_tokens.stride(0),
            top_tokens.stride(1),
            sequence.stride(0),
            sequence.stride(1),
            sequence.stride(2),
            out_token_ids.stride(0),
            out_token_ids.stride(1),
            out_token_index.stride(0),
            out_token_index.stride(1),
            out_log_probs.stride(0),
            out_log_probs.stride(1),
            out_sequence.stride(0),
            out_sequence.stride(1),
            out_sequence.stride(2),
            beam_width=beam_width,
            candidate_width=candidate_width,
            candidate_count=candidate_count,
            index_mask=(1 << index_bits) - 1,
            current_step=current_step,
            total_rounds=total_rounds,
            BLOCK_SIZE_T=block_size_t,
        )
        return (
            out_token_ids.reshape(-1, 1),
            out_token_index.reshape(-1, 1),
            out_log_probs.reshape(-1, 1),
            out_sequence,
        )

    scores = (
        log_probs.reshape(batch_size, beam_width, 1)
        + top_probs.reshape(batch_size, beam_width, candidate_width)
    ).reshape(batch_size, -1)
    flat_indices = _stable_topk_indices(scores, result_width)
    _final_select_output_kernel[(batch_size, result_width)](
        flat_indices,
        scores,
        top_tokens,
        sequence,
        out_token_ids,
        out_token_index,
        out_log_probs,
        out_sequence,
        flat_indices.stride(0),
        flat_indices.stride(1),
        scores.stride(0),
        scores.stride(1),
        top_tokens.stride(0),
        top_tokens.stride(1),
        sequence.stride(0),
        sequence.stride(1),
        sequence.stride(2),
        out_token_ids.stride(0),
        out_token_ids.stride(1),
        out_token_index.stride(0),
        out_token_index.stride(1),
        out_log_probs.stride(0),
        out_log_probs.stride(1),
        out_sequence.stride(0),
        out_sequence.stride(1),
        out_sequence.stride(2),
        beam_width=beam_width,
        candidate_width=candidate_width,
        current_step=current_step,
        total_rounds=total_rounds,
        BLOCK_SIZE_T=block_size_t,
    )
    return (
        out_token_ids.reshape(-1, 1),
        out_token_index.reshape(-1, 1),
        out_log_probs.reshape(-1, 1),
        out_sequence,
    )


@triton.jit
def _group_step0_kernel(
    top_tokens,
    out_token_ids,
    out_token_index,
    out_log_probs,
    out_prefix,
    out_sequence,
    top_tokens_stride_row,
    top_tokens_stride_candidate,
    out_token_stride_batch,
    out_token_stride_result,
    out_index_stride_batch,
    out_index_stride_result,
    out_score_stride_batch,
    out_score_stride_result,
    out_prefix_stride_batch,
    out_prefix_stride_beam,
    out_sequence_stride_batch,
    out_sequence_stride_result,
    out_sequence_stride_step,
    beam_width: tl.constexpr,
    top_k: tl.constexpr,
):
    request = tl.program_id(0)
    result = tl.program_id(1)
    copy_token = result < beam_width
    token = tl.load(
        top_tokens + (request * beam_width + result) * top_tokens_stride_row,
        mask=copy_token,
        other=0,
    )
    tl.store(
        out_token_ids + request * out_token_stride_batch + result * out_token_stride_result,
        0,
    )
    tl.store(
        out_token_index + request * out_index_stride_batch + result * out_index_stride_result,
        0,
    )
    tl.store(
        out_log_probs + request * out_score_stride_batch + result * out_score_stride_result,
        0.0,
    )
    tl.store(
        out_sequence + request * out_sequence_stride_batch + result * out_sequence_stride_result,
        token,
    )
    tl.store(
        out_prefix + request * out_prefix_stride_batch + result * out_prefix_stride_beam,
        0,
        mask=result < beam_width,
    )


@triton.jit
def _group_prefix_kernel(
    flat_indices,
    out_prefix,
    flat_stride_batch,
    flat_stride_result,
    out_prefix_stride_batch,
    out_prefix_stride_beam,
    beam_width: tl.constexpr,
    effective_k: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    request = tl.program_id(0)
    parent_limit = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_SIZE_K)
    mask = offsets < effective_k
    flat = tl.load(
        flat_indices + request * flat_stride_batch + offsets * flat_stride_result,
        mask=mask,
        other=0,
    )
    parent = flat // beam_width
    cumulative_count = tl.sum(tl.where(mask & (parent <= parent_limit), 1, 0))
    tl.store(
        out_prefix + request * out_prefix_stride_batch + parent_limit * out_prefix_stride_beam,
        cumulative_count + request * beam_width,
    )


@triton.jit
def _group_keys_parent_sort_kernel(
    selection_keys,
    grouped_keys,
    selection_stride_batch,
    selection_stride_result,
    grouped_stride_batch,
    grouped_stride_result,
    beam_width: tl.constexpr,
    candidate_count: tl.constexpr,
    index_mask: tl.constexpr,
    effective_k: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Group selected keys by parent, preserving score order within a parent."""

    request = tl.program_id(0)
    rank = tl.arange(0, BLOCK_SIZE_K)
    valid = rank < effective_k
    key = tl.load(
        selection_keys + request * selection_stride_batch + rank * selection_stride_result,
        mask=valid,
        other=0,
    )
    flat = candidate_count - 1 - (key & index_mask)
    parent = flat // beam_width
    grouping_key = tl.where(valid, parent * BLOCK_SIZE_K + rank, (beam_width + 1) * BLOCK_SIZE_K)
    grouped_rank = tl.sort(grouping_key) % BLOCK_SIZE_K
    grouped = tl.load(
        selection_keys + request * selection_stride_batch + grouped_rank * selection_stride_result,
        mask=valid,
        other=0,
    )
    tl.store(
        grouped_keys + request * grouped_stride_batch + rank * grouped_stride_result,
        grouped,
        mask=valid,
    )


@triton.jit
def _group_keys_prefix_kernel(
    selection_keys,
    out_prefix,
    selection_stride_batch,
    selection_stride_result,
    out_prefix_stride_batch,
    out_prefix_stride_beam,
    beam_width: tl.constexpr,
    candidate_count: tl.constexpr,
    index_mask: tl.constexpr,
    effective_k: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    request = tl.program_id(0)
    parent_limit = tl.program_id(1)
    rank = tl.arange(0, BLOCK_SIZE_K)
    valid = rank < effective_k
    key = tl.load(
        selection_keys + request * selection_stride_batch + rank * selection_stride_result,
        mask=valid,
        other=0,
    )
    flat = candidate_count - 1 - (key & index_mask)
    parent = flat // beam_width
    cumulative_count = tl.sum(tl.where(valid & (parent <= parent_limit), 1, 0))
    tl.store(
        out_prefix + request * out_prefix_stride_batch + parent_limit * out_prefix_stride_beam,
        cumulative_count + request * beam_width,
    )


@triton.jit
def _group_keys_output_kernel(
    grouped_keys,
    log_probs,
    top_probs,
    top_tokens,
    sequence,
    out_token_ids,
    out_token_index,
    out_log_probs,
    out_sequence,
    grouped_stride_batch,
    grouped_stride_result,
    log_probs_stride_row,
    top_probs_stride_row,
    top_probs_stride_candidate,
    top_tokens_stride_row,
    top_tokens_stride_candidate,
    sequence_stride_batch,
    sequence_stride_beam,
    sequence_stride_step,
    out_token_stride_batch,
    out_token_stride_result,
    out_index_stride_batch,
    out_index_stride_result,
    out_score_stride_batch,
    out_score_stride_result,
    out_sequence_stride_batch,
    out_sequence_stride_result,
    out_sequence_stride_step,
    beam_width: tl.constexpr,
    candidate_count: tl.constexpr,
    index_mask: tl.constexpr,
    current_step: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    request = tl.program_id(0)
    result = tl.program_id(1)
    key = tl.load(grouped_keys + request * grouped_stride_batch + result * grouped_stride_result)
    flat = candidate_count - 1 - (key & index_mask)
    parent = flat // beam_width
    candidate = flat % beam_width
    global_parent = request * beam_width + parent
    token = tl.load(
        top_tokens + global_parent * top_tokens_stride_row + candidate * top_tokens_stride_candidate
    )
    score = tl.load(log_probs + global_parent * log_probs_stride_row) + tl.load(
        top_probs + global_parent * top_probs_stride_row + candidate * top_probs_stride_candidate
    )

    tl.store(
        out_token_ids + request * out_token_stride_batch + result * out_token_stride_result,
        token,
    )
    tl.store(
        out_token_index + request * out_index_stride_batch + result * out_index_stride_result,
        global_parent,
    )
    tl.store(
        out_log_probs + request * out_score_stride_batch + result * out_score_stride_result,
        score,
    )

    step = tl.arange(0, BLOCK_SIZE_T)
    history = tl.load(
        sequence
        + request * sequence_stride_batch
        + parent * sequence_stride_beam
        + step * sequence_stride_step,
        mask=step < current_step,
        other=0,
    )
    output = tl.where(step == current_step, token, history)
    tl.store(
        out_sequence
        + request * out_sequence_stride_batch
        + result * out_sequence_stride_result
        + step * out_sequence_stride_step,
        output,
        mask=step <= current_step,
    )


@triton.jit
def _group_output_kernel(
    flat_indices,
    group_order,
    scores,
    top_tokens,
    sequence,
    out_token_ids,
    out_token_index,
    out_log_probs,
    out_sequence,
    flat_stride_batch,
    flat_stride_result,
    group_order_stride_batch,
    group_order_stride_result,
    scores_stride_batch,
    scores_stride_candidate,
    top_tokens_stride_row,
    top_tokens_stride_candidate,
    sequence_stride_batch,
    sequence_stride_beam,
    sequence_stride_step,
    out_token_stride_batch,
    out_token_stride_result,
    out_index_stride_batch,
    out_index_stride_result,
    out_score_stride_batch,
    out_score_stride_result,
    out_sequence_stride_batch,
    out_sequence_stride_result,
    out_sequence_stride_step,
    beam_width: tl.constexpr,
    effective_k: tl.constexpr,
    current_step: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    request = tl.program_id(0)
    result = tl.program_id(1)
    valid = result < effective_k
    selected_rank = tl.load(
        group_order + request * group_order_stride_batch + result * group_order_stride_result,
        mask=valid,
        other=0,
    )
    flat = tl.load(
        flat_indices + request * flat_stride_batch + selected_rank * flat_stride_result,
        mask=valid,
        other=0,
    )
    parent = flat // beam_width
    candidate = flat % beam_width
    global_parent = request * beam_width + parent
    token = tl.load(
        top_tokens
        + global_parent * top_tokens_stride_row
        + candidate * top_tokens_stride_candidate,
        mask=valid,
        other=0,
    )
    score = tl.load(
        scores + request * scores_stride_batch + flat * scores_stride_candidate,
        mask=valid,
        other=0.0,
    )

    tl.store(
        out_token_ids + request * out_token_stride_batch + result * out_token_stride_result,
        token,
    )
    tl.store(
        out_token_index + request * out_index_stride_batch + result * out_index_stride_result,
        tl.where(valid, global_parent, 0),
    )
    tl.store(
        out_log_probs + request * out_score_stride_batch + result * out_score_stride_result,
        score,
    )

    step = tl.arange(0, BLOCK_SIZE_T)
    step_mask = step <= current_step
    history_mask = valid & (step < current_step)
    history = tl.load(
        sequence
        + request * sequence_stride_batch
        + parent * sequence_stride_beam
        + step * sequence_stride_step,
        mask=history_mask,
        other=0,
    )
    output = tl.where(valid & (step == current_step), token, history)
    tl.store(
        out_sequence
        + request * out_sequence_stride_batch
        + result * out_sequence_stride_result
        + step * out_sequence_stride_step,
        output,
        mask=step_mask,
    )


def beam_search_group_triton(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable beam grouping with hierarchical top-k and fused grouping kernels."""

    batch_size, beam_width, effective_k = _validate_group(
        log_probs,
        top_tokens,
        top_probs,
        sequence,
        current_step,
        top_k,
    )
    if log_probs.device.type != "cuda":
        raise ValueError("Triton operators require CUDA tensors")
    if effective_k > _MAX_TRITON_GROUP_K:
        return _beam_search_group_torch(
            log_probs,
            top_tokens,
            top_probs,
            sequence,
            current_step,
            top_k,
        )

    out_token_ids = torch.empty((batch_size, top_k), dtype=torch.int32, device=sequence.device)
    out_token_index = torch.empty_like(out_token_ids)
    out_log_probs = torch.empty((batch_size, top_k), dtype=torch.float32, device=sequence.device)
    out_prefix = (
        torch.zeros((batch_size, beam_width), dtype=torch.int32, device=sequence.device)
        if current_step == 0
        else torch.empty((batch_size, beam_width), dtype=torch.int32, device=sequence.device)
    )
    out_sequence = torch.empty(
        (batch_size, top_k, current_step + 1),
        dtype=torch.int32,
        device=sequence.device,
    )

    if current_step == 0:
        _group_step0_kernel[(batch_size, top_k)](
            top_tokens,
            out_token_ids,
            out_token_index,
            out_log_probs,
            out_prefix,
            out_sequence,
            top_tokens.stride(0),
            top_tokens.stride(1),
            out_token_ids.stride(0),
            out_token_ids.stride(1),
            out_token_index.stride(0),
            out_token_index.stride(1),
            out_log_probs.stride(0),
            out_log_probs.stride(1),
            out_prefix.stride(0),
            out_prefix.stride(1),
            out_sequence.stride(0),
            out_sequence.stride(1),
            out_sequence.stride(2),
            beam_width=beam_width,
            top_k=top_k,
        )
        return (
            out_token_ids,
            out_token_index,
            out_log_probs,
            out_prefix,
            out_sequence,
        )

    if (
        top_k == effective_k
        and effective_k == _FUSED_SCORE_TOPK_WIDTH
        and effective_k <= _SCORE_TOPK_CHUNK_SIZE
    ):
        selection_keys, index_bits = _stable_score_topk_keys_triton(
            log_probs,
            top_probs,
            batch_size,
            beam_width,
            beam_width,
            effective_k,
        )
        grouped_keys = torch.empty_like(selection_keys)
        block_size_k = triton.next_power_of_2(effective_k)
        candidate_count = beam_width * beam_width
        index_mask = (1 << index_bits) - 1
        _group_keys_parent_sort_kernel[(batch_size,)](
            selection_keys,
            grouped_keys,
            selection_keys.stride(0),
            selection_keys.stride(1),
            grouped_keys.stride(0),
            grouped_keys.stride(1),
            beam_width=beam_width,
            candidate_count=candidate_count,
            index_mask=index_mask,
            effective_k=effective_k,
            BLOCK_SIZE_K=block_size_k,
            num_warps=4,
        )
        _group_keys_prefix_kernel[(batch_size, beam_width)](
            selection_keys,
            out_prefix,
            selection_keys.stride(0),
            selection_keys.stride(1),
            out_prefix.stride(0),
            out_prefix.stride(1),
            beam_width=beam_width,
            candidate_count=candidate_count,
            index_mask=index_mask,
            effective_k=effective_k,
            BLOCK_SIZE_K=block_size_k,
        )
        block_size_t = triton.next_power_of_2(current_step + 1)
        _group_keys_output_kernel[(batch_size, top_k)](
            grouped_keys,
            log_probs,
            top_probs,
            top_tokens,
            sequence,
            out_token_ids,
            out_token_index,
            out_log_probs,
            out_sequence,
            grouped_keys.stride(0),
            grouped_keys.stride(1),
            log_probs.stride(0),
            top_probs.stride(0),
            top_probs.stride(1),
            top_tokens.stride(0),
            top_tokens.stride(1),
            sequence.stride(0),
            sequence.stride(1),
            sequence.stride(2),
            out_token_ids.stride(0),
            out_token_ids.stride(1),
            out_token_index.stride(0),
            out_token_index.stride(1),
            out_log_probs.stride(0),
            out_log_probs.stride(1),
            out_sequence.stride(0),
            out_sequence.stride(1),
            out_sequence.stride(2),
            beam_width=beam_width,
            candidate_count=candidate_count,
            index_mask=index_mask,
            current_step=current_step,
            BLOCK_SIZE_T=block_size_t,
        )
        return (
            out_token_ids,
            out_token_index,
            out_log_probs,
            out_prefix,
            out_sequence,
        )

    scores = (
        log_probs.reshape(batch_size, beam_width, 1)
        + top_probs.reshape(batch_size, beam_width, beam_width)
    ).reshape(batch_size, -1)
    flat_indices = _stable_topk_indices(scores, effective_k)
    parent = torch.div(flat_indices, beam_width, rounding_mode="floor")
    group_order = torch.argsort(parent, dim=1, stable=True)
    block_size_k = triton.next_power_of_2(effective_k)
    _group_prefix_kernel[(batch_size, beam_width)](
        flat_indices,
        out_prefix,
        flat_indices.stride(0),
        flat_indices.stride(1),
        out_prefix.stride(0),
        out_prefix.stride(1),
        beam_width=beam_width,
        effective_k=effective_k,
        BLOCK_SIZE_K=block_size_k,
    )
    block_size_t = triton.next_power_of_2(current_step + 1)
    _group_output_kernel[(batch_size, top_k)](
        flat_indices,
        group_order,
        scores,
        top_tokens,
        sequence,
        out_token_ids,
        out_token_index,
        out_log_probs,
        out_sequence,
        flat_indices.stride(0),
        flat_indices.stride(1),
        group_order.stride(0),
        group_order.stride(1),
        scores.stride(0),
        scores.stride(1),
        top_tokens.stride(0),
        top_tokens.stride(1),
        sequence.stride(0),
        sequence.stride(1),
        sequence.stride(2),
        out_token_ids.stride(0),
        out_token_ids.stride(1),
        out_token_index.stride(0),
        out_token_index.stride(1),
        out_log_probs.stride(0),
        out_log_probs.stride(1),
        out_sequence.stride(0),
        out_sequence.stride(1),
        out_sequence.stride(2),
        beam_width=beam_width,
        effective_k=effective_k,
        current_step=current_step,
        BLOCK_SIZE_T=block_size_t,
    )
    return (
        out_token_ids,
        out_token_index,
        out_log_probs,
        out_prefix,
        out_sequence,
    )


__all__ = [
    "beam_search_group_triton",
    "beam_search_rec_final_select_triton",
    "select_unshared_kv_triton",
]
