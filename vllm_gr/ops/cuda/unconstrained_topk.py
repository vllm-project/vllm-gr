# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Full-vocabulary Beam Top-K implemented with Triton kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from vllm_gr.ops.cuda.topk_sort_key import pack_score_index, unpack_score

# Triton JIT functions intentionally use its untyped DSL values.
# mypy: disable-error-code="misc,no-untyped-def"

LOCAL_TILE_SIZE = 1024


@dataclass(frozen=True, slots=True)
class UnconstrainedTopKScratch:
    """Persistent storage for full-vocabulary Top-K and logsumexp reduction."""

    local_scores: torch.Tensor
    local_indices: torch.Tensor
    partial_max: torch.Tensor
    partial_sum: torch.Tensor
    merge_scores: torch.Tensor
    merge_indices: torch.Tensor
    vocab_size: int
    top_k: int


@triton.jit
def _local_topk_kernel(
    logits,
    temperatures,
    local_scores,
    local_indices,
    partial_max,
    partial_sum,
    vocab_size,
    logits_stride,
    temperature_count,
    scratch_stride,
    partial_stride,
    select_width,
    BLOCK: tl.constexpr,
    SELECT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    flat = tile * BLOCK + offsets
    valid = flat < vocab_size
    values = tl.load(logits + row * logits_stride + flat, mask=valid, other=-float("inf"))
    values = values.to(tl.float32)
    temperature_index = tl.where(temperature_count == 1, 0, row)
    temperature = tl.load(temperatures + temperature_index)
    temperature = tl.where(temperature == 0.0, 1.0, temperature)
    values = values / temperature
    # Non-finite logits cannot produce valid Beam candidates or probability mass.
    values = tl.where((values > -float("inf")) & (values < float("inf")), values, -float("inf"))
    # Keep the normalization statistics beside the local candidates so the
    # final stage never has to rescan the full-vocabulary logits.
    tile_max = tl.max(values, axis=0)
    # An entirely masked or -inf tile contributes zero probability mass.
    tile_sum = tl.sum(tl.exp(values - tl.where(tile_max == -float("inf"), 0.0, tile_max)), axis=0)
    tl.store(partial_max + row * partial_stride + tile, tile_max)
    tl.store(partial_sum + row * partial_stride + tile, tile_sum)
    packed = pack_score_index(values, flat, valid)
    # topk selects only the candidates we retain.  Complementing the packed
    # key reverses its signed ordering without changing score or index ties.
    if SELECT_BLOCK == 1:
        # The smallest packed key is the highest score, then lowest token ID.
        sorted_pairs = tl.min(packed, axis=0)
    else:
        sorted_pairs = ~tl.topk(~packed, SELECT_BLOCK)
    selected_offsets = tl.arange(0, SELECT_BLOCK)
    output = row * scratch_stride + tile * select_width + selected_offsets
    tl.store(
        local_scores + output,
        unpack_score(sorted_pairs),
        mask=selected_offsets < select_width,
    )
    tl.store(
        local_indices + output,
        (sorted_pairs & 0x00000000FFFFFFFF).to(tl.int32),
        mask=selected_offsets < select_width,
    )


@triton.jit
def _merge_topk_kernel(
    input_scores,
    input_indices,
    output_scores,
    output_indices,
    list_count,
    select_width,
    scratch_stride,
    BLOCK: tl.constexpr,
    SELECT_BLOCK: tl.constexpr,
    FAN_IN: tl.constexpr,
):
    row = tl.program_id(0)
    output_list = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    source_list = output_list * FAN_IN + offsets // select_width
    source_rank = offsets % select_width
    valid = (offsets < FAN_IN * select_width) & (source_list < list_count)
    source = row * scratch_stride + source_list * select_width + source_rank
    scores = tl.load(input_scores + source, mask=valid, other=-float("inf"))
    indices = tl.load(input_indices + source, mask=valid, other=-1)
    valid = valid & (indices >= 0)
    packed = pack_score_index(scores, indices, valid)
    if SELECT_BLOCK == 1:
        sorted_pairs = tl.min(packed, axis=0)
    else:
        sorted_pairs = ~tl.topk(~packed, SELECT_BLOCK)
    selected_offsets = tl.arange(0, SELECT_BLOCK)
    output = row * scratch_stride + output_list * select_width + selected_offsets
    tl.store(
        output_scores + output,
        unpack_score(sorted_pairs),
        mask=selected_offsets < select_width,
    )
    tl.store(
        output_indices + output,
        (sorted_pairs & 0x00000000FFFFFFFF).to(tl.int32),
        mask=selected_offsets < select_width,
    )


@triton.jit
def _finalize_topk_kernel(
    selected_scores,
    selected_indices,
    partial_max,
    partial_sum,
    output_ids,
    output_logprobs,
    output_valid,
    output_counts,
    partial_count,
    selected_stride,
    partial_stride,
    output_stride,
    top_k,
    SCAN_BLOCK: tl.constexpr,
    OUTPUT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile_offsets = tl.arange(0, SCAN_BLOCK)
    tile_valid = tile_offsets < partial_count
    tile_maxes = tl.load(
        partial_max + row * partial_stride + tile_offsets,
        mask=tile_valid,
        other=-float("inf"),
    )
    tile_sums = tl.load(
        partial_sum + row * partial_stride + tile_offsets,
        mask=tile_valid,
        other=0.0,
    )
    global_max = tl.max(tile_maxes, axis=0)
    has_mass = global_max != -float("inf")
    safe_max = tl.where(has_mass, global_max, 0.0)
    denominator = safe_max + tl.log(tl.sum(tile_sums * tl.exp(tile_maxes - safe_max), axis=0))

    ranks = tl.arange(0, OUTPUT_BLOCK)
    selected = ranks < top_k
    selected_offset = row * selected_stride + ranks
    tokens = tl.load(selected_indices + selected_offset, mask=selected, other=-1)
    scores = tl.load(selected_scores + selected_offset, mask=selected, other=-float("inf"))
    valid = selected & (tokens >= 0) & has_mass & (scores > -float("inf")) & (scores < float("inf"))
    output_offset = row * output_stride + ranks
    tl.store(output_ids + output_offset, tokens, mask=selected)
    tl.store(
        output_logprobs + output_offset,
        tl.where(valid, scores - denominator, -float("inf")),
        mask=selected,
    )
    tl.store(output_valid + output_offset, valid, mask=selected)
    tl.store(output_counts + row, tl.sum(valid.to(tl.int32), axis=0))


def allocate_unconstrained_topk_scratch(
    batch_size: int,
    vocab_size: int,
    top_k: int,
    device: torch.device,
) -> UnconstrainedTopKScratch:
    if min(batch_size, vocab_size, top_k) <= 0:
        raise ValueError("batch_size, vocab_size, and top_k must be positive")
    if top_k > LOCAL_TILE_SIZE:
        raise ValueError(f"top_k must be <= {LOCAL_TILE_SIZE}")
    list_count = (vocab_size + LOCAL_TILE_SIZE - 1) // LOCAL_TILE_SIZE
    slots = list_count * top_k
    return UnconstrainedTopKScratch(
        local_scores=torch.empty((batch_size, slots), dtype=torch.float32, device=device),
        local_indices=torch.empty((batch_size, slots), dtype=torch.int32, device=device),
        partial_max=torch.empty((batch_size, list_count), dtype=torch.float32, device=device),
        partial_sum=torch.empty((batch_size, list_count), dtype=torch.float32, device=device),
        merge_scores=torch.empty((batch_size, slots), dtype=torch.float32, device=device),
        merge_indices=torch.empty((batch_size, slots), dtype=torch.int32, device=device),
        vocab_size=vocab_size,
        top_k=top_k,
    )


def unconstrained_topk_out(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int,
    scratch: UnconstrainedTopKScratch,
    out_token_ids: torch.Tensor,
    out_logprobs: torch.Tensor,
    out_valid_mask: torch.Tensor,
    out_valid_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write full-vocabulary Top-K/logprobs with one logits scan.

    The local Triton stage computes both candidate scores and per-tile
    logsumexp statistics after applying the per-row temperature.  Merge/finalize
    stages consume only scratch data.
    """

    if (
        logits.ndim != 2
        or not logits.is_cuda
        or logits.stride(1) != 1
        or logits.stride(0) < logits.shape[1]
    ):
        raise ValueError("logits must be a CUDA matrix with contiguous vocabulary rows")
    rows, vocab_size = (int(logits.shape[0]), int(logits.shape[1]))
    if (
        temperatures.device != logits.device
        or temperatures.dtype != torch.float32
        or temperatures.ndim != 1
        or not temperatures.is_contiguous()
        or temperatures.numel() not in (1, rows)
    ):
        raise ValueError(
            "temperatures must be contiguous CUDA float32 with one value or one per row"
        )
    if vocab_size != scratch.vocab_size or top_k != scratch.top_k:
        raise ValueError("unconstrained Top-K scratch shape does not match logits")
    if rows > scratch.local_scores.shape[0]:
        raise ValueError("unconstrained Top-K scratch has insufficient rows")
    expected = (rows, top_k)
    for name, tensor, dtype in (
        ("out_token_ids", out_token_ids, torch.int32),
        ("out_logprobs", out_logprobs, torch.float32),
        ("out_valid_mask", out_valid_mask, torch.bool),
    ):
        if (
            tensor.device != logits.device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != expected
        ):
            raise ValueError(f"{name} must be {dtype} CUDA tensor with shape {expected}")
    if (
        out_valid_counts.device != logits.device
        or out_valid_counts.dtype != torch.int32
        or tuple(out_valid_counts.shape) != (rows,)
    ):
        raise ValueError("out_valid_counts must be a CUDA int32 tensor")

    list_count = (vocab_size + LOCAL_TILE_SIZE - 1) // LOCAL_TILE_SIZE
    scratch_stride = int(scratch.local_scores.shape[1])
    select_block = triton.next_power_of_2(top_k)
    _local_topk_kernel[(rows, list_count)](
        logits,
        temperatures,
        scratch.local_scores,
        scratch.local_indices,
        scratch.partial_max,
        scratch.partial_sum,
        vocab_size,
        logits.stride(0),
        temperatures.numel(),
        scratch_stride,
        list_count,
        top_k,
        BLOCK=LOCAL_TILE_SIZE,
        SELECT_BLOCK=select_block,
        num_warps=8,
    )
    input_scores, input_indices = scratch.local_scores, scratch.local_indices
    output_scores, output_indices = scratch.merge_scores, scratch.merge_indices
    current_count = list_count
    # Four-way reduction halves the depth of the merge tree for the beam
    # widths used by OneRec while keeping each CTA's candidate set bounded.
    merge_fan_in = 4 if top_k <= 128 else 2
    merge_block = triton.next_power_of_2(merge_fan_in * top_k)
    while current_count > 1:
        next_count = (current_count + merge_fan_in - 1) // merge_fan_in
        _merge_topk_kernel[(rows, next_count)](
            input_scores,
            input_indices,
            output_scores,
            output_indices,
            current_count,
            top_k,
            scratch_stride,
            BLOCK=merge_block,
            SELECT_BLOCK=select_block,
            FAN_IN=merge_fan_in,
            num_warps=8,
        )
        input_scores, output_scores = output_scores, input_scores
        input_indices, output_indices = output_indices, input_indices
        current_count = next_count
    _finalize_topk_kernel[(rows,)](
        input_scores,
        input_indices,
        scratch.partial_max,
        scratch.partial_sum,
        out_token_ids,
        out_logprobs,
        out_valid_mask,
        out_valid_counts,
        list_count,
        scratch_stride,
        list_count,
        out_token_ids.stride(0),
        top_k,
        SCAN_BLOCK=triton.next_power_of_2(list_count),
        OUTPUT_BLOCK=triton.next_power_of_2(top_k),
        num_warps=8,
    )
    return out_token_ids, out_logprobs, out_valid_mask, out_valid_counts


__all__ = [
    "UnconstrainedTopKScratch",
    "allocate_unconstrained_topk_scratch",
    "unconstrained_topk_out",
]
