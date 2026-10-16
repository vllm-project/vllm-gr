# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic hierarchical CUDA Top-K for OneRec Beam decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from vllm_gr.config.models import MAX_BEAM_WIDTH

# Triton JIT functions intentionally use its untyped DSL values.
# mypy: disable-error-code="misc,no-untyped-def"


LOCAL_TILE_SIZE = 256
MAX_CANDIDATE_WIDTH = 512
MAX_REQUESTS = 48


@triton.jit
def _pack_score_flat(score, flat_index, valid):
    """Pack (score DESC, flat_index ASC) into ascending signed-i64 order."""

    min_i32: tl.constexpr = -2147483648
    score = tl.where(score == score, score, -float("inf"))
    score = tl.where(valid, score, -float("inf"))
    score = tl.where(score == 0.0, 0.0, score)
    score_bits = score.to(tl.int32, bitcast=True)
    sign = score_bits >> 31
    sort_key = tl.where(sign == 0, score_bits ^ -1, score_bits ^ min_i32)
    key64 = sort_key.to(tl.int64) & 0x00000000FFFFFFFF
    # Triton 3.6 type-checks both ``where`` branches before applying a
    # trailing cast.  Promote the live index first so the unsigned-32 sentinel
    # is represented in i64 instead of overflowing the inferred i32 branch.
    tie_key = tl.where(valid, flat_index.to(tl.int64), 0x00000000FFFFFFFF)
    return (key64 << 32) | tie_key


@triton.jit
def _unpack_score(packed):
    min_i32: tl.constexpr = -2147483648
    sort_key = ((packed >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    sign = sort_key >> 31
    score_bits = tl.where(sign < 0, sort_key ^ -1, sort_key ^ min_i32)
    return score_bits.to(tl.float32, bitcast=True)


@triton.jit(
    do_not_specialize=[
        "beam_width",
        "candidate_width",
        "select_width",
        "scratch_stride",
    ]
)
def _local_beam_topk_kernel(
    cumulative_logprobs,
    top_logprobs,
    candidate_valid_mask,
    local_scores,
    local_flat_indices,
    beam_width,
    candidate_width,
    select_width,
    scratch_stride,
    HAS_VALID_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
    MASK_INVALID_INDICES: tl.constexpr = False,
):
    request = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    candidate_count = beam_width * candidate_width
    flat = tile * BLOCK + offsets
    lane_valid = flat < candidate_count
    parent = flat // candidate_width
    candidate = flat - parent * candidate_width
    row = request * beam_width + parent
    candidate_valid = lane_valid
    if HAS_VALID_MASK:
        candidate_valid = candidate_valid & tl.load(
            candidate_valid_mask + row * candidate_width + candidate,
            mask=lane_valid,
            other=False,
        )
    cumulative = tl.load(
        cumulative_logprobs + row,
        mask=candidate_valid,
        other=-float("inf"),
    )
    candidate_score = tl.load(
        top_logprobs + row * candidate_width + candidate,
        mask=candidate_valid,
        other=-float("inf"),
    )
    score = cumulative + candidate_score
    packed = _pack_score_flat(score, flat, candidate_valid if MASK_INVALID_INDICES else lane_valid)
    sorted_pairs = tl.sort(packed, descending=False)
    selected_scores = _unpack_score(sorted_pairs)
    selected_flat = (sorted_pairs & 0x00000000FFFFFFFF).to(tl.int32)
    output_mask = offsets < select_width
    output = request * scratch_stride + tile * select_width + offsets
    tl.store(local_scores + output, selected_scores, mask=output_mask)
    tl.store(local_flat_indices + output, selected_flat, mask=output_mask)


@triton.jit(
    do_not_specialize=[
        "list_count",
        "select_width",
        "scratch_stride",
    ]
)
def _merge_beam_topk_kernel(
    input_scores,
    input_flat_indices,
    output_scores,
    output_flat_indices,
    list_count,
    select_width,
    scratch_stride,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    output_list = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    source_list = output_list * 2 + offsets // select_width
    source_rank = offsets % select_width
    valid = (offsets < 2 * select_width) & (source_list < list_count)
    source = request * scratch_stride + source_list * select_width + source_rank
    score = tl.load(input_scores + source, mask=valid, other=-float("inf"))
    flat = tl.load(input_flat_indices + source, mask=valid, other=-1)
    valid = valid & (flat >= 0)
    packed = _pack_score_flat(score, flat, valid)
    sorted_pairs = tl.sort(packed, descending=False)
    selected_scores = _unpack_score(sorted_pairs)
    selected_flat = (sorted_pairs & 0x00000000FFFFFFFF).to(tl.int32)
    output_mask = offsets < select_width
    output = request * scratch_stride + output_list * select_width + offsets
    tl.store(output_scores + output, selected_scores, mask=output_mask)
    tl.store(output_flat_indices + output, selected_flat, mask=output_mask)


@triton.jit
def _beam_group_topk_kernel(
    cumulative_logprobs,
    top_logprobs,
    candidate_valid_mask,
    out_scores,
    out_flat_indices,
    scratch_stride,
    beam_width,
    HAS_VALID_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Select global Top-W by merging W already-sorted candidate rows.

    RecConstrainedTopK guarantees score-descending rows.  Adding one parent
    cumulative score preserves that order, so BeamSearchGroup is a W-way
    merge rather than a sort of W*W values.  Keep one cursor per parent and
    emit W winners with score-desc/flat-index-asc ordering.  ``beam_width`` is
    a runtime loop bound to avoid unrolling 128/256 iterations in Triton JIT.
    """

    request = tl.program_id(0)
    parent = tl.arange(0, BLOCK)
    lane = parent < beam_width
    cursor = tl.zeros((BLOCK,), dtype=tl.int32)
    max_i32: tl.constexpr = 2147483647
    for output_rank in range(0, beam_width):
        candidate = cursor
        row = request * beam_width + parent
        flat = parent * beam_width + candidate
        candidate_valid = lane & (candidate < beam_width)
        if HAS_VALID_MASK:
            candidate_valid = candidate_valid & tl.load(
                candidate_valid_mask + row * beam_width + candidate,
                mask=lane & (candidate < beam_width),
                other=False,
            )
        cumulative = tl.load(
            cumulative_logprobs + row,
            mask=candidate_valid,
            other=-float("inf"),
        )
        candidate_score = tl.load(
            top_logprobs + row * beam_width + candidate,
            mask=candidate_valid,
            other=-float("inf"),
        )
        score = cumulative + candidate_score
        score = tl.where(score == score, score, -float("inf"))
        best_score = tl.max(score, axis=0)
        has_candidate = tl.sum(candidate_valid.to(tl.int32), axis=0) > 0
        # Stable global tie break: lower flattened parent/candidate index.
        best_flat = tl.min(
            tl.where(candidate_valid & (score == best_score), flat, max_i32),
            axis=0,
        )
        best_score = tl.where(has_candidate, best_score, -float("inf"))
        best_flat = tl.where(has_candidate, best_flat, -1)
        tl.store(
            out_scores + request * scratch_stride + output_rank,
            best_score,
        )
        tl.store(
            out_flat_indices + request * scratch_stride + output_rank,
            best_flat,
        )
        cursor += (has_candidate & lane & (parent == best_flat // beam_width)).to(tl.int32)


@triton.jit
def _beam_group_finalize_kernel(
    selected_scores,
    selected_flat_indices,
    top_tokens,
    sequence,
    candidate_counts,
    out_token_ids,
    out_parent_indices,
    out_scores,
    out_grouped_flat_indices,
    out_sequence,
    out_status,
    out_payload,
    scratch_stride,
    active_count,
    current_step,
    HAS_COUNTS: tl.constexpr,
    WRITE_RESULT: tl.constexpr,
    BEAM_WIDTH: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Group survivors and write all Worker-owned fixed outputs.

    Parent grouping, sequence update, deferred status, and wire packing share
    the same W-lane program.  This replaces the separate group/write/pack
    launches on the fixed CUDA Worker path.
    """

    request = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    lane = rank < BEAM_WIDTH
    selected_offset = request * scratch_stride + rank
    flat = tl.load(selected_flat_indices + selected_offset, mask=lane, other=-1)
    valid = lane & (flat >= 0) & (flat < BEAM_WIDTH * BEAM_WIDTH)
    parent = flat // BEAM_WIDTH

    # BeamSearchGroup's public ordering is parent-major while retaining the
    # global score/flat-index order within one parent.
    group_key = (parent.to(tl.int64) << 32) | rank.to(tl.int64)
    group_key = tl.where(valid, group_key, 0x7FFFFFFFFFFFFFFF)
    grouped = tl.sort(group_key, descending=False)
    grouped_rank = (grouped & 0x00000000FFFFFFFF).to(tl.int64)
    grouped_valid = lane & (grouped_rank < BEAM_WIDTH)
    safe_rank = tl.where(grouped_valid, grouped_rank, 0)
    grouped_flat = tl.load(
        selected_flat_indices + request * scratch_stride + safe_rank,
        mask=grouped_valid,
        other=-1,
    )
    grouped_valid = grouped_valid & (grouped_flat >= 0)
    grouped_parent = grouped_flat // BEAM_WIDTH
    grouped_score = tl.load(
        selected_scores + request * scratch_stride + safe_rank,
        mask=grouped_valid,
        other=-float("inf"),
    )
    token = tl.load(
        top_tokens + request * BEAM_WIDTH * BEAM_WIDTH + grouped_flat,
        mask=grouped_valid,
        other=0,
    ).to(tl.int32)

    # A -1 selection is the explicit exhaustion sentinel emitted by the merge
    # kernel.  Treat it as a failed row even if caller-provided counts are
    # inconsistent with the candidate mask.
    bad = tl.sum((lane & ~valid).to(tl.int32), axis=0) > 0
    if HAS_COUNTS:
        count = tl.load(
            candidate_counts + request * BEAM_WIDTH + rank,
            mask=lane,
            other=0,
        )
        bad = bad | (tl.sum((lane & (count <= 0)).to(tl.int32), axis=0) > 0)
    output_valid = grouped_valid & ~bad
    safe_parent = tl.where(output_valid, grouped_parent, 0)
    token = tl.where(output_valid, token, 0)
    grouped_score = tl.where(output_valid, grouped_score, -float("inf"))
    grouped_flat = tl.where(output_valid, grouped_flat, 0)

    output = request * BEAM_WIDTH + rank
    tl.store(out_token_ids + output, token, mask=lane)
    tl.store(
        out_parent_indices + output,
        request * BEAM_WIDTH + safe_parent,
        mask=lane,
    )
    tl.store(out_scores + output, grouped_score, mask=lane)
    tl.store(out_grouped_flat_indices + output, grouped_flat, mask=lane)

    for step in tl.static_range(0, MAX_STEPS):
        history = tl.load(
            sequence + (request * BEAM_WIDTH + safe_parent) * MAX_STEPS + step,
            mask=lane & output_valid & (step < current_step),
            other=0,
        )
        value = tl.where(
            step < current_step,
            history,
            tl.where(step == current_step, token, 0),
        )
        tl.store(
            out_sequence + output * MAX_STEPS + step,
            value,
            mask=lane,
        )

    if WRITE_RESULT:
        score_bits = grouped_score.to(tl.int32, bitcast=True)
        tl.store(out_payload + output, token, mask=lane)
        tl.store(
            out_payload + active_count + output,
            request * BEAM_WIDTH + safe_parent,
            mask=lane,
        )
        tl.store(out_payload + 2 * active_count + output, score_bits, mask=lane)
        tl.store(out_status + request, tl.where(bad, 1, 0))


@triton.jit
def _beam_result_pack_kernel(
    token_ids,
    parent_indices,
    scores,
    status,
    out_payload,
    active_count,
    BEAM_WIDTH: tl.constexpr,
    RESULT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack terminal Beam winners while preserving deferred row status."""

    request = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    lane = rank < RESULT_WIDTH
    offset = request * RESULT_WIDTH + rank
    bad = tl.load(status + request) != 0
    token = tl.load(token_ids + offset, mask=lane, other=0).to(tl.int32)
    parent = tl.load(parent_indices + offset, mask=lane, other=0).to(tl.int32)
    score = tl.load(scores + offset, mask=lane, other=-float("inf")).to(tl.float32)
    token = tl.where(bad, 0, token)
    parent = tl.where(bad, request * BEAM_WIDTH, parent)
    score = tl.where(bad, -float("inf"), score)
    tl.store(out_payload + offset, token, mask=lane)
    tl.store(out_payload + active_count + offset, parent, mask=lane)
    tl.store(
        out_payload + 2 * active_count + offset,
        score.to(tl.int32, bitcast=True),
        mask=lane,
    )


@triton.jit(
    do_not_specialize=[
        "beam_width",
        "candidate_width",
        "scratch_stride",
    ]
)
def _group_beam_outputs_kernel(
    selected_scores,
    selected_flat_indices,
    top_tokens,
    out_token_ids,
    out_parent_indices,
    out_scores,
    out_grouped_flat_indices,
    beam_width,
    candidate_width,
    scratch_stride,
    WRITE_FLAT_INDICES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    selected_offset = request * scratch_stride + rank
    flat = tl.load(selected_flat_indices + selected_offset, mask=rank < beam_width, other=-1)
    valid = (rank < beam_width) & (flat >= 0) & (flat < beam_width * candidate_width)
    parent = flat // candidate_width

    group_key = (parent.to(tl.int64) << 32) | rank.to(tl.int64)
    group_key = tl.where(valid, group_key, 0x7FFFFFFFFFFFFFFF)
    grouped = tl.sort(group_key, descending=False)
    grouped_rank = (grouped & 0x00000000FFFFFFFF).to(tl.int64)
    grouped_valid = (rank < beam_width) & (grouped_rank < beam_width)
    safe_rank = tl.where(grouped_valid, grouped_rank, 0)
    grouped_flat = tl.load(
        selected_flat_indices + request * scratch_stride + safe_rank,
        mask=grouped_valid,
        other=-1,
    )
    grouped_valid = grouped_valid & (grouped_flat >= 0)
    grouped_parent = grouped_flat // candidate_width
    grouped_score = tl.load(
        selected_scores + request * scratch_stride + safe_rank,
        mask=grouped_valid,
        other=-float("inf"),
    )
    token = tl.load(
        top_tokens + request * beam_width * candidate_width + grouped_flat,
        mask=grouped_valid,
        other=0,
    )

    output = request * beam_width + rank
    tl.store(out_token_ids + output, token, mask=rank < beam_width)
    tl.store(
        out_parent_indices + output,
        tl.where(grouped_valid, request * beam_width + grouped_parent, -1),
        mask=rank < beam_width,
    )
    tl.store(out_scores + output, grouped_score, mask=rank < beam_width)
    if WRITE_FLAT_INDICES:
        tl.store(
            out_grouped_flat_indices + output,
            grouped_flat,
            mask=rank < beam_width,
        )


@triton.jit(do_not_specialize=["beam_width"])
def _bootstrap_beam_outputs_kernel(
    cumulative_logprobs,
    top_tokens,
    top_logprobs,
    candidate_valid_mask,
    out_token_ids,
    out_parent_indices,
    out_scores,
    out_flat_indices,
    beam_width,
    HAS_VALID_MASK: tl.constexpr,
    COMPACT_INPUT: tl.constexpr,
    WRITE_FLAT_INDICES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Expand the first live parent of every request from one beam to W."""

    request = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    # Canonical prefill produces one RecConstrainedTopK row per request.  The
    # legacy standalone ABI supplies B*W rows and only row 0 of every request
    # is live.  Accept both layouts so serving does not allocate/copy a dummy
    # W*W tensor merely to satisfy the bootstrap step.
    row = request if COMPACT_INPUT else request * beam_width
    candidate = row * beam_width + rank
    lane_valid = rank < beam_width
    candidate_valid = lane_valid
    if HAS_VALID_MASK:
        candidate_valid = candidate_valid & tl.load(
            candidate_valid_mask + candidate,
            mask=lane_valid,
            other=False,
        )
    token = tl.load(top_tokens + candidate, mask=lane_valid, other=0)
    score = tl.load(cumulative_logprobs + row) + tl.load(
        top_logprobs + candidate,
        mask=lane_valid,
        other=-float("inf"),
    )
    score = tl.where(candidate_valid & (score == score), score, -float("inf"))
    output = request * beam_width + rank
    tl.store(out_token_ids + output, token, mask=rank < beam_width)
    tl.store(
        out_parent_indices + output,
        tl.where(lane_valid, request * beam_width, -1),
        mask=rank < beam_width,
    )
    tl.store(out_scores + output, score, mask=rank < beam_width)
    if WRITE_FLAT_INDICES:
        tl.store(out_flat_indices + output, rank, mask=rank < beam_width)


@triton.jit(do_not_specialize=["beam_width", "result_width"])
def _grouped_final_outputs_kernel(
    selected_token_ids,
    selected_parent_indices,
    selected_scores,
    out_token_ids,
    out_parent_indices,
    out_scores,
    out_group_positions,
    beam_width,
    result_width,
    BLOCK: tl.constexpr,
):
    """Restore score order after BeamSearchGroup's parent grouping."""

    request = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    valid = rank < beam_width
    source = request * beam_width + rank
    score = tl.load(selected_scores + source, mask=valid, other=-float("inf"))
    # BeamSearchGroup is ordered by parent and preserves its original stable
    # (score, flat-index) order within each parent.  For exact score ties this
    # grouped rank is therefore the same ordering as the original flat key.
    packed = _pack_score_flat(score, rank, valid)
    ordered = tl.sort(packed, descending=False)
    group_position = (ordered & 0x00000000FFFFFFFF).to(tl.int32)
    output_valid = rank < result_width
    safe_position = tl.where(output_valid, group_position, 0)
    selected = request * beam_width + safe_position
    output = request * result_width + rank
    tl.store(
        out_token_ids + output,
        tl.load(selected_token_ids + selected, mask=output_valid, other=0),
        mask=output_valid,
    )
    tl.store(
        out_parent_indices + output,
        tl.load(selected_parent_indices + selected, mask=output_valid, other=-1),
        mask=output_valid,
    )
    tl.store(
        out_scores + output,
        tl.load(selected_scores + selected, mask=output_valid, other=-float("inf")),
        mask=output_valid,
    )
    tl.store(out_group_positions + output, group_position, mask=output_valid)


@triton.jit(do_not_specialize=["beam_width", "result_width", "max_steps"])
def _grouped_final_sequence_kernel(
    grouped_sequence,
    group_positions,
    out_sequence,
    beam_width,
    result_width,
    max_steps,
    BLOCK_STEPS: tl.constexpr,
):
    request = tl.program_id(0)
    rank = tl.program_id(1)
    step = tl.arange(0, BLOCK_STEPS)
    group_position = tl.load(group_positions + request * result_width + rank)
    source = (request * beam_width + group_position) * max_steps + step
    output = (request * result_width + rank) * max_steps + step
    mask = step < max_steps
    tl.store(
        out_sequence + output,
        tl.load(grouped_sequence + source, mask=mask, other=0),
        mask=mask,
    )


@triton.jit(do_not_specialize=["beam_width"])
def _parent_count_prefix_kernel(
    grouped_parent_indices,
    out_parent_count_prefix,
    beam_width,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    parent_id = tl.program_id(1)
    rank = tl.arange(0, BLOCK)
    global_parent = tl.load(
        grouped_parent_indices + request * beam_width + rank,
        mask=rank < beam_width,
        other=-1,
    )
    local_parent = global_parent - request * beam_width
    count = tl.sum(
        (rank < beam_width) & (local_parent >= 0) & (local_parent <= parent_id),
        axis=0,
    )
    tl.store(
        out_parent_count_prefix + request * beam_width + parent_id,
        request * beam_width + count,
    )


@triton.jit(
    do_not_specialize=[
        "beam_width",
        "candidate_width",
        "result_width",
        "scratch_stride",
    ]
)
def _final_beam_outputs_kernel(
    selected_scores,
    selected_flat_indices,
    top_tokens,
    out_token_ids,
    out_parent_indices,
    out_scores,
    beam_width,
    candidate_width,
    result_width,
    scratch_stride,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    selected_offset = request * scratch_stride + rank
    flat = tl.load(selected_flat_indices + selected_offset, mask=rank < result_width, other=-1)
    valid = (rank < result_width) & (flat >= 0) & (flat < beam_width * candidate_width)
    parent = flat // candidate_width
    token = tl.load(
        top_tokens + request * beam_width * candidate_width + flat,
        mask=valid,
        other=0,
    )
    score = tl.load(selected_scores + selected_offset, mask=valid, other=-float("inf"))
    output = request * result_width + rank
    tl.store(out_token_ids + output, token, mask=rank < result_width)
    tl.store(
        out_parent_indices + output,
        tl.where(valid, request * beam_width + parent, -1),
        mask=rank < result_width,
    )
    tl.store(out_scores + output, score, mask=rank < result_width)


@triton.jit(
    do_not_specialize=[
        "beam_width",
        "output_width",
        "max_steps",
        "current_step",
    ]
)
def _write_beam_sequence_kernel(
    sequence,
    out_token_ids,
    out_parent_indices,
    out_sequence,
    beam_width,
    output_width,
    max_steps,
    current_step,
    BLOCK_STEPS: tl.constexpr,
):
    request = tl.program_id(0)
    child = tl.program_id(1)
    step = tl.arange(0, BLOCK_STEPS)
    output_row = request * output_width + child
    global_parent = tl.load(out_parent_indices + output_row)
    local_parent = global_parent - request * beam_width
    valid_parent = (local_parent >= 0) & (local_parent < beam_width)
    history_mask = (step < max_steps) & (step < current_step) & valid_parent
    history = tl.load(
        sequence + (request * beam_width + local_parent) * max_steps + step,
        mask=history_mask,
        other=0,
    )
    token = tl.load(out_token_ids + output_row)
    value = tl.where(step < current_step, history, tl.where(step == current_step, token, 0))
    output = (request * output_width + child) * max_steps + step
    tl.store(out_sequence + output, value, mask=step < max_steps)


@dataclass(frozen=True, slots=True)
class BeamSelectionScratch:
    """Caller-owned ping-pong storage for hierarchical Beam Top-K."""

    local_scores: torch.Tensor
    local_flat_indices: torch.Tensor
    merge_scores: torch.Tensor
    merge_flat_indices: torch.Tensor


def beam_selection_scratch_slots(
    beam_width: int,
    candidate_width: int,
    selection_width: int,
) -> int:
    if min(beam_width, candidate_width, selection_width) <= 0:
        raise ValueError("beam, candidate, and selection widths must be positive")
    candidate_count = beam_width * candidate_width
    tile_count = (candidate_count + LOCAL_TILE_SIZE - 1) // LOCAL_TILE_SIZE
    return tile_count * selection_width


def allocate_beam_selection_scratch(
    batch_size: int,
    beam_width: int,
    candidate_width: int,
    selection_width: int,
    device: torch.device,
) -> BeamSelectionScratch:
    """Allocate convenience scratch; graph/runtime code should retain it."""

    slots = beam_selection_scratch_slots(beam_width, candidate_width, selection_width)
    return BeamSelectionScratch(
        local_scores=torch.empty((batch_size, slots), dtype=torch.float32, device=device),
        local_flat_indices=torch.empty((batch_size, slots), dtype=torch.int32, device=device),
        merge_scores=torch.empty((batch_size, slots), dtype=torch.float32, device=device),
        merge_flat_indices=torch.empty((batch_size, slots), dtype=torch.int32, device=device),
    )


def _require_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_beam_inputs(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    candidate_valid_mask: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if isinstance(current_step, bool) or not isinstance(current_step, int):
        raise TypeError("current_step must be an int")
    if cumulative_logprobs.device.type != "cuda":
        raise ValueError("CUDA Beam selection requires CUDA tensors")
    device = cumulative_logprobs.device
    if cumulative_logprobs.dtype != torch.float32:
        raise TypeError("cumulative_logprobs must use torch.float32")
    if cumulative_logprobs.ndim != 2 or cumulative_logprobs.shape[1] != 1:
        raise ValueError("cumulative_logprobs must have shape [B, 1] or [B*W, 1]")
    if not cumulative_logprobs.is_contiguous():
        raise ValueError("cumulative_logprobs must be contiguous")
    if sequence.dtype != torch.int32 or sequence.ndim != 3 or not sequence.is_contiguous():
        raise ValueError("sequence must be contiguous int32 [B, W, max_steps]")
    if sequence.device != device:
        raise ValueError("sequence must be on the cumulative_logprobs device")
    batch_size, beam_width, max_steps = sequence.shape
    if not 1 <= batch_size <= MAX_REQUESTS:
        raise ValueError(f"batch size must be in [1, {MAX_REQUESTS}]")
    if not 1 <= beam_width <= MAX_BEAM_WIDTH:
        raise ValueError(f"beam width must be in [1, {MAX_BEAM_WIDTH}]")
    if current_step < 0 or current_step >= max_steps:
        raise ValueError("current_step is outside the fixed sequence extent")
    compact_bootstrap = current_step == 0 and cumulative_logprobs.shape[0] == batch_size
    expected_rows = batch_size if compact_bootstrap else batch_size * beam_width
    if cumulative_logprobs.shape[0] != expected_rows:
        raise ValueError(
            "cumulative_logprobs rows must equal B for compact step-0 bootstrap or B*W otherwise"
        )
    if top_tokens.dtype != torch.int32 or top_tokens.ndim != 2:
        raise TypeError("top_tokens must be a 2-D torch.int32 tensor")
    if top_logprobs.dtype != torch.float32 or top_logprobs.ndim != 2:
        raise TypeError("top_logprobs must be a 2-D torch.float32 tensor")
    if top_tokens.device != device or top_logprobs.device != device:
        raise ValueError("candidate tensors must be on the cumulative_logprobs device")
    if not top_tokens.is_contiguous() or not top_logprobs.is_contiguous():
        raise ValueError("candidate tensors must be contiguous")
    if top_tokens.shape != top_logprobs.shape:
        raise ValueError("top_tokens and top_logprobs must have equal shapes")
    candidate_width = top_tokens.shape[1]
    if top_tokens.shape[0] != expected_rows:
        raise ValueError(
            "candidate rows must equal B for compact step-0 bootstrap or B*W otherwise"
        )
    if not 1 <= candidate_width <= MAX_CANDIDATE_WIDTH:
        raise ValueError(f"candidate width must be in [1, {MAX_CANDIDATE_WIDTH}]")
    if candidate_valid_mask is not None:
        _require_cuda_tensor(
            "candidate_valid_mask",
            candidate_valid_mask,
            device=device,
            dtype=torch.bool,
            shape=tuple(top_tokens.shape),
        )
    return batch_size, beam_width, candidate_width, max_steps


def _validate_scratch(
    scratch: BeamSelectionScratch,
    *,
    device: torch.device,
    batch_size: int,
    required_slots: int,
) -> int:
    tensors = (
        ("local_scores", scratch.local_scores, torch.float32),
        ("local_flat_indices", scratch.local_flat_indices, torch.int32),
        ("merge_scores", scratch.merge_scores, torch.float32),
        ("merge_flat_indices", scratch.merge_flat_indices, torch.int32),
    )
    stride = scratch.local_scores.shape[1] if scratch.local_scores.ndim == 2 else -1
    for name, tensor, dtype in tensors:
        if tensor.device != device or tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype} on {device}")
        if tensor.ndim != 2 or tensor.shape[0] < batch_size or tensor.shape[1] < required_slots:
            raise ValueError(f"{name} must be [B, >= {required_slots}], got {tuple(tensor.shape)}")
        if tensor.shape[1] != stride or not tensor.is_contiguous():
            raise ValueError("all Beam scratch tensors must share one contiguous stride")
    return int(stride)


def _hierarchical_topk_out(
    cumulative_logprobs: torch.Tensor,
    top_logprobs: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None,
    batch_size: int,
    beam_width: int,
    candidate_width: int,
    selection_width: int,
    scratch: BeamSelectionScratch,
    *,
    mask_invalid_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    candidate_count = beam_width * candidate_width
    list_count = (candidate_count + LOCAL_TILE_SIZE - 1) // LOCAL_TILE_SIZE
    local_block = triton.next_power_of_2(min(candidate_count, LOCAL_TILE_SIZE))
    required_slots = list_count * selection_width
    scratch_stride = _validate_scratch(
        scratch,
        device=cumulative_logprobs.device,
        batch_size=batch_size,
        required_slots=required_slots,
    )
    validity = candidate_valid_mask if candidate_valid_mask is not None else top_logprobs
    _local_beam_topk_kernel[(batch_size, list_count)](
        cumulative_logprobs,
        top_logprobs,
        validity,
        scratch.local_scores,
        scratch.local_flat_indices,
        beam_width,
        candidate_width,
        selection_width,
        scratch_stride,
        HAS_VALID_MASK=candidate_valid_mask is not None,
        BLOCK=local_block,
        MASK_INVALID_INDICES=mask_invalid_indices,
        num_warps=8 if local_block >= 128 else 4,
    )
    input_scores = scratch.local_scores
    input_indices = scratch.local_flat_indices
    output_scores = scratch.merge_scores
    output_indices = scratch.merge_flat_indices
    merge_block = triton.next_power_of_2(2 * selection_width)
    while list_count > 1:
        next_count = (list_count + 1) // 2
        _merge_beam_topk_kernel[(batch_size, next_count)](
            input_scores,
            input_indices,
            output_scores,
            output_indices,
            list_count,
            selection_width,
            scratch_stride,
            BLOCK=merge_block,
            num_warps=8 if merge_block >= 256 else 4,
        )
        input_scores, output_scores = output_scores, input_scores
        input_indices, output_indices = output_indices, input_indices
        list_count = next_count
    return input_scores, input_indices, scratch_stride


def beam_search_group_cuda_worker_out(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    out_token_ids: torch.Tensor,
    out_parent_indices: torch.Tensor,
    out_scores: torch.Tensor,
    out_sequence: torch.Tensor,
    out_grouped_flat_indices: torch.Tensor,
    scratch_scores: torch.Tensor,
    scratch_flat_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None,
    candidate_counts: torch.Tensor | None,
    out_status: torch.Tensor | None,
    out_payload: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-kernel fixed-width BeamSearchGroup for the CUDA Worker.

    The surrounding workspace has already been validated and warmed.  Keep
    the checked public operator unchanged; this private path deliberately
        specializes W so Triton can perform one request-wide W-way merge and
        one parent-group/finalize launch.
    """

    batch_size, beam_width, candidate_width, max_steps = _validate_beam_inputs(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        candidate_valid_mask,
    )
    if current_step == 0:
        raise ValueError("Worker BeamSearchGroup requires initialized decode state")
    if beam_width not in (64, 128, 256) or candidate_width != beam_width:
        raise ValueError("Worker BeamSearchGroup supports fixed W=K in 64/128/256")
    device = cumulative_logprobs.device
    specs = (
        ("out_token_ids", out_token_ids, torch.int32, (batch_size, beam_width)),
        (
            "out_parent_indices",
            out_parent_indices,
            torch.int32,
            (batch_size, beam_width),
        ),
        ("out_scores", out_scores, torch.float32, (batch_size, beam_width)),
        (
            "out_sequence",
            out_sequence,
            torch.int32,
            (batch_size, beam_width, max_steps),
        ),
        (
            "out_grouped_flat_indices",
            out_grouped_flat_indices,
            torch.int32,
            (batch_size, beam_width),
        ),
    )
    for name, tensor, dtype, shape in specs:
        _require_cuda_tensor(name, tensor, device=device, dtype=dtype, shape=shape)
    if out_sequence.data_ptr() == sequence.data_ptr():
        raise ValueError("out_sequence must not alias sequence")
    if scratch_scores.device != device or scratch_scores.dtype != torch.float32:
        raise ValueError("scratch_scores must be CUDA fp32")
    if scratch_flat_indices.device != device or scratch_flat_indices.dtype != torch.int32:
        raise ValueError("scratch_flat_indices must be CUDA int32")
    if scratch_scores.ndim != 2 or scratch_flat_indices.shape != scratch_scores.shape:
        raise ValueError("Worker BeamSearchGroup scratch must use equal 2-D shapes")
    if scratch_scores.shape[0] < batch_size or scratch_scores.shape[1] < beam_width:
        raise ValueError("Worker BeamSearchGroup scratch capacity is too small")
    write_result = out_status is not None and out_payload is not None
    if write_result:
        if candidate_counts is None:
            raise ValueError("candidate_counts are required when writing deferred status")
        if candidate_counts.device != device or candidate_counts.dtype != torch.int32:
            raise ValueError("candidate_counts must be CUDA int32")
        if candidate_counts.numel() < batch_size * beam_width:
            raise ValueError("candidate_counts capacity is too small")
        if out_status.device != device or out_status.dtype != torch.int32:
            raise ValueError("out_status must be CUDA int32")
        if out_status.numel() < batch_size:
            raise ValueError("out_status capacity is too small")
        if out_payload.device != device or out_payload.dtype != torch.int32:
            raise ValueError("out_payload must be CUDA int32")
        if out_payload.numel() < 3 * batch_size * beam_width:
            raise ValueError("out_payload capacity is too small")

    _beam_search_group_cuda_worker_out_unchecked(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        out_token_ids,
        out_parent_indices,
        out_scores,
        out_sequence,
        out_grouped_flat_indices,
        scratch_scores,
        scratch_flat_indices,
        candidate_valid_mask,
        candidate_counts,
        out_status,
        out_payload,
    )
    return out_token_ids, out_parent_indices, out_scores, out_sequence


def _beam_search_group_cuda_worker_out_unchecked(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    out_token_ids: torch.Tensor,
    out_parent_indices: torch.Tensor,
    out_scores: torch.Tensor,
    out_sequence: torch.Tensor,
    out_grouped_flat_indices: torch.Tensor,
    scratch_scores: torch.Tensor,
    scratch_flat_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None,
    candidate_counts: torch.Tensor | None,
    out_status: torch.Tensor | None,
    out_payload: torch.Tensor | None,
) -> None:
    """Launch the prevalidated persistent Worker BeamGroup buffers."""

    batch_size, beam_width = out_token_ids.shape
    max_steps = out_sequence.shape[2]
    scratch_stride = int(scratch_scores.shape[1])
    write_result = out_status is not None and out_payload is not None
    block = triton.next_power_of_2(beam_width)
    validity = candidate_valid_mask if candidate_valid_mask is not None else top_logprobs
    _beam_group_topk_kernel[(batch_size,)](
        cumulative_logprobs,
        top_logprobs,
        validity,
        scratch_scores,
        scratch_flat_indices,
        scratch_stride,
        beam_width,
        HAS_VALID_MASK=candidate_valid_mask is not None,
        BLOCK=block,
        num_warps=8 if block >= 128 else 4,
    )
    group_block = triton.next_power_of_2(beam_width)
    _beam_group_finalize_kernel[(batch_size,)](
        scratch_scores,
        scratch_flat_indices,
        top_tokens,
        sequence,
        candidate_counts if candidate_counts is not None else scratch_flat_indices,
        out_token_ids,
        out_parent_indices,
        out_scores,
        out_grouped_flat_indices,
        out_sequence,
        out_status if out_status is not None else scratch_flat_indices,
        out_payload if out_payload is not None else scratch_flat_indices,
        scratch_stride,
        batch_size * beam_width,
        current_step,
        HAS_COUNTS=candidate_counts is not None,
        WRITE_RESULT=write_result,
        BEAM_WIDTH=beam_width,
        MAX_STEPS=max_steps,
        BLOCK=group_block,
        num_warps=8 if group_block >= 128 else 4,
    )


def beam_result_pack_cuda_worker_out(
    token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    scores: torch.Tensor,
    status: torch.Tensor,
    out_payload: torch.Tensor,
    *,
    beam_width: int,
) -> None:
    """Pack fixed Worker outputs into ``token|parent|score_bits`` staging.

    This terminal-step helper is deliberately unchecked beyond the compact
    output ABI: the runner owns and prevalidates all buffers.  Deferred error
    rows are sanitized without reading a device scalar on the host.
    """

    if token_ids.device.type != "cuda" or token_ids.dtype != torch.int32:
        raise ValueError("token_ids must be CUDA int32")
    if token_ids.ndim != 2 or not token_ids.is_contiguous():
        raise ValueError("token_ids must be contiguous [B, Wout]")
    if parent_indices.shape != token_ids.shape or parent_indices.dtype != torch.int32:
        raise ValueError("parent_indices must be contiguous int32 with token_ids shape")
    if scores.shape != token_ids.shape or scores.dtype != torch.float32:
        raise ValueError("scores must be contiguous fp32 with token_ids shape")
    if not parent_indices.is_contiguous() or not scores.is_contiguous():
        raise ValueError("terminal Beam outputs must be contiguous")
    batch_size, result_width = token_ids.shape
    if status.device != token_ids.device or status.dtype != torch.int32:
        raise ValueError("status must be CUDA int32")
    if status.numel() < batch_size:
        raise ValueError("status capacity is too small")
    if out_payload.device != token_ids.device or out_payload.dtype != torch.int32:
        raise ValueError("out_payload must be CUDA int32")
    active_count = batch_size * result_width
    if out_payload.numel() < 3 * active_count:
        raise ValueError("out_payload capacity is too small")
    block = triton.next_power_of_2(result_width)
    _beam_result_pack_kernel[(batch_size,)](
        token_ids,
        parent_indices,
        scores,
        status,
        out_payload,
        active_count,
        BEAM_WIDTH=beam_width,
        RESULT_WIDTH=result_width,
        BLOCK=block,
        num_warps=8 if block >= 128 else 4,
    )


def beam_search_group_cuda_out(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    top_k: int,
    out_token_ids: torch.Tensor,
    out_parent_indices: torch.Tensor,
    out_scores: torch.Tensor,
    out_parent_count_prefix: torch.Tensor,
    out_sequence: torch.Tensor,
    local_scores: torch.Tensor,
    local_flat_indices: torch.Tensor,
    merge_scores: torch.Tensor,
    merge_flat_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None = None,
    out_grouped_flat_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write fixed-width, parent-grouped Beam survivors into caller buffers."""

    batch_size, beam_width, candidate_width, max_steps = _validate_beam_inputs(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        candidate_valid_mask,
    )
    if current_step not in (0, 1, 2):
        raise ValueError("CUDA BeamSearchGroup supports current_step 0, 1, or 2")
    if candidate_width != beam_width:
        raise ValueError("BeamSearchGroup candidate width must equal beam width")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    if top_k != beam_width:
        raise ValueError("CUDA BeamSearchGroup requires fixed top_k == beam_width")
    device = cumulative_logprobs.device
    output_specs = (
        ("out_token_ids", out_token_ids, torch.int32, (batch_size, beam_width)),
        ("out_parent_indices", out_parent_indices, torch.int32, (batch_size, beam_width)),
        ("out_scores", out_scores, torch.float32, (batch_size, beam_width)),
        (
            "out_parent_count_prefix",
            out_parent_count_prefix,
            torch.int32,
            (batch_size, beam_width),
        ),
        (
            "out_sequence",
            out_sequence,
            torch.int32,
            (batch_size, beam_width, max_steps),
        ),
    )
    for name, tensor, dtype, shape in output_specs:
        _require_cuda_tensor(name, tensor, device=device, dtype=dtype, shape=shape)
    if out_grouped_flat_indices is not None:
        _require_cuda_tensor(
            "out_grouped_flat_indices",
            out_grouped_flat_indices,
            device=device,
            dtype=torch.int32,
            shape=(batch_size, beam_width),
        )
    if out_sequence.data_ptr() == sequence.data_ptr():
        raise ValueError("out_sequence must not alias sequence")

    scratch = BeamSelectionScratch(
        local_scores,
        local_flat_indices,
        merge_scores,
        merge_flat_indices,
    )
    block = triton.next_power_of_2(beam_width)
    if current_step == 0:
        # Bootstrap launches no hierarchical Top-K kernel; retain a one-slot
        # scratch ABI for the caller-owned/out variant without allocating W-by-W
        # temporary storage.
        required_slots = 1
        _validate_scratch(
            scratch,
            device=device,
            batch_size=batch_size,
            required_slots=required_slots,
        )
        validity = candidate_valid_mask if candidate_valid_mask is not None else top_logprobs
        _bootstrap_beam_outputs_kernel[(batch_size,)](
            cumulative_logprobs,
            top_tokens,
            top_logprobs,
            validity,
            out_token_ids,
            out_parent_indices,
            out_scores,
            (
                out_grouped_flat_indices
                if out_grouped_flat_indices is not None
                else out_parent_indices
            ),
            beam_width,
            HAS_VALID_MASK=candidate_valid_mask is not None,
            COMPACT_INPUT=cumulative_logprobs.shape[0] == batch_size,
            WRITE_FLAT_INDICES=out_grouped_flat_indices is not None,
            BLOCK=block,
            num_warps=8 if block >= 128 else 4,
        )
        _parent_count_prefix_kernel[(batch_size, beam_width)](
            out_parent_indices,
            out_parent_count_prefix,
            beam_width,
            BLOCK=block,
            num_warps=8 if block >= 128 else 4,
        )
        step_block = triton.next_power_of_2(max_steps)
        _write_beam_sequence_kernel[(batch_size, beam_width)](
            sequence,
            out_token_ids,
            out_parent_indices,
            out_sequence,
            beam_width,
            beam_width,
            max_steps,
            current_step,
            BLOCK_STEPS=step_block,
            num_warps=4,
        )
        return (
            out_token_ids,
            out_parent_indices,
            out_scores,
            out_parent_count_prefix,
            out_sequence,
        )

    selected_scores, selected_indices, scratch_stride = _hierarchical_topk_out(
        cumulative_logprobs,
        top_logprobs,
        candidate_valid_mask,
        batch_size,
        beam_width,
        candidate_width,
        beam_width,
        scratch,
    )
    _group_beam_outputs_kernel[(batch_size,)](
        selected_scores,
        selected_indices,
        top_tokens,
        out_token_ids,
        out_parent_indices,
        out_scores,
        (out_grouped_flat_indices if out_grouped_flat_indices is not None else out_parent_indices),
        beam_width,
        candidate_width,
        scratch_stride,
        WRITE_FLAT_INDICES=out_grouped_flat_indices is not None,
        BLOCK=block,
        num_warps=8 if block >= 128 else 4,
    )
    _parent_count_prefix_kernel[(batch_size, beam_width)](
        out_parent_indices,
        out_parent_count_prefix,
        beam_width,
        BLOCK=block,
        num_warps=8 if block >= 128 else 4,
    )
    step_block = triton.next_power_of_2(max_steps)
    _write_beam_sequence_kernel[(batch_size, beam_width)](
        sequence,
        out_token_ids,
        out_parent_indices,
        out_sequence,
        beam_width,
        beam_width,
        max_steps,
        current_step,
        BLOCK_STEPS=step_block,
        num_warps=4,
    )
    return (
        out_token_ids,
        out_parent_indices,
        out_scores,
        out_parent_count_prefix,
        out_sequence,
    )


def beam_search_rec_final_select_grouped_cuda_out(
    selected_token_ids: torch.Tensor,
    selected_parent_indices: torch.Tensor,
    selected_scores: torch.Tensor,
    grouped_sequence: torch.Tensor,
    result_width: int,
    out_token_ids: torch.Tensor,
    out_parent_indices: torch.Tensor,
    out_scores: torch.Tensor,
    out_sequence: torch.Tensor,
    out_group_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finalize parent-grouped Beam winners into the current score order.

    This is a Worker-only downstream stage for the fixed
    ``TopKOut -> GroupOut -> FinalSelectOut`` pipeline.  The public OneRec
    FinalSelect operator retains its existing W*K candidate ABI.
    """

    if selected_token_ids.device.type != "cuda":
        raise ValueError("grouped CUDA FinalSelect requires CUDA tensors")
    device = selected_token_ids.device
    if selected_token_ids.ndim != 2:
        raise ValueError("selected_token_ids must be [B, W]")
    batch_size, beam_width = selected_token_ids.shape
    if not 1 <= batch_size <= MAX_REQUESTS:
        raise ValueError(f"batch size must be in [1, {MAX_REQUESTS}]")
    if not 1 <= beam_width <= MAX_BEAM_WIDTH:
        raise ValueError(f"beam width must be in [1, {MAX_BEAM_WIDTH}]")
    if isinstance(result_width, bool) or not isinstance(result_width, int):
        raise TypeError("result_width must be an int")
    if not 1 <= result_width <= beam_width:
        raise ValueError("result_width must be in [1, beam_width]")
    if grouped_sequence.ndim != 3 or tuple(grouped_sequence.shape[:2]) != (
        batch_size,
        beam_width,
    ):
        raise ValueError("grouped_sequence must be [B, W, max_steps]")
    max_steps = grouped_sequence.shape[2]
    input_specs = (
        ("selected_token_ids", selected_token_ids, torch.int32, (batch_size, beam_width)),
        (
            "selected_parent_indices",
            selected_parent_indices,
            torch.int32,
            (batch_size, beam_width),
        ),
        ("selected_scores", selected_scores, torch.float32, (batch_size, beam_width)),
        (
            "grouped_sequence",
            grouped_sequence,
            torch.int32,
            (batch_size, beam_width, max_steps),
        ),
    )
    output_specs = (
        ("out_token_ids", out_token_ids, torch.int32, (batch_size, result_width)),
        (
            "out_parent_indices",
            out_parent_indices,
            torch.int32,
            (batch_size, result_width),
        ),
        ("out_scores", out_scores, torch.float32, (batch_size, result_width)),
        (
            "out_sequence",
            out_sequence,
            torch.int32,
            (batch_size, result_width, max_steps),
        ),
        (
            "out_group_positions",
            out_group_positions,
            torch.int32,
            (batch_size, result_width),
        ),
    )
    for name, tensor, dtype, shape in (*input_specs, *output_specs):
        _require_cuda_tensor(name, tensor, device=device, dtype=dtype, shape=shape)

    block = triton.next_power_of_2(beam_width)
    _grouped_final_outputs_kernel[(batch_size,)](
        selected_token_ids,
        selected_parent_indices,
        selected_scores,
        out_token_ids,
        out_parent_indices,
        out_scores,
        out_group_positions,
        beam_width,
        result_width,
        BLOCK=block,
        num_warps=8 if block >= 128 else 4,
    )
    step_block = triton.next_power_of_2(max_steps)
    _grouped_final_sequence_kernel[(batch_size, result_width)](
        grouped_sequence,
        out_group_positions,
        out_sequence,
        beam_width,
        result_width,
        max_steps,
        BLOCK_STEPS=step_block,
        num_warps=4,
    )
    return out_token_ids, out_parent_indices, out_scores, out_sequence


def beam_search_group_cuda(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    top_k: int,
    candidate_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocating convenience wrapper for CUDA BeamSearchGroup."""

    batch_size, beam_width, candidate_width, max_steps = _validate_beam_inputs(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        candidate_valid_mask,
    )
    if current_step not in (0, 1, 2):
        raise ValueError("CUDA BeamSearchGroup supports current_step 0, 1, or 2")
    if candidate_width != beam_width:
        raise ValueError("BeamSearchGroup candidate width must equal beam width")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    if top_k != beam_width:
        raise ValueError("CUDA BeamSearchGroup requires fixed top_k == beam_width")
    device = cumulative_logprobs.device
    scratch = allocate_beam_selection_scratch(
        batch_size,
        1 if current_step == 0 else beam_width,
        1 if current_step == 0 else candidate_width,
        1 if current_step == 0 else beam_width,
        device,
    )
    outputs = (
        torch.empty((batch_size, beam_width), dtype=torch.int32, device=device),
        torch.empty((batch_size, beam_width), dtype=torch.int32, device=device),
        torch.empty((batch_size, beam_width), dtype=torch.float32, device=device),
        torch.empty((batch_size, beam_width), dtype=torch.int32, device=device),
        torch.empty(
            (batch_size, beam_width, max_steps),
            dtype=torch.int32,
            device=device,
        ),
    )
    return beam_search_group_cuda_out(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        top_k,
        *outputs,
        scratch.local_scores,
        scratch.local_flat_indices,
        scratch.merge_scores,
        scratch.merge_flat_indices,
        candidate_valid_mask,
    )


def beam_search_rec_final_select_cuda_out(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
    out_token_ids: torch.Tensor,
    out_parent_indices: torch.Tensor,
    out_scores: torch.Tensor,
    out_sequence: torch.Tensor,
    local_scores: torch.Tensor,
    local_flat_indices: torch.Tensor,
    merge_scores: torch.Tensor,
    merge_flat_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write score-ordered final Beam results into caller-owned CUDA tensors."""

    batch_size, beam_width, candidate_width, max_steps = _validate_beam_inputs(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        candidate_valid_mask,
    )
    if current_step not in (0, 1, 2):
        raise ValueError("CUDA FinalSelect supports current_step 0, 1, or 2")
    if isinstance(result_width, bool) or not isinstance(result_width, int):
        raise TypeError("result_width must be an int")
    if not 1 <= result_width <= beam_width:
        raise ValueError("CUDA FinalSelect requires result_width in [1, beam_width]")
    device = cumulative_logprobs.device
    output_specs = (
        ("out_token_ids", out_token_ids, torch.int32, (batch_size * result_width, 1)),
        (
            "out_parent_indices",
            out_parent_indices,
            torch.int32,
            (batch_size * result_width, 1),
        ),
        ("out_scores", out_scores, torch.float32, (batch_size * result_width, 1)),
        (
            "out_sequence",
            out_sequence,
            torch.int32,
            (batch_size, result_width, max_steps),
        ),
    )
    for name, tensor, dtype, shape in output_specs:
        _require_cuda_tensor(name, tensor, device=device, dtype=dtype, shape=shape)
    if out_sequence.data_ptr() == sequence.data_ptr():
        raise ValueError("out_sequence must not alias sequence")

    scratch = BeamSelectionScratch(
        local_scores,
        local_flat_indices,
        merge_scores,
        merge_flat_indices,
    )
    selected_scores, selected_indices, scratch_stride = _hierarchical_topk_out(
        cumulative_logprobs,
        top_logprobs,
        candidate_valid_mask,
        batch_size,
        beam_width,
        candidate_width,
        result_width,
        scratch,
    )
    block = triton.next_power_of_2(result_width)
    _final_beam_outputs_kernel[(batch_size,)](
        selected_scores,
        selected_indices,
        top_tokens,
        out_token_ids,
        out_parent_indices,
        out_scores,
        beam_width,
        candidate_width,
        result_width,
        scratch_stride,
        BLOCK=block,
        num_warps=8 if block >= 128 else 4,
    )
    step_block = triton.next_power_of_2(max_steps)
    _write_beam_sequence_kernel[(batch_size, result_width)](
        sequence,
        out_token_ids,
        out_parent_indices,
        out_sequence,
        beam_width,
        result_width,
        max_steps,
        current_step,
        BLOCK_STEPS=step_block,
        num_warps=4,
    )
    return out_token_ids, out_parent_indices, out_scores, out_sequence


def beam_search_rec_final_select_cuda(
    cumulative_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
    candidate_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocating convenience wrapper for CUDA OneRec FinalSelect."""

    batch_size, beam_width, candidate_width, max_steps = _validate_beam_inputs(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        candidate_valid_mask,
    )
    if current_step not in (0, 1, 2):
        raise ValueError("CUDA FinalSelect supports current_step 0, 1, or 2")
    if isinstance(result_width, bool) or not isinstance(result_width, int):
        raise TypeError("result_width must be an int")
    if not 1 <= result_width <= beam_width:
        raise ValueError("CUDA FinalSelect requires result_width in [1, beam_width]")
    device = cumulative_logprobs.device
    scratch = allocate_beam_selection_scratch(
        batch_size, beam_width, candidate_width, result_width, device
    )
    outputs = (
        torch.empty((batch_size * result_width, 1), dtype=torch.int32, device=device),
        torch.empty((batch_size * result_width, 1), dtype=torch.int32, device=device),
        torch.empty((batch_size * result_width, 1), dtype=torch.float32, device=device),
        torch.empty(
            (batch_size, result_width, max_steps),
            dtype=torch.int32,
            device=device,
        ),
    )
    return beam_search_rec_final_select_cuda_out(
        cumulative_logprobs,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        result_width,
        *outputs,
        scratch.local_scores,
        scratch.local_flat_indices,
        scratch.merge_scores,
        scratch.merge_flat_indices,
        candidate_valid_mask,
    )


__all__ = [
    "BeamSelectionScratch",
    "LOCAL_TILE_SIZE",
    "MAX_BEAM_WIDTH",
    "MAX_CANDIDATE_WIDTH",
    "MAX_REQUESTS",
    "allocate_beam_selection_scratch",
    "beam_search_group_cuda",
    "beam_search_group_cuda_out",
    "beam_search_rec_final_select_cuda",
    "beam_search_rec_final_select_cuda_out",
    "beam_search_rec_final_select_grouped_cuda_out",
    "beam_selection_scratch_slots",
]
