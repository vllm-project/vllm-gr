# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA persistent Beam updates. No scheduling, transport, or host readback."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import triton
import triton.language as tl

# mypy: disable-error-code="misc,no-untyped-def"

ERROR_NOT_INITIALIZED = 1
ERROR_ALREADY_INITIALIZED = 2
ERROR_INVALID_CANDIDATE = 3
ERROR_CAPACITY = 4
ERROR_NO_CANDIDATES = 5
FINISH_STOP = 1
FINISH_LENGTH = 2


@triton.jit
def _reset_session_state_kernel(
    tensors,
    SIZES: tl.constexpr,
    STARTS: tl.constexpr,
    NEGATIVE_INFINITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Reset disjoint contiguous rows in one launch, without touching config."""
    block = tl.program_id(0)
    for i in tl.static_range(len(SIZES)):
        if (block >= STARTS[i]) & (block < STARTS[i + 1]):
            offset = (block - STARTS[i]) * BLOCK + tl.arange(0, BLOCK)
            if NEGATIVE_INFINITY[i]:
                value = -float("inf")
            else:
                value = 0
            tl.store(tensors[i] + offset, value, mask=offset < SIZES[i])


@dataclass(slots=True)
class BeamStateScratch:
    """Optional postprocess scratch; contains no cross-step request state."""

    parent_scores: torch.Tensor
    selection_mask: torch.Tensor
    terminal_ranks: torch.Tensor
    finish_reasons: torch.Tensor
    cumulative_scores: torch.Tensor
    row_counts: torch.Tensor
    row_errors: torch.Tensor
    row_valid_counts: torch.Tensor
    row_offsets: torch.Tensor
    execute: torch.Tensor
    reorder: torch.Tensor
    next_completed: torch.Tensor

    @classmethod
    def allocate(
        cls,
        batch: int,
        width: int,
        device: torch.device,
        *,
        candidate_width: int | None = None,
    ) -> BeamStateScratch:
        def empty(shape: tuple[int, ...], dtype: torch.dtype = torch.int32) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=device)

        candidate_width = width if candidate_width is None else candidate_width
        rows, candidates = (batch, width), (batch, width, candidate_width)
        return cls(
            empty(rows, torch.float32),
            empty(candidates, torch.bool),
            empty(candidates),
            empty(candidates),
            empty(candidates, torch.float32),
            empty(rows),
            empty(rows),
            empty(rows),
            empty(rows),
            empty((batch,), torch.bool),
            empty((batch,), torch.bool),
            empty((batch,)),
        )

    def device_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, field.name) for field in fields(self))


_ERROR_NOT_INITIALIZED = tl.constexpr(ERROR_NOT_INITIALIZED)
_ERROR_ALREADY_INITIALIZED = tl.constexpr(ERROR_ALREADY_INITIALIZED)
_ERROR_INVALID_CANDIDATE = tl.constexpr(ERROR_INVALID_CANDIDATE)
_ERROR_CAPACITY = tl.constexpr(ERROR_CAPACITY)
_ERROR_NO_CANDIDATES = tl.constexpr(ERROR_NO_CANDIDATES)
_FINISH_STOP = tl.constexpr(FINISH_STOP)
_FINISH_LENGTH = tl.constexpr(FINISH_LENGTH)


@triton.jit
def _rank_final_beams_kernel(
    scores,
    lengths,
    reasons,
    ranking_scores,
    CAPACITY: tl.constexpr,
    PROMPT_LENGTH: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    LENGTH_PENALTY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    score = tl.load(scores + i, mask=i < CAPACITY, other=-float("inf"))
    length = tl.load(lengths + i, mask=i < CAPACITY, other=0)
    reason = tl.load(reasons + i, mask=i < CAPACITY, other=0)
    # Match the public Beam score's total decoder length, excluding the
    # terminal stop token. Keep raw cumulative scores for the output payload.
    effective = tl.maximum(PROMPT_LENGTH + length - (reason == _FINISH_STOP), 1)
    # The common scale preserves full-length scores exactly, avoiding new
    # rounding ties for fixed-depth recommendation models.
    ratio = (PROMPT_LENGTH + MAX_TOKENS) / effective.to(tl.float32)
    ranked = score * tl.exp(tl.log(ratio) * LENGTH_PENALTY)
    tl.store(ranking_scores + i, ranked, mask=i < CAPACITY)


@triton.jit
def _prepare_candidates_kernel(
    tokens,
    logprobs,
    valid_mask,
    beam_scores,
    active_mask,
    generated,
    max_tokens,
    stops,
    num_stops,
    ignore_stops,
    parent_scores,
    selection_mask,
    terminal_ranks,
    finish_reasons,
    cumulative_scores,
    row_counts,
    row_errors,
    row_valid_counts,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    PREFILL: tl.constexpr,
):
    parent = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    lane = rank < WIDTH
    if PREFILL:
        base = 0.0
        live = True
        step = 0
    else:
        base = tl.load(beam_scores + parent)
        live = tl.load(active_mask + parent)
        step = tl.load(generated)
    offset = parent * WIDTH + rank
    valid = lane & live & tl.load(valid_mask + offset, mask=lane, other=False)
    token = tl.load(tokens + offset, mask=valid, other=0)
    local = tl.load(logprobs + offset, mask=valid, other=0.0)
    total = base + local
    bad = valid & (
        (token < 0)
        | (tl.abs(local) == float("inf"))
        | (local != local)
        | (tl.abs(total) == float("inf"))
        | (total != total)
    )
    stop = tl.full((BLOCK,), False, tl.int1)
    if not tl.load(ignore_stops):
        for index in range(tl.load(num_stops)):
            stop = stop | (token == tl.load(stops + index))
    last = step + 1 >= tl.load(max_tokens)
    terminal = valid & (stop | last)
    reason = tl.where(terminal, tl.where(stop, _FINISH_STOP, _FINISH_LENGTH), 0)
    tl.store(parent_scores + parent, base)
    tl.store(selection_mask + offset, valid & ~terminal, mask=lane)
    tl.store(terminal_ranks + offset, tl.cumsum(terminal.to(tl.int32)) - 1, mask=lane)
    tl.store(finish_reasons + offset, reason, mask=lane)
    tl.store(cumulative_scores + offset, total, mask=lane)
    tl.store(row_counts + parent, tl.sum(terminal.to(tl.int32)))
    tl.store(row_errors + parent, tl.sum(bad.to(tl.int32)))
    tl.store(row_valid_counts + parent, tl.sum(valid.to(tl.int32)))


@triton.jit
def _validate_step_kernel(
    initialized,
    finished,
    error,
    generated,
    decode_steps,
    max_tokens,
    completed,
    row_counts,
    row_errors,
    row_valid_counts,
    row_offsets,
    execute,
    reorder,
    next_completed,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
    KV_CAPACITY: tl.constexpr,
    PREFILL: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    counts = tl.load(row_counts + row, mask=row < ROWS, other=0)
    offsets = tl.cumsum(counts) - counts
    previous = tl.load(completed)
    next_count = previous + tl.sum(counts)
    valid_count = tl.sum(tl.load(row_valid_counts + row, mask=row < ROWS, other=0))
    bad = tl.sum(tl.load(row_errors + row, mask=row < ROWS, other=0)) > 0
    old_error = tl.load(error)
    init = tl.load(initialized)
    done = tl.load(finished)
    step = tl.load(generated)
    code = tl.full((), 0, tl.int32)
    if PREFILL:
        code = tl.where(init, _ERROR_ALREADY_INITIALIZED, code)
        # Cancellation may finish a row before its first output chunk.
        # Preserve the duplicate-initialization error for initialized rows.
        enabled = (old_error == 0) & (~done | init)
    else:
        code = tl.where(~init, _ERROR_NOT_INITIALIZED, code)
        enabled = (old_error == 0) & ~done
    capacity_bad = next_count > CAPACITY
    if not PREFILL:
        capacity_bad = capacity_bad | (step < 1) | (step >= tl.load(max_tokens))
        capacity_bad = capacity_bad | (tl.load(decode_steps) != step - 1)
        capacity_bad = capacity_bad | (tl.load(decode_steps) >= KV_CAPACITY)
    code = tl.where((code == 0) & capacity_bad, _ERROR_CAPACITY, code)
    code = tl.where((code == 0) & bad, _ERROR_INVALID_CANDIDATE, code)
    code = tl.where((code == 0) & (valid_count == 0) & (previous == 0), _ERROR_NO_CANDIDATES, code)
    run = enabled & (code == 0)
    tl.store(error, code, mask=enabled & (code != 0))
    tl.store(execute, run)
    # This is a step-entry latch. Commit writes finished only after KV.
    tl.store(reorder, run & ~PREFILL & (valid_count > tl.sum(counts)))
    tl.store(next_completed, next_count)
    tl.store(row_offsets + row, previous + offsets, mask=row < ROWS)


@triton.jit
def _append_completed_kernel(
    tokens,
    terminal_ranks,
    finish_reasons,
    cumulative_scores,
    row_offsets,
    generated,
    execute,
    completed_tokens,
    completed_scores,
    completed_parents,
    completed_lengths,
    completed_reasons,
    completed_ties,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    PREFILL: tl.constexpr,
):
    parent = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    offset = parent * WIDTH + rank
    reason = tl.load(finish_reasons + offset, mask=rank < WIDTH, other=0)
    valid = (rank < WIDTH) & (reason != 0) & tl.load(execute)
    dest = tl.load(row_offsets + parent) + tl.load(
        terminal_ranks + offset, mask=rank < WIDTH, other=0
    )
    length = 1 if PREFILL else tl.load(generated) + 1
    token = tl.load(tokens + offset, mask=valid, other=0)
    score = tl.load(cumulative_scores + offset, mask=valid, other=0.0)
    tl.store(completed_tokens + dest, token, mask=valid)
    tl.store(completed_scores + dest, score, mask=valid)
    tl.store(completed_parents + dest, parent, mask=valid)
    tl.store(completed_lengths + dest, length, mask=valid)
    tl.store(completed_reasons + dest, reason, mask=valid)
    tl.store(completed_ties + dest, offset, mask=valid)


@triton.jit
def _stage_sequence_kernel(
    sequence,
    selected_tokens,
    selected_parents,
    selected_flat,
    next_sequence,
    generated,
    execute,
    WIDTH: tl.constexpr,
    SEQUENCE_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    PREFILL: tl.constexpr,
):
    child = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    step = 0 if PREFILL else tl.load(generated)
    live = tl.load(selected_flat + child) >= 0
    parent = tl.where(live, tl.load(selected_parents + child), 0)
    token = tl.load(selected_tokens + child)
    old = tl.load(
        sequence + parent * SEQUENCE_STRIDE + column,
        mask=live & (column < step) & (column < SEQUENCE_STRIDE),
        other=0,
    )
    value = tl.where(column < step, old, tl.where(live & (column == step), token, 0))
    tl.store(
        next_sequence + child * OUT_STRIDE + column,
        value,
        mask=(column < SEQUENCE_STRIDE) & tl.load(execute),
    )


@triton.jit
def _commit_beams_kernel(
    selected_tokens,
    selected_parents,
    selected_scores,
    selected_flat,
    next_sequence,
    sequence,
    scores,
    input_ids,
    parent_ids,
    active_mask,
    lengths,
    history_tokens,
    history_parents,
    generated,
    execute,
    WIDTH: tl.constexpr,
    SEQUENCE_STRIDE: tl.constexpr,
    NEXT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    PREFILL: tl.constexpr,
):
    child = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    run = tl.load(execute)
    live = tl.load(selected_flat + child) >= 0
    parent = tl.where(live, tl.load(selected_parents + child), 0)
    token = tl.where(live, tl.load(selected_tokens + child), 0)
    score = tl.where(live, tl.load(selected_scores + child), -float("inf"))
    step = 0 if PREFILL else tl.load(generated)
    value = tl.load(
        next_sequence + child * NEXT_STRIDE + column, mask=column < SEQUENCE_STRIDE, other=0
    )
    tl.store(
        sequence + child * SEQUENCE_STRIDE + column, value, mask=run & (column < SEQUENCE_STRIDE)
    )
    tl.store(scores + child, score, mask=run)
    tl.store(input_ids + child, token, mask=run)
    tl.store(parent_ids + child, parent, mask=run)
    tl.store(active_mask + child, live, mask=run)
    tl.store(lengths + child, tl.where(live, step + 1, 0), mask=run)
    tl.store(history_tokens + step * WIDTH + child, token, mask=run)
    tl.store(history_parents + step * WIDTH + child, parent, mask=run)


@triton.jit
def _commit_step_kernel(
    active_mask,
    generated,
    decode_steps,
    num_active,
    completed,
    initialized,
    finished,
    next_completed,
    execute,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    PREFILL: tl.constexpr,
):
    rank = tl.arange(0, BLOCK)
    count = tl.sum(tl.load(active_mask + rank, mask=rank < WIDTH, other=False).to(tl.int32))
    run = tl.load(execute)
    step = 1 if PREFILL else tl.load(generated) + 1
    decode = 0 if PREFILL else tl.load(decode_steps) + 1
    tl.store(generated, step, mask=run)
    tl.store(decode_steps, decode, mask=run)
    tl.store(num_active, count, mask=run)
    tl.store(completed, tl.load(next_completed), mask=run)
    tl.store(initialized, True, mask=run)
    tl.store(finished, count == 0, mask=run)


@triton.jit
def _pack_final_beams_kernel(
    selected_indices,
    completed_count,
    completed_tokens,
    completed_scores,
    completed_parents,
    completed_lengths,
    completed_reasons,
    history_tokens,
    history_parents,
    output,
    WIDTH: tl.constexpr,
    STEPS: tl.constexpr,
    RESULT_WIDTH: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    """Materialize selected completed paths directly into the final wire buffer."""

    rank = tl.program_id(0)
    count = tl.load(completed_count)
    candidate = tl.load(selected_indices + rank)
    valid = (rank < RESULT_WIDTH) & (candidate >= 0) & (candidate < count)
    length = tl.load(completed_lengths + candidate, mask=valid, other=0)
    token = tl.load(completed_tokens + candidate, mask=valid, other=0)
    parent = tl.load(completed_parents + candidate, mask=valid, other=0)
    score = tl.load(completed_scores + candidate, mask=valid, other=-float("inf"))
    reason = tl.load(completed_reasons + candidate, mask=valid, other=0)
    final_token = token
    row = 1 + rank * ROW_STRIDE

    tl.store(output, tl.minimum(count, RESULT_WIDTH), mask=rank == 0)
    tl.store(output + row + 2 * STEPS, score.to(tl.int32, bitcast=True))
    tl.store(output + row + 2 * STEPS + 1, length)
    tl.store(output + row + 2 * STEPS + 2, reason)
    tl.store(output + row + 2 * STEPS + 3, tl.where(reason == _FINISH_STOP, final_token, -1))

    # Walk the append-only lineage backwards. Every row is initialized first,
    # so short stop-completions cannot expose stale values from a previous run.
    columns = tl.arange(0, ROW_BLOCK)
    tl.store(output + row + columns, 0, mask=columns < 2 * STEPS)
    for offset in range(STEPS):
        position = length - 1 - offset
        write = valid & (position >= 0)
        tl.store(output + row + position, token, mask=write)
        tl.store(output + row + STEPS + position, parent, mask=write)
        previous = write & (position > 0)
        history_offset = (position - 1) * WIDTH + parent
        token = tl.load(history_tokens + history_offset, mask=previous, other=0)
        parent = tl.load(history_parents + history_offset, mask=previous, other=0)
