# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure NumPy CPU reference implementation for Beam candidate selection.

The reference starts after candidate generation. It does not query a
Constraint Table, parse vLLM FlatLogprobs, or manipulate Beam KV buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeGuard

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class BeamItemCandidates:
    """Logical Beam parents for one item, independent of physical row order."""

    item_index: int
    parent_beam_indices: npt.NDArray[np.integer]
    token_ids: npt.NDArray[np.integer]
    logprobs: npt.NDArray[np.floating]
    valid_mask: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class BeamReferenceBeam:
    is_active: bool
    token_ids: tuple[int, ...]
    cumulative_logprob: np.float32
    parent_beam_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BeamReferenceCompletedBeam:
    token_ids: tuple[int, ...]
    cumulative_logprob: np.float32
    parent_beam_indices: tuple[int, ...]
    stop_token_id: int
    generation_step: int
    tie_break_index: int


@dataclass(frozen=True, slots=True)
class BeamReferenceItemState:
    item_index: int
    beams: tuple[BeamReferenceBeam, ...]
    num_active_beams: int
    completed_beams: tuple[BeamReferenceCompletedBeam, ...]


@dataclass(frozen=True, slots=True)
class BeamReferenceState:
    beam_width: int
    num_generated_tokens: int
    items: tuple[BeamReferenceItemState, ...]


@dataclass(frozen=True, slots=True)
class BeamReferenceOutputSequence:
    token_ids: tuple[int, ...]
    cumulative_logprob: float | None
    parent_beam_indices: tuple[int, ...]
    finish_reason: Literal["stop", "length"]
    stop_token_id: int | None


@dataclass(frozen=True, slots=True)
class BeamReferenceItemOutput:
    item_index: int
    sequences: tuple[BeamReferenceOutputSequence, ...]


@dataclass(frozen=True, slots=True)
class BeamReferenceBatchOutput:
    items: tuple[BeamReferenceItemOutput, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    parent_beam_index: int
    candidate_rank: int
    tie_break_index: int
    token_id: int
    cumulative_logprob: np.float32
    token_ids: tuple[int, ...]
    parent_beam_indices: tuple[int, ...]
    is_terminal: bool
    generation_step: int


@dataclass(frozen=True, slots=True)
class _OutputCandidate:
    token_ids: tuple[int, ...]
    cumulative_logprob: np.float32
    parent_beam_indices: tuple[int, ...]
    finish_reason: Literal["stop", "length"]
    stop_token_id: int | None
    generation_step: int
    tie_break_index: int


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(name: str, value: object) -> int:
    if not _is_int(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if not _is_int(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _matrix(name: str, value: object) -> npt.NDArray[np.generic]:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D rectangular array")
    return array


def _parent_beam_indices(
    value: object,
    *,
    num_rows: int,
) -> npt.NDArray[np.int64]:
    indices = np.asarray(value)
    if indices.ndim != 1 or indices.shape[0] != num_rows:
        raise ValueError("parent_beam_indices must be a 1-D array with one entry per row")
    if not np.issubdtype(indices.dtype, np.integer) or np.issubdtype(indices.dtype, np.bool_):
        raise ValueError("parent_beam_indices must contain integers")
    normalized = np.asarray(indices, dtype=np.int64)
    if np.any(normalized < 0):
        raise ValueError("parent_beam_indices must be non-negative")
    if np.unique(normalized).size != normalized.size:
        raise ValueError("parent_beam_indices must be unique")
    return normalized


def _normalize_common(
    *,
    item_index: int,
    parent_beam_indices: object,
    token_ids: object,
    logprobs: object,
    valid_mask: object,
) -> BeamItemCandidates:
    _non_negative_int("item_index", item_index)
    tokens = _matrix("token_ids", token_ids)
    scores = _matrix("logprobs", logprobs)
    mask = _matrix("valid_mask", valid_mask)
    if tokens.shape != scores.shape or tokens.shape != mask.shape:
        raise ValueError("token_ids, logprobs, and valid_mask must have equal shapes")
    if tokens.shape[1] == 0:
        raise ValueError("candidate width K must be positive")
    if not np.issubdtype(tokens.dtype, np.integer) or np.issubdtype(tokens.dtype, np.bool_):
        raise ValueError("token_ids must contain integers")
    if not np.issubdtype(scores.dtype, np.floating):
        raise ValueError("logprobs must contain floating-point values")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError("valid_mask must contain booleans")
    parents = _parent_beam_indices(
        parent_beam_indices,
        num_rows=tokens.shape[0],
    )
    return BeamItemCandidates(
        item_index=item_index,
        parent_beam_indices=parents,
        token_ids=np.asarray(tokens, dtype=np.int64),
        logprobs=np.asarray(scores, dtype=np.float32),
        valid_mask=np.asarray(mask, dtype=np.bool_),
    )


def normalize_sampler_candidates(
    *,
    item_index: int,
    parent_beam_indices: object,
    token_ids: object,
    logprobs: object,
    valid_mask: object | None = None,
) -> BeamItemCandidates:
    """Normalize sampled plus Top-K to exactly K candidates."""

    raw_tokens = _matrix("token_ids", token_ids)
    raw_logprobs = _matrix("logprobs", logprobs)
    if raw_tokens.shape != raw_logprobs.shape:
        raise ValueError("token_ids and logprobs must have equal shapes")
    if raw_tokens.shape[1] < 2:
        raise ValueError("sampler rows must contain sampled plus at least one Top-K slot")
    if valid_mask is None:
        raw_mask = np.isfinite(raw_logprobs)
    else:
        raw_mask = _matrix("valid_mask", valid_mask)
        if raw_mask.shape != raw_tokens.shape:
            raise ValueError("valid_mask must match the sampler shape")
        if not np.issubdtype(raw_mask.dtype, np.bool_):
            raise ValueError("valid_mask must contain booleans")

    num_rows, raw_width = raw_tokens.shape
    candidate_width = raw_width - 1
    out_tokens = np.empty((num_rows, candidate_width), dtype=np.int64)
    out_logprobs = np.empty((num_rows, candidate_width), dtype=np.float32)
    out_mask = np.empty((num_rows, candidate_width), dtype=np.bool_)
    for row in range(num_rows):
        sampled_token = raw_tokens[row, 0]
        sampled_in_topk = bool(
            np.any(
                (raw_tokens[row, 1:] == sampled_token)
                & np.asarray(raw_mask[row, 1:], dtype=np.bool_)
            )
        )
        selected = slice(1, raw_width) if sampled_in_topk else slice(0, candidate_width)
        out_tokens[row] = raw_tokens[row, selected]
        out_logprobs[row] = raw_logprobs[row, selected]
        out_mask[row] = raw_mask[row, selected]
    return _normalize_common(
        item_index=item_index,
        parent_beam_indices=parent_beam_indices,
        token_ids=out_tokens,
        logprobs=out_logprobs,
        valid_mask=out_mask,
    )


def normalize_constraint_candidates(
    *,
    item_index: int,
    parent_beam_indices: object,
    token_ids: object,
    logprobs: object,
    valid_mask: object,
) -> BeamItemCandidates:
    """Normalize Constraint TopK without sampled-slot removal or dedup."""

    return _normalize_common(
        item_index=item_index,
        parent_beam_indices=parent_beam_indices,
        token_ids=token_ids,
        logprobs=logprobs,
        valid_mask=valid_mask,
    )


def _inactive_beam() -> BeamReferenceBeam:
    return BeamReferenceBeam(
        is_active=False,
        token_ids=(),
        cumulative_logprob=np.float32(-np.inf),
        parent_beam_indices=(),
    )


def _initial_beam() -> BeamReferenceBeam:
    return BeamReferenceBeam(
        is_active=True,
        token_ids=(),
        cumulative_logprob=np.float32(0.0),
        parent_beam_indices=(),
    )


def _validate_candidate_batch(
    candidate_items: tuple[BeamItemCandidates, ...],
) -> None:
    if not isinstance(candidate_items, tuple) or not candidate_items:
        raise ValueError("candidate_items must be a non-empty tuple")
    for expected_index, candidates in enumerate(candidate_items):
        if not isinstance(candidates, BeamItemCandidates):
            raise TypeError("candidate_items must contain BeamItemCandidates")
        if candidates.item_index != expected_index:
            raise ValueError("candidate item_index values must be ordered and contiguous")
        _normalize_common(
            item_index=candidates.item_index,
            parent_beam_indices=candidates.parent_beam_indices,
            token_ids=candidates.token_ids,
            logprobs=candidates.logprobs,
            valid_mask=candidates.valid_mask,
        )


def _stop_ids(stop_token_ids: tuple[int, ...]) -> set[int]:
    result: set[int] = set()
    for index, token_id in enumerate(stop_token_ids):
        normalized = _non_negative_int(f"stop_token_ids[{index}]", token_id)
        if normalized in result:
            raise ValueError("stop_token_ids must be unique")
        result.add(normalized)
    return result


def _enumerate_candidates(
    candidates: BeamItemCandidates,
    *,
    parent_beams: tuple[BeamReferenceBeam, ...],
    generation_step: int,
    stop_ids: set[int],
    ignore_stop_tokens: bool,
) -> list[_Candidate]:
    candidate_width = candidates.token_ids.shape[1]
    values: list[_Candidate] = []
    for physical_row, parent_index_raw in enumerate(candidates.parent_beam_indices):
        parent_index = int(parent_index_raw)
        if parent_index >= len(parent_beams) or not parent_beams[parent_index].is_active:
            raise ValueError(f"parent_beam_index {parent_index} does not identify an active beam")
        parent = parent_beams[parent_index]
        for candidate_rank in range(candidate_width):
            if not bool(candidates.valid_mask[physical_row, candidate_rank]):
                continue
            token_id = int(candidates.token_ids[physical_row, candidate_rank])
            if token_id < 0:
                raise ValueError("valid candidate token_ids must be non-negative")
            logprob = np.float32(candidates.logprobs[physical_row, candidate_rank])
            if not np.isfinite(logprob):
                raise ValueError("valid candidate logprobs must be finite")
            cumulative_logprob = np.float32(parent.cumulative_logprob) + logprob
            if not np.isfinite(cumulative_logprob):
                raise ValueError("FP32 cumulative_logprob must remain finite")
            tie_break_index = parent_index * candidate_width + candidate_rank
            values.append(
                _Candidate(
                    parent_beam_index=parent_index,
                    candidate_rank=candidate_rank,
                    tie_break_index=tie_break_index,
                    token_id=token_id,
                    cumulative_logprob=np.float32(cumulative_logprob),
                    token_ids=parent.token_ids + (token_id,),
                    parent_beam_indices=parent.parent_beam_indices + (parent_index,),
                    is_terminal=(not ignore_stop_tokens and token_id in stop_ids),
                    generation_step=generation_step,
                )
            )
    values.sort(key=lambda candidate: candidate.tie_break_index)
    return values


def _score_order(candidates: list[_Candidate]) -> list[_Candidate]:
    if not candidates:
        return []
    scores = np.asarray(
        [candidate.cumulative_logprob for candidate in candidates],
        dtype=np.float32,
    )
    tie_break_indices = np.asarray(
        [candidate.tie_break_index for candidate in candidates],
        dtype=np.int64,
    )
    order = np.lexsort((tie_break_indices, -scores))
    return [candidates[int(index)] for index in order]


def _to_completed(candidate: _Candidate) -> BeamReferenceCompletedBeam:
    return BeamReferenceCompletedBeam(
        token_ids=candidate.token_ids,
        cumulative_logprob=candidate.cumulative_logprob,
        parent_beam_indices=candidate.parent_beam_indices,
        stop_token_id=candidate.token_id,
        generation_step=candidate.generation_step,
        tie_break_index=candidate.tie_break_index,
    )


def _next_item_state(
    *,
    item_index: int,
    beam_width: int,
    previous_completed: tuple[BeamReferenceCompletedBeam, ...],
    candidates: list[_Candidate],
) -> BeamReferenceItemState:
    terminal = [candidate for candidate in candidates if candidate.is_terminal]
    active = [candidate for candidate in candidates if not candidate.is_terminal]
    selected = _score_order(active)[:beam_width]
    # Score order decides survival. Stable parent grouping defines the
    # execution Beam order consumed by the next Beam KV reorder.
    selected = sorted(selected, key=lambda candidate: candidate.parent_beam_index)
    beams = [
        BeamReferenceBeam(
            is_active=True,
            token_ids=candidate.token_ids,
            cumulative_logprob=candidate.cumulative_logprob,
            parent_beam_indices=candidate.parent_beam_indices,
        )
        for candidate in selected
    ]
    beams.extend(_inactive_beam() for _ in range(beam_width - len(beams)))
    return BeamReferenceItemState(
        item_index=item_index,
        beams=tuple(beams),
        num_active_beams=len(selected),
        completed_beams=previous_completed
        + tuple(_to_completed(candidate) for candidate in terminal),
    )


def select_initial_beams(
    candidate_items: tuple[BeamItemCandidates, ...],
    *,
    beam_width: int,
    stop_token_ids: tuple[int, ...] = (),
    ignore_stop_tokens: bool = False,
) -> BeamReferenceState:
    """Select the first generated token from exactly one parent per item."""

    width = _positive_int("beam_width", beam_width)
    if not isinstance(ignore_stop_tokens, bool):
        raise TypeError("ignore_stop_tokens must be a bool")
    _validate_candidate_batch(candidate_items)
    stops = _stop_ids(stop_token_ids)
    item_states: list[BeamReferenceItemState] = []
    for candidates in candidate_items:
        if (
            candidates.parent_beam_indices.shape != (1,)
            or int(candidates.parent_beam_indices[0]) != 0
        ):
            raise ValueError("initial Beam selection requires exactly one parent beam index 0")
        values = _enumerate_candidates(
            candidates,
            parent_beams=(_initial_beam(),),
            generation_step=0,
            stop_ids=stops,
            ignore_stop_tokens=ignore_stop_tokens,
        )
        item_states.append(
            _next_item_state(
                item_index=candidates.item_index,
                beam_width=width,
                previous_completed=(),
                candidates=values,
            )
        )
    return BeamReferenceState(
        beam_width=width,
        num_generated_tokens=1,
        items=tuple(item_states),
    )


def _validate_state_candidates(
    state: BeamReferenceState,
    candidate_items: tuple[BeamItemCandidates, ...],
) -> None:
    if not isinstance(state, BeamReferenceState):
        raise TypeError("state must be BeamReferenceState")
    _positive_int("state.beam_width", state.beam_width)
    _positive_int("state.num_generated_tokens", state.num_generated_tokens)
    _validate_candidate_batch(candidate_items)
    if len(candidate_items) != len(state.items):
        raise ValueError("candidate batch size must match state")
    for candidates, item in zip(candidate_items, state.items):
        if candidates.item_index != item.item_index:
            raise ValueError("candidate item identity must match state")
        if len(item.beams) != state.beam_width:
            raise ValueError("state beams must have fixed beam_width")
        if not 0 <= item.num_active_beams <= state.beam_width:
            raise ValueError("state num_active_beams is invalid")
        expected = set(range(item.num_active_beams))
        actual = {int(index) for index in candidates.parent_beam_indices}
        if actual != expected:
            raise ValueError("Decode parent_beam_indices must cover every active execution beam")


def select_next_beams(
    state: BeamReferenceState,
    candidate_items: tuple[BeamItemCandidates, ...],
    *,
    stop_token_ids: tuple[int, ...] = (),
    ignore_stop_tokens: bool = False,
) -> BeamReferenceState:
    """Select one non-final Decode round and construct next execution beams."""

    if not isinstance(ignore_stop_tokens, bool):
        raise TypeError("ignore_stop_tokens must be a bool")
    _validate_state_candidates(state, candidate_items)
    stops = _stop_ids(stop_token_ids)
    next_items: list[BeamReferenceItemState] = []
    for candidates, item in zip(candidate_items, state.items):
        values = _enumerate_candidates(
            candidates,
            parent_beams=item.beams,
            generation_step=state.num_generated_tokens,
            stop_ids=stops,
            ignore_stop_tokens=ignore_stop_tokens,
        )
        next_items.append(
            _next_item_state(
                item_index=item.item_index,
                beam_width=state.beam_width,
                previous_completed=item.completed_beams,
                candidates=values,
            )
        )
    return BeamReferenceState(
        beam_width=state.beam_width,
        num_generated_tokens=state.num_generated_tokens + 1,
        items=tuple(next_items),
    )


def _output_order(candidates: list[_OutputCandidate]) -> list[_OutputCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.cumulative_logprob),
            candidate.generation_step,
            candidate.tie_break_index,
        ),
    )


def select_final_sequences(
    state: BeamReferenceState | None,
    candidate_items: tuple[BeamItemCandidates, ...],
    *,
    beam_width: int,
    num_return_sequences: int,
    stop_token_ids: tuple[int, ...] = (),
    ignore_stop_tokens: bool = False,
    include_stop_token: bool = True,
    include_cumulative_logprob: bool = True,
    include_parent_beam_indices: bool = True,
) -> BeamReferenceBatchOutput:
    """Select Final Top-R without constructing another Beam execution state."""

    width = _positive_int("beam_width", beam_width)
    num_sequences = _positive_int("num_return_sequences", num_return_sequences)
    if num_sequences > width:
        raise ValueError("num_return_sequences must be <= beam_width")
    for name, value in (
        ("ignore_stop_tokens", ignore_stop_tokens),
        ("include_stop_token", include_stop_token),
        ("include_cumulative_logprob", include_cumulative_logprob),
        ("include_parent_beam_indices", include_parent_beam_indices),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
    stops = _stop_ids(stop_token_ids)

    if state is None:
        _validate_candidate_batch(candidate_items)
        generation_step = 0
        parent_beams_by_item = [(_initial_beam(),) for _ in candidate_items]
        completed_by_item = [() for _ in candidate_items]
        for candidates in candidate_items:
            if (
                candidates.parent_beam_indices.shape != (1,)
                or int(candidates.parent_beam_indices[0]) != 0
            ):
                raise ValueError("one-token final selection requires one parent beam index 0")
    else:
        _validate_state_candidates(state, candidate_items)
        if state.beam_width != width:
            raise ValueError("beam_width must match state")
        generation_step = state.num_generated_tokens
        parent_beams_by_item = [item.beams for item in state.items]
        completed_by_item = [item.completed_beams for item in state.items]

    outputs: list[BeamReferenceItemOutput] = []
    for candidates, parent_beams, completed_beams in zip(
        candidate_items,
        parent_beams_by_item,
        completed_by_item,
    ):
        final_candidates = _enumerate_candidates(
            candidates,
            parent_beams=parent_beams,
            generation_step=generation_step,
            stop_ids=stops,
            ignore_stop_tokens=ignore_stop_tokens,
        )
        pool = [
            _OutputCandidate(
                token_ids=completed.token_ids,
                cumulative_logprob=completed.cumulative_logprob,
                parent_beam_indices=completed.parent_beam_indices,
                finish_reason="stop",
                stop_token_id=completed.stop_token_id,
                generation_step=completed.generation_step,
                tie_break_index=completed.tie_break_index,
            )
            for completed in completed_beams
        ]
        pool.extend(
            _OutputCandidate(
                token_ids=candidate.token_ids,
                cumulative_logprob=candidate.cumulative_logprob,
                parent_beam_indices=candidate.parent_beam_indices,
                finish_reason="stop" if candidate.is_terminal else "length",
                stop_token_id=candidate.token_id if candidate.is_terminal else None,
                generation_step=candidate.generation_step,
                tie_break_index=candidate.tie_break_index,
            )
            for candidate in final_candidates
        )

        selected = _output_order(pool)[:num_sequences]
        sequences: list[BeamReferenceOutputSequence] = []
        for candidate in selected:
            token_output = candidate.token_ids
            if candidate.finish_reason == "stop" and not include_stop_token:
                token_output = token_output[:-1]
            sequences.append(
                BeamReferenceOutputSequence(
                    token_ids=token_output,
                    cumulative_logprob=(
                        float(candidate.cumulative_logprob) if include_cumulative_logprob else None
                    ),
                    parent_beam_indices=(
                        candidate.parent_beam_indices if include_parent_beam_indices else ()
                    ),
                    finish_reason=candidate.finish_reason,
                    stop_token_id=candidate.stop_token_id,
                )
            )
        outputs.append(
            BeamReferenceItemOutput(
                item_index=candidates.item_index,
                sequences=tuple(sequences),
            )
        )
    return BeamReferenceBatchOutput(items=tuple(outputs))


__all__ = [
    "BeamItemCandidates",
    "BeamReferenceBatchOutput",
    "BeamReferenceBeam",
    "BeamReferenceCompletedBeam",
    "BeamReferenceItemOutput",
    "BeamReferenceItemState",
    "BeamReferenceOutputSequence",
    "BeamReferenceState",
    "normalize_constraint_candidates",
    "normalize_sampler_candidates",
    "select_final_sequences",
    "select_initial_beams",
    "select_next_beams",
]
