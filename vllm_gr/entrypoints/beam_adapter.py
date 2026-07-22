# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adapters between vLLM beam objects and the pure GR beam domain."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.logprobs import FlatLogprobs

from vllm_gr.beam import (
    BeamCandidateSegment,
    BeamGroup,
    BeamRequestContext,
    BeamState,
    BeamStepInput,
    deduplicate_segment,
    length_penalized_score,
    run_beam_step,
)

FlatStepData = tuple[
    list[int],
    list[float],
    list[int | None],
    list[str | None],
]


class FlatLogprobsView(Protocol):
    """Structural view needed to split vLLM flat logprob payloads."""

    start_indices: list[int]
    end_indices: list[int]
    token_ids: list[int]
    logprobs: list[float]
    ranks: list[int | None]
    decoded_tokens: list[str | None]


@dataclass(slots=True)
class VllmBeamStepResult:
    """Materialized vLLM objects produced by one pure domain transition."""

    active: list[BeamSearchSequence]
    completed: list[BeamSearchSequence]
    fork_info: list[tuple[int, int]]


def candidate_segment_from_flat(
    flat: FlatLogprobsView,
    candidate_count: int,
    *,
    start_offset: int | None = None,
    end_offset: int | None = None,
) -> BeamCandidateSegment:
    """Extract and deduplicate one candidate segment from FlatLogprobs."""

    if start_offset is None:
        start_offset = flat.start_indices[0]
    if end_offset is None:
        end_offset = flat.end_indices[0]
    return deduplicate_segment(
        flat.token_ids,
        flat.logprobs,
        flat.ranks,
        flat.decoded_tokens,
        start_offset=start_offset,
        end_offset=end_offset,
        candidate_count=candidate_count,
    )


def candidate_segments_from_combined_flat(
    flat: FlatLogprobsView,
    parent_count: int,
    candidate_count: int,
) -> tuple[BeamCandidateSegment, ...]:
    """Split grouped FlatLogprobs into one segment per active parent beam."""

    if parent_count <= 0:
        raise ValueError("parent_count must be greater than zero")
    total_logprobs = len(flat.token_ids)
    if total_logprobs % parent_count != 0:
        raise ValueError("grouped beam logprobs must divide evenly across active parent beams")
    stride = total_logprobs // parent_count
    return tuple(
        candidate_segment_from_flat(
            flat,
            candidate_count,
            start_offset=parent_index * stride,
            end_offset=(parent_index + 1) * stride,
        )
        for parent_index in range(parent_count)
    )


def _to_state(sequence: BeamSearchSequence) -> BeamState:
    return BeamState(
        token_ids=tuple(sequence.tokens),
        cumulative_logprob=sequence.cum_logprob,
    )


def _to_flat_step_data(
    segment: BeamCandidateSegment,
    allowed_token_ids: frozenset[int] | None,
) -> FlatStepData:
    logprobs = [
        logprob if allowed_token_ids is None or token_id in allowed_token_ids else -float("inf")
        for token_id, logprob in zip(segment.token_ids, segment.logprobs)
    ]
    size = len(segment.token_ids)
    return (
        list(segment.token_ids),
        logprobs,
        list(segment.ranks) if segment.ranks is not None else [None] * size,
        list(segment.decoded_tokens) if segment.decoded_tokens is not None else [None] * size,
    )


def _materialize_sequence(
    parent: BeamSearchSequence,
    state: BeamState,
    segment: BeamCandidateSegment,
    initial_logprobs: FlatLogprobs,
    allowed_token_ids: frozenset[int] | None,
) -> BeamSearchSequence:
    sequence = BeamSearchSequence(
        orig_prompt=parent.orig_prompt,
        tokens=list(state.token_ids),
        logprobs=initial_logprobs,  # type: ignore[arg-type]
        lora_request=parent.lora_request,
        cum_logprob=state.cumulative_logprob,
        finish_reason=state.finish_reason.value if state.finish_reason is not None else None,
        stop_reason=state.stop_reason,
    )
    setattr(sequence, "_lp_parent", parent)
    setattr(sequence, "_lp_step_data", _to_flat_step_data(segment, allowed_token_ids))
    return sequence


def apply_vllm_beam_step(
    active_beams: Sequence[BeamSearchSequence],
    candidate_segments: Sequence[BeamCandidateSegment],
    context: BeamRequestContext,
    initial_logprobs: FlatLogprobs,
    *,
    allowed_token_ids: Sequence[frozenset[int] | None] | None = None,
) -> VllmBeamStepResult:
    """Run a pure beam step and restore vLLM sequence/logprob metadata."""

    allowed = tuple(allowed_token_ids) if allowed_token_ids is not None else None
    step_result = run_beam_step(
        BeamStepInput(
            group=BeamGroup(active=tuple(_to_state(beam) for beam in active_beams)),
            candidate_segments=tuple(candidate_segments),
            context=context,
            allowed_token_ids=allowed,
        )
    )

    active = [
        _materialize_sequence(
            active_beams[candidate.parent_index],
            state,
            candidate_segments[candidate.parent_index],
            initial_logprobs,
            None if allowed is None else allowed[candidate.parent_index],
        )
        for candidate, state in zip(step_result.selected_candidates, step_result.group.active)
    ]
    completed = [
        _materialize_sequence(
            active_beams[candidate.parent_index],
            state,
            candidate_segments[candidate.parent_index],
            initial_logprobs,
            None if allowed is None else allowed[candidate.parent_index],
        )
        for candidate, state in zip(
            step_result.finished_candidates,
            step_result.finished_this_step,
        )
    ]
    fork_info = [
        (mapping.parent_index, mapping.token_id) for mapping in step_result.parent_child_map
    ]
    return VllmBeamStepResult(active=active, completed=completed, fork_info=fork_info)


def select_best_beams(
    beams: Sequence[BeamSearchSequence],
    context: BeamRequestContext,
) -> list[BeamSearchSequence]:
    """Return final vLLM beams using the shared length-penalized score."""

    return sorted(
        beams,
        key=lambda beam: length_penalized_score(
            tuple(beam.tokens),
            beam.cum_logprob,
            context.eos_token_id,
            context.length_penalty,
            prompt_length=context.prompt_length,
        ),
        reverse=True,
    )[: context.beam_width]
