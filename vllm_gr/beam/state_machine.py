# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure beam candidate expansion and state transitions."""

from collections.abc import Sequence

from vllm_gr.beam.domain import (
    BeamCandidate,
    BeamCandidateSegment,
    BeamFinishedReason,
    BeamGroup,
    BeamParentChild,
    BeamState,
    BeamStepInput,
    BeamStepResult,
)
from vllm_gr.beam.pruning import prune_candidates
from vllm_gr.beam.ranking import rank_states


def deduplicate_segment(
    raw_token_ids: Sequence[int],
    raw_logprobs: Sequence[float],
    raw_ranks: Sequence[int | None] | None,
    raw_decoded_tokens: Sequence[str | None] | None,
    *,
    start_offset: int,
    end_offset: int,
    candidate_count: int,
) -> BeamCandidateSegment:
    """Extract one flat-logprob segment and remove a duplicate sampled token.

    Flat logprobs may contain the sampled token at index zero in addition to
    the requested top-K values. This preserves the existing GR extraction
    rule while returning an immutable, transport-independent value.
    """

    if candidate_count <= 0:
        raise ValueError("candidate_count must be greater than zero")
    if start_offset < 0 or end_offset < start_offset:
        raise ValueError("invalid segment offsets")

    token_ids = tuple(raw_token_ids[start_offset:end_offset])
    logprobs = tuple(raw_logprobs[start_offset:end_offset])
    ranks = tuple(raw_ranks[start_offset:end_offset]) if raw_ranks is not None else None
    decoded_tokens = (
        tuple(raw_decoded_tokens[start_offset:end_offset])
        if raw_decoded_tokens is not None
        else None
    )
    segment = BeamCandidateSegment(token_ids, logprobs, ranks, decoded_tokens)

    selected = slice(0, candidate_count)
    if len(segment.token_ids) > candidate_count:
        sampled_token = segment.token_ids[0]
        if sampled_token in segment.token_ids[1 : candidate_count + 1]:
            selected = slice(1, candidate_count + 1)

    return BeamCandidateSegment(
        token_ids=segment.token_ids[selected],
        logprobs=segment.logprobs[selected],
        ranks=segment.ranks[selected] if segment.ranks is not None else None,
        decoded_tokens=(
            segment.decoded_tokens[selected] if segment.decoded_tokens is not None else None
        ),
    )


def expand_candidates(step_input: BeamStepInput) -> tuple[BeamCandidate, ...]:
    """Expand per-parent segments and apply optional catalog constraints."""

    candidates: list[BeamCandidate] = []
    for parent_index, (parent, segment) in enumerate(
        zip(step_input.group.active, step_input.candidate_segments)
    ):
        allowed = (
            None
            if step_input.allowed_token_ids is None
            else step_input.allowed_token_ids[parent_index]
        )
        for candidate_index, (token_id, token_logprob) in enumerate(
            zip(segment.token_ids, segment.logprobs)
        ):
            if allowed is not None and token_id not in allowed:
                token_logprob = -float("inf")
            candidates.append(
                BeamCandidate(
                    parent_index=parent_index,
                    token_id=token_id,
                    token_logprob=token_logprob,
                    cumulative_logprob=parent.cumulative_logprob + token_logprob,
                    rank=segment.ranks[candidate_index] if segment.ranks is not None else None,
                    decoded_token=(
                        segment.decoded_tokens[candidate_index]
                        if segment.decoded_tokens is not None
                        else None
                    ),
                )
            )
    return tuple(candidates)


def _extend_state(
    parent: BeamState,
    candidate: BeamCandidate,
    *,
    finish_reason: BeamFinishedReason | None = None,
    stop_reason: int | str | None = None,
    append_token: bool = True,
) -> BeamState:
    token_ids = parent.token_ids + (candidate.token_id,) if append_token else parent.token_ids
    return BeamState(
        token_ids=token_ids,
        cumulative_logprob=candidate.cumulative_logprob,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )


def run_beam_step(step_input: BeamStepInput) -> BeamStepResult:
    """Apply candidate expansion, EOS handling, pruning, and state update."""

    context = step_input.context
    candidates = expand_candidates(step_input)
    finished: list[BeamState] = []
    finished_candidates: list[BeamCandidate] = []
    expandable: list[BeamCandidate] = []

    for candidate in candidates:
        if not context.ignore_eos and candidate.token_id == context.eos_token_id:
            if candidate.cumulative_logprob != -float("inf"):
                finished_candidates.append(candidate)
                finished.append(
                    _extend_state(
                        step_input.group.active[candidate.parent_index],
                        candidate,
                        finish_reason=BeamFinishedReason.STOP,
                        stop_reason=context.eos_token_id,
                        append_token=context.include_stop_token,
                    )
                )
        else:
            expandable.append(candidate)

    selected = prune_candidates(expandable, context.beam_width)
    active = tuple(
        _extend_state(step_input.group.active[candidate.parent_index], candidate)
        for candidate in selected
    )
    parent_child_map = tuple(
        BeamParentChild(candidate.parent_index, child_index, candidate.token_id)
        for child_index, candidate in enumerate(selected)
    )
    group = BeamGroup(
        active=active,
        completed=step_input.group.completed + tuple(finished),
    )
    return BeamStepResult(
        group=group,
        selected_candidates=selected,
        finished_candidates=tuple(finished_candidates),
        parent_child_map=parent_child_map,
        finished_this_step=tuple(finished),
    )


def finalize_beam_group(
    group: BeamGroup,
    *,
    eos_token_id: int,
    beam_width: int,
    length_penalty: float = 1.0,
    prompt_length: int = 0,
) -> tuple[BeamState, ...]:
    """Mark remaining beams length-finished and return the best final beams."""

    if beam_width <= 0:
        raise ValueError("beam_width must be greater than zero")
    length_finished = tuple(
        BeamState(
            token_ids=beam.token_ids,
            cumulative_logprob=beam.cumulative_logprob,
            finish_reason=BeamFinishedReason.LENGTH,
        )
        for beam in group.active
    )
    return rank_states(
        group.completed + length_finished,
        eos_token_id=eos_token_id,
        length_penalty=length_penalty,
        prompt_length=prompt_length,
    )[:beam_width]
