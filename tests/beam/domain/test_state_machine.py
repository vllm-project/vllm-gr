# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_gr.beam import (
    BeamCandidateSegment,
    BeamFinishedReason,
    BeamGroup,
    BeamRequestContext,
    BeamState,
    BeamStepInput,
    deduplicate_segment,
    expand_candidates,
    finalize_beam_group,
    run_beam_step,
)


def _step_input(
    *,
    include_stop_token: bool = True,
    ignore_eos: bool = False,
    allowed_token_ids: tuple[frozenset[int] | None, ...] | None = None,
) -> BeamStepInput:
    return BeamStepInput(
        group=BeamGroup(
            active=(
                BeamState(token_ids=(10,), cumulative_logprob=-0.5),
                BeamState(token_ids=(20,), cumulative_logprob=-0.2),
            )
        ),
        candidate_segments=(
            BeamCandidateSegment(
                token_ids=(2, 11),
                logprobs=(-0.1, -0.3),
                ranks=(1, 2),
                decoded_tokens=("</s>", "a"),
            ),
            BeamCandidateSegment(token_ids=(21, 22), logprobs=(-0.2, -0.4)),
        ),
        context=BeamRequestContext(
            beam_width=2,
            eos_token_id=2,
            include_stop_token=include_stop_token,
            ignore_eos=ignore_eos,
        ),
        allowed_token_ids=allowed_token_ids,
    )


def test_deduplicate_segment_removes_sampled_token_when_repeated_in_top_k() -> None:
    segment = deduplicate_segment(
        [7, 3, 7, 5],
        [-0.01, -0.1, -0.2, -0.3],
        [1, 2, 3, 4],
        ["seven", "three", "seven", "five"],
        start_offset=0,
        end_offset=4,
        candidate_count=3,
    )

    assert segment.token_ids == (3, 7, 5)
    assert segment.logprobs == (-0.1, -0.2, -0.3)
    assert segment.ranks == (2, 3, 4)
    assert segment.decoded_tokens == ("three", "seven", "five")


def test_deduplicate_segment_keeps_sampled_token_when_not_repeated() -> None:
    segment = deduplicate_segment(
        [7, 3, 4, 5],
        [-0.01, -0.1, -0.2, -0.3],
        None,
        None,
        start_offset=0,
        end_offset=4,
        candidate_count=3,
    )

    assert segment.token_ids == (7, 3, 4)
    assert segment.logprobs == (-0.01, -0.1, -0.2)


def test_deduplicate_segment_respects_flat_vector_offsets() -> None:
    segment = deduplicate_segment(
        [90, 91, 7, 3, 7],
        [-9.0, -9.1, -0.01, -0.1, -0.2],
        None,
        None,
        start_offset=2,
        end_offset=5,
        candidate_count=2,
    )

    assert segment.token_ids == (3, 7)


def test_expand_candidates_accumulates_parent_scores_and_metadata() -> None:
    candidates = expand_candidates(_step_input(ignore_eos=True))

    assert candidates[0].cumulative_logprob == -0.6
    assert candidates[0].rank == 1
    assert candidates[0].decoded_token == "</s>"
    assert candidates[2].cumulative_logprob == -0.4


def test_expand_candidates_applies_per_parent_catalog_mask() -> None:
    candidates = expand_candidates(
        _step_input(
            ignore_eos=True,
            allowed_token_ids=(frozenset({11}), frozenset({21})),
        )
    )

    assert [candidate.cumulative_logprob for candidate in candidates] == [
        -float("inf"),
        -0.8,
        -0.4,
        -float("inf"),
    ]


def test_run_step_finishes_eos_and_prunes_active_beams() -> None:
    result = run_beam_step(_step_input())

    assert [beam.token_ids for beam in result.group.active] == [(20, 21), (20, 22)]
    assert result.finished_this_step == result.group.completed
    assert result.finished_this_step[0].token_ids == (10, 2)
    assert result.finished_this_step[0].finish_reason is BeamFinishedReason.STOP
    assert result.finished_this_step[0].stop_reason == 2
    assert result.finished_candidates[0].parent_index == 0
    assert result.finished_candidates[0].token_id == 2
    assert [
        (item.parent_index, item.child_index, item.token_id) for item in result.parent_child_map
    ] == [
        (1, 0, 21),
        (1, 1, 22),
    ]


def test_run_step_can_omit_eos_from_completed_tokens() -> None:
    result = run_beam_step(_step_input(include_stop_token=False))

    assert result.finished_this_step[0].token_ids == (10,)
    assert result.finished_this_step[0].stop_reason == 2


def test_run_step_expands_eos_when_eos_is_ignored() -> None:
    result = run_beam_step(_step_input(ignore_eos=True))

    assert not result.finished_this_step
    assert [beam.token_ids for beam in result.group.active] == [(20, 21), (10, 2)]


def test_run_step_retains_previously_completed_beams() -> None:
    previous = BeamState(
        token_ids=(99, 2),
        cumulative_logprob=-0.1,
        finish_reason=BeamFinishedReason.STOP,
        stop_reason=2,
    )
    step_input = _step_input()
    step_input = BeamStepInput(
        group=BeamGroup(active=step_input.group.active, completed=(previous,)),
        candidate_segments=step_input.candidate_segments,
        context=step_input.context,
    )

    result = run_beam_step(step_input)

    assert result.group.completed[0] is previous
    assert len(result.group.completed) == 2


def test_run_step_returns_no_active_beams_when_catalog_masks_everything() -> None:
    result = run_beam_step(_step_input(allowed_token_ids=(frozenset(), frozenset())))

    assert result.group.active == ()
    assert result.parent_child_map == ()


def test_finalize_combines_completed_and_length_finished_beams() -> None:
    completed = BeamState(
        token_ids=(10, 2),
        cumulative_logprob=-1.2,
        finish_reason=BeamFinishedReason.STOP,
        stop_reason=2,
    )
    group = BeamGroup(
        active=(BeamState(token_ids=(10, 11, 12), cumulative_logprob=-0.9),),
        completed=(completed,),
    )

    final = finalize_beam_group(group, eos_token_id=2, beam_width=2)

    assert final[0].finish_reason is BeamFinishedReason.LENGTH
    assert final[1] is completed
