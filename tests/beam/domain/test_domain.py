# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_gr.beam import (
    BeamCandidateSegment,
    BeamFinishedReason,
    BeamGroup,
    BeamRequestContext,
    BeamState,
    BeamStepInput,
)


def test_beam_state_reports_finished_status() -> None:
    active = BeamState(token_ids=(1, 2))
    completed = BeamState(
        token_ids=(1, 2, 3),
        finish_reason=BeamFinishedReason.STOP,
        stop_reason=3,
    )

    assert not active.is_finished
    assert completed.is_finished


def test_beam_group_rejects_finished_active_beam() -> None:
    beam = BeamState(token_ids=(1,), finish_reason=BeamFinishedReason.LENGTH)

    with pytest.raises(ValueError, match="active beams"):
        BeamGroup(active=(beam,))


def test_beam_group_rejects_unfinished_completed_beam() -> None:
    with pytest.raises(ValueError, match="completed beams"):
        BeamGroup(active=(), completed=(BeamState(token_ids=(1,)),))


@pytest.mark.parametrize("beam_width", [0, -1])  # type: ignore[misc]
def test_request_context_rejects_non_positive_beam_width(beam_width: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        BeamRequestContext(beam_width=beam_width, eos_token_id=2)


@pytest.mark.parametrize("beam_width", [True, 2.0])  # type: ignore[misc]
def test_request_context_rejects_non_integer_beam_width(beam_width: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        BeamRequestContext(beam_width=beam_width, eos_token_id=2)


def test_candidate_segment_requires_aligned_metadata() -> None:
    with pytest.raises(ValueError, match="ranks"):
        BeamCandidateSegment(token_ids=(1, 2), logprobs=(-0.1, -0.2), ranks=(1,))


def test_step_input_requires_one_segment_per_active_beam() -> None:
    group = BeamGroup(active=(BeamState(token_ids=(1,)),))

    with pytest.raises(ValueError, match="candidate segment"):
        BeamStepInput(
            group=group,
            candidate_segments=(),
            context=BeamRequestContext(beam_width=1, eos_token_id=2),
        )
