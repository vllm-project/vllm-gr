# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from typing import Any

import pytest
import torch

from vllm_gr.v1.beam.attention_context import (
    BeamAttentionContext,
    BeamDecisionBuffers,
)
from vllm_gr.v1.beam.session import BeamSearchSession
from vllm_gr.v1.worker.gpu.beam_state import BeamSearchState

# pytest's parametrization decorators are untyped.
# mypy: disable-error-code="misc"


def _decision() -> BeamDecisionBuffers:
    return BeamDecisionBuffers(2, 4, 8, torch.device("cpu"), torch.float16)


def test_state_aliases_existing_storage_and_accounts_only_new_memory() -> None:
    with pytest.raises(ValueError, match="batch row"):
        BeamSearchState(
            BeamDecisionBuffers(0, 4, 8, torch.device("cpu"), torch.float16),
            beam_width=4,
            max_new_tokens=8,
        )
    decision = _decision()
    state = BeamSearchState(decision, beam_width=3, max_new_tokens=5, stop_token_ids=(99,))
    assert state.sequence.data_ptr() == decision.sequence.data_ptr()
    assert state.beam_scores.data_ptr() == decision.beam_scores.data_ptr()
    assert state.sequence.shape == (2, 3, 8)
    assert state.history_token_ids.shape == (2, 5, 3)
    assert state.completed_capacity == 3 + 4 * 9
    assert state.stop_token_ids.tolist() == [[99], [99]]
    owned = state.owned_tensors()
    assert len({t.data_ptr() for t in owned}) == len(owned)
    assert state.device_memory_bytes == sum(t.numel() * t.element_size() for t in owned)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("beam_width", "candidate_width", "max_new_tokens", "expected"),
    [
        (4, None, 8, 4 + 7 * 16),  # unset K follows W: the historical bound
        (4, 4, 8, 4 + 7 * 16),  # explicit K == W is identical
        (4, 64, 1, 64),  # prefill contributes one row's candidates
        (4, 64, 3, 64 + 2 * 4 * 64),  # multi-step: W rows of K per step
        (4, 2, 3, 2 + 2 * 4 * 2),  # K < W still holds every finishing candidate
    ],
)
def test_completed_capacity_covers_every_finishing_candidate(
    beam_width: int,
    candidate_width: int | None,
    max_new_tokens: int,
    expected: int,
) -> None:
    """Every finishing candidate appends one record, so K is part of the bound.

    The pre-decoupling bound ``W + (N-1) * W**2`` assumed K == W; with ``K > W``
    a session overflows the completed buffer at the first step and the state
    reports ERROR_CAPACITY instead of producing a result.
    """
    state = BeamSearchState(
        _decision(),
        beam_width=beam_width,
        max_new_tokens=max_new_tokens,
        candidate_width=candidate_width,
    )

    assert state.completed_capacity == expected
    assert state.completed_token_ids.shape == (2, expected)
    resolved = beam_width if candidate_width is None else candidate_width
    assert state.candidate_width == resolved


def test_beam_state_rejects_a_non_positive_candidate_width() -> None:
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="candidate_width"):
            BeamSearchState(_decision(), beam_width=4, max_new_tokens=8, candidate_width=bad)


def test_reset_preserves_other_rows_configuration_and_addresses() -> None:
    decision = _decision()
    state = BeamSearchState(decision, beam_width=4, max_new_tokens=8, stop_token_ids=(7, 9))
    tensors = (*state.owned_tensors(), state.sequence, state.beam_scores)
    addresses = [t.data_ptr() for t in tensors]
    for tensor in tensors:
        tensor[1].fill_(1)
    other = [t[1].clone() for t in tensors]
    state.num_completed[0] = 9
    state.is_finished[0] = True
    state.reset(0)
    assert [t.data_ptr() for t in tensors] == addresses
    for tensor, expected in zip(tensors, other):
        torch.testing.assert_close(tensor[1], expected)
    assert state.num_completed[0] == 0
    assert not state.is_finished[0]
    assert state.stop_token_ids[0].tolist() == [7, 9]
    assert state.max_new_tokens_tensor[0] == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beam_width": 0},
        {"beam_width": True},
        {"beam_width": 5},
        {"max_new_tokens": 0},
        {"max_new_tokens": 9},
        {"stop_token_ids": (-1,)},
        {"stop_token_ids": (1, 1)},
    ],
)
def test_invalid_capacity_or_configuration(kwargs: dict[str, Any]) -> None:
    config = {"beam_width": 4, "max_new_tokens": 8} | kwargs
    with pytest.raises((ValueError, TypeError)):
        BeamSearchState(_decision(), **config)


def test_session_begin_can_reset_one_decision_row() -> None:
    context = BeamAttentionContext(2, 4, 1, 8, 8, 2, torch.device("cpu"), torch.float16)
    slot = context.registry.allocate("s", 4, 0)
    session = BeamSearchSession("s", slot, context)
    context.decision.sequence.fill_(17)
    context.decision.beam_scores.fill_(-3)
    context.decision.worker_decode_steps = 7
    session.begin_session(4, 0, decision_index=1)
    assert torch.all(context.decision.sequence[0] == 17)
    assert torch.all(context.decision.sequence[1] == 0)
    assert context.decision.worker_decode_steps == 7
    session.begin_session(4, 0)
    assert torch.all(context.decision.sequence == 0)
    assert context.decision.worker_decode_steps == 0


pytestmark = pytest.mark.cpu_test
