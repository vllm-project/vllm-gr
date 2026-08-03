# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm_gr.beam_search_step import apply_beam_search_step
from vllm_gr.entrypoints.beam_search_config import BeamSearchStep
from vllm_gr.entrypoints.openai import serving_engine


def _parent(tokens: list[int], score: float) -> SimpleNamespace:
    return SimpleNamespace(
        orig_prompt={"prompt_token_ids": [0]},
        tokens=tokens,
        cum_logprob=score,
        lora_request=None,
        get_prompt=lambda: {"prompt_token_ids": tokens},
    )


def _output(token_ids: list[int], scores: list[float]) -> SimpleNamespace:
    flat = SimpleNamespace(
        token_ids=token_ids,
        logprobs=scores,
        ranks=[1] * len(token_ids),
        decoded_tokens=[None] * len(token_ids),
        start_indices=[0],
        end_indices=[len(token_ids)],
    )
    return SimpleNamespace(outputs=[SimpleNamespace(logprobs=flat, finish_reason=None)])


def test_shared_step_selects_candidates_globally_across_parents() -> None:
    result = apply_beam_search_step(
        [_parent([0], 0.0), _parent([0, 9], -0.3)],
        [_output([1, 2], [-0.2, -0.4]), _output([3, 4], [-0.05, -0.5])],
        BeamSearchStep(expand_width=2, keep_width=2),
        SimpleNamespace(),
    )
    selected = sorted((beam.tokens[-1], beam.cum_logprob) for beam in result.active)
    assert [token_id for token_id, _ in selected] == [1, 3]
    assert [score for _, score in selected] == pytest.approx([-0.2, -0.35])
    assert result.completed == []


def test_shared_step_applies_parent_constraint_before_selection() -> None:
    result = apply_beam_search_step(
        [_parent([0], 0.0)],
        [_output([1, 2], [-0.1, -0.2])],
        BeamSearchStep(expand_width=2, keep_width=1),
        SimpleNamespace(),
        allowed_token_ids_by_parent=[{2}],
        catalog_strategy="null",
    )
    assert [beam.tokens[-1] for beam in result.active] == [2]


def test_shared_step_validates_engine_output_count_and_logprobs() -> None:
    parent = _parent([0], 0.0)
    step = BeamSearchStep(expand_width=1, keep_width=1)
    with pytest.raises(ValueError, match="one engine output"):
        apply_beam_search_step([parent], [], step, SimpleNamespace())
    with pytest.raises(ValueError, match="non-empty logprobs"):
        apply_beam_search_step(
            [parent],
            [SimpleNamespace(outputs=[SimpleNamespace(logprobs=None)])],
            step,
            SimpleNamespace(),
        )


def test_shared_step_splits_grouped_engine_output_and_returns_fork_route() -> None:
    parents = [_parent([0, 10], 0.0), _parent([0, 20], -0.5)]
    flat = SimpleNamespace(
        token_ids=[1, 1, 2, 3, 3, 4],
        logprobs=[-0.1, -0.1, -0.2, -0.05, -0.05, -0.3],
        ranks=[1] * 6,
        decoded_tokens=[None] * 6,
        start_indices=[0],
        end_indices=[6],
    )
    result = apply_beam_search_step(
        parents,
        [SimpleNamespace(outputs=[SimpleNamespace(logprobs=flat)])],
        BeamSearchStep(expand_width=2, keep_width=2),
        SimpleNamespace(),
        grouped_output=True,
    )
    assert [beam.tokens[-1] for beam in result.active] == [1, 2]
    assert result.fork_info == [(0, 1), (0, 2)]


def test_async_step_runner_reuses_grouped_beam_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_output = _output([1, 2], [-0.1, -0.2])
    grouped_flat = SimpleNamespace(
        token_ids=[3, 4, 5, 6],
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        ranks=[1] * 4,
        decoded_tokens=[None] * 4,
        start_indices=[0],
        end_indices=[4],
    )
    grouped_output = SimpleNamespace(
        outputs=[SimpleNamespace(logprobs=grouped_flat, finish_reason=None)]
    )
    add_step = AsyncMock(return_value=([first_output], ["prefill-id"]))
    beam_step = AsyncMock(return_value=([grouped_output], ["child-0", "child-1"]))
    cleanup = AsyncMock()
    monkeypatch.setattr(serving_engine, "_add_batch_step", add_step)
    monkeypatch.setattr(serving_engine, "_beam_request_step", beam_step)
    monkeypatch.setattr(serving_engine, "_beam_request_cleanup", cleanup)
    engine_client = SimpleNamespace(beam_request_step_update=AsyncMock())

    result = asyncio.run(
        serving_engine.run_beam_search_steps_async(
            engine_client,
            [_parent([0], 0.0)],
            [
                BeamSearchStep(expand_width=2, keep_width=2),
                BeamSearchStep(expand_width=2, keep_width=2),
            ],
            SimpleNamespace(),
            "request",
        )
    )

    assert add_step.await_count == 1
    assert beam_step.await_count == 1
    assert beam_step.call_args.kwargs["fork_info"] == [(0, 1), (0, 2)]
    cleanup.assert_awaited_once_with(
        engine_client,
        "prefill-id",
        ["child-0", "child-1"],
        None,
    )
    assert len(result) == 2
