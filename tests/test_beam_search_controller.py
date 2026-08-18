# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from vllm_gr.constraints.trie_resource import TrieConstraintCompiler
from vllm_gr.v1.beam.beam_search_controller import BeamSearchController
from vllm_gr.v1.beam.controller_types import (
    BeamControllerRequest,
    BeamControllerStepOutput,
)


def _request(**overrides) -> BeamControllerRequest:
    values = {
        "beam_session_id": "beam-session-0",
        "external_request_id": "request-0",
        "beam_width": 2,
        "max_tokens": 2,
        "prompt_token_ids": [10],
        "initial_tokens": [10],
        "prompt_length": 1,
        "eos_token_id": 2,
    }
    values.update(overrides)
    return BeamControllerRequest(**values)


def _step(token_ids, logprobs) -> BeamControllerStepOutput:
    return BeamControllerStepOutput(
        token_ids=token_ids,
        logprobs=logprobs,
        ranks=list(range(1, len(token_ids) + 1)),
        decoded_tokens=[None] * len(token_ids),
    )


def test_controller_runs_two_steps_and_finalizes_history() -> None:
    controller = BeamSearchController.from_request(_request())
    first = controller.consume_output(_step([11, 11, 12], [-0.1, -0.1, -0.2]))
    assert first.finished is False
    assert first.next_tokens == [11, 12]
    assert first.parent_beam_indices == [0, 0]

    second = controller.consume_output(
        _step(
            [21, 21, 22, 31, 31, 32],
            [-0.1, -0.1, -0.2, -5.0, -5.0, -6.0],
        )
    )
    assert second.finished is True
    result = controller.finalize()
    assert [beam.token_ids for beam in result.beams] == [[11, 21], [11, 22]]
    assert [beam.cumulative_logprob for beam in result.beams] == pytest.approx([-0.2, -0.3])
    assert all(len(beam.logprobs["start_indices"]) == 2 for beam in result.beams)
    with pytest.raises(RuntimeError, match="terminal"):
        controller.consume_output(_step([], []))


def test_controller_completes_eos_before_topk() -> None:
    controller = BeamSearchController.from_request(_request(max_tokens=1))
    decision = controller.consume_output(_step([2, 2, 11], [-0.01, -0.01, -0.2]))
    assert decision.finished is True
    result = controller.finalize()
    assert result.beams[0].finish_reason == "stop"
    assert result.beams[0].stop_reason == 2
    assert result.beams[0].token_ids == []


def test_zero_execution_steps_finalize_without_worker_output() -> None:
    controller = BeamSearchController.from_request(_request(max_tokens=1, end_token_id=3))
    assert controller.requires_step is False
    result = controller.finalize()
    assert result.beams[0].token_ids == [3]


def test_controller_rejects_unsupported_constraint_backend() -> None:
    with pytest.raises(NotImplementedError, match="constraint_table"):
        BeamSearchController.from_request(_request(constraint_backend="constraint_table"))


class _Tokenizer:
    ids: ClassVar = {"<begin>": 1, "a": 11, "b": 12, "<end>": 3}

    def convert_tokens_to_ids(self, tokens):
        return [self.ids.get(token, -1) for token in tokens]


def test_controller_owns_trie_state_and_null_filtering(tmp_path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([["<begin>", "a", "<end>"]]))
    resource = TrieConstraintCompiler.from_catalog(_Tokenizer(), str(catalog))
    controller = BeamSearchController.from_request(
        _request(
            max_tokens=3,
            initial_tokens=[10, 1],
            begin_token_id=1,
            catalog_strategy="null",
            constraint_resource_id=resource.digest,
        ),
        resource,
    )
    first = controller.consume_output(_step([12, 12, 11], [-0.01, -0.01, -0.2]))
    assert first.next_tokens == [11]
    assert controller.active_constraint_states == [
        resource.advance(resource.advance(resource.root_state(), 1), 11)
    ]
