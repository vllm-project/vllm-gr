# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.outputs import LogprobsLists

from vllm_gr.config.parser import BEAM_CONTROL_MODE_KEY
from vllm_gr.v1.beam.controller_types import BeamControllerRequest
from vllm_gr.v1.engine.core import (
    _ensure_beam_state,
    cancel_controller,
    clear_controllers,
    consume_step_output,
    create_controller,
    get_controller,
    remove_controller,
)


class _Engine:
    def __init__(self) -> None:
        self.vllm_config = SimpleNamespace(
            additional_config={BEAM_CONTROL_MODE_KEY: "engine_per_step_request"}
        )
        self.output_queue: queue.Queue = queue.Queue()
        self.beam_cleanup_updates = []
        self.added_requests = []
        self.beam_cache = {}
        self.beam_cache_lock = None
        _ensure_beam_state(self)

    def _cache_beam_request(self, request) -> None:
        self.beam_cache[request.request_id] = {"request": request}

    def add_request(self, request, current_wave: int) -> None:
        self.added_requests.append((request, current_wave))

    def _handle_beam_request_step_update(self, update) -> None:
        self.beam_cleanup_updates.append(update)


def _start(session_id: str = "session-0", *, max_tokens: int = 2):
    controller = BeamControllerRequest(
        beam_session_id=session_id,
        external_request_id="external-0",
        beam_width=2,
        max_tokens=max_tokens,
        prompt_token_ids=[10],
        initial_tokens=[10],
        prompt_length=1,
        eos_token_id=2,
    )
    return SimpleNamespace(
        controller=controller,
        client_index=3,
        call_id=17,
        current_wave=4,
        data_parallel_rank=None,
        constraint_resource=None,
    )


def _request(session_id: str = "session-0"):
    return SimpleNamespace(
        request_id=session_id,
        sampling_params=SamplingParams(logprobs=2, max_tokens=1),
        lora_request=None,
        cache_salt=None,
        trace_headers=None,
        priority=5,
    )


def test_registry_rejects_duplicate_and_unknown_sessions() -> None:
    engine = _Engine()
    start = _start()
    request = _request()

    controller = create_controller(engine, start, request)
    assert get_controller(engine, "session-0") is controller
    assert len(engine.added_requests) == 1
    with pytest.raises(ValueError, match="duplicate beam_session_id"):
        create_controller(engine, start, request)
    with pytest.raises(KeyError, match="unknown beam controller"):
        get_controller(engine, "missing")
    with pytest.raises(KeyError, match="unknown beam controller"):
        consume_step_output(engine, "missing", SimpleNamespace())


def test_request_admission_rejects_mode_mismatch() -> None:
    engine = _Engine()
    engine.vllm_config.additional_config.clear()

    with pytest.raises(ValueError, match="beam_control_mode is 'serving'"):
        create_controller(engine, _start(), _request())
    assert not engine.beam_controllers


def test_cancel_and_remove_cleanup_are_idempotent() -> None:
    engine = _Engine()
    create_controller(engine, _start(), _request())

    assert cancel_controller(engine, "session-0") is True
    assert "session-0" not in engine.beam_controllers
    assert engine.beam_cleanup_updates[-1].beam_width == 0
    assert cancel_controller(engine, "session-0") is False
    assert remove_controller(engine, "session-0") is False
    with pytest.raises(KeyError, match="unknown beam controller"):
        cancel_controller(engine, "never-seen")


def test_shutdown_clears_controller_and_resource_registries() -> None:
    engine = _Engine()
    create_controller(engine, _start(), _request())

    clear_controllers(engine)
    clear_controllers(engine)

    assert not engine.beam_controllers
    assert not engine.beam_controller_contexts
    assert not engine.beam_seen_session_ids
    assert not engine.beam_cancelled_session_ids
    assert len(engine.beam_constraint_resources) == 0


def test_zero_step_controller_returns_one_terminal_result() -> None:
    engine = _Engine()

    controller = create_controller(engine, _start(max_tokens=0), _request())

    assert controller.state.value == "finalized"
    assert "session-0" not in engine.beam_controllers
    client_index, outputs = engine.output_queue.get_nowait()
    assert client_index == 3
    assert outputs.utility_output.call_id == 17
    result = outputs.utility_output.result.result
    assert result["finish_reason"] == "completed"
    assert result["beam_session_id"] == "session-0"
    assert engine.output_queue.empty()


def _output(token_ids: list[list[int]], logprobs: list[list[float]]) -> EngineCoreOutput:
    return EngineCoreOutput(
        request_id="session-0",
        new_token_ids=[row[0] for row in token_ids],
        new_logprobs=LogprobsLists(
            logprob_token_ids=np.asarray(token_ids, dtype=np.int32),
            logprobs=np.asarray(logprobs, dtype=np.float32),
            sampled_token_ranks=np.ones(len(token_ids), dtype=np.int32),
        ),
    )


def test_consume_step_output_drives_internal_update_then_terminal_result() -> None:
    engine = _Engine()
    create_controller(engine, _start(), _request())

    first = consume_step_output(
        engine,
        "session-0",
        _output([[11, 11, 12]], [[-0.1, -0.1, -0.2]]),
    )
    assert first.finished is False
    internal_update = engine.beam_cleanup_updates[-1]
    assert internal_update.controller_owned is True
    assert internal_update.decode_step == 0
    assert internal_update.beam_tokens == [[11], [12]]

    second = consume_step_output(
        engine,
        "session-0",
        _output(
            [[21, 21, 22], [31, 31, 32]],
            [[-0.1, -0.1, -0.2], [-5.0, -5.0, -6.0]],
        ),
    )
    assert second.finished is True
    assert engine.beam_cleanup_updates[-1].beam_width == 0
    _, outputs = engine.output_queue.get_nowait()
    result = outputs.utility_output.result.result
    assert [beam["token_ids"] for beam in result["beams"]] == [[11, 21], [11, 22]]
