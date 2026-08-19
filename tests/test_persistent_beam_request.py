# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
from types import MethodType, SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_gr.config import BeamControlMode
from vllm_gr.config.parser import BEAM_CONTROL_MODE_KEY
from vllm_gr.v1.beam.controller_types import BeamControllerRequest
from vllm_gr.v1.engine.core import _ensure_beam_state, create_controller
from vllm_gr.v1.engine.persistent_beam import (
    PersistentBeamStage,
    begin_persistent_finalization,
    complete_persistent_step,
    continue_persistent_request,
    finish_dispatched_persistent_drains,
    mark_persistent_request_free,
    mark_persistent_step_scheduled,
    mark_persistent_teardown_dispatched,
)
from vllm_gr.v1.engine.scheduler_metadata import build_beam_request_metadata


class _SchedulerHarness:
    def __init__(self) -> None:
        self.requests = {}
        self.num_waiting_for_streaming_input = 0
        self.log_stats = False
        self._beam_finished_session_ids = []
        self._update_request_as_session = MethodType(Scheduler._update_request_as_session, self)


class _PersistentEngine:
    def __init__(self) -> None:
        self.vllm_config = SimpleNamespace(
            additional_config={
                BEAM_CONTROL_MODE_KEY: BeamControlMode.ENGINE_PERSISTENT_REQUEST.value
            },
            scheduler_config=SimpleNamespace(async_scheduling=False),
        )
        self.scheduler = _SchedulerHarness()
        self.output_queue = queue.Queue()
        self.aborts_queue = queue.Queue()
        self.added_requests = []
        self.internal_add_updates = []
        self.beam_cache = {}
        self.beam_cache_lock = None
        self.create_controller = MethodType(create_controller, self)
        _ensure_beam_state(self)

    def add_request(self, request, current_wave: int) -> None:
        self.added_requests.append((request, current_wave))
        self.scheduler.requests[request.request_id] = request

    def _cache_beam_request(self, request) -> None:
        pytest.fail("persistent admission must not populate the Beam request cache")

    def _handle_beam_request_step_update(self, update) -> None:
        self.internal_add_updates.append(update)


def _start(session_id: str = "persistent-session", width: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        controller=BeamControllerRequest(
            beam_session_id=session_id,
            external_request_id=f"external-{session_id}",
            beam_width=width,
            max_tokens=3,
            prompt_token_ids=[10, 20],
            initial_tokens=[10, 20],
            prompt_length=2,
            eos_token_id=2,
            beam_control_mode=BeamControlMode.ENGINE_PERSISTENT_REQUEST.value,
        ),
        client_index=0,
        call_id=7,
        current_wave=0,
        data_parallel_rank=None,
        constraint_resource=None,
    )


def _request(session_id: str = "persistent-session") -> Request:
    return Request(
        request_id=session_id,
        prompt_token_ids=[10, 20],
        sampling_params=SamplingParams(max_tokens=1, logprobs=2),
        pooling_params=None,
        arrival_time=123.0,
    )


def _decision(step: int = 0, tokens: tuple[int, ...] = (31, 32)) -> SimpleNamespace:
    return SimpleNamespace(
        completed_step=step,
        next_tokens=list(tokens),
        parent_beam_indices=[0] * len(tokens),
        active_width=len(tokens),
        finished=False,
    )


def _scheduler_output(session_id: str = "persistent-session", width: int = 2):
    return SimpleNamespace(num_scheduled_tokens={session_id: width})


def _park_prefill(engine: _PersistentEngine, request: Request) -> None:
    output = _scheduler_output(request.request_id, 2)
    mark_persistent_step_scheduled(engine.scheduler, output)
    request.num_computed_tokens = 2
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    engine.scheduler.num_waiting_for_streaming_input = 1
    complete_persistent_step(engine.scheduler, output)


def test_persistent_continuation_reuses_one_request_without_internal_add(monkeypatch) -> None:
    engine = _PersistentEngine()
    request = _request()
    controller = create_controller(engine, _start(), request)

    assert len(engine.added_requests) == 1
    assert engine.scheduler.requests[request.request_id] is request
    assert request.resumable is True
    assert not engine.beam_cache
    state = request.gr_persistent_beam_state
    assert state.stage is PersistentBeamStage.PREFILLING
    _park_prefill(engine, request)
    assert state.stage is PersistentBeamStage.PARKED

    monkeypatch.setattr(
        "vllm.v1.request.Request",
        lambda *args, **kwargs: pytest.fail("continuation constructed a Request"),
    )
    continue_persistent_request(
        engine,
        request.request_id,
        _decision(),
        controller,
        request.sampling_params,
    )

    assert engine.scheduler.requests[request.request_id] is request
    assert len(engine.added_requests) == 1
    assert not engine.internal_add_updates
    assert request.status is RequestStatus.WAITING
    assert request.prompt_token_ids == [10, 20, 31, 32]
    assert request.num_computed_tokens == 2
    assert state.stage is PersistentBeamStage.BEAM_DECODING
    assert state.logical_step == 0
    assert state.next_epoch == 1

    metadata = build_beam_request_metadata(request, 2)
    assert metadata is not None
    assert metadata["cache_len"] == 2
    assert metadata["beam_stage"] == "beam_decoding"
    assert metadata["scheduled_width"] == metadata["active_width"] == 2


def test_parallel_scheduler_view_is_replaced_and_position_does_not_drift() -> None:
    engine = _PersistentEngine()
    request = _request()
    controller = create_controller(engine, _start(), request)
    _park_prefill(engine, request)

    for step, tokens in enumerate(((31, 32), (41, 42), (51, 52))):
        continue_persistent_request(
            engine,
            request.request_id,
            _decision(step, tokens),
            controller,
            request.sampling_params,
        )
        assert request.prompt_token_ids == [10, 20, *tokens]
        assert request.num_prompt_tokens == 4
        assert request.num_computed_tokens == 2

        output = _scheduler_output(width=2)
        mark_persistent_step_scheduled(engine.scheduler, output)
        request.num_computed_tokens = 4
        request._all_token_ids.append(999)
        request._output_token_ids.append(999)
        request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        engine.scheduler.num_waiting_for_streaming_input = 1
        complete_persistent_step(engine.scheduler, output)

    assert request.gr_persistent_beam_state.completed_epoch == 3


def test_epoch_and_finalization_side_effects_are_single_shot() -> None:
    engine = _PersistentEngine()
    request = _request()
    create_controller(engine, _start(), request)
    state = request.gr_persistent_beam_state
    output = _scheduler_output()

    mark_persistent_step_scheduled(engine.scheduler, output)
    with pytest.raises(RuntimeError, match="more than once"):
        mark_persistent_step_scheduled(engine.scheduler, output)
    request.num_computed_tokens = 2
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    complete_persistent_step(engine.scheduler, output)

    assert begin_persistent_finalization(engine, request.request_id)
    assert not begin_persistent_finalization(engine, request.request_id)
    assert state.stage is PersistentBeamStage.DRAINING
    assert engine.scheduler._beam_finished_session_ids == [request.request_id]
    assert engine.aborts_queue.get_nowait() == request.request_id

    mark_persistent_teardown_dispatched(engine.scheduler, [request.request_id])
    mark_persistent_request_free(request)
    assert state.worker_teardown_sent and state.scheduler_freed
    with pytest.raises(RuntimeError, match="more than once"):
        mark_persistent_request_free(request)


def test_admission_waits_for_worker_drain_before_reusing_slot() -> None:
    engine = _PersistentEngine()
    first = _request("first")
    create_controller(engine, _start("first", width=128), first)
    _park_prefill(engine, first)
    begin_persistent_finalization(engine, "first")

    second = _request("second")
    assert create_controller(engine, _start("second", width=1), second) is None
    assert engine.persistent_active_session_id == "first"
    assert "second" not in engine.scheduler.requests
    assert len(engine.persistent_pending_starts) == 1

    mark_persistent_teardown_dispatched(engine.scheduler, ["first"])
    mark_persistent_request_free(first)
    engine.scheduler.requests.pop("first")
    finish_dispatched_persistent_drains(engine)

    assert engine.persistent_active_session_id == "second"
    assert engine.scheduler.requests["second"] is second
    assert len(engine.added_requests) == 2
    assert not engine.persistent_pending_starts


def test_persistent_mode_rejects_async_scheduling_transactionally() -> None:
    engine = _PersistentEngine()
    engine.vllm_config.scheduler_config.async_scheduling = True
    with pytest.raises(NotImplementedError, match="async_scheduling"):
        create_controller(engine, _start(), _request())
    assert not engine.beam_controllers


def test_zero_step_persistent_controller_does_not_take_worker_slot() -> None:
    engine = _PersistentEngine()
    start = _start()
    start.controller.max_tokens = 0
    controller = create_controller(engine, start, _request())

    assert controller.requires_step is False
    assert not engine.added_requests
    assert engine.persistent_active_session_id is None
    assert not engine.beam_controllers
    _, outputs = engine.output_queue.get_nowait()
    assert outputs.utility_output.call_id == start.call_id
