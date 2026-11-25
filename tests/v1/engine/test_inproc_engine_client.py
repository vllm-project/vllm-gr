# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""In-process EngineCore terminal handling and failure isolation."""

from __future__ import annotations

import asyncio
import gc
import weakref
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm")

from vllm.v1.engine import EngineCoreOutputs  # noqa: E402
from vllm.v1.engine.core import EngineCore  # noqa: E402
from vllm.v1.engine.core_client import InprocClient  # noqa: E402
from vllm.v1.engine.exceptions import EngineDeadError  # noqa: E402

from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchOutput,
    BeamBatchParams,
    BeamBatchResult,
    BeamOutputOptions,
)
from vllm_gr.v1.beam.dispatch import (
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    GRDeviceStateRef,
    GRStageKind,
    GRStageMetadata,
    GRWorkerResult,
)
from vllm_gr.v1.core.sched.beam_session import GRSessionStatus
from vllm_gr.v1.engine.core import install_inproc_gr_terminal_step
from vllm_gr.v1.engine.core_client import inproc_engine_failure
from vllm_gr.v1.engine.core_client_patch import apply_engine_client_patches


class _FakeCore:
    """EngineCore stand-in whose every step raises ``error`` when one is given."""

    def __init__(self, error: BaseException | None) -> None:
        self.error = error
        self.steps = 0
        self.added: list[Any] = []
        self.aborted: list[Any] = []
        self.submitted: list[Any] = []

    def step_fn(self) -> tuple[Any, bool]:
        self.steps += 1
        if self.error is not None:
            raise self.error
        return {}, True

    def post_step(self, model_executed: bool) -> None:
        del model_executed

    def preprocess_add_request(self, request: Any) -> tuple[Any, int]:
        return request, 0

    def add_request(self, request: Any, request_wave: int = 0) -> None:
        del request_wave
        self.added.append(request)

    def abort_requests(self, request_ids: Any) -> None:
        self.aborted.append(request_ids)

    def _handle_submit_gr_session(self, batch_request: Any) -> None:
        self.submitted.append(batch_request)

    def shutdown(self) -> None:
        self.shut_down = True


def _client(error: BaseException | None) -> tuple[Any, _FakeCore]:
    apply_engine_client_patches()
    core = _FakeCore(error)
    client = object.__new__(InprocClient)
    client.engine_core = core
    return client, core


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.cpu_test  # type: ignore[misc]
@pytest.mark.parametrize("device_type", ["cuda", "npu"])  # type: ignore[misc]
@pytest.mark.parametrize("blocked_by", [None, "pending", "abort", "other_request"])  # type: ignore[misc]
def test_inproc_terminal_step_checks_readiness_and_work(
    monkeypatch: pytest.MonkeyPatch, device_type: str, blocked_by: str | None
) -> None:
    client, _ = _client(None)
    core = object.__new__(EngineCore)
    client.engine_core = core
    core.vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device=SimpleNamespace(type=device_type))
    )
    core.post_step = Mock()
    core.capture_iteration_details = Mock(return_value=nullcontext())
    core.log_error_detail = Mock(return_value=nullcontext())
    core._process_aborts_queue = Mock()
    core._attach_iteration_details = Mock()
    native_output = EngineCoreOutputs()
    native_step = Mock()

    def step(_core: EngineCore) -> tuple[dict[int, EngineCoreOutputs], bool]:
        native_step()
        return {0: native_output}, True

    monkeypatch.setattr(EngineCore, "step_with_batch_queue", step)
    core.step_fn = core.step_with_batch_queue

    session_id = "session"
    state = GRDeviceStateRef(session_id, 1)
    stage = GRStageMetadata(
        session_id=session_id,
        dispatch_id=1,
        stage_index=1,
        kind=GRStageKind.DECODE,
        decode_step=0,
        is_last_stage=True,
        input_device_state=GRDeviceStateRef(session_id, 0),
        output_device_state=state,
        beam_params=BeamBatchParams(2, 2, 0.0, (), True),
        output_options=BeamOutputOptions(2, False, True, False),
        num_prompt_tokens=2,
    )
    scheduler_output = SimpleNamespace(**{GR_STAGE_OUTPUT_ATTR: {session_id: stage}})
    terminal = BeamBatchResult(BEAM_BATCH_PROTOCOL_VERSION, session_id, BeamBatchOutput(()), None)
    receipt = GRWorkerResult(session_id, 1, 1, 1, state, True, terminal, resources_released=True)
    model_output = SimpleNamespace(**{GR_WORKER_RESULT_ATTR: {session_id: receipt}})
    future: Future[SimpleNamespace] = Future()
    if blocked_by == "pending":
        monkeypatch.setattr(
            future, "result", Mock(side_effect=AssertionError("must not wait for a pending result"))
        )
    else:
        future.set_result(model_output)
    core.batch_queue = deque([(future, scheduler_output, future)])
    core.aborts_queue = Queue()
    if blocked_by == "abort":
        core.aborts_queue.put_nowait([session_id])

    request = object()
    session = SimpleNamespace(
        status=GRSessionStatus.RUNNING,
        owner_generation=stage.owner_generation,
        _in_flight_dispatches={stage.dispatch_id},
        _dispatch_records={stage.dispatch_id: stage},
        request=request,
    )
    published = EngineCoreOutputs()
    core.scheduler = SimpleNamespace(
        _gr_v1_sessions={session_id: session},
        _gr_owner=session_id,
        requests={"r": SimpleNamespace(gr_batch_request=request)},
        update_from_output=Mock(return_value={0: published}),
    )
    if blocked_by == "other_request":
        core.scheduler.requests["other"] = SimpleNamespace()
    install_inproc_gr_terminal_step(core)

    if blocked_by is None:
        assert client.get_output() is published
        assert not core.batch_queue
        native_step.assert_not_called()
        core._process_aborts_queue.assert_called_once()
        core.scheduler.update_from_output.assert_called_once_with(scheduler_output, model_output)
        core._attach_iteration_details.assert_called_once_with({0: published}, None)
        core.post_step.assert_called_once_with(model_executed=False)
    else:
        assert client.get_output() is native_output
        assert len(core.batch_queue) == 1
        native_step.assert_called_once()
        core.scheduler.update_from_output.assert_not_called()
        core.post_step.assert_called_once_with(model_executed=True)


def test_a_failed_step_marks_the_inproc_engine_dead() -> None:
    oom = RuntimeError("CUDA out of memory (test)")
    client, core = _client(oom)

    with pytest.raises(EngineDeadError) as raised:
        client.get_output()
    assert raised.value.__cause__ is oom
    assert inproc_engine_failure(client) is oom
    assert core.steps == 1

    # Every later entry point fails fast, keeps the root cause, and never
    # reaches EngineCore again.
    for call in (
        client.get_output,
        lambda: client.add_request(SimpleNamespace(request_id="next")),
        lambda: client.submit_gr_session(object()),
    ):
        with pytest.raises(EngineDeadError) as raised:
            call()
        assert raised.value.__cause__ is oom
    # Like SyncMPClient, abort on a dead engine sends nothing and does not raise.
    assert client.abort_requests(["session-0"]) is None
    assert core.steps == 1
    assert core.added == core.aborted == core.submitted == []


def test_a_healthy_inproc_engine_is_unchanged() -> None:
    client, core = _client(None)

    client.get_output()
    client.get_output()
    client.add_request(SimpleNamespace(request_id="next"))
    client.abort_requests(["session-0"])
    client.abort_requests([])
    batch = object()
    client.submit_gr_session(batch)

    assert core.steps == 2
    assert [request.request_id for request in core.added] == ["next"]
    # An empty abort stays a no-op, as in the native clients.
    assert core.aborted == [["session-0"]]
    assert core.submitted == [batch]
    assert inproc_engine_failure(client) is None


def test_a_dead_engine_does_not_affect_another_client() -> None:
    dead, _ = _client(RuntimeError("boom"))
    healthy, healthy_core = _client(None)

    with pytest.raises(EngineDeadError):
        dead.get_output()
    healthy.get_output()

    assert inproc_engine_failure(dead) is not None
    assert inproc_engine_failure(healthy) is None
    assert healthy_core.steps == 1


def test_installing_the_patches_twice_wraps_each_method_once() -> None:
    names = ("get_output", "add_request", "abort_requests", "submit_gr_session", "shutdown")
    apply_engine_client_patches()
    first = {name: getattr(InprocClient, name) for name in names}
    # Without the identity check, the second install would record the guard
    # itself as the native method and every call would recurse.
    apply_engine_client_patches()
    assert {name: getattr(InprocClient, name) for name in names} == first

    client, core = _client(RuntimeError("boom"))
    with pytest.raises(EngineDeadError):
        client.get_output()
    assert core.steps == 1


class _Probe:
    """Stands in for a tensor that a failing step holds as a local variable."""


def test_shutdown_releases_the_failure_frames_but_keeps_the_cause() -> None:
    probe_ref: list[weakref.ref[_Probe]] = []

    class _LeakyCore(_FakeCore):
        def step_fn(self) -> tuple[Any, bool]:
            logits = _Probe()  # noqa: F841 - kept alive only by this frame
            probe_ref.append(weakref.ref(logits))
            raise RuntimeError("CUDA out of memory (test)")

    apply_engine_client_patches()
    client = object.__new__(InprocClient)
    client.engine_core = _LeakyCore(None)
    try:
        client.get_output()
    except EngineDeadError:
        pass
    gc.collect()
    # The recorded failure keeps the failing frame, and its locals, alive.
    assert probe_ref[0]() is not None

    client.shutdown()
    gc.collect()
    assert probe_ref[0]() is None
    assert client.engine_core.shut_down
    # The cause survives with its message; only the frame locals are gone.
    with pytest.raises(EngineDeadError) as raised:
        client.get_output()
    assert str(raised.value.__cause__) == "CUDA out of memory (test)"


def _frontend_engine(core: Any) -> tuple[Any, dict[str, Any]]:
    states: dict[str, Any] = {}

    def add_request(request: Any, prompt: Any, queue: Any = None) -> None:
        del prompt, queue
        states[request.request_id] = request

    def abort_requests(ids: Any, internal: bool) -> list[str]:
        assert internal
        return [request_id for request_id in ids if states.pop(request_id, None) is not None]

    engine = SimpleNamespace(
        engine_core=core,
        output_processor=SimpleNamespace(add_request=add_request, abort_requests=abort_requests),
        _run_output_handler=lambda: None,
    )
    return engine, states


def _dead_inproc_client() -> Any:
    client, _ = _client(RuntimeError("boom"))
    with pytest.raises(EngineDeadError):
        client.get_output()
    return client


def _dead_mp_client() -> Any:
    def send(batch_request: Any) -> None:
        raise EngineDeadError()

    return SimpleNamespace(submit_gr_session=send)


@pytest.mark.parametrize("make_core", [_dead_inproc_client, _dead_mp_client])  # type: ignore[misc]
def test_a_rejected_submission_leaves_no_frontend_state(make_core: Any) -> None:
    from tests.helpers.beam_requests import make_batch_request
    from vllm_gr.v1.engine.beam_session_client import submit_gr_session_sync

    engine, states = _frontend_engine(make_core())
    for _ in range(3):
        with pytest.raises(EngineDeadError):
            submit_gr_session_sync(engine, make_batch_request())
    assert states == {}


def test_a_rejected_async_submission_leaves_no_frontend_state() -> None:
    from tests.helpers.beam_requests import make_batch_request
    from vllm_gr.v1.engine.beam_session_client import submit_gr_session_async

    async def dead(batch_request: Any) -> None:
        raise EngineDeadError()

    engine, states = _frontend_engine(SimpleNamespace(submit_gr_session_async=dead))

    async def run() -> None:
        for _ in range(3):
            with pytest.raises(EngineDeadError):
                await submit_gr_session_async(engine, make_batch_request())

    asyncio.run(run())
    assert states == {}


def _ordinary_request() -> Any:
    """A native request before vLLM assigns its external and internal IDs."""
    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

    return EngineCoreRequest(
        request_id="session-0",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _native_request_ids() -> Any:
    from vllm.v1.engine.input_processor import InputProcessor

    # The real ID assignment: external ID kept, internal ID made unique.
    return SimpleNamespace(assign_request_id=InputProcessor.assign_request_id)


def _output_processor() -> Any:
    from vllm.v1.engine.output_processor import OutputProcessor

    return OutputProcessor(None, log_stats=False)


def _frontend_state(processor: Any) -> tuple[int, int, int]:
    return (
        len(processor.request_states),
        len(processor.external_req_ids),
        len(processor.parent_requests),
    )


def _dead_mp_core() -> Any:
    def add_request(request: Any) -> None:
        raise EngineDeadError()

    async def add_request_async(request: Any) -> None:
        raise EngineDeadError()

    return SimpleNamespace(
        add_request=add_request, add_request_async=add_request_async, shutdown=lambda **kwargs: None
    )


@pytest.mark.parametrize("n", [1, 2])  # type: ignore[misc]
@pytest.mark.parametrize("make_core", [_dead_inproc_client, _dead_mp_core])  # type: ignore[misc]
def test_ordinary_requests_to_a_dead_engine_leave_no_frontend_state(make_core: Any, n: int) -> None:
    from vllm.v1.engine.llm_engine import LLMEngine

    from vllm_gr.v1.engine.core_client_patch import apply_async_llm_patches

    apply_async_llm_patches()
    engine = object.__new__(LLMEngine)
    engine.output_processor = _output_processor()
    engine.input_processor = _native_request_ids()
    engine.engine_core = make_core()

    for _ in range(3):
        request = _ordinary_request()
        request.sampling_params.n = n
        with pytest.raises(EngineDeadError):
            engine.add_request(request.request_id, request, request.sampling_params)
        assert _frontend_state(engine.output_processor) == (0, 0, 0)


def test_ordinary_async_requests_to_a_dead_engine_leave_no_frontend_state() -> None:
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm_gr.v1.engine.core_client_patch import apply_async_llm_patches

    apply_async_llm_patches()
    engine = object.__new__(AsyncLLM)
    engine.output_processor = _output_processor()
    engine.engine_core = _dead_mp_core()
    engine.scheduler_config = SimpleNamespace(max_num_queued_reqs=None, max_num_queued_tokens=None)
    engine.log_requests = False

    async def run() -> None:
        for _ in range(3):
            request = _ordinary_request()
            _native_request_ids().assign_request_id(request)
            with pytest.raises(EngineDeadError):
                await engine._add_request(request, None, None, 0, None)
            assert _frontend_state(engine.output_processor) == (0, 0, 0)

    asyncio.run(run())


def test_frontend_rollback_keeps_a_live_registration_on_other_errors() -> None:
    from vllm.v1.engine.llm_engine import LLMEngine

    from vllm_gr.v1.engine.core_client_patch import apply_async_llm_patches

    def reject(request: Any) -> None:
        raise ValueError("not a dead engine")

    apply_async_llm_patches()
    engine = object.__new__(LLMEngine)
    engine.output_processor = _output_processor()
    engine.input_processor = _native_request_ids()
    engine.engine_core = SimpleNamespace(add_request=reject)
    request = _ordinary_request()
    with pytest.raises(ValueError, match="not a dead engine"):
        engine.add_request(request.request_id, request, request.sampling_params)
    # Native behavior: only a dead engine triggers the rollback.
    assert _frontend_state(engine.output_processor) == (1, 1, 0)


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.cpu_test  # type: ignore[misc]
def test_sync_validation_is_bound_to_request_scheduler_and_live_trie_count() -> None:
    from vllm_gr.v1.engine.beam_admission import _sync_validation, has_sync_validation

    request: Any = object()
    config = object()
    scheduler = SimpleNamespace(vllm_config=config, _gr_trie_count=3)
    token = _sync_validation.set((scheduler, request, config, 3))
    try:
        assert has_sync_validation(scheduler, request)
        assert not has_sync_validation(scheduler, object())
        assert not has_sync_validation(
            SimpleNamespace(vllm_config=config, _gr_trie_count=3), request
        )
        scheduler._gr_trie_count = 4
        assert not has_sync_validation(scheduler, request)
        scheduler._gr_trie_count = 3
        scheduler.vllm_config = object()
        assert not has_sync_validation(scheduler, request)
    finally:
        _sync_validation.reset(token)
    assert not has_sync_validation(scheduler, request)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.cpu_test  # type: ignore[misc]
def test_terminal_owned_columns_reject_wrong_primitive_types() -> None:
    from msgspec.structs import replace as replace_struct

    from vllm_gr.v1.beam.batch_contracts import (
        BEAM_BATCH_PROTOCOL_VERSION,
        BeamBatchOutput,
        BeamBatchResult,
        BeamRequestOutput,
        BeamSequenceColumns,
    )
    from vllm_gr.v1.engine.beam_session_client import _has_owned_columns

    column = BeamSequenceColumns(1, 1, bytes(4), bytes(4), None)
    result = BeamBatchResult(
        BEAM_BATCH_PROTOCOL_VERSION,
        "request",
        BeamBatchOutput((BeamRequestOutput(0, "request", column),)),
        None,
    )
    assert _has_owned_columns(result)
    for bad in (
        replace_struct(column, count=True),
        replace_struct(column, steps=0),
        replace_struct(column, token_ids=bytearray(4)),
    ):
        output = BeamBatchOutput((BeamRequestOutput(0, "request", bad),))
        assert not _has_owned_columns(replace_struct(result, output=output))
