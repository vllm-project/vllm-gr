# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the CUDA V1 fixed-control output adapter."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
import ctypes
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker import cuda_async_output as adapter


class _FakeSnapshot:
    registration_failed = False
    device_done = True
    device_status = 0

    @property
    def device_succeeded(self) -> bool:
        return True


class _FakeTicket:
    def __init__(self, generation: int, calls: list[Any]):
        self.generation = generation
        self.calls = calls
        self.closed = False
        self.cancelled = False

    def register(self, stream_pointer: int) -> None:
        self.calls.append(("register", stream_pointer))

    def snapshot(self) -> _FakeSnapshot:
        return _FakeSnapshot()

    def wait_delivery(self, timeout_ms: int) -> bool:
        self.calls.append(("wait_delivery", timeout_ms))
        return True

    def cancel_delivery(self) -> None:
        self.cancelled = True

    def close(self) -> bool:
        if self.closed:
            return False
        self.closed = True
        self.calls.append(("close", self.generation))
        return True


class _FakeAPI:
    def __init__(self, calls: list[Any]):
        self.calls = calls
        self.tickets: list[_FakeTicket] = []

    def new_ticket(self, generation: int) -> _FakeTicket:
        self.calls.append(("acquire", generation))
        ticket = _FakeTicket(generation, self.calls)
        self.tickets.append(ticket)
        return ticket


class _FakeDeviceTensor:
    def __init__(self, address: int, device: Any, calls: list[Any]):
        self._address = address
        self.device = device
        self.dtype = torch.int32
        self.shape = (1, 1)
        self.calls = calls
        self.value = 0

    def is_contiguous(self) -> bool:
        return True

    def data_ptr(self) -> int:
        return self._address

    def fill_(self, value: int) -> "_FakeDeviceTensor":
        self.calls.append(("fill", self._address, value))
        self.value = value
        return self


class _FakeHostTensor:
    def __init__(self, calls: list[Any]):
        self.calls = calls
        self.value = ctypes.c_int32()

    def data_ptr(self) -> int:
        return ctypes.addressof(self.value)

    def copy_(self, source: _FakeDeviceTensor, *, non_blocking: bool):
        self.calls.append(("copy", source.data_ptr(), non_blocking))
        self.value.value = source.value
        return self


class _FakeEvent:
    def __init__(self, calls: list[Any]):
        self.calls = calls

    def record(self) -> None:
        self.calls.append("event_record")


class _FakeStream:
    def __init__(
        self,
        name: str,
        pointer: int,
        calls: list[Any],
        device: torch.device | None = None,
    ):
        self.name = name
        self.cuda_stream = pointer
        self.calls = calls
        self.device = torch.device("cuda:0") if device is None else device

    def wait_stream(self, stream: "_FakeStream") -> None:
        self.calls.append(("wait_stream", self.name, stream.name))

    def wait_event(self, event: _FakeEvent) -> None:
        self.calls.append(("wait_event", self.name, event))


@pytest.fixture
def cuda_fakes(monkeypatch: pytest.MonkeyPatch):
    calls: list[Any] = []
    hosts: list[_FakeHostTensor] = []
    cuda0 = torch.device("cuda:0")
    producer = _FakeStream("producer", 101, calls, cuda0)
    notification = _FakeStream("notification", 202, calls, cuda0)

    def empty(*args, **kwargs):
        calls.append(("empty", args, kwargs))
        host = _FakeHostTensor(calls)
        hosts.append(host)
        return host

    def event():
        calls.append("event_create")
        return _FakeEvent(calls)

    def stream(*, device):
        calls.append(("make_notification_stream", device))
        return notification

    def current_stream(device):
        calls.append(("current_stream", device))
        return producer

    def synchronize(device):
        calls.append(("synchronize", device))

    @contextmanager
    def use_stream(selected):
        calls.append(("enter_stream", selected.name))
        yield
        calls.append(("exit_stream", selected.name))

    monkeypatch.setattr(adapter.torch, "empty", empty)
    monkeypatch.setattr(adapter.torch.cuda, "Event", event)
    monkeypatch.setattr(adapter.torch.cuda, "Stream", stream)
    monkeypatch.setattr(adapter.torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(adapter.torch.cuda, "stream", use_stream)
    monkeypatch.setattr(adapter.torch.cuda, "synchronize", synchronize)
    yield SimpleNamespace(
        calls=calls,
        hosts=hosts,
        producer=producer,
        notification=notification,
    )
    adapter._QUARANTINED.clear()


def _workspace(calls: list[Any], base: int = 4096):
    device = torch.device("cuda:0")
    return SimpleNamespace(
        gr_control_output_0=_FakeDeviceTensor(base, device, calls),
        gr_control_output_1=_FakeDeviceTensor(base + 64, device, calls),
    )


def _runner(workspace, api: _FakeAPI, *, output_token_ids=None):
    batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(
            output_token_ids=[] if output_token_ids is None else output_token_ids
        )
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(_obj=batch),
        use_async_scheduling=True,
        routed_experts_initialized=False,
        device=workspace.gr_control_output_0.device,
        _gr_model_runner_states={},
    )
    transport = adapter.CudaOutputTransport(runner, api)
    runner._gr_cuda_output_transport = transport
    return runner, transport


def _activate(runner, workspace, request_id: str = "gr", device_step: int = 0):
    backend_state = SimpleNamespace(
        workspace=workspace,
        control_epoch=None,
        control_phase=None,
    )
    state = SimpleNamespace(
        device_step=device_step,
        backend_state=backend_state,
    )
    runner._gr_model_runner_states[request_id] = state
    return state


def _scheduler_output(stage: str = "decode", request_id: str = "gr"):
    metadata = {"persistent_runner_state": True}
    if stage == "decode":
        metadata["is_beam_decode"] = True
    elif stage == "last_prefill":
        metadata.update(is_constraint_prefill=True, is_last_prefill=True)
    elif stage == "partial_prefill":
        metadata.update(is_constraint_prefill=True, is_last_prefill=False)
    return SimpleNamespace(beam_data={request_id: metadata})


def _model_output(request_id: str = "gr"):
    return SimpleNamespace(
        req_ids=[request_id],
        req_id_to_index={request_id: 0},
        sampled_token_ids=None,
        logprobs="untouched",
    )


def _output_class():
    class NativeOutput:
        def __init__(
            self,
            model_runner_output,
            sampled_token_ids,
            logprobs_tensors,
            invalid_req_indices,
            async_output_copy_stream,
            vocab_size,
            routed_experts=None,
        ):
            self.native_init = (
                model_runner_output,
                sampled_token_ids,
                logprobs_tensors,
                invalid_req_indices,
                async_output_copy_stream,
                vocab_size,
                routed_experts,
            )
            self.native_result = object()

        def get_output(self):
            return self.native_result

    return NativeOutput


def _construct(Output, model_output, sampled, copy_stream, **kwargs):
    return Output(
        model_output,
        sampled,
        kwargs.get("logprobs_tensors"),
        kwargs.get("invalid_req_indices", []),
        copy_stream,
        32000,
        routed_experts=kwargs.get("routed_experts"),
    )


def test_install_preserves_class_identity_and_fully_delegates_ordinary_output():
    Output = _output_class()
    identity = Output
    adapter.install_async_output_adapter(Output)
    patched = (Output.__init__, Output.get_output)
    adapter.install_async_output_adapter(Output)

    model_output = _model_output("ordinary")
    sampled = object()
    logprobs = object()
    copy_stream = object()
    routed = object()
    result = _construct(
        Output,
        model_output,
        sampled,
        copy_stream,
        logprobs_tensors=logprobs,
        invalid_req_indices=[2],
        routed_experts=routed,
    )

    assert Output is identity
    assert (Output.__init__, Output.get_output) == patched
    assert result.native_init == (
        model_output,
        sampled,
        logprobs,
        [2],
        copy_stream,
        32000,
        routed,
    )
    assert result.get_output() is result.native_result


@pytest.mark.parametrize(
    ("stage", "expected"),
    [("decode", "gr"), ("last_prefill", "gr"), ("partial_prefill", None)],
)
def test_persistent_output_request_only_admits_output_bearing_stages(stage, expected):
    assert adapter.persistent_output_request(_scheduler_output(stage)) == expected
    assert adapter.persistent_output_request(SimpleNamespace(beam_data={})) is None


def test_sampling_scope_restores_context_and_fill_requires_active_frame(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, _ = _runner(workspace, api)
    _activate(runner, workspace)

    with pytest.raises(RuntimeError, match="active persistent sampling lease"):
        adapter.fill_control(workspace, 1, 7)

    Output = _output_class()
    adapter.install_async_output_adapter(Output)
    copy_stream = _FakeStream("copy", 303, cuda_fakes.calls)
    with adapter.sampling_scope(runner, _scheduler_output()):
        sampled = adapter.fill_control(workspace, 1, 7)
        output = _construct(Output, _model_output(), sampled, copy_stream)
        assert adapter._FRAME.get() is not None
    assert adapter._FRAME.get() is None
    assert output.get_output().sampled_token_ids == [[7]]
    assert cuda_fakes.calls.index(("acquire", 1)) < cuda_fakes.calls.index(
        ("fill", sampled.data_ptr(), 7)
    )


def test_prepare_workspace_is_fixed_and_two_inflight_controls_do_not_alias(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, transport = _runner(workspace, api)
    state = _activate(runner, workspace)
    first_pool = transport.prepare_workspace(workspace)
    assert transport.prepare_workspace(workspace) is first_pool
    assert len(cuda_fakes.hosts) == 2
    allocations = [call for call in cuda_fakes.calls if call[0] == "empty"]
    assert len(allocations) == 2
    assert all(
        args == ((1, 1),)
        and kwargs == {"dtype": torch.int32, "device": "cpu", "pin_memory": True}
        for _, args, kwargs in allocations
    )
    assert cuda_fakes.calls.count("event_create") == 2
    assert sum(
        call[0] == "make_notification_stream"
        for call in cuda_fakes.calls
        if isinstance(call, tuple)
    ) == 1

    Output = _output_class()
    adapter.install_async_output_adapter(Output)
    copy_stream = _FakeStream("copy", 303, cuda_fakes.calls)
    outputs = []
    for device_step, token_id in ((1, 17), (2, 29)):
        with adapter.sampling_scope(runner, _scheduler_output()):
            sampled = adapter.fill_control(workspace, device_step, token_id)
            outputs.append(_construct(Output, _model_output(), sampled, copy_stream))
        state.device_step = device_step

    assert first_pool.occupied_slots == 2
    assert transport.high_watermark == 2
    assert outputs[0].sampled_token_ids_cpu is not outputs[1].sampled_token_ids_cpu
    assert outputs[0].get_output().sampled_token_ids == [[17]]
    assert outputs[1].get_output().sampled_token_ids == [[29]]
    assert first_pool.occupied_slots == 0
    assert transport.copies == transport.consumed == 2


def test_host_read_is_the_ack_and_repeated_get_uses_detached_result(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, transport = _runner(workspace, api)
    _activate(runner, workspace)
    Output = _output_class()
    adapter.install_async_output_adapter(Output)
    copy_stream = _FakeStream("copy", 303, cuda_fakes.calls)

    with adapter.sampling_scope(runner, _scheduler_output()):
        sampled = adapter.fill_control(workspace, 1, 41)
        output = _construct(Output, _model_output(), sampled, copy_stream)

    pool = output._gr_control_frame.pool
    ticket = api.tickets[0]
    assert pool.occupied_slots == 1
    assert not ticket.closed and transport.consumed == 0
    first = output.get_output()
    assert first.sampled_token_ids == [[41]]
    assert ticket.closed and pool.occupied_slots == 0
    assert output.get_output() is first
    assert transport.consumed == 1


@pytest.mark.parametrize("single_value", [True, False])
def test_upstream_async_output_future_consumes_adapter_once(
    cuda_fakes, single_value
):
    from vllm.v1.executor.uniproc_executor import AsyncOutputFuture

    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, transport = _runner(workspace, api)
    _activate(runner, workspace)
    Output = _output_class()
    adapter.install_async_output_adapter(Output)

    with adapter.sampling_scope(runner, _scheduler_output()):
        sampled = adapter.fill_control(workspace, 1, 47)
        async_output = _construct(
            Output,
            _model_output(),
            sampled,
            _FakeStream("copy", 303, cuda_fakes.calls),
        )

    future = AsyncOutputFuture(async_output, single_value=single_value)
    result = future.result()
    expected = async_output._model_runner_output
    assert result is expected if single_value else result == [expected]
    assert future.result() is result
    assert transport.consumed == 1


def test_new_session_uses_other_slot_until_old_host_frame_is_consumed(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, transport = _runner(workspace, api)
    state = _activate(runner, workspace, request_id="old")
    backend_state = state.backend_state
    Output = _output_class()
    adapter.install_async_output_adapter(Output)
    copy_stream = _FakeStream("copy", 303, cuda_fakes.calls)

    with adapter.sampling_scope(runner, _scheduler_output(request_id="old")):
        old_device = adapter.fill_control(workspace, 1, 101)
        old_output = _construct(Output, _model_output("old"), old_device, copy_stream)
    old_frame = old_output._gr_control_frame
    old_epoch = backend_state.control_epoch
    old_phase = backend_state.control_phase
    assert old_epoch == 1
    assert old_phase is not None
    assert old_frame.pool.occupied_slots == 1

    # GRModelRunnerState may be cached, but backend bootstrap resets only the
    # per-session epoch/phase before the first control reservation.
    runner._gr_model_runner_states.pop("old")
    backend_state.control_epoch = None
    backend_state.control_phase = None
    state.device_step = 0
    runner._gr_model_runner_states["new"] = state
    with adapter.sampling_scope(runner, _scheduler_output(request_id="new")):
        new_device = adapter.fill_control(workspace, 1, 202)
        new_output = _construct(Output, _model_output("new"), new_device, copy_stream)
    new_frame = new_output._gr_control_frame

    assert new_frame.lease.slot != old_frame.lease.slot
    assert backend_state.control_epoch == old_epoch + 1
    assert backend_state.control_phase != old_phase
    assert old_frame.pool.occupied_slots == 2
    assert new_output.get_output().sampled_token_ids == [[202]]
    assert old_output.get_output().sampled_token_ids == [[101]]
    assert old_frame.pool.occupied_slots == 0
    assert transport.copies == transport.consumed == 2


def _queued_control(cuda_fakes, runner, workspace, request_id, token_id):
    """Real executor Future over the adapter; CUDA completion is simulated."""
    from vllm.v1.executor.uniproc_executor import AsyncOutputFuture

    Output = _output_class()
    adapter.install_async_output_adapter(Output)
    scheduled = _scheduler_output(request_id=request_id)
    scheduled.total_num_scheduled_tokens = 1
    scheduled.pending_structured_output_tokens = False
    with adapter.sampling_scope(runner, scheduled):
        sampled = adapter.fill_control(workspace, 1, token_id)
        output = _construct(
            Output, _model_output(request_id), sampled,
            _FakeStream("copy", 303, cuda_fakes.calls),
        )
    executed = Future()
    executed.set_result(None)
    return (AsyncOutputFuture(output, single_value=True), scheduled, executed), output


def _queue_core(entry, update, process_aborts=lambda: None):
    return SimpleNamespace(
        batch_queue=deque([entry]),
        batch_queue_size=2,
        scheduler=SimpleNamespace(
            has_requests=lambda: False, update_from_output=update,
        ),
        log_error_detail=lambda _: nullcontext(),
        log_iteration_details=lambda _: nullcontext(),
        _process_aborts_queue=process_aborts,
        is_ec_consumer=True,
        is_pooling_model=False,
        use_spec_decode=False,
    )


@pytest.mark.parametrize("abort_during_result", [False, True])
def test_real_batch_queue_consumes_control_before_stale_abort_reconcile(
    cuda_fakes, abort_during_result,
):
    from vllm.v1.engine.core import EngineCore

    workspace = _workspace(cuda_fakes.calls)
    runner, transport = _runner(workspace, _FakeAPI(cuda_fakes.calls))
    _activate(runner, workspace)
    entry, output = _queued_control(cuda_fakes, runner, workspace, "gr", 83)
    pool = output._gr_control_frame.pool
    order = []
    if not abort_during_result:
        runner._gr_model_runner_states.pop("gr")

    def process_aborts():
        assert transport.consumed == 1 and pool.occupied_slots == 0
        runner._gr_model_runner_states.pop("gr", None)
        order.append("abort")

    def update(scheduled, result):
        assert scheduled is entry[1]
        assert result.sampled_token_ids == [[83]]
        assert not runner._gr_model_runner_states
        assert pool.occupied_slots == 0
        order.append("stale_reconcile")
        return {}  # Scheduler drops the aborted request's detached output.

    core = _queue_core(entry, update, process_aborts)
    assert pool.occupied_slots == 1 and not entry[0].done()
    assert EngineCore.step_with_batch_queue(core) == ({}, False)
    assert order == ["abort", "stale_reconcile"]
    assert not core.batch_queue and entry[0].done()
    assert transport.copies == transport.consumed == 1
    transport.shutdown()


def test_real_batch_queue_new_session_does_not_overwrite_aborted_control(cuda_fakes):
    from vllm.v1.engine.core import EngineCore

    workspace = _workspace(cuda_fakes.calls)
    runner, transport = _runner(workspace, _FakeAPI(cuda_fakes.calls))
    _activate(runner, workspace, request_id="old")
    old_entry, old_output = _queued_control(cuda_fakes, runner, workspace, "old", 101)
    runner._gr_model_runner_states.pop("old")
    _activate(runner, workspace, request_id="new")
    reconciled = []
    new_outputs = []

    def update(scheduled, result):
        reconciled.append((result.req_ids[0], result.sampled_token_ids))
        return {}

    core = _queue_core(old_entry, update)
    new_scheduled = _scheduler_output(request_id="new")
    new_scheduled.total_num_scheduled_tokens = 1
    new_scheduled.pending_structured_output_tokens = False
    core.scheduler.has_requests = lambda: True
    core.scheduler.schedule = lambda: new_scheduled
    core.scheduler.get_grammar_bitmask = lambda _: None

    def sample_tokens(grammar, *, non_block):
        assert grammar is None and non_block
        entry, output = _queued_control(cuda_fakes, runner, workspace, "new", 202)
        new_outputs.append(output)
        assert output._gr_control_frame.lease.slot != old_output._gr_control_frame.lease.slot
        assert output._gr_control_frame.pool.occupied_slots == 2
        return entry[0]

    core.model_executor = SimpleNamespace(
        execute_model=lambda scheduled, non_block: old_entry[2],
        sample_tokens=sample_tokens,
    )
    assert EngineCore.step_with_batch_queue(core) == ({}, True)
    assert reconciled == [("old", [[101]])]
    assert len(core.batch_queue) == 1 and transport.consumed == 1
    assert not core.batch_queue[-1][0].done()
    core.scheduler.has_requests = lambda: False
    assert EngineCore.step_with_batch_queue(core) == ({}, False)
    assert reconciled == [("old", [[101]]), ("new", [[202]])]
    assert transport.copies == transport.consumed == 2
    assert transport.high_watermark == 2
    assert not core.batch_queue
    assert new_outputs[0]._gr_control_frame.pool.occupied_slots == 0
    transport.shutdown()


def test_stopped_inprocess_queue_keeps_control_owned_until_terminal_drain(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, transport = _runner(workspace, api)
    _activate(runner, workspace)
    entry, output = _queued_control(cuda_fakes, runner, workspace, "gr", 97)
    core = _queue_core(entry, lambda *_: pytest.fail("no stepping after abort"))
    runner._gr_model_runner_states.pop("gr")
    pool = output._gr_control_frame.pool
    # InprocClient has no background queue consumer. Device completion alone
    # is not a Host acknowledgement and must not permit slot reuse.
    assert not core.scheduler.has_requests() and core.batch_queue
    assert pool.occupied_slots == 1 and transport.consumed == 0
    assert not api.tickets[0].closed
    transport.shutdown()
    assert ("synchronize", torch.device("cuda:0")) in cuda_fakes.calls
    assert pool.occupied_slots == 0 and api.tickets[0].closed
    assert transport.consumed == 0 and not entry[0].done()


def test_real_batch_queue_delivery_error_quarantines_until_terminal_drain(
    cuda_fakes, monkeypatch,
):
    from vllm.v1.engine.core import EngineCore

    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, transport = _runner(workspace, api)
    _activate(runner, workspace)
    entry, output = _queued_control(cuda_fakes, runner, workspace, "gr", 109)
    error = RuntimeError("injected completion failure")

    def fail_wait(timeout_ms):
        raise error

    monkeypatch.setattr(api.tickets[0], "wait_delivery", fail_wait)
    core = _queue_core(
        entry, lambda *_: pytest.fail("failed output must not reconcile"),
        lambda: pytest.fail("failed output must not reach abort reconciliation"),
    )
    with pytest.raises(RuntimeError, match="injected completion failure") as caught:
        EngineCore.step_with_batch_queue(core)
    assert caught.value is error
    assert not core.batch_queue and entry[0].exception() is error
    assert transport in adapter._QUARANTINED
    assert output._gr_control_frame.pool.occupied_slots == 1
    assert not api.tickets[0].closed and transport.consumed == 0
    with pytest.raises(RuntimeError, match="quarantined"):
        transport.check_live()
    transport.shutdown()
    assert api.tickets[0].closed
    assert output._gr_control_frame.pool.occupied_slots == 0
    assert transport not in adapter._QUARANTINED


def test_invalid_request_zero_yields_empty_control_row(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, _ = _runner(workspace, api)
    _activate(runner, workspace)
    Output = _output_class()
    adapter.install_async_output_adapter(Output)

    with adapter.sampling_scope(runner, _scheduler_output()):
        sampled = adapter.fill_control(workspace, 1, 53)
        output = _construct(
            Output,
            _model_output(),
            sampled,
            _FakeStream("copy", 303, cuda_fakes.calls),
            invalid_req_indices=[0],
        )
    assert output.get_output().sampled_token_ids == [[]]


def test_constructor_works_after_final_state_release(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, _ = _runner(workspace, api)
    _activate(runner, workspace)
    Output = _output_class()
    adapter.install_async_output_adapter(Output)

    with adapter.sampling_scope(runner, _scheduler_output("last_prefill")):
        sampled = adapter.fill_control(workspace, 1, 61)
        runner._gr_model_runner_states.pop("gr")
        output = _construct(
            Output,
            _model_output(),
            sampled,
            _FakeStream("copy", 303, cuda_fakes.calls),
        )
    assert output.get_output().sampled_token_ids == [[61]]


@pytest.mark.parametrize("bad_field", ["device", "request_id", "logprobs"])
def test_constructor_rejects_mismatched_control_abi_without_starting_dma(
    cuda_fakes, bad_field
):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, _ = _runner(workspace, api)
    _activate(runner, workspace)
    Output = _output_class()
    adapter.install_async_output_adapter(Output)
    copy_stream = _FakeStream("copy", 303, cuda_fakes.calls)

    with pytest.raises(RuntimeError, match="ABI mismatch|does not own"):
        with adapter.sampling_scope(runner, _scheduler_output()):
            sampled = adapter.fill_control(workspace, 1, 71)
            model_output = _model_output("wrong" if bad_field == "request_id" else "gr")
            _construct(
                Output,
                model_output,
                object() if bad_field == "device" else sampled,
                copy_stream,
                logprobs_tensors=object() if bad_field == "logprobs" else None,
            )

    assert not any(call[0] == "copy" for call in cuda_fakes.calls if isinstance(call, tuple))
    assert not any(
        call[0] == "wait_stream" for call in cuda_fakes.calls if isinstance(call, tuple)
    )
    assert api.tickets and not any(
        call[0] == "register" for call in cuda_fakes.calls if isinstance(call, tuple)
    )


def test_second_host_history_consumer_is_rejected_at_scope_and_constructor(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, _ = _runner(workspace, api, output_token_ids=[[1]])
    _activate(runner, workspace)
    with pytest.raises(RuntimeError, match="Host output-token history"):
        with adapter.sampling_scope(runner, _scheduler_output()):
            pytest.fail("scope admitted a second Host consumer")

    runner.input_batch._obj.sampling_metadata.output_token_ids = []
    Output = _output_class()
    adapter.install_async_output_adapter(Output)
    with pytest.raises(RuntimeError, match="Host output-token history"):
        with adapter.sampling_scope(runner, _scheduler_output()):
            sampled = adapter.fill_control(workspace, 1, 79)
            runner.input_batch._obj.sampling_metadata.output_token_ids = [[1]]
            _construct(
                Output,
                _model_output(),
                sampled,
                _FakeStream("copy", 303, cuda_fakes.calls),
            )
    assert not any(call[0] == "copy" for call in cuda_fakes.calls if isinstance(call, tuple))


def test_constructor_observes_producer_before_entering_copy_stream(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, _ = _runner(workspace, api)
    _activate(runner, workspace)
    Output = _output_class()
    adapter.install_async_output_adapter(Output)

    with adapter.sampling_scope(runner, _scheduler_output()):
        sampled = adapter.fill_control(workspace, 1, 83)
        _construct(
            Output,
            _model_output(),
            sampled,
            _FakeStream("copy", 303, cuda_fakes.calls),
        )

    current = next(
        index
        for index, call in enumerate(cuda_fakes.calls)
        if isinstance(call, tuple) and call[0] == "current_stream"
    )
    entered_copy = next(
        index
        for index, call in enumerate(cuda_fakes.calls)
        if call == ("enter_stream", "copy")
    )
    assert current < entered_copy


def test_notification_wait_and_callback_do_not_switch_current_stream(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, _ = _runner(workspace, api)
    _activate(runner, workspace)
    Output = _output_class()
    adapter.install_async_output_adapter(Output)
    copy_stream = _FakeStream("copy", 303, cuda_fakes.calls)

    with adapter.sampling_scope(runner, _scheduler_output()):
        sampled = adapter.fill_control(workspace, 1, 87)
        before = adapter.torch.cuda.current_stream(sampled.device)
        output = _construct(Output, _model_output(), sampled, copy_stream)
        after = adapter.torch.cuda.current_stream(sampled.device)

    assert before is after is cuda_fakes.producer
    assert cuda_fakes.calls.count(("enter_stream", "copy")) == 1
    assert cuda_fakes.calls.count(("exit_stream", "copy")) == 1
    assert ("enter_stream", "notification") not in cuda_fakes.calls
    assert ("exit_stream", "notification") not in cuda_fakes.calls

    copy_index = next(
        index
        for index, call in enumerate(cuda_fakes.calls)
        if isinstance(call, tuple) and call[0] == "copy"
    )
    record_index = cuda_fakes.calls.index("event_record")
    notify_index = next(
        index
        for index, call in enumerate(cuda_fakes.calls)
        if isinstance(call, tuple)
        and call[0] == "wait_event"
        and call[1] == "notification"
    )
    callback = ("register", cuda_fakes.notification.cuda_stream)
    callback_index = cuda_fakes.calls.index(callback)
    assert copy_index < record_index < notify_index < callback_index
    assert api.tickets[0].calls is cuda_fakes.calls
    assert output.get_output().sampled_token_ids == [[87]]


def test_prepare_workspace_rejects_slots_on_a_different_runner_device(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, transport = _runner(workspace, api)
    runner.device = torch.device("cuda:1")

    with pytest.raises(RuntimeError, match="runner device"):
        transport.prepare_workspace(workspace)


def test_constructor_rechecks_runner_device_before_copy(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, _ = _runner(workspace, api)
    _activate(runner, workspace)
    Output = _output_class()
    adapter.install_async_output_adapter(Output)

    with pytest.raises(RuntimeError, match="runner device"):
        with adapter.sampling_scope(runner, _scheduler_output()):
            sampled = adapter.fill_control(workspace, 1, 89)
            runner.device = torch.device("cuda:1")
            _construct(
                Output,
                _model_output(),
                sampled,
                _FakeStream("copy", 303, cuda_fakes.calls),
            )
    assert not any(call[0] == "copy" for call in cuda_fakes.calls if isinstance(call, tuple))


@pytest.mark.parametrize("stream_kind", ["copy", "notification"])
def test_constructor_rejects_stream_on_another_device_before_copy(
    cuda_fakes, stream_kind
):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, _ = _runner(workspace, api)
    _activate(runner, workspace)
    Output = _output_class()
    adapter.install_async_output_adapter(Output)
    copy_stream = _FakeStream("copy", 303, cuda_fakes.calls)
    if stream_kind == "copy":
        copy_stream.device = torch.device("cuda:1")
    else:
        cuda_fakes.notification.device = torch.device("cuda:1")

    with pytest.raises(RuntimeError, match="stream.*device"):
        with adapter.sampling_scope(runner, _scheduler_output()):
            sampled = adapter.fill_control(workspace, 1, 97)
            _construct(Output, _model_output(), sampled, copy_stream)
    assert not any(call[0] == "copy" for call in cuda_fakes.calls if isinstance(call, tuple))


def test_transport_shutdown_drains_the_runner_device(cuda_fakes):
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, transport = _runner(workspace, api)
    transport.prepare_workspace(workspace)

    assert transport.shutdown() is None
    assert ("synchronize", torch.device("cuda:0")) in cuda_fakes.calls
    assert not transport._pools
