# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="no-untyped-def,no-untyped-call,misc"
"""Pause native GR queues and drain controlled CPU Futures without a GPU."""

from concurrent.futures import Future
from contextlib import nullcontext

import pytest
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from tests.v1.engine.test_gr_admission_queue import batch
from tests.v1.engine.test_gr_admission_queue import core as _core_fixture
from tests.v1.engine.test_gr_admission_queue import scheduler as _scheduler_fixture
from tests.v1.engine.test_gr_admission_queue import worker_receipt
from vllm_gr.v1.beam.dispatch import GR_RETIRE_OUTPUT_ATTR, GR_STAGE_OUTPUT_ATTR
from vllm_gr.v1.core.sched.beam_session import GRResourceStatus, GRSessionStatus

admission_core = _core_fixture
scheduler = _scheduler_fixture


class Deferred(Future[ModelRunnerOutput]):
    def __init__(self, executor, scheduled):
        super().__init__()
        self.executor = executor
        self.scheduled = scheduled
        self.output = worker_receipt(executor.core.scheduler, scheduled)
        if executor.immediate:
            self.set_result(self.output)

    def result(self, timeout=None):
        if not self.done():
            if self.executor.before_consume is not None:
                self.executor.before_consume(self.scheduled)
            self.set_result(self.output)
        return super().result(timeout)


class Executor:
    def __init__(self, core):
        self.core = core
        self.immediate = False
        self.before_consume = None
        self.submitted = []

    def collective_rpc(self, method):
        assert method == "synchronize_device"
        # Completing pause must synchronize only after every native/GR frame drains.
        assert not self.core.batch_queue
        assert not self.core.has_work()

    def execute_model(self, scheduled, non_block):
        assert non_block
        self.current = scheduled
        self.submitted.append(scheduled)
        return Deferred(self, scheduled)

    def sample_tokens(self, grammar, non_block):
        assert non_block
        return Deferred(self, self.current)


@pytest.fixture
def core(admission_core):
    core = admission_core
    # Restore the actual native idle callback path, which admission tests do
    # not exercise. Pause completion must occur through the input-loop check.
    core._idle_state_callbacks = []
    core._notify_idle_state_callbacks = EngineCoreProc._notify_idle_state_callbacks.__get__(core)
    core.model_executor = Executor(core)
    core.log_error_detail = lambda _: nullcontext()
    # Exercise native capture_iteration_details with logging disabled.
    core.log_stats = False
    core._process_aborts_queue = lambda: None
    core.is_ec_consumer = True
    core.is_pooling_model = False
    return core


def submit(core, name, **kwargs):
    core._handle_submit_gr_session(batch(name, **kwargs))
    core._process_input_queue()
    return core.scheduler._gr_v1_sessions[name]


def step(core):
    outputs, executed = core.step_with_batch_queue()
    for item in (outputs or {}).items():
        core.output_queue.put_nowait(item)
    return executed


def drain(core):
    for _ in range(30):
        if not core.has_work():
            break
        step(core)
    assert not core.has_work(), "pause/drain still counts retained sessions as runnable"
    core._process_input_queue()


def model_dispatches(core):
    return [
        (m.session_id, m.stage_index)
        for scheduled in core.model_executor.submitted
        for m in getattr(scheduled, GR_STAGE_OUTPUT_ATTR, {}).values()
    ]


def assert_finished(core, names):
    results: list[EngineCoreOutput] = []
    while not core.output_queue.empty():
        _, output = core.output_queue.get_nowait()
        results.extend(out for out in output.outputs if out.finish_reason is not None)
    assert [out.request_id for out in results] == names
    assert all(out.finish_reason is FinishReason.LENGTH for out in results)
    for out in results:
        terminal = out.kv_transfer_params["gr_batch_result"]
        assert terminal.failure is None
        assert terminal.batch_id == out.request_id
        assert all(
            sequence.finish_reason == "length"
            for item in terminal.output.items
            for sequence in item.sequences
        )
    assert not core.scheduler.requests
    assert not core.scheduler._gr_v1_sessions


@pytest.mark.parametrize("mode", ["keep", "wait"])
def test_pause_with_only_queued_gr_sessions_completes_and_resumes(core, mode):
    first = submit(core, "A")
    second = submit(core, "B")
    # GR resource ownership is assigned at ADD; native execution has not
    # started. PAUSED_NEW must follow the native running queue, not this enum.
    assert first.resource_status is GRResourceStatus.RUNNING
    assert core.scheduler.requests["A"].status is RequestStatus.WAITING
    assert core.pause_scheduler(mode=mode, clear_cache=False) is None
    assert not core.has_work()
    assert core.scheduler._gr_v1_sessions == {"A": first, "B": second}
    assert not first.worker_resources_allocated
    assert model_dispatches(core) == []
    core.resume_scheduler()
    drain(core)
    assert_finished(core, ["A", "B"])


@pytest.mark.parametrize("mode", ["keep", "wait"])
def test_new_gr_add_while_paused_stays_queued_until_resume(core, mode):
    assert core.pause_scheduler(mode=mode, clear_cache=False) is None
    session = submit(core, "A")
    assert not core.has_work()
    assert not session.worker_resources_allocated
    assert core.scheduler.requests["A"].status is RequestStatus.WAITING
    assert model_dispatches(core) == []
    assert core.pause_scheduler(mode=mode, clear_cache=False) is None
    core.resume_scheduler()
    drain(core)
    assert_finished(core, ["A"])


@pytest.mark.parametrize(("queue_size", "in_flight"), [(1, False), (2, False), (2, True)])
def test_keep_retains_active_state_and_kv_after_in_flight_drain(core, queue_size, in_flight):
    active = submit(core, "A")
    waiting = submit(core, "B")
    core.batch_queue_size = queue_size
    core.model_executor.immediate = not in_flight
    step(core)
    core.model_executor.immediate = False
    # 0.30 retains even a completed Future when the native batch queue has room.
    pending = queue_size > 1
    assert bool(core.batch_queue) is pending
    if pending:
        assert core.batch_queue[-1][0].done() is not in_flight
    issued = active.issued_stages
    generation = active.owner_generation
    native = core.scheduler.requests["A"]
    free = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    paused = core.pause_scheduler(mode="keep", clear_cache=False)
    if pending:
        assert isinstance(paused, Future) and not paused.done()
        assert not core.scheduler.has_requests()
        assert core.has_work()  # Native batch queue still needs draining.
    else:
        assert paused is None
    drain(core)
    if paused is not None:
        assert paused.done() and paused.result() is None
    assert active.issued_stages == issued == 1
    assert core.scheduler.requests["A"].num_output_tokens == 1
    assert active.owner_generation == generation
    assert active.resource_status is GRResourceStatus.RUNNING
    assert waiting.resource_status is GRResourceStatus.WAITING
    assert core.scheduler.requests["A"] is native
    assert core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    assert model_dispatches(core) == [("A", 0)]
    assert not any(hasattr(s, GR_RETIRE_OUTPUT_ATTR) for s in core.model_executor.submitted)
    core.resume_scheduler()
    drain(core)
    assert_finished(core, ["A", "B"])


def test_wait_finishes_running_request_and_retirement_but_preserves_waiting(core):
    active = submit(core, "A")
    waiting = submit(core, "B")
    step(core)
    paused = core.pause_scheduler(mode="wait", clear_cache=False)
    assert isinstance(paused, Future) and not paused.done()
    assert core.scheduler.pause_state is PauseState.PAUSED_NEW
    drain(core)
    assert paused.done() and paused.result() is None
    assert active.resource_status is GRResourceStatus.CLOSED
    assert core.scheduler._gr_v1_sessions == {"B": waiting}
    assert not waiting.worker_resources_allocated
    assert core.scheduler.requests["B"].status is RequestStatus.WAITING
    assert model_dispatches(core) == [("A", 0), ("A", 1), ("A", 2)]
    core.resume_scheduler()
    drain(core)
    assert_finished(core, ["A", "B"])


def test_keep_delivers_terminal_result_and_releases_without_cleanup_frame(core):
    active = submit(core, "A", n=1)
    waiting = submit(core, "B")
    step(core)
    paused = core.pause_scheduler(mode="keep", clear_cache=False)
    assert isinstance(paused, Future) and not paused.done()
    step(core)  # Consume the already-issued terminal Prefill result.
    assert active._terminal_emitted
    assert active.resource_status is GRResourceStatus.CLOSED
    # The fused release drains the last GR cleanup work inside the consuming
    # frame, so the native input-loop idle check completes the pause without
    # another engine step.
    core._process_input_queue()
    assert paused.done() and paused.result() is None
    assert core.scheduler._gr_v1_sessions == {"B": waiting}
    assert model_dispatches(core) == [("A", 0)]
    core.resume_scheduler()
    drain(core)
    assert_finished(core, ["A", "B"])


@pytest.mark.parametrize("mode", ["keep", "wait"])
def test_pause_after_last_cancel_waits_for_dispatches_and_release_proof(core, mode):
    free = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    session = submit(core, "A")
    step(core)
    assert session._in_flight_dispatches == {0}
    core.abort_requests(["A"])
    assert session.status is GRSessionStatus.ABORTED
    assert session.resource_status is GRResourceStatus.DRAINING
    assert core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() < free
    paused = core.pause_scheduler(mode=mode, clear_cache=False)
    assert isinstance(paused, Future) and not paused.done()
    observed_proof = []

    def before_consume(scheduled):
        if not hasattr(scheduled, GR_RETIRE_OUTPUT_ATTR):
            return
        assert not paused.done()
        assert core.has_work()
        assert core.scheduler._gr_v1_sessions == {"A": session}
        assert session.resource_status is GRResourceStatus.RETIRING
        assert not session._in_flight_dispatches
        assert scheduled.total_num_scheduled_tokens == 0
        observed_proof.append(session.retirement)

    core.model_executor.before_consume = before_consume
    drain(core)
    assert len(observed_proof) == 1
    assert paused.done() and paused.result() is None
    assert session.resource_status is GRResourceStatus.CLOSED
    assert not core.scheduler._gr_v1_sessions
    assert core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    assert model_dispatches(core) == [("A", 0)]


pytestmark = pytest.mark.cpu_test
