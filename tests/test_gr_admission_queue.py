# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="no-untyped-def,no-untyped-call,misc"
"""GR rejection through native EngineCore ADD queues, without a model/GPU."""

from collections import deque
from queue import Queue
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import msgspec
import numpy as np
import pytest
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequestType, FinishReason
from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from tests.test_async_scheduling_scheduler import receipt, release_receipt
from tests.test_async_scheduling_scheduler import scheduler as _scheduler_fixture
from tests.test_gr_async_scheduler import _batch_request, _multi_batch_request
from vllm_gr.v1.beam.batch_contracts import (
    ADMISSION_REJECTED_CODE,
    BeamBatchOutput,
    BeamBatchResult,
    BeamOutputSequence,
    BeamRequestOutput,
)
from vllm_gr.v1.engine.beam_search_v1 import GRResourceStatus, GRSessionStatus
from vllm_gr.v1.engine.core import _handle_submit_gr_session
from vllm_gr.v1.engine.engine_core_patch import (
    apply_engine_core_request_patches,
    make_patched_engine_init,
)
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_BATCH_RESULT_ATTR,
    GR_RETIRE_OUTPUT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GRAdmissionError,
    consume_gr_retirement,
)

scheduler = _scheduler_fixture


def batch(name, *, prompt=7, n=3, client=3):
    result = _batch_request(n)
    request = result.items[0].request
    request.request_id = name
    request.prompt_token_ids = np.full(prompt, 1, dtype=np.int64)
    request.client_index = client
    return msgspec.structs.replace(result, batch_id=name)


def bare_core(scheduler, cls):
    """Keep the actual preprocessing, dispatch, ADD and ERROR implementations."""
    core = object.__new__(cls)
    core.scheduler = scheduler
    core.vllm_config = scheduler.vllm_config
    core.batch_queue = deque()
    core.batch_queue_size = 2
    core.mm_receiver_cache = None
    if not hasattr(kv_cache_utils, "NONE_HASH"):
        kv_cache_utils.init_none_hash(sha256)
    core.request_block_hasher = get_request_block_hasher(16, sha256)
    return core


@pytest.fixture
def core(scheduler):
    apply_engine_core_request_patches()
    core = bare_core(scheduler, EngineCoreProc)
    core.input_queue = Queue()
    core.output_queue = Queue()
    core.aborts_queue = Queue()
    core.engines_running = False
    core.shutdown_state = EngineShutdownState.RUNNING
    core.process_input_queue_block = False
    core._notify_idle_state_callbacks = lambda: None
    return core


def assert_one_error(core, name, client=3):
    target, output = core.output_queue.get_nowait()
    assert target == client
    assert output.finished_requests == [name]
    assert len(output.outputs) == 1
    assert output.outputs[0].request_id == name
    assert output.outputs[0].finish_reason is FinishReason.ERROR
    assert output.outputs[0].new_token_ids == []
    # A bare error output says only that something went wrong, so the frontend
    # can only report a server fault. The rejection is the request's answer and
    # has to travel as the terminal result the session never produced.
    rejection = output.outputs[0].kv_transfer_params["gr_batch_result"]
    assert rejection.batch_id == name
    assert rejection.output is None
    assert rejection.failure.code == ADMISSION_REJECTED_CODE
    assert rejection.failure.message.strip()
    assert core.output_queue.empty()


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("via_socket", [False, True])
def test_invalid_trie_rejected_before_native_add_and_engine_remains_usable(core, count, via_socket):
    core._gr_trie_count = core.scheduler._gr_trie_count = count
    bad = batch("bad-trie")
    bad = msgspec.structs.replace(
        bad, items=(msgspec.structs.replace(bad.items[0], trie_id=count),)
    )
    if via_socket:
        core._handle_submit_gr_session(bad)
        assert core.input_queue.empty()
    else:
        core._handle_client_request(EngineCoreRequestType.SUBMIT_GR_SESSION, bad)
    assert_one_error(core, "bad-trie")
    assert not core.scheduler.requests
    assert not getattr(core.scheduler, "_gr_v1_sessions", {})
    good = batch("good-trie")
    good = msgspec.structs.replace(
        good, items=(msgspec.structs.replace(good.items[0], trie_id=count - 1),)
    )
    core._handle_submit_gr_session(good)
    core._process_input_queue()
    complete(core, "good-trie")


@pytest.mark.parametrize("via_socket", [False, True])
def test_negative_tokens_rejected_before_preprocessing(core, monkeypatch, via_socket):
    bad = batch("negative-token")
    bad.items[0].request.prompt_token_ids = np.array([-1, 2, 3], dtype=np.int64)
    preprocess = Mock(wraps=core.preprocess_add_request)
    monkeypatch.setattr(core, "preprocess_add_request", preprocess)
    free = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    if via_socket:
        core._handle_submit_gr_session(bad)
    else:
        core._handle_client_request(EngineCoreRequestType.SUBMIT_GR_SESSION, bad)
    assert core.input_queue.empty()
    preprocess.assert_not_called()
    assert_one_error(core, "negative-token")
    assert not core.scheduler.requests
    assert not getattr(core.scheduler, "_gr_v1_sessions", {})
    assert core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free

    core._handle_submit_gr_session(batch("valid-after-negative"))
    core._process_input_queue()
    complete(core, "valid-after-negative")


@pytest.mark.parametrize("counts", [[1], [2, 2], [1, 2], [], [0]])
def test_engine_startup_publishes_live_trie_count(counts):
    from vllm_gr.v1.engine.core import loaded_constraint_trie_count

    engine = SimpleNamespace()

    def original(self):
        self.vllm_config = SimpleNamespace(additional_config={"vllm_gr_beam_execution_mode": "v1"})
        self.scheduler = SimpleNamespace()
        self.model_executor = SimpleNamespace(collective_rpc=Mock(return_value=counts))

    initialize = make_patched_engine_init(original)
    assert make_patched_engine_init(initialize) is initialize
    if not counts or len(set(counts)) != 1 or counts[0] < 1:
        with pytest.raises(RuntimeError, match="Trie count"):
            initialize(engine)
        assert not hasattr(engine, "_gr_trie_count")
    else:
        initialize(engine)
        assert engine._gr_trie_count == engine.scheduler._gr_trie_count == counts[0]
        engine.model_executor.collective_rpc.assert_called_once_with(loaded_constraint_trie_count)


def test_worker_reports_loaded_table_not_configuration():
    from vllm_gr.v1.engine.core import loaded_constraint_trie_count

    runner = SimpleNamespace(constraint_backend=None)
    worker = SimpleNamespace(model_runner=runner)
    assert loaded_constraint_trie_count(worker) == 1
    runner.constraint_backend = SimpleNamespace(
        info=SimpleNamespace(table_metadata=SimpleNamespace(trie_count=3))
    )
    assert loaded_constraint_trie_count(worker) == 3


def worker_receipt(scheduler: Any, scheduled: Any) -> ModelRunnerOutput:
    """Build deterministic CPU stage, final-output and fallback retirement receipts."""
    if getattr(scheduled, GR_RETIRE_OUTPUT_ATTR, None) is not None:
        worker_output = release_receipt(scheduled)
    elif getattr(scheduled, GR_STAGE_OUTPUT_ATTR, None):
        worker_output = receipt(scheduled)
        results = {}
        for session_id, metadata in getattr(scheduled, GR_STAGE_OUTPUT_ATTR).items():
            if not metadata.is_last_stage:
                continue
            request = scheduler._gr_v1_sessions[session_id].request
            # CPU fixture for the terminal consumer contract. This tests
            # delivery/release, not model generation or the A5 adapter.
            sequences = tuple(
                BeamOutputSequence(
                    token_ids=(3,) * request.beam_params.max_tokens,
                    cumulative_logprob=-float(i),
                    parent_beam_indices=(0,) * request.beam_params.max_tokens,
                    finish_reason="length",
                    stop_token_id=None,
                )
                for i in range(request.output_options.num_return_sequences)
            )
            results[session_id] = BeamBatchResult(
                protocol_version=request.protocol_version,
                batch_id=session_id,
                output=BeamBatchOutput((BeamRequestOutput(0, session_id, sequences),)),
                failure=None,
            )
        setattr(worker_output, GR_BATCH_RESULT_ATTR, results)
    else:
        ids = list(scheduled.num_scheduled_tokens)
        worker_output = ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(ids)},
            sampled_token_ids=[[3] for _ in ids],
        )
    return worker_output


def complete(core, name):
    """Drive the real Scheduler with deterministic Worker terminal receipts."""
    outputs: list[EngineCoreOutput] = []
    for _ in range(20):
        if not core.scheduler.has_requests():
            break
        scheduled = core.scheduler.schedule()
        worker_output = worker_receipt(core.scheduler, scheduled)
        result = core.scheduler.update_from_output(scheduled, worker_output)
        outputs.extend(out for group in result.values() for out in group.outputs)
    assert not core.scheduler.has_requests()
    assert name not in core.scheduler.requests
    assert not getattr(core.scheduler, "_gr_v1_sessions", {})
    assert getattr(core.scheduler, "_gr_owner", None) is None
    terminal = [out for out in outputs if out.request_id == name and out.finish_reason is not None]
    assert len(terminal) == 1
    assert terminal[0].finish_reason is not FinishReason.ERROR


@pytest.mark.parametrize("rejection", ["kv_capacity", "width", "budget"])
def test_rejected_queued_gr_add_returns_error_then_valid_request_completes(core, rejection):
    bad = batch("bad")
    if rejection == "kv_capacity":
        # Keep the model-length limit larger than the total physical KV pool.
        core.scheduler.kv_cache_manager.max_model_len = 256
        bad.items[0].request.prompt_token_ids = np.full(129, 1, dtype=np.int64)
    else:
        params = msgspec.structs.replace(
            bad.beam_params, **({"beam_width": 8} if rejection == "width" else {"max_tokens": 4})
        )
        bad = msgspec.structs.replace(bad, beam_params=params)
    free = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    core._handle_submit_gr_session(bad)
    # Invalid shared semantics fail during preprocessing; capacity/owner
    # admission still runs on the Core thread without leaving ghost sessions.
    assert core.input_queue.qsize() == (0 if rejection == "budget" else 1)
    if rejection != "budget":
        assert core.output_queue.empty()
    assert not getattr(core.scheduler, "_gr_v1_sessions", {})
    core._process_input_queue()
    assert_one_error(core, "bad")
    assert core.is_running()
    assert not core.scheduler.requests
    assert not core.scheduler.waiting
    assert not getattr(core.scheduler, "_gr_v1_sessions", {})
    assert getattr(core.scheduler, "_gr_owner_generation", 0) == 0
    assert getattr(core.scheduler, "_gr_execution_capacity", None) is None
    assert core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    core._handle_submit_gr_session(batch("good"))
    core._process_input_queue()
    complete(core, "good")
    assert core.output_queue.empty()
    assert core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


@pytest.mark.parametrize("rejection", ["ordinary", "duplicate", "oversized_gr"])
def test_queued_rejection_preserves_existing_gr_owner_and_in_flight_work(core, rejection):
    core._handle_submit_gr_session(batch("active"))
    core._process_input_queue()
    scheduler = core.scheduler
    session = scheduler._gr_v1_sessions["active"]
    native = scheduler.requests["active"]
    first = scheduler.schedule()
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    generation = scheduler._gr_owner_generation
    pending = set(session._in_flight_dispatches)
    name = "active" if rejection == "duplicate" else "bad"
    rejected = batch(name, client=9, prompt=65 if rejection == "oversized_gr" else 7)
    if rejection == "ordinary":
        core.input_queue.put_nowait(
            (EngineCoreRequestType.ADD, core.preprocess_add_request(rejected.items[0].request))
        )
    else:
        core._handle_submit_gr_session(rejected)
    core._process_input_queue()
    assert_one_error(core, name, client=9)
    assert scheduler._gr_v1_sessions == {"active": session}
    assert scheduler.requests["active"] is native
    assert scheduler._gr_owner == "active"
    assert scheduler._gr_owner_generation == generation
    assert session._in_flight_dispatches == pending
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    scheduler.update_from_output(first, receipt(first))
    complete(core, "active")
    assert session.resource_status is GRResourceStatus.CLOSED
    assert core.output_queue.empty()


def test_queued_gr_add_cannot_displace_existing_ordinary_request(core):
    existing = batch("ordinary", n=1, client=2).items[0].request
    core.input_queue.put_nowait((EngineCoreRequestType.ADD, core.preprocess_add_request(existing)))
    core._process_input_queue()
    native = core.scheduler.requests["ordinary"]
    core._handle_submit_gr_session(batch("bad"))
    core._process_input_queue()
    assert_one_error(core, "bad")
    assert core.scheduler.requests == {"ordinary": native}
    assert not getattr(core.scheduler, "_gr_v1_sessions", {})
    complete(core, "ordinary")


def test_input_loop_consumes_valid_add_after_rejection_in_same_queue(core):
    core._handle_submit_gr_session(batch("bad", prompt=65))
    core._handle_submit_gr_session(batch("good"))
    core._process_input_queue()
    assert_one_error(core, "bad")
    assert core.input_queue.empty()
    assert set(core.scheduler.requests) == {"good"}
    complete(core, "good")


def test_public_abort_finishes_all_gr_batch_members(core):
    request = _multi_batch_request()
    core.scheduler.scheduler_config.max_num_seqs = 2
    core.scheduler.scheduler_config.max_num_batched_tokens = 8
    core._handle_submit_gr_session(request)
    core._process_input_queue()

    assert set(core.scheduler.requests) == {"session-0", "session-0:1"}
    core.abort_requests([request.batch_id])

    assert not core.scheduler.requests
    assert not getattr(core.scheduler, "_gr_v1_sessions", {})
    assert not getattr(core.scheduler, "_gr_v1_request_sessions", {})


@pytest.mark.parametrize("selector", ["member", "generator", "all"])
@pytest.mark.parametrize("in_flight", [False, True])
def test_native_finish_returns_entire_gr_batch_and_drains_before_release(core, selector, in_flight):
    request = _multi_batch_request()
    scheduler = core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    initial_free = pool.get_num_free_blocks()
    core._handle_submit_gr_session(request)
    core._process_input_queue()
    members = dict(scheduler.requests)
    session = scheduler._gr_v1_sessions[request.batch_id]
    pending = [scheduler.schedule(), scheduler.schedule()] if in_flight else []
    request_ids = {
        "member": "session-0:1",
        "generator": (name for name in ("missing", "session-0:1", "session-0:1")),
        "all": None,
    }[selector]

    finished = scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    assert {item.request_id for item in finished} == set(members)
    assert len(finished) == 2
    assert all(isinstance(item, Request) and item is members[item.request_id] for item in finished)
    assert all(item.status is RequestStatus.FINISHED_ABORTED for item in finished)
    assert not scheduler.requests
    assert not scheduler.running
    assert not scheduler.waiting
    assert not scheduler.skipped_waiting
    assert session.status is GRSessionStatus.ABORTED
    assert scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED) == []

    if in_flight:
        assert session.resource_status is GRResourceStatus.DRAINING
        assert pool.get_num_free_blocks() < initial_free
        for scheduled in pending:
            scheduler.update_from_output(scheduled, receipt(scheduled))
        assert not session._kv_dispatch_leases
        assert session.resource_status is GRResourceStatus.DRAINING
        retirement = scheduler.schedule()
        scheduler.update_from_output(retirement, release_receipt(retirement))
        # Duplicate completion must not free another generation's resources.
        assert consume_gr_retirement(scheduler, retirement, release_receipt(retirement)) is None
    assert session.resource_status is GRResourceStatus.CLOSED
    assert pool.get_num_free_blocks() == initial_free
    assert not scheduler._gr_v1_sessions
    assert not scheduler._gr_v1_request_sessions
    assert getattr(scheduler, "_gr_owner", None) is None


def test_native_abort_all_preserves_none_sentinel(core):
    finished = core.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

    assert finished == []


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_native_add_failure_still_escapes_without_publishing_gr_state(
    core, monkeypatch, error_type
):
    failure = error_type("injected native insertion failure")

    def fail(request):
        raise failure

    # Failure after GR validation, exactly where native queue insertion runs.
    monkeypatch.setattr(core.scheduler, "_enqueue_waiting_request", fail)
    core._handle_submit_gr_session(batch("bad"))
    with pytest.raises(error_type) as raised:
        core._process_input_queue()
    assert raised.value is failure
    assert core.output_queue.empty()
    assert not core.scheduler.requests
    assert not getattr(core.scheduler, "_gr_v1_sessions", {})
    assert getattr(core.scheduler, "_gr_owner_generation", 0) == 0
    assert getattr(core.scheduler, "_gr_execution_capacity", None) is None


def test_inprocess_admission_error_propagates_to_caller(scheduler):
    core = bare_core(scheduler, EngineCore)
    with pytest.raises(GRAdmissionError, match="total native KV capacity"):
        _handle_submit_gr_session(core, batch("bad", prompt=65))
    assert not scheduler.requests
    assert not getattr(scheduler, "_gr_v1_sessions", {})


def test_shutdown_add_keeps_native_abort_semantics(core):
    core.shutdown_state = EngineShutdownState.SHUTTING_DOWN
    core._handle_submit_gr_session(batch("stopping", prompt=65))
    core._process_input_queue()
    client, result = core.output_queue.get_nowait()
    assert client == 3
    assert result.outputs[0].request_id == "stopping"
    assert result.outputs[0].finish_reason is FinishReason.ABORT
    assert core.output_queue.empty()
    assert not core.scheduler.requests
    assert not getattr(core.scheduler, "_gr_v1_sessions", {})


pytestmark = pytest.mark.cpu_test
