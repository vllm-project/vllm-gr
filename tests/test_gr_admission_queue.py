# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="no-untyped-def,no-untyped-call,misc"
"""GR rejection through native EngineCore ADD queues, without a model/GPU."""

from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequestType, FinishReason
from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState
from vllm.v1.outputs import ModelRunnerOutput

from tests.test_async_scheduling_scheduler import receipt, release_receipt
from tests.test_async_scheduling_scheduler import scheduler as _scheduler_fixture
from tests.test_gr_async_scheduler import _batch_request
from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchOutput,
    BeamBatchResult,
    BeamOutputSequence,
    BeamRequestOutput,
)
from vllm_gr.v1.engine.beam_search_v1 import GRResourceStatus, initialize_gr_admission
from vllm_gr.v1.engine.core import _handle_submit_gr_session
from vllm_gr.v1.engine.engine_core_patch import apply_engine_core_request_patches
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_BATCH_RESULT_ATTR,
    GR_RETIRE_OUTPUT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GRAdmissionError,
)

scheduler = _scheduler_fixture


def batch(name, *, prompt=7, n=3, client=3):
    result = _batch_request(n)
    request = result.items[0].request
    request.request_id = name
    request.prompt_token_ids = [1] * prompt
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
    initialize_gr_admission(core)
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
    assert core.output_queue.empty()


def worker_receipt(scheduler, scheduled):
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
            # delivery/release, not model generation or the device adapter.
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
        bad.items[0].request.prompt_token_ids = [1] * 129
    else:
        params = msgspec.structs.replace(
            bad.beam_params, **({"beam_width": 8} if rejection == "width" else {"max_tokens": 4})
        )
        bad = msgspec.structs.replace(bad, beam_params=params)
    free = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    core._handle_submit_gr_session(bad)
    # Public budget mismatches fail before ADD. Runtime KV/width admission
    # remains on the core thread, before registering any owner.
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


def test_core_checks_once_and_scheduler_consumes_each_submission(core, monkeypatch):
    from vllm_gr.v1.engine import beam_search_v1 as contract
    from vllm_gr.v1.engine import gr_async_scheduler

    validate = Mock(wraps=contract.validate_gr_request)
    unexpected = Mock(side_effect=AssertionError("repeated admission check"))
    monkeypatch.setattr(contract, "validate_gr_request", validate)
    for module in (contract, gr_async_scheduler):
        for name in ("validate_gr_execution_config", "validate_gr_device_config"):
            monkeypatch.setattr(module, name, unexpected)
    monkeypatch.setattr(gr_async_scheduler, "validate_gr_request", unexpected)
    for name in ("first", "second"):
        core._handle_submit_gr_session(batch(name))
        _, (native, _) = core.input_queue.queue[0]
        admission = native._gr_request_admission
        assert admission.runtime is core.scheduler._gr_admission_runtime
        core._process_input_queue()
        assert not hasattr(native, "_gr_request_admission")
        complete(core, name)
    assert validate.call_count == 2
    unexpected.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "changed"),
        ("prompt_token_ids", []),
        ("prompt_embeds", object()),
        ("mm_features", [object()]),
        ("lora_request", object()),
        ("pooling_params", object()),
        ("sampling_params", None),
        ("ignore_eos", False),
        ("max_tokens", 1),
        ("n", 2),
        ("prompt_length", 8),
    ],
)
def test_mutation_between_core_validation_and_scheduler_add_is_rejected(core, field, value):
    request = batch("changed-in-queue")
    core._handle_submit_gr_session(request)
    native = request.items[0].request
    if field in ("ignore_eos", "max_tokens", "n"):
        setattr(native.sampling_params, field, value)
    elif field == "prompt_length":
        native.prompt_token_ids.append(value)
    else:
        setattr(native, field, value)
    core._process_input_queue()
    assert_one_error(core, "changed-in-queue")
    assert not core.scheduler.requests
    assert not getattr(core.scheduler, "_gr_v1_sessions", {})
    assert getattr(core.scheduler, "_gr_owner_generation", 0) == 0


@pytest.mark.parametrize("replacement", ["other_request", "other_runtime", "invalid_receipt"])
def test_admission_cannot_be_reused_for_another_request_or_runtime(core, replacement):
    from dataclasses import replace

    request = batch("target")
    core._handle_submit_gr_session(request)
    _, (native, _) = core.input_queue.queue[0]
    admission = native._gr_request_admission
    if replacement == "other_request":
        native.gr_batch_request = batch("target")
    elif replacement == "other_runtime":
        native._gr_request_admission = replace(admission, runtime=replace(admission.runtime))
    else:
        native._gr_request_admission = True
    core._process_input_queue()
    assert_one_error(core, "target")
    assert not core.scheduler.requests


@pytest.mark.parametrize("container", ["items", "stop_token_ids"])
def test_mutable_contract_containers_keep_scheduler_validation(core, container):
    request = batch("mutable-contract")
    if container == "items":
        request = msgspec.structs.replace(request, items=list(request.items))
    else:
        params = msgspec.structs.replace(request.beam_params, stop_token_ids=[2])
        request = msgspec.structs.replace(request, beam_params=params)
    core._handle_submit_gr_session(request)
    _, (native, _) = core.input_queue.queue[0]
    assert not hasattr(native, "_gr_request_admission")
    if container == "items":
        request.items[0] = msgspec.structs.replace(request.items[0], item_index=1)
    else:
        request.beam_params.stop_token_ids.append(-1)
    core._process_input_queue()
    assert_one_error(core, "mutable-contract")
    assert not core.scheduler.requests
    assert not getattr(core.scheduler, "_gr_v1_sessions", {})


def _startup_core():
    config = SimpleNamespace(
        additional_config={"vllm_gr_beam_execution_mode": "v1", "vllm_gr_beam_max_width": 4},
        scheduler_config=SimpleNamespace(async_scheduling=True, enable_chunked_prefill=False),
        device_config=SimpleNamespace(device="cuda"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        model_config=SimpleNamespace(dtype="torch.bfloat16"),
    )
    return SimpleNamespace(
        vllm_config=config,
        scheduler=SimpleNamespace(vllm_config=config),
        batch_queue=deque(),
        batch_queue_size=2,
    )


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("scheduler_config", "async_scheduling", False, "async_scheduling=True"),
        ("scheduler_config", "enable_chunked_prefill", True, "enable_chunked_prefill=False"),
        ("scheduler_config", "long_prefill_token_threshold", 1, "partial Prefill"),
        ("device_config", "device", "npu", "GPU V1 required"),
        ("parallel_config", "tensor_parallel_size", 2, "TP/PP/DP/DCP/PCP=1"),
        ("model_config", "dtype", "torch.float32", "FP16 or BF16"),
        ("model_config", "is_multimodal_model", True, "uniform text decoder"),
        ("model_config", "quantization", "awq", "quantization"),
    ],
)
def test_invalid_static_contract_is_rejected_at_initialization(section, field, value, message):
    core = _startup_core()
    setattr(getattr(core.vllm_config, section), field, value)
    with pytest.raises(ValueError, match=message):
        initialize_gr_admission(core)
    assert not hasattr(core, "_gr_admission_runtime")
    assert not hasattr(core.scheduler, "_gr_admission_runtime")


@pytest.mark.parametrize("size", [0, 1])
def test_missing_batch_queue_is_rejected_at_initialization(size):
    core = _startup_core()
    core.batch_queue_size = size
    with pytest.raises(RuntimeError, match="non-blocking EngineCore batch queue"):
        initialize_gr_admission(core)


def test_missing_fixed_width_is_rejected_at_initialization():
    core = _startup_core()
    del core.vllm_config.additional_config["vllm_gr_beam_max_width"]
    with pytest.raises(ValueError, match="beam_max_width to be fixed at startup"):
        initialize_gr_admission(core)


def test_legacy_initialization_does_not_require_v1_constraints(monkeypatch):
    from vllm_gr.v1.engine import beam_search_v1 as contract

    core = _startup_core()
    core.vllm_config.additional_config = {}
    core.vllm_config.scheduler_config.async_scheduling = False
    core.vllm_config.device_config.device = "cpu"
    unexpected = Mock(side_effect=AssertionError("legacy initialized a V1 contract"))
    monkeypatch.setattr(contract, "validate_gr_execution_config", unexpected)
    initialize_gr_admission(core)
    assert core._gr_admission_runtime is None
    assert core.scheduler._gr_admission_runtime is None
    unexpected.assert_not_called()


def test_nested_engine_constructors_validate_before_input_start_once(monkeypatch):
    from vllm_gr.v1.engine import beam_search_v1 as contract
    from vllm_gr.v1.engine.engine_core_patch import make_patched_engine_core_init

    validate = Mock(wraps=contract.validate_gr_execution_config)
    monkeypatch.setattr(contract, "validate_gr_execution_config", validate)

    def base_init(core):
        core.__dict__.update(_startup_core().__dict__)

    base = make_patched_engine_core_init(base_init)

    def proc_init(core):
        base(core)
        assert core._gr_admission_runtime is core.scheduler._gr_admission_runtime
        assert core._gr_admission_runtime.capacity.beam_width == 4
        core.input_started = True

    core = SimpleNamespace()
    make_patched_engine_core_init(proc_init)(core)
    assert core.input_started
    validate.assert_called_once()
