# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="no-untyped-def,no-untyped-call,misc"
"""Asynchronous scheduling admission/retirement against the installed native AsyncScheduler."""

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.platforms import current_platform
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from tests.test_gr_async_scheduler import _batch_request
from vllm_gr.v1.engine.beam_search_v1 import (
    GRResourceStatus,
    GRRetireResult,
    GRWorkerResult,
)
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_RETIRE_OUTPUT_ATTR,
    GR_RETIRE_RESULT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    consume_gr_retirement,
)

pytestmark = pytest.mark.skipif(
    current_platform.device_type == "npu",
    reason="V1 async scheduler tests are temporarily disabled on NPU",
)


@pytest.fixture
def scheduler(tmp_path, request):
    if current_platform.device_type == "npu":
        pytest.skip("V1 async scheduler tests are temporarily disabled on NPU")

    from vllm_gr.v1.engine.engine_core_patch import apply_scheduler_patch

    apply_scheduler_patch()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "vocab_size": 16,
                "max_position_embeddings": 64,
            }
        )
    )
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            max_model_len=64,
            enforce_eager=True,
            dtype="half",
        ),
        cache_config=CacheConfig(
            block_size=16, enable_prefix_caching=getattr(request, "param", False)
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=64,
            is_encoder_decoder=False,
            max_num_batched_tokens=64,
            max_num_seqs=4,
            async_scheduling=True,
            enable_chunked_prefill=False,
        ),
        additional_config={"vllm_gr_beam_execution_mode": "v1", "vllm_gr_beam_max_width": 4},
    )
    kv = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16),
            )
        ],
    )
    config.cache_config.num_gpu_blocks = 8
    return AsyncScheduler(
        config, kv, SimpleNamespace(should_advance=lambda _: False), block_size=16
    )


def add(scheduler, name, prompt=7, n=3):
    batch = _batch_request(n)
    batch.items[0].request.request_id = name
    batch.items[0].request.prompt_token_ids = [1] * prompt
    batch.items[0].request.sampling_params.ignore_eos = True
    batch.items[0].request.sampling_params.stop_token_ids = []
    batch = msgspec.structs.replace(batch, batch_id=name)
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher

    if not hasattr(kv_cache_utils, "NONE_HASH"):
        kv_cache_utils.init_none_hash(sha256)

    native = Request.from_engine_core_request(
        batch.items[0].request, block_hasher=get_request_block_hasher(16, sha256)
    )
    native.gr_batch_request = batch
    scheduler.add_request(native)
    return scheduler._gr_v1_sessions[name]


def receipt(scheduled, *, finished=None, released=None):
    mapping = getattr(scheduled, GR_STAGE_OUTPUT_ATTR, {})
    ids, sampled, results = [], [], {}
    for name, m in mapping.items():
        ids.append(name)
        sampled.append([0] if m.produces_output else [])
        worker_finished = m.is_last_stage if finished is None else finished
        resources_released = (
            bool(worker_finished and m.is_last_stage) if released is None else released
        )
        results[name] = GRWorkerResult(
            name,
            m.dispatch_id,
            m.stage_index,
            int(m.produces_output),
            m.output_device_state,
            worker_finished,
            resources_released=resources_released,
            owner_generation=m.owner_generation,
        )
    out = ModelRunnerOutput(
        req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)}, sampled_token_ids=sampled
    )
    setattr(out, GR_WORKER_RESULT_ATTR, results)
    return out


def release_receipt(scheduled: Any) -> ModelRunnerOutput:
    m = getattr(scheduled, GR_RETIRE_OUTPUT_ATTR)
    out = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
    setattr(
        out,
        GR_RETIRE_RESULT_ATTR,
        GRRetireResult(m.session_id, m.owner_generation, m.retire_id, True),
    )
    return out


def test_fifo_owner_handoffs_on_terminal_receipt_without_retirement(scheduler):
    a = add(scheduler, "A")
    b = add(scheduler, "B")
    c = add(scheduler, "C")
    first, second = scheduler.schedule(), scheduler.schedule()
    assert list(first.num_scheduled_tokens) == list(second.num_scheduled_tokens) == ["A"]
    assert a._in_flight_dispatches == {0, 1}
    assert b.resource_status is c.resource_status is GRResourceStatus.WAITING
    assert scheduler.schedule().total_num_scheduled_tokens == 0
    assert [r.request_id for r in scheduler.waiting] == ["B", "C"]
    scheduler.finish_requests("B", RequestStatus.FINISHED_ABORTED)
    assert b.resource_status is GRResourceStatus.CLOSED
    scheduler.update_from_output(first, receipt(first))
    third = scheduler.schedule()
    assert list(third.num_scheduled_tokens) == ["A"]
    scheduler.update_from_output(second, receipt(second))
    scheduler.update_from_output(third, receipt(third))
    assert a.resource_status is GRResourceStatus.CLOSED
    assert a._terminal_emitted
    assert list(scheduler.schedule().num_scheduled_tokens) == ["C"]


def test_early_terminal_releases_on_last_lookahead_receipt(scheduler):
    session = add(scheduler, "A")
    add(scheduler, "B")
    first, successor = scheduler.schedule(), scheduler.schedule()

    scheduler.update_from_output(first, receipt(first, finished=True, released=False))
    assert session.resource_status is GRResourceStatus.DRAINING
    assert session._in_flight_dispatches

    scheduler.update_from_output(successor, receipt(successor, finished=True, released=True))
    assert session.resource_status is GRResourceStatus.CLOSED
    assert session._terminal_emitted
    next_output = scheduler.schedule()
    assert getattr(next_output, GR_RETIRE_OUTPUT_ATTR, None) is None
    assert list(next_output.num_scheduled_tokens) == ["B"]


def test_chunked_prefill_rejected_before_registration(scheduler):
    scheduler.scheduler_config.enable_chunked_prefill = True
    with pytest.raises(ValueError, match="enable_chunked_prefill=False"):
        add(scheduler, "A", prompt=49)
    assert not getattr(scheduler, "_gr_v1_sessions", {})
    assert not getattr(scheduler, "_gr_owner", None)


def test_stale_owner_release_proof_cannot_finalize_draining_session(scheduler):
    session = add(scheduler, "A")
    add(scheduler, "B")
    first, successor = scheduler.schedule(), scheduler.schedule()
    scheduler.update_from_output(first, receipt(first, finished=True, released=False))

    stale = receipt(successor, finished=True, released=True)
    results = getattr(stale, GR_WORKER_RESULT_ATTR)
    results["A"] = replace(results["A"], owner_generation=session.owner_generation + 1)
    scheduler.update_from_output(successor, stale)

    assert session.worker_resources_allocated
    assert session.resource_status is GRResourceStatus.DRAINING
    assert not session._terminal_emitted
    assert scheduler._gr_owner == "A"
    assert session._pending_terminal_result.failure.code == "worker_protocol_error"

    retirement = scheduler.schedule()
    assert getattr(retirement, GR_RETIRE_OUTPUT_ATTR).session_id == "A"
    scheduler.update_from_output(retirement, release_receipt(retirement))
    assert session.resource_status is GRResourceStatus.CLOSED
    assert session._terminal_emitted
    assert list(scheduler.schedule().num_scheduled_tokens) == ["B"]


def test_last_cancelled_request_keeps_work_until_release(scheduler):
    session = add(scheduler, "A")
    first, second = scheduler.schedule(), scheduler.schedule()
    scheduler.finish_requests("A", RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(first, receipt(first))
    assert session.resource_status is GRResourceStatus.DRAINING
    assert getattr(scheduler.schedule(), GR_RETIRE_OUTPUT_ATTR, None) is None
    scheduler.update_from_output(second, receipt(second))
    assert scheduler.has_requests()
    retirement = scheduler.schedule()
    scheduler.update_from_output(retirement, release_receipt(retirement))
    assert session.resource_status is GRResourceStatus.CLOSED
    assert not scheduler.has_requests()


def test_wrong_and_duplicate_retirement_cannot_release_another_owner(scheduler):
    a = add(scheduler, "A", n=1)
    add(scheduler, "B")
    first = scheduler.schedule()
    scheduler.update_from_output(first, receipt(first, released=False))
    retire = scheduler.schedule()
    out = release_receipt(retire)
    correct = getattr(out, GR_RETIRE_RESULT_ATTR)
    setattr(
        out, GR_RETIRE_RESULT_ATTR, replace(correct, owner_generation=correct.owner_generation + 1)
    )
    with pytest.raises(RuntimeError, match="prove release"):
        consume_gr_retirement(scheduler, retire, out)
    assert scheduler._gr_owner == "A"
    scheduler.update_from_output(retire, release_receipt(retire))
    scheduler.schedule()
    consume_gr_retirement(scheduler, retire, release_receipt(retire))
    assert a.resource_status is GRResourceStatus.CLOSED
    assert scheduler._gr_owner == "B"


def test_oversized_request_fails_before_owner_or_worker_allocation(scheduler):
    scheduler.kv_cache_manager.max_model_len = 256
    with pytest.raises(ValueError, match="total native KV capacity"):
        add(scheduler, "too-large", prompt=129)
    assert not getattr(scheduler, "_gr_v1_sessions", {})
    assert getattr(scheduler, "_gr_owner", None) is None


def test_temporary_kv_shortage_applies_backpressure(scheduler):
    pool = scheduler.kv_cache_manager.block_pool
    held = pool.get_new_blocks(6)
    session = add(scheduler, "A", prompt=33)
    assert scheduler.schedule().total_num_scheduled_tokens == 0
    assert not session.worker_resources_allocated
    pool.free_blocks(held)
    assert scheduler.schedule().num_scheduled_tokens == {"A": 33}
    assert len(scheduler.kv_cache_manager.get_blocks("A").blocks[0]) == 3


def test_native_batch_queue_submits_successor_before_consuming_receipt(scheduler):
    from collections import deque
    from concurrent.futures import Future
    from contextlib import nullcontext

    from vllm.v1.engine.core import EngineCore

    events = []

    class Deferred(Future[ModelRunnerOutput]):
        def __init__(self, output, identity):
            super().__init__()
            self.output, self.identity = output, identity

        def result(self, timeout=None):
            events.append(("consume", self.identity))
            return self.output

    class Executor:
        def execute_model(self, output, non_block):
            assert non_block
            self.current = output
            m = getattr(output, GR_STAGE_OUTPUT_ATTR, {}).get("A")
            identity = m.dispatch_id if m is not None else "cleanup"
            events.append(("execute", identity))
            result = (
                release_receipt(output)
                if hasattr(output, GR_RETIRE_OUTPUT_ATTR)
                else receipt(output)
            )
            return Deferred(result, identity)

        def sample_tokens(self, grammar, non_block):
            m = getattr(self.current, GR_STAGE_OUTPUT_ATTR)["A"]
            events.append(("sample", m.dispatch_id))
            return Deferred(receipt(self.current), m.dispatch_id)

    core = SimpleNamespace(
        scheduler=scheduler,
        batch_queue=deque(),
        batch_queue_size=2,
        is_ec_consumer=True,
        is_pooling_model=False,
        model_executor=Executor(),
        log_error_detail=lambda _: nullcontext(),
        log_iteration_details=lambda _: nullcontext(),
        _process_aborts_queue=lambda: None,
    )
    add(scheduler, "A")
    first, ran = EngineCore.step_with_batch_queue(core)
    assert first is None and ran
    EngineCore.step_with_batch_queue(core)
    assert events[:5] == [
        ("execute", 0),
        ("sample", 0),
        ("execute", 1),
        ("sample", 1),
        ("consume", 0),
    ]
    for _ in range(10):
        if not scheduler.has_requests() and not core.batch_queue:
            break
        EngineCore.step_with_batch_queue(core)
    assert not scheduler.has_requests() and not core.batch_queue


def test_async_output_failure_reaches_native_multiproc_failure_channel():
    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.outputs import AsyncModelRunnerOutput

    from vllm_gr.v1.worker.gr_async_output import install_async_output_guard

    class BrokenD2H(AsyncModelRunnerOutput):
        def get_output(self):
            raise RuntimeError("injected D2H failure")

    install_async_output_guard()
    responses: list[tuple[Any, object]] = []
    worker = SimpleNamespace(worker_response_mq=SimpleNamespace(enqueue=responses.append))
    WorkerProc.enqueue_output(worker, BrokenD2H())
    WorkerProc.enqueue_output(worker, "next output")
    assert responses == [
        (WorkerProc.ResponseStatus.FAILURE, "injected D2H failure"),
        (WorkerProc.ResponseStatus.SUCCESS, "next output"),
    ]


@pytest.mark.parametrize("scheduler", [True], indirect=True)
def test_prefix_cache_hit_reuses_only_full_prompt_blocks(scheduler):
    add(scheduler, "A", prompt=33, n=1)
    while True:
        output = scheduler.schedule()
        scheduler.update_from_output(output, receipt(output))
        if "A" not in scheduler._gr_v1_sessions:
            break
    add(scheduler, "B", prompt=33, n=3)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"B": 1}
    assert output.scheduled_new_reqs[0].num_computed_tokens == 32
    m = getattr(output, GR_STAGE_OUTPUT_ATTR)["B"]
    assert m.produces_output and m.stage_index == 0
    assert len(scheduler.kv_cache_manager.get_blocks("B").blocks[0]) == 3


def test_duplicate_receipt_never_reenters_native_accounting(scheduler):
    session = add(scheduler, "A")
    first = scheduler.schedule()
    second = scheduler.schedule()
    native = scheduler.requests["A"]
    assert session.issued_stages == 2
    scheduler.update_from_output(first, receipt(first))
    before = (
        native.num_output_tokens,
        native.num_output_placeholders,
        tuple(block.ref_cnt for block in scheduler.kv_cache_manager.get_blocks("A").blocks[0]),
    )
    assert scheduler.update_from_output(first, receipt(first)) == {}
    after = (
        native.num_output_tokens,
        native.num_output_placeholders,
        tuple(block.ref_cnt for block in scheduler.kv_cache_manager.get_blocks("A").blocks[0]),
    )
    assert before == after
    assert session._in_flight_dispatches == {1}
    assert scheduler.requests["A"] is native
    scheduler.update_from_output(second, receipt(second))


def test_worker_input_detaches_retained_native_blocks(scheduler):
    from vllm_gr.v1.worker.gr_execution import worker_input

    add(scheduler, "A")
    scheduled = scheduler.schedule()
    view = worker_input(scheduled)
    original = scheduled.scheduled_new_reqs[0]
    local = view.scheduled_new_reqs[0]
    original_blocks = tuple(tuple(group) for group in original.block_ids)
    local.block_ids[0].append(999)
    local.prompt_token_ids.append(999)
    local.sampling_params.stop_token_ids = [999]
    view.num_scheduled_tokens["A"] = 256
    view.total_num_scheduled_tokens = 256
    assert tuple(tuple(group) for group in original.block_ids) == original_blocks
    assert original.prompt_token_ids == [1] * 7
    assert original.sampling_params.stop_token_ids == []
    assert scheduled.num_scheduled_tokens == {"A": 7}
    assert scheduled.total_num_scheduled_tokens == 7
    assert view.gr_stage_metadata is scheduled.gr_stage_metadata
