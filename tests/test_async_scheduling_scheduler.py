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
            max_num_batched_tokens=16,
            max_num_seqs=4,
            async_scheduling=True,
            enable_chunked_prefill=True,
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


def receipt(scheduled, *, finished=None):
    mapping = getattr(scheduled, GR_STAGE_OUTPUT_ATTR, {})
    ids, sampled, results = [], [], {}
    for name, m in mapping.items():
        ids.append(name)
        sampled.append([0] if m.produces_output else [])
        results[name] = GRWorkerResult(
            name,
            m.dispatch_id,
            m.stage_index,
            int(m.produces_output),
            m.output_device_state,
            m.is_last_stage if finished is None else finished,
        )
    out = ModelRunnerOutput(
        req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)}, sampled_token_ids=sampled
    )
    setattr(out, GR_WORKER_RESULT_ATTR, results)
    return out


def release_receipt(scheduled):
    m = getattr(scheduled, GR_RETIRE_OUTPUT_ATTR)
    out = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
    setattr(
        out,
        GR_RETIRE_RESULT_ATTR,
        GRRetireResult(m.session_id, m.owner_generation, m.retire_id, True),
    )
    return out


def test_fifo_owner_retains_second_slot_and_waits_for_retirement(scheduler):
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
    assert a.resource_status is GRResourceStatus.DRAINING
    assert a._terminal_emitted
    retirement = scheduler.schedule()
    assert retirement.total_num_scheduled_tokens == 0
    assert a.resource_status is GRResourceStatus.RETIRING
    assert scheduler.schedule().total_num_scheduled_tokens == 0
    scheduler.update_from_output(retirement, release_receipt(retirement))
    assert a.resource_status is GRResourceStatus.CLOSED
    assert list(scheduler.schedule().num_scheduled_tokens) == ["C"]


def test_chunked_prefill_saturates_without_output_placeholders(scheduler):
    a = add(scheduler, "A", prompt=49)
    add(scheduler, "B")
    first, second = scheduler.schedule(), scheduler.schedule()
    assert first.num_scheduled_tokens == second.num_scheduled_tokens == {"A": 16}
    assert scheduler.requests["A"].num_output_placeholders == 0
    assert a.issued_stages == 0
    assert len(scheduler.kv_cache_manager.get_blocks("A").blocks[0]) == 4
    assert scheduler.schedule().total_num_scheduled_tokens == 0
    assert scheduler.requests["A"].num_computed_tokens == 32
    scheduler.update_from_output(first, receipt(first))
    assert scheduler.schedule().num_scheduled_tokens == {"A": 16}


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
    scheduler.update_from_output(first, receipt(first))
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
    assert scheduler.schedule().num_scheduled_tokens == {"A": 16}
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
        if scheduler._gr_v1_sessions["A"]._terminal_emitted:
            break
    retirement = scheduler.schedule()
    scheduler.update_from_output(retirement, release_receipt(retirement))
    add(scheduler, "B", prompt=33, n=3)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"B": 1}
    assert output.scheduled_new_reqs[0].num_computed_tokens == 32
    m = getattr(output, GR_STAGE_OUTPUT_ATTR)["B"]
    assert m.produces_output and m.stage_index == 0
    assert len(scheduler.kv_cache_manager.get_blocks("B").blocks[0]) == 3
