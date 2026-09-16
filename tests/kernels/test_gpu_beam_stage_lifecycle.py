# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native input staging and cancellation across asynchronous Beam dispatches."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from tests.kernels.test_gpu_beam_stage_runner import (
    dispatch,
    install_decode_model,
    metadata,
    prepare_binding,
    runtime,
)
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
)
from vllm_gr.v1.worker.gpu_beam_stage_runner import AsyncGPUBeamOutput

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
# mypy: disable-error-code="no-untyped-def,no-untyped-call"


class InputPrepEvent:
    def __init__(self):
        self.event = torch.Event()
        self.waits = 0
        self.records = 0

    def synchronize(self):
        self.waits += 1
        self.event.synchronize()

    def record(self):
        self.records += 1
        self.event.record()


def release(owner):
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)


def warm_stage_kernels(dtype):
    # Both control addresses must be warm: their different alignment can
    # specialize Triton and accidentally hide the pending-DMA race.
    owner = runtime(dtype)
    receipts: list[AsyncGPUBeamOutput] = []
    for stage in range(3):
        if stage == 2:
            receipts[0].get_output()
        execution = owner.admit(dispatch(metadata(stage)))
        if stage:
            prepare_binding(owner, execution.entries[0])
        owner.logits = torch.zeros((1 if stage == 0 else 4, 6), device="cuda", dtype=dtype)
        receipts.append(owner.sample())
    for receipt in receipts:
        receipt.get_output()
    release(owner)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_prefill_dma_drains_before_decode_bookkeeping(monkeypatch, dtype, chunked, graph):
    warm_stage_kernels(dtype)
    owner = runtime(dtype)
    worker = owner.worker
    pinned = torch.zeros(1, pin_memory=True, dtype=torch.int32)
    copied = torch.empty(1, device="cuda", dtype=torch.int32)
    positions = torch.empty(7, device="cuda", dtype=torch.int64)
    offsets = torch.arange(7, device="cuda", dtype=torch.int64)
    # A small KV stand-in makes wrong native positions observable without a
    # model. The native updater, event context and pinned H2D are real.
    kv = torch.zeros(16, device="cuda", dtype=dtype)
    values = torch.ones(7, device="cuda", dtype=dtype)
    prefill_logits = torch.zeros((1, 6), device="cuda", dtype=dtype)
    decode_logits = torch.zeros((4, 6), device="cuda", dtype=dtype)
    dma_done = torch.Event()

    def noop(*args, **kwargs):
        return None

    worker.prepare_inputs_event = InputPrepEvent()
    worker.synchronize_input_prep = MethodType(GPUModelRunner.synchronize_input_prep, worker)
    worker.input_batch = SimpleNamespace(
        req_id_to_index={"session-0": 0},
        num_computed_tokens_cpu_tensor=pinned,
        num_computed_tokens_cpu=pinned.numpy(),
        update_req_spec_token_ids=noop,
        condense=noop,
        refresh_metadata=noop,
    )
    worker.requests = {
        "session-0": SimpleNamespace(
            prev_num_draft_len=0, num_computed_tokens=0, output_token_ids=[], block_ids=[[0]]
        )
    }
    worker.late_interaction_runner = SimpleNamespace(on_requests_finished=noop)
    worker.num_prompt_logprobs = {}
    worker.encoder_cache = {}
    worker.speculative_config = None
    worker.use_async_spec_decode = False
    worker.use_async_scheduling = True
    worker._may_reorder_batch = noop
    worker.model_config = SimpleNamespace(
        enforce_eager=not graph,
        max_model_len=16,
        get_num_attention_heads=lambda _: 1,
        get_num_kv_heads=lambda _: 1,
    )
    worker.parallel_config = SimpleNamespace()
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer"], kv_cache_spec=SimpleNamespace(block_size=16))
        ]
    )
    owner._forward = lambda: decode_logits + owner.input_ids[:, None].to(dtype)
    if graph:
        worker.device = owner.beam.state.device
        worker._beam_context = owner.beam.session.context
        worker.vllm_config = SimpleNamespace()
        worker._vllm_gr_config = SimpleNamespace(beam_max_width=4)
        install_decode_model(worker, lambda ids, pos: decode_logits + ids[:, None].to(dtype))
        owner.prepared = owner.prepare_cpu(metadata(1), 0)
        owner._bind_attention(metadata(1))
        owner.input_ids.zero_()
        owner.capture_forward()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )
    monkeypatch.setattr(
        "vllm_gr.v1.worker.model_runner_common._get_gr_config",
        lambda _: SimpleNamespace(beam_graph_enabled=graph),
    )

    def update(scheduled):
        m = getattr(scheduled, GR_STAGE_OUTPUT_ATTR)["session-0"]
        if m.stage_index == 1:
            assert dma_done.query(), "Decode overwrites a live native Prefill DMA source"
        return GPUModelRunner._update_states(worker, scheduled)

    worker._update_states = update

    def schedule(m, computed, count):
        result = dispatch(m)
        result.num_scheduled_tokens[m.session_id] = count
        result.finished_req_ids = set()
        result.new_block_ids_to_zero = None
        result.free_encoder_mm_hashes = []
        result.scheduled_new_reqs = []
        result.scheduled_spec_decode_tokens = {}
        result.scheduled_cached_reqs = SimpleNamespace(
            req_ids=[m.session_id],
            resumed_req_ids=set(),
            num_computed_tokens=[computed],
            new_block_ids=[None],
            num_output_tokens=[0],
            new_token_ids=[],
        )
        return result

    def native_prefill(runner, scheduled):
        count = scheduled.num_scheduled_tokens["session-0"]
        m = getattr(scheduled, GR_STAGE_OUTPUT_ATTR)["session-0"]
        with runner.synchronize_input_prep():
            runner._update_states(scheduled)
            if m.produces_output:
                torch.cuda._sleep(2_000_000_000)
            copied.copy_(runner.input_batch.num_computed_tokens_cpu_tensor, non_blocking=True)
            dma_done.record()
            torch.add(copied.expand(count), offsets[:count], out=positions[:count])
        kv.index_copy_(0, positions[:count], values[:count])
        runner.execute_model_state = SimpleNamespace(logits=prefill_logits)
        return None

    # Warm Torch's copy/add/index kernels before delaying the producer stream.
    for count in (3, 4, 7):
        copied.copy_(pinned, non_blocking=True)
        torch.add(copied.expand(count), offsets[:count], out=positions[:count])
        kv.index_copy_(0, positions[:count], values[:count])
    kv.zero_()
    torch.accelerator.synchronize()

    dispatch_id = 0
    start = 0
    if chunked:
        owner.execute(schedule(metadata(output=False), 0, 3), native_prefill)
        owner.sample().get_output()
        dispatch_id, start = 1, 3
    owner.execute(schedule(metadata(dispatch=dispatch_id), start, 7 - start), native_prefill)
    prefill = owner.sample()
    assert not dma_done.query(), "test must overlap native H2D with Decode dispatch"
    event = worker.prepare_inputs_event
    waits, records = event.waits, event.records
    owner.execute(schedule(metadata(1, dispatch=dispatch_id + 1), 7, 1), native_prefill)
    first_decode = owner.sample()
    assert event.waits == waits + 1
    assert event.records == records
    prefill.get_output()

    owner.execute(schedule(metadata(2, dispatch=dispatch_id + 2), 8, 1), native_prefill)
    last_decode = owner.sample()
    assert event.waits == waits + 1, "continuous Decode must not wait for native input staging"
    assert event.records == records
    assert sum(owner.control_busy) == 2
    first_decode.get_output()
    last_decode.get_output()
    assert copied.cpu().tolist() == [start]
    assert positions[: 7 - start].cpu().tolist() == list(range(start, 7))
    assert kv.cpu().tolist() == [1] * 7 + [0] * 9
    assert (owner.graph is not None) == graph
    release(owner)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("consume_first", [False, True])
def test_cancelled_chunk_drains_without_initialization(dtype, n, consume_first):
    owner = runtime(dtype, n)
    owner.admit(dispatch(metadata(output=False, n=n)))
    first = owner.sample()
    if consume_first:
        first.get_output()
    queued = dispatch(metadata(dispatch=1, n=n))
    owner.cancel(owner.binding)
    state = owner.beam.state
    snapshot = {
        name: value.clone()
        for name, value in vars(state).items()
        if isinstance(value, torch.Tensor)
    }
    owner.admit(queued)
    # Cancellation must not require or inspect logits from a drain-only chunk.
    owner.logits = None
    final = owner.sample()
    before = getattr(first.get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
    after = getattr(final.get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
    assert before.dispatch_id == 0 and not before.finished and before.produced_token_count == 0
    assert after.dispatch_id == 1 and after.finished and after.produced_token_count == 0
    assert after.terminal_result is None
    for name, value in snapshot.items():
        torch.testing.assert_close(getattr(state, name), value, rtol=0, atol=0)
    release(owner)
    owner.beam.state.reset(owner.binding.state_row)
    assert not state.is_finished[owner.binding.state_row]
    assert not state.is_initialized[owner.binding.state_row]
    for name, value in snapshot.items():
        torch.testing.assert_close(getattr(state, name)[:2], value[:2], rtol=0, atol=0)
