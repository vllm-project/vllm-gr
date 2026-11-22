# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="no-untyped-def,no-untyped-call,misc"
"""CUDA asynchronous scheduling preparation, output drain and fixed-address graph contracts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
import torch

from tests.kernels.beam_stage_test_utils import (
    dispatch,
    metadata,
    prepare_binding,
    runtime,
)
from vllm_gr.v1.attention.backends import beam_attn_metadata
from vllm_gr.v1.attention.backends import beam_attn_metadata_layout as layout
from vllm_gr.v1.attention.backends import beam_attn_metadata_tensors as tensors
from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadataBuilder,
    build_async,
)
from vllm_gr.v1.attention.backends.beam_attn_metadata_layout import build_async_layout
from vllm_gr.v1.beam.dispatch import (
    GR_RETIRE_RESULT_ATTR,
    GR_WORKER_RESULT_ATTR,
    GRRetireMetadata,
)
from vllm_gr.v1.worker import gr_gc as _gc_runtime
from vllm_gr.v1.worker.cuda.gr_inputs import ERROR_DISPATCH_STATE, GRInputBuffers
from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("width", [1, 4, 128, 256])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_each_decode_rebuilds_without_cache_or_device_reads(monkeypatch, width, n):
    buffers = GRInputBuffers(width, 96, 16, "cuda")
    binding = SimpleNamespace(session_id="session-0", native_request_id="session-0")
    # Prefix planning reads the KV geometry, never the tensor contents.
    context = SimpleNamespace(
        max_decode_steps=n,
        dtype=torch.float16,
        pools=[
            SimpleNamespace(
                unshared_key=torch.empty((1, width, 1, n, 64), device="cuda", dtype=torch.float16)
            )
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Asynchronous scheduling used a cache or read generated device state on the CPU"
        )

    first = None
    for step in range(n - 1):
        m = replace(
            metadata(step + 1, n=n),
            num_prompt_tokens=17 + step,
            num_prompt_tokens_by_request=(17 + step,),
            beam_params=msgspec.structs.replace(metadata(n=n).beam_params, beam_width=width),
        )
        with monkeypatch.context() as patch:
            patch.setattr(BeamAttentionMetadataBuilder, "_geometry_cache", forbidden)
            # Future async implementations must not fall back to the legacy
            # geometry/layout caches or the H2D upload helpers.
            for module, name in (
                (beam_attn_metadata, "_geom_cache_key"),
                (layout, "_geom_cache_key"),
                (layout, "_segment_requests"),
                (layout, "_build_beam_mappings"),
                (tensors, "cached_seqlens_views"),
                (tensors, "upload_incremental_seqlens"),
                (tensors, "pack_int32_to_dev"),
                (tensors, "get_schedule"),
            ):
                patch.setattr(module, name, forbidden)
            for name in ("item", "cpu", "tolist"):
                patch.setattr(torch.Tensor, name, forbidden)
            prepared = buffers.prepare_cpu(m, binding, block_ids=[[2 + step, 4]], slot=step % 2)
            views = buffers.upload(prepared)
            built = build_async(
                layout=prepared.layout,
                views=views,
                session=SimpleNamespace(is_active=True, context=context),
                mask=torch.ones(width, device="cuda", dtype=torch.bool),
                slots=torch.full((width,), -1, device="cuda", dtype=torch.int64),
                block_size=16,
                max_model_len=96,
                num_heads=1,
                num_kv_heads=1,
            )
        assert built.num_reqs == 1
        assert built.query_start_loc.tolist() == [0, width]
        assert built.seq_lens.tolist() == [17 + 2 * step + 1]
        assert built.suffix_seqlens_kv_device.tolist() == [step + 1] * width
        assert built.suffix_cu_seqlens_k_device.tolist() == [
            i * (step + 1) for i in range(width + 1)
        ]
        assert prepared.layout["positions"] == [17 + 2 * step] * width
        assert built.prefix_seqlens_kv_cpu == [17 + step]
        if first is not None:
            assert built is not first
            assert built.prefix_block_table.data_ptr() == first.prefix_block_table.data_ptr()
            assert first.prefix_seqlens_kv_cpu == [17 + step - 1]
        first = built


@pytest.mark.parametrize("mirror", [None, "device"])
def test_missing_cpu_mirrors_fail_without_device_fallback(mirror, monkeypatch):
    query = None if mirror is None else torch.tensor([0, 4], device="cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("device mirror was read on the Host")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    with pytest.raises(ValueError, match="CPU query/sequence mirrors"):
        build_async_layout(
            width=4,
            prompt_len=7,
            decode_step=0,
            block_ids=[1],
            max_blocks=4,
            query_start_loc_cpu=query,
            seq_lens_cpu=None,
        )


def test_staging_slot_waits_for_its_own_h2d_before_host_write():
    buffers = GRInputBuffers(4, 64, 16, "cuda")
    binding = SimpleNamespace(session_id="session-0", native_request_id="session-0")
    first = buffers.prepare_cpu(metadata(1), binding, block_ids=[[1]], slot=0)
    buffers.upload(first)
    torch.accelerator.synchronize()
    saved = buffers.host[0].clone()
    waiting, allow = Event(), Event()

    class DelayedCopy:
        def query(self):
            return False

        def synchronize(self):
            waiting.set()
            assert allow.wait(10)

    buffers.h2d_done[0] = DelayedCopy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(buffers.prepare_cpu, metadata(2), binding, block_ids=[[2]], slot=0)
        assert waiting.wait(10)
        assert torch.equal(buffers.host[0], saved)
        # The other dispatch has an independent pinned source.
        buffers.prepare_cpu(metadata(2), binding, block_ids=[[3]], slot=1)
        assert torch.equal(buffers.host[0], saved)
        allow.set()
        future.result(timeout=10)
    assert not torch.equal(buffers.host[0], saved)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_planned_state_mismatch_is_a_device_error_with_no_generated_successor(dtype):
    owner = runtime(dtype)
    owner.admit(dispatch(metadata()))
    owner.logits = torch.zeros((1, 6), device="cuda", dtype=dtype)
    first = owner.sample()
    owner.beam.state.num_decode_steps[0:1].fill_(1)  # Planned Decode is zero.
    execution = owner.admit(dispatch(metadata(1)))
    prepare_binding(owner, execution.entries[0])
    assert owner.beam.state.error_code[0].item() == ERROR_DISPATCH_STATE
    assert owner.input_ids.tolist() == [0] * 4
    assert owner.positions.tolist() == [0] * 4
    assert not owner.execution_mask.any().item()
    assert owner.beam.session.context.decode_step_buf.item() == -1
    assert owner.metadata.suffix_seqlens_kv_device.tolist() == [1] * 4
    owner.logits = torch.zeros((4, 6), device="cuda", dtype=dtype)
    second = owner.sample()
    assert not first._output  # Successor was submitted before predecessor consumption.
    first.get_output()
    result = getattr(second.get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
    assert result.produced_token_count == 0
    assert result.finished and result.terminal_result.failure.code == "beam_device_error"


def test_retirement_waits_for_output_consumer_and_returns_release_proof(monkeypatch):
    freezes: list[str] = []
    monkeypatch.setattr(
        _gc_runtime,
        "schedule_warmup_gc_freeze",
        lambda: freezes.append("scheduled"),
    )
    owner = runtime()
    owner.admit(dispatch(metadata()))
    owner.logits = torch.zeros((1, 6), device="cuda", dtype=torch.float16)
    receipt = owner.sample()
    retirement = owner.request_release(GRRetireMetadata("session-0", 0, 1))
    assert owner.cancelled and not owner.closed
    started = Event()

    def consume():
        started.set()
        return retirement.get_output()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(consume)
        assert started.wait(10)
        assert not future.done()
        assert not owner.closed
        receipt.get_output()
        output = future.result(timeout=10)
    assert getattr(output, GR_RETIRE_RESULT_ATTR).released
    assert owner.closed
    assert owner.beam.session.context.registry.num_active_sessions == 0
    assert retirement.get_output() is output
    assert freezes == ["scheduled"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_legal_eos_is_an_ordinary_candidate_with_delayed_receipts(dtype):
    owner = runtime(dtype, stop_token_ids=(1,))
    receipts: list[Any] = []
    for step in range(3):
        if step == 2:
            receipts[0].get_output()
        m = metadata(step)
        m = replace(m, beam_params=msgspec.structs.replace(m.beam_params, stop_token_ids=(1,)))
        execution = owner.admit(dispatch(m))
        if step:
            prepare_binding(owner, execution.entries[0])
        owner.logits = torch.zeros((1 if step == 0 else 4, 6), device="cuda", dtype=dtype)
        receipts.append(owner.sample())
        if step == 0:
            assert owner.beam.state.active_mask[0].sum().item() == 2
            assert owner.beam.state.num_completed[0].item() == 0
    for step, receipt in enumerate(receipts):
        result = getattr(receipt.get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
        assert result.produced_token_count == 1
        if step < len(receipts) - 1:
            assert not result.finished and result.terminal_result is None
        else:
            assert result.finished
            assert result.terminal_result is not None
            assert result.terminal_result.failure is None
            assert result.resources_released
    count = owner.beam.state.num_completed[0].item()
    lengths = owner.beam.state.completed_lengths[0, :count].tolist()
    assert lengths and all(length == 3 for length in lengths)
    assert 1 in owner.beam.state.history_token_ids[0, 0].tolist()
    assert owner.closed


@torch.inference_mode()
def test_startup_capture_is_reused_across_prompt_lengths_and_budgets(monkeypatch):
    from tests.kernels.beam_stage_test_utils import (
        admission_worker,
        install_decode_model,
    )
    from vllm_gr.v1.worker.cuda.beam_decode_graph import BeamDecodeGraph
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
        GPUBeamStageRunner,
        get_stage_runner,
    )
    from vllm_gr.v1.worker.gr_startup import capture_async_forward

    captures: list[tuple[int, int, int]] = []
    original_capture = BeamDecodeGraph.capture

    def record_capture(self, *args, **kwargs):
        if not self.ready:
            signature = self.signature
            captures.append((signature.batch_size, signature.beam_width, signature.num_model_rows))
        return original_capture(self, *args, **kwargs)

    monkeypatch.setattr(BeamDecodeGraph, "capture", record_capture)
    worker = admission_worker()
    freezes: list[str] = []
    monkeypatch.setattr(
        _gc_runtime,
        "schedule_warmup_gc_freeze",
        lambda: freezes.append("scheduled"),
    )
    worker.requests = {}
    worker.prepare_inputs_event = None
    worker.model_config.enforce_eager = False
    worker.model_config.max_model_len = 64
    worker._vllm_gr_config.beam_graph_enabled = True
    worker._vllm_gr_config.beam_worker_decision = True

    worker.vllm_config.scheduler_config = SimpleNamespace(async_scheduling=True, max_num_seqs=1)
    worker.kv_cache_config.kv_cache_groups[0].kv_cache_spec = SimpleNamespace(block_size=16)
    worker._update_states = lambda _: None
    install_decode_model(
        worker,
        lambda ids, positions: ids[:, None].to(torch.float16).expand(-1, 6) + positions[:, None],
    )
    assert capture_async_forward(worker, hooks_type=CudaStageHooksV1)
    assert freezes == []
    owner = worker._gr_gpu_stage_runner
    graph = owner.graph
    assert owner.closed and graph is not None
    assert captures == [(1, 4, 4)]
    fixed_inputs = (owner.input_ids, owner.positions, owner.slots, owner.execution_mask)
    input_buffers = owner.inputs
    previous_state = owner.beam.state

    def unexpected_forward(self):
        raise AssertionError(
            "requests must replay the startup graph without capturing or eager Forward"
        )

    monkeypatch.setattr(GPUBeamStageRunner, "_forward", unexpected_forward)
    for prompt, budget in [(7, 1), (13, 2), (21, 3), (11, 1)]:
        first = replace(
            metadata(n=budget),
            num_prompt_tokens=prompt,
            num_prompt_tokens_by_request=(prompt,),
        )
        assert get_stage_runner(worker, dispatch(first), hooks_type=CudaStageHooksV1) is owner
        assert owner.beam.state.max_new_tokens == budget
        if budget != previous_state.max_new_tokens:
            assert owner.beam.state is not previous_state
        previous_state = owner.beam.state
        assert owner.inputs is input_buffers
        assert all(
            current is fixed
            for current, fixed in zip(
                (owner.input_ids, owner.positions, owner.slots, owner.execution_mask),
                fixed_inputs,
            )
        )
        worker.requests["session-0"] = SimpleNamespace(block_ids=[[1, 2]])

        def prefill(worker, output):
            worker.execute_model_state = SimpleNamespace(
                logits=torch.zeros((1, 6), device="cuda", dtype=torch.float16)
            )

        receipts: list[Any] = []
        for step in range(budget):
            if step == 2:
                receipts[0].get_output()
            m = replace(
                metadata(step, n=budget),
                num_prompt_tokens=prompt,
                num_prompt_tokens_by_request=(prompt,),
            )
            owner.execute(dispatch(m), prefill)
            assert worker._beam_decode_graph is graph
            assert captures == [(1, 4, 4)]
            if step == 0:
                final = owner.final_output
                assert final.layout.width == owner.beam.state.beam_width
                assert final.layout.steps == budget
                assert final.device.numel() == final.layout.size
                assert final.host.numel() == final.layout.size
                assert final._ranking_scores.shape == (
                    1,
                    owner.beam.state.completed_capacity,
                )
                assert final.prompt_lengths == (prompt,)
            receipts.append(owner.sample())
        terminal = None
        for receipt in receipts:
            result = getattr(receipt.get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
            terminal = result.terminal_result or terminal
        assert terminal is not None and terminal.output is not None
        for sequence in terminal.output.items[0].sequences:
            assert len(sequence.token_ids) == budget
            assert sequence.cumulative_logprob is not None
        assert owner.closed
        assert worker._beam_decode_graph is graph
