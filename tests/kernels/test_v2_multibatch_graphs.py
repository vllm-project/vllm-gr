# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Actual CUDA replay survives B switches and a shape-local capture refusal."""

import gc
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tests.constraints.test_table import make_table
from tests.kernels.beam_stage_test_utils import (
    admission_worker,
    dispatch,
    install_decode_model,
    metadata,
)
from vllm_gr.constraints.cuda_backend import CUDAConstraintBackend
from vllm_gr.constraints.loader import _upload_cuda_backend
from vllm_gr.constraints.versioned_table import VersionedConstraintTable
from vllm_gr.v1.engine.beam_search_v1 import GRStageKind
from vllm_gr.v1.worker.gpu_beam_stage_runner import (
    get_stage_runner,
)
from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import (
    CudaStageHooksV2,
)
from vllm_gr.v1.worker.gpu_model_runner_v2_graph import capture_multibatch_forward
from vllm_gr.v1.worker.gpu_model_runner_v2_resources import close_v2_runtime
from vllm_gr.v1.worker.model_runner_patch import make_load_model_with_constraint_backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@torch.inference_mode()
def test_runtime_recapture_refusal_preserves_live_request(monkeypatch):
    gc.collect()
    worker = admission_worker(max_rows=32)
    worker.model_config.enforce_eager = False
    worker._vllm_gr_config.beam_graph_enabled = True
    worker._vllm_gr_config.beam_worker_decision = True
    worker._vllm_gr_config.beam_online_batch_sizes = (1,)
    worker._vllm_gr_config.constraint_table = SimpleNamespace(enabled=True)
    worker.vllm_config.scheduler_config = SimpleNamespace(async_scheduling=True, max_num_seqs=8)
    install_decode_model(worker, lambda ids, pos: (ids + pos)[:, None].to(torch.float16))
    assert capture_multibatch_forward(worker)
    graph = worker._beam_decode_graph
    assert graph.ready
    graph.clear()
    worker._gr_dynamic_graphs_discarded = True
    m = metadata()
    scheduled = dispatch(m)
    owner = get_stage_runner(worker, scheduled, hooks_type=CudaStageHooksV2)
    decode = replace(m, kind=GRStageKind.DECODE, stage_index=1, decode_step=0, kv_blocks=((1,),))
    owner.prepared = owner.prepare_cpu(decode, 0)
    owner._bind_attention(decode)
    owner.input_ids.fill_(3)
    owner.positions.fill_(10)
    original = owner._forward_hidden
    attempts = []

    def refuse(capture):
        if capture:
            attempts.append(True)
            raise RuntimeError("operation not supported when stream is capturing")
        return original(capture)

    monkeypatch.setattr(owner, "_forward_hidden", refuse)
    for _ in range(2):
        result = owner.hooks.decode_logits(owner)
        torch.testing.assert_close(result[:, 0], torch.full_like(result[:, 0], 13))
    assert len(attempts) == 1
    assert owner.graph is None
    assert not torch.cuda.is_current_stream_capturing()
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)
    assert owner.closed and not worker._beam_sessions


@torch.inference_mode()  # type: ignore[misc]
def test_model_reload_retries_a_refused_decode_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """A replacement runtime must not inherit the previous one's refusals."""
    gc.collect()
    worker = admission_worker(max_rows=32)
    worker.model_config.enforce_eager = False
    worker._vllm_gr_config.beam_graph_enabled = True
    worker._vllm_gr_config.beam_worker_decision = True
    worker._vllm_gr_config.beam_online_batch_sizes = (1,)
    worker._vllm_gr_config.constraint_table = SimpleNamespace(enabled=True)
    worker.vllm_config.scheduler_config = SimpleNamespace(async_scheduling=True, max_num_seqs=8)
    install_decode_model(worker, lambda ids, pos: (ids + pos)[:, None].to(torch.float16))
    assert capture_multibatch_forward(worker)
    # Refuse every capture attempt the way an unsupported op does. Only the
    # capture call is refused, so the shape stays servable through eager.
    refuse = [True]
    original = CudaStageHooksV2.forward_hidden

    def forward(
        self: Any,
        ids: torch.Tensor,
        positions: torch.Tensor,
        attention: Any,
        slots: Any,
        capture: bool,
    ) -> Any:
        if capture and refuse[0]:
            raise RuntimeError("operation not supported when stream is capturing")
        return original(self, ids, positions, attention, slots, capture)

    monkeypatch.setattr(CudaStageHooksV2, "forward_hidden", forward)
    slots = iter((0, 1))

    def serve_decode() -> Any:
        m = metadata()
        scheduled = dispatch(m)
        owner = get_stage_runner(worker, scheduled, hooks_type=CudaStageHooksV2)
        decode = replace(
            m, kind=GRStageKind.DECODE, stage_index=1, decode_step=0, kv_blocks=((1,),)
        )
        owner.prepared = owner.prepare_cpu(decode, next(slots))
        owner._bind_attention(decode)
        owner.input_ids.fill_(3)
        owner.positions.fill_(10)
        result = owner.hooks.decode_logits(owner)
        torch.testing.assert_close(result[:, 0], torch.full_like(result[:, 0], 13))
        done = torch.Event()
        done.record()
        owner.release(owner.binding, consumer_done=done)
        assert owner.closed and not worker._beam_sessions
        return owner

    # The runtime capture is refused: eager serves the shape and records the one
    # geometry whose graph could not be built.
    worker._beam_decode_graph.clear()
    worker._gr_dynamic_graphs_discarded = True
    assert serve_decode().graph is None
    assert list(worker._gr_runtime_decode_capture_failures) == [(1, 4)]

    # A complete load replaces this runtime; the record describes graphs that
    # only the retired runtime owned and must not outlive it.  Drive the real
    # entry point: it closes the previous runtime, then loads weights and
    # republishes a constraint table, which is what a model reload does.
    loaded: list[Any] = []

    def native_load(runner: Any) -> None:
        fresh, _ = _upload_cuda_backend(
            make_table(), device=runner.device, max_top_k=4, max_rows=32
        )
        fresh._mark_ready()
        runner.constraint_backend = VersionedConstraintTable(fresh, loader=lambda _: fresh)
        loaded.append(runner)

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        lambda **kwargs: pytest.fail(f"this runner configures no constraint table: {kwargs}"),
    )
    make_load_model_with_constraint_backend(native_load, close_runtime=close_v2_runtime)(worker)
    assert loaded == [worker]
    assert isinstance(worker.constraint_backend.get(), CUDAConstraintBackend)
    assert not hasattr(worker, "_gr_runtime_decode_capture_failures")

    # The replacement recaptures the same geometry and really replays it.
    refuse[0] = False
    assert capture_multibatch_forward(worker)
    replay = Mock(wraps=worker._beam_decode_graph.replay)
    monkeypatch.setattr(worker._beam_decode_graph, "replay", replay)
    forced = serve_decode()
    assert forced.graph is not None
    assert replay.call_count == 1


@pytest.mark.parametrize("fail_shape", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("sizes", [(1, 2, 4), (1, 3, 7), (1, 6, 8)])  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_all_batches_capture_at_startup_and_reuse_storage(
    monkeypatch: pytest.MonkeyPatch, fail_shape: bool, sizes: tuple[int, ...]
) -> None:
    # Reclaim cycles from the previous parametrized worker before capture.
    # CUDAGraph destruction during another graph's capture invalidates CUDA.
    gc.collect()
    # The V1 startup override must not replace any of the V2 capture shapes.
    monkeypatch.setenv("VLLM_GR_V1_STARTUP_CAPTURE_BATCH_SIZE", "4")
    worker = admission_worker(max_rows=32)
    del worker.use_async_scheduling  # V2 reads the resolved scheduler config.
    worker.model_config.enforce_eager = False
    worker._vllm_gr_config.beam_graph_enabled = True
    worker._vllm_gr_config.beam_worker_decision = True
    worker._vllm_gr_config.beam_online_batch_sizes = sizes
    worker._vllm_gr_config.constraint_table = SimpleNamespace(enabled=True)
    worker.vllm_config.scheduler_config = SimpleNamespace(async_scheduling=True, max_num_seqs=8)
    install_decode_model(worker, lambda ids, pos: (ids + pos)[:, None].to(torch.float16))
    original = CudaStageHooksV2.forward_hidden

    def forward(
        self: Any,
        ids: torch.Tensor,
        positions: torch.Tensor,
        attention: Any,
        slots: Any,
        capture: bool,
    ) -> Any:
        if fail_shape and capture and ids.numel() == sizes[1] * 4:
            raise RuntimeError("operation not supported when stream is capturing")
        return original(self, ids, positions, attention, slots, capture)

    monkeypatch.setattr(CudaStageHooksV2, "forward_hidden", forward)
    assert capture_multibatch_forward(worker)
    graph = worker._beam_decode_graph
    assert {s.batch_size for s in graph._bindings} == (
        set(sizes) - {sizes[1]} if fail_shape else set(sizes)
    )
    buffers = worker._gr_input_buffers
    addresses = (
        buffers.device.data_ptr(),
        buffers.input_ids.data_ptr(),
        buffers.positions.data_ptr(),
    )

    def forbid_capture(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("serving must never capture")

    monkeypatch.setattr(graph, "capture", forbid_capture)
    replay = Mock(wraps=graph.replay)
    monkeypatch.setattr(graph, "replay", replay)
    unselected = next(b for b in range(1, 9) if b not in sizes)
    for iteration, b in enumerate((1, sizes[-1], sizes[1], unselected, sizes[-1], 1)):
        ids = ("session-0",) + tuple(f"member-{i}" for i in range(1, b))
        m = replace(
            metadata(),
            request_ids=ids,
            num_prompt_tokens=7 * b,
            num_prompt_tokens_by_request=(7,) * b,
        )
        scheduled = dispatch(m)
        scheduled.num_scheduled_tokens = dict.fromkeys(ids, 7)
        owner = get_stage_runner(worker, scheduled, hooks_type=CudaStageHooksV2)
        decode = replace(
            m, kind=GRStageKind.DECODE, stage_index=1, decode_step=0, kv_blocks=((1,),) * b
        )
        owner.prepared = owner.prepare_cpu(decode, iteration % 2)
        owner._bind_attention(decode)
        assert owner.inputs is buffers
        assert (
            buffers.device.data_ptr(),
            owner.input_ids.data_ptr(),
            owner.positions.data_ptr(),
        ) == addresses
        owner.input_ids.copy_(torch.arange(b * 4, device=worker.device) + iteration)
        owner.positions.fill_(iteration + 10)
        before = replay.call_count
        result = owner.hooks.decode_logits(owner)
        assert replay.call_count - before == int(b in sizes and not (fail_shape and b == sizes[1]))
        torch.testing.assert_close(
            result[:, 0], (owner.input_ids + owner.positions).to(torch.float16)
        )
        done = torch.Event()
        done.record()
        owner.release(owner.binding, consumer_done=done)
        assert owner.closed and not worker._beam_sessions
    assert capture_multibatch_forward(worker)
