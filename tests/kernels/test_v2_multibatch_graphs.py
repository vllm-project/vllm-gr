# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual CUDA replay survives B switches and a shape-local capture refusal."""

import gc
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tests.kernels.test_gpu_beam_stage_runner import (
    admission_worker,
    dispatch,
    install_decode_model,
    metadata,
)
from vllm_gr.v1.engine.beam_search_v1 import GRStageKind
from vllm_gr.v1.worker.gpu_beam_stage_runner import (
    get_stage_runner,
)
from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import (
    V2StageExecutionHooks,
)
from vllm_gr.v1.worker.gpu_model_runner_v2_graph import capture_multibatch_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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
    original = V2StageExecutionHooks.forward_hidden

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

    monkeypatch.setattr(V2StageExecutionHooks, "forward_hidden", forward)
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
        owner = get_stage_runner(worker, scheduled, hooks_type=V2StageExecutionHooks)
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
