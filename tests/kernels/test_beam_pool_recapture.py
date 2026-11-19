# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Real Beam graph captures survive retirement while outputs remain alive."""

import gc
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.kernels.beam_stage_test_utils import (
    admission_worker,
    install_decode_model,
)
from vllm_gr.v1.worker.cuda.beam_decode_graph import BeamDecodeGraph
from vllm_gr.v1.worker.cuda.gpu_startup import _capture_batch
from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2
from vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_resources import (
    capture_multibatch_forward,
)

pytestmark = [
    pytest.mark.gpu_test,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.l2  # type: ignore[misc]
@pytest.mark.parametrize("retirement", ["clear", "discard"])  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_recapture_after_last_graph_retired_with_live_output(
    monkeypatch: pytest.MonkeyPatch, retirement: str
) -> None:
    gc.collect()
    worker = admission_worker(max_rows=32)
    worker.model_config.enforce_eager = False
    worker._vllm_gr_config.beam_graph_enabled = True
    worker._vllm_gr_config.beam_worker_decision = True
    worker._vllm_gr_config.beam_online_batch_sizes = (1,)
    worker._vllm_gr_config.constraint_table = SimpleNamespace(enabled=True)
    worker.vllm_config.scheduler_config = SimpleNamespace(async_scheduling=True, max_num_seqs=8)
    install_decode_model(worker, lambda ids, pos: (ids + pos)[:, None].to(torch.float16))

    # The vLLM entry stores weak output references. Preserve the actual tensor
    # returned from capture to keep its old pool allocation alive.
    retained: list[torch.Tensor] = []
    original = BeamDecodeGraph.model_forward

    def retain_capture_output(graph: BeamDecodeGraph, *args: Any, **kwargs: Any) -> Any:
        result = original(graph, *args, **kwargs)
        if graph.capturing and isinstance(result, torch.Tensor):
            retained.append(result)
        return result

    monkeypatch.setattr(BeamDecodeGraph, "model_forward", retain_capture_output)
    assert capture_multibatch_forward(worker)
    graph = worker._beam_decode_graph
    assert graph.ready and retained
    previous = graph.wrapper.concrete_cudagraph_entries[graph.descriptor].cudagraph
    old_graph = weakref.ref(previous)
    del previous
    if retirement == "clear":
        graph.clear()
    else:
        graph.discard_geometry(1, 4)
    gc.collect()
    assert old_graph() is None
    assert retained[0].numel() > 0

    assert (
        _capture_batch(worker, 1, hooks_type=CudaStageHooksV2, allow_eager_fallback=False) is None
    )
    assert worker._beam_decode_graph is graph
    assert graph.select(1, 4).ready
    owner = worker._gr_gpu_stage_runner
    owner.input_ids.fill_(3)
    owner.positions.fill_(10)
    torch.testing.assert_close(
        graph.replay(), torch.full((4, 1), 13, dtype=torch.float16, device="cuda")
    )


@pytest.mark.l2  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_recapture_preserves_other_live_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gc.collect()
    worker = admission_worker(max_rows=32)
    worker.model_config.enforce_eager = False
    worker._vllm_gr_config.beam_graph_enabled = True
    worker._vllm_gr_config.beam_worker_decision = True
    worker._vllm_gr_config.beam_online_batch_sizes = (1, 2)
    worker._vllm_gr_config.constraint_table = SimpleNamespace(enabled=True)
    worker.vllm_config.scheduler_config = SimpleNamespace(async_scheduling=True, max_num_seqs=8)
    install_decode_model(worker, lambda ids, pos: (ids + pos)[:, None].to(torch.float16))

    retained: list[torch.Tensor] = []
    original = BeamDecodeGraph.model_forward

    def retain_capture_output(graph: BeamDecodeGraph, *args: Any, **kwargs: Any) -> Any:
        result = original(graph, *args, **kwargs)
        if graph.capturing and isinstance(result, torch.Tensor):
            retained.append(result)
        return result

    monkeypatch.setattr(BeamDecodeGraph, "model_forward", retain_capture_output)
    assert capture_multibatch_forward(worker)
    graph = worker._beam_decode_graph
    assert len(retained) == 2
    pool = graph.wrapper.graph_pool
    graph.discard_geometry(1, 4)
    assert graph.select(2, 4).ready
    survivor = weakref.ref(graph.wrapper.concrete_cudagraph_entries[graph.descriptor].cudagraph)
    assert (
        _capture_batch(worker, 1, hooks_type=CudaStageHooksV2, allow_eager_fallback=False) is None
    )
    assert graph.wrapper.graph_pool == pool
    assert survivor() is not None
    owner = worker._gr_gpu_stage_runner
    owner.inputs.input_ids.fill_(3)
    owner.inputs.positions.fill_(10)
    torch.testing.assert_close(
        graph.select(2, 4).replay(),
        torch.full((8, 1), 13, dtype=torch.float16, device="cuda"),
    )

    graph.discard_geometry(1, 4)
    graph.discard_geometry(2, 4)
    gc.collect()
    assert survivor() is None
    assert (
        _capture_batch(worker, 1, hooks_type=CudaStageHooksV2, allow_eager_fallback=False) is None
    )
    assert graph.select(1, 4).ready
    assert graph.wrapper.graph_pool != pool
    owner = worker._gr_gpu_stage_runner
    owner.input_ids.fill_(3)
    owner.positions.fill_(10)
    torch.testing.assert_close(
        graph.replay(), torch.full((4, 1), 13, dtype=torch.float16, device="cuda")
    )
