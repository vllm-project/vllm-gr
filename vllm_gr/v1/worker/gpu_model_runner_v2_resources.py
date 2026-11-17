# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Worker-owned input storage, graph bindings and Decode startup for ModelRunner V2."""

from collections.abc import Callable
from typing import Any

import torch

from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeGraph
from vllm_gr.v1.worker.gr_inputs import GRInputBuffers


class V2InputBuffers(GRInputBuffers):
    """Keep model inputs at fixed addresses across V2 batch-size changes."""

    def __init__(
        self,
        width: int,
        max_model_len: int,
        block_size: int,
        device: torch.device | str,
        *,
        max_batch: int,
    ) -> None:
        super().__init__(width, max_model_len, block_size, device, max_batch=max_batch)
        rows = max_batch * width
        self.input_ids = torch.empty(rows, dtype=torch.int32, device=device)
        self.positions = torch.empty(rows, dtype=torch.int64, device=device)
        self.slots = torch.full((rows,), -1, dtype=torch.int64, device=device)
        self.execution_mask = torch.empty(rows, dtype=torch.bool, device=device)


class V2BeamDecodeGraph(BeamDecodeGraph):
    """Reject replay after replacement of V2's captured storage or model."""

    def __init__(self, worker: Any, width: int) -> None:
        super().__init__(worker, width)
        self.input_buffers: Any = None

    def _check_context(self) -> None:
        super()._check_context()
        if self.input_buffers is not getattr(self.worker, "_gr_input_buffers", None):
            raise RuntimeError("Beam graph input storage changed after capture")
        if self.worker.model is not self.original_model and self.worker.model is not self.wrapper:
            raise RuntimeError("Beam graph model changed after capture")

    def capture(self, forward: Callable[[bool], Any], *, persistent_install: bool) -> None:
        if not self.ready:
            self.input_buffers = getattr(self.worker, "_gr_input_buffers", None)
        super().capture(forward, persistent_install=persistent_install)


def close_v2_runtime(runner: Any) -> None:
    """Invalidate V2 captures before replacing or shutting down the model."""
    from vllm_gr.v1.worker.gpu_prefill_graph_runner_v2 import (
        uninstall_prefill_graph_manager_v2,
    )

    uninstall_prefill_graph_manager_v2(runner)
    graph = getattr(runner, "_beam_decode_graph", None)
    if graph is not None:
        graph.uninstall()
        graph.clear()
        runner._beam_decode_graph = None
    # Both capture-failure records describe graphs that only this runtime owns:
    # the startup record is keyed by batch size, the runtime record by
    # ``(batch, width)``.  A later runtime recaptures the same shapes, so a
    # record left behind would veto a graph that has just been captured.
    for name in (
        "_gr_input_buffers",
        "_gr_gpu_stage_runner",
        "_gr_gpu_stage_runner_cache",
        "_gr_gpu_stage_runner_cache_key",
        "_gr_decode_capture_failures",
        "_gr_runtime_decode_capture_failures",
        "_gr_dynamic_graphs_discarded",
        "_gr_async_decode_graph_capture_failed",
        "_gr_async_decode_graph_capture_error",
        "_gr_unconstrained_head_layout",
    ):
        if hasattr(runner, name):
            delattr(runner, name)


def capture_multibatch_forward(worker: Any, *, allow_eager_fallback: bool = True) -> bool:
    """Publish configured V2 Decode shapes, with shape-local fallback."""
    with torch.inference_mode():
        from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import CudaStageHooksV2
        from vllm_gr.v1.worker.gpu_stage_hooks import validate_stage_hooks_type
        from vllm_gr.v1.worker.gr_startup import _capture_batch, _capture_enabled
        from vllm_gr.v1.worker.model_runner_common import _get_gr_config

        if not _capture_enabled(worker) or str(worker.device).split(":")[0] != "cuda":
            return False
        cfg = _get_gr_config(worker)
        max_reqs = worker.vllm_config.scheduler_config.max_num_seqs
        batches = tuple(b for b in cfg.beam_online_batch_sizes if b <= max_reqs)
        if hasattr(worker, "_gr_decode_capture_failures"):
            owner = getattr(worker, "_gr_gpu_stage_runner", None)
            if owner is None or not owner.closed:
                raise RuntimeError(
                    "Asynchronous scheduling startup encountered an incomplete capture state"
                )
            validate_stage_hooks_type(owner.hooks, CudaStageHooksV2)
            graph = getattr(worker, "_beam_decode_graph", None)
            for batch in batches:
                if batch not in worker._gr_decode_capture_failures and (
                    graph is None or not graph.select(batch, cfg.beam_max_width).ready
                ):
                    raise RuntimeError("Published Decode graph is not ready")
            return len(worker._gr_decode_capture_failures) < len(batches)
        if getattr(worker, "_gr_gpu_stage_runner", None) is not None:
            raise RuntimeError(
                "Asynchronous scheduling startup encountered an incomplete capture state"
            )
        from vllm_gr.v1.attention.backends.beam_prefix_calibrate import (
            ensure_prefix_calibration,
        )

        ensure_prefix_calibration(worker)
        failures = {}
        for batch in batches:
            error = _capture_batch(
                worker,
                batch,
                hooks_type=CudaStageHooksV2,
                allow_eager_fallback=allow_eager_fallback,
            )
            if error is not None:
                failures[batch] = error
        worker._gr_decode_capture_failures = failures
        worker._gr_async_decode_graph_capture_failed = len(failures) == len(batches)
        worker._gr_async_decode_graph_capture_error = next(iter(failures.values()), None)
        return len(failures) < len(batches)
