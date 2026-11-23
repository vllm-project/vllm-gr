# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Worker-owned input storage, graph bindings and Decode startup for ModelRunner V2."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from vllm_gr.v1.worker.cuda.beam_decode_graph import BeamDecodeGraph
from vllm_gr.v1.worker.cuda.gr_inputs import GRInputBuffers


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


@dataclass
class V2RuntimeState:
    """One generation of Worker-owned fixed inputs and capture records.

    Replacing this object invalidates borrowed graph bindings. Stage owners,
    shared graph registries, contexts/workspaces and model head caches retain
    their own allocation and retirement rules.
    """

    inputs: V2InputBuffers | None = None
    startup_failures: dict[int, str] | None = None
    runtime_failures: dict[tuple[int, ...], str] = field(default_factory=dict)
    capture_failed: bool = False
    capture_error: str | None = None
    graphs_discarded: bool = False


def get_v2_runtime(worker: Any) -> V2RuntimeState:
    """Obtain the current generation; storage remains lazily allocated."""
    runtime = getattr(worker, "_gr_v2_runtime", None)
    if runtime is None:
        runtime = V2RuntimeState()
        worker._gr_v2_runtime = runtime
    return cast(V2RuntimeState, runtime)


class V2BeamDecodeGraph(BeamDecodeGraph):
    """Reject replay after replacement of V2's captured storage or model."""

    def __init__(self, worker: Any, width: int) -> None:
        super().__init__(worker, width)
        self.input_buffers: V2InputBuffers | None = None
        self.runtime: V2RuntimeState | None = None

    def _check_context(self) -> None:
        super()._check_context()
        self._check_runtime_binding()

    def _check_runtime_binding(self) -> None:
        runtime = getattr(self.worker, "_gr_v2_runtime", None)
        if runtime is None or runtime is not self.runtime:
            raise RuntimeError("Beam graph runtime generation changed after capture")
        if self.input_buffers is not runtime.inputs:
            raise RuntimeError("Beam graph input storage changed after capture")
        if self.worker.model is not self.original_model and self.worker.model is not self.wrapper:
            raise RuntimeError("Beam graph model changed after capture")

    def capture(self, forward: Callable[[bool], Any], *, persistent_install: bool) -> None:
        if self.runtime is None:
            self.runtime = get_v2_runtime(self.worker)
            self.input_buffers = self.runtime.inputs
        # Selecting an uncaptured shape must not rebind a graph registry whose
        # other shapes still borrow a previous generation, input set or model.
        self._check_runtime_binding()
        super().capture(forward, persistent_install=persistent_install)


def _require_released_v2_runtime(runner: Any) -> None:
    """Check the lifecycle caller's drain proof without waiting or cancelling.

    Worker RPC serialization prevents new dispatch admission during teardown.
    GPUBeamStageRunner.release closes only after Beam end_session synchronizes
    its last work and final consumer event. Host output slots must also be free.
    Fused terminal release keeps its drain proof incomplete through native
    request cleanup, even after the GPU owner has closed.
    """
    owners = [getattr(runner, "_gr_gpu_stage_runner", None)]
    owners.extend(getattr(runner, "_gr_gpu_stage_runner_cache", {}).values())
    retirement = getattr(runner, "_gr_retirement_output", None)
    if retirement is not None:
        owners.append(retirement.owner)
        if retirement._output is None:
            raise RuntimeError("V2 runtime cleanup requires retirement output-consumer drain")
    seen: set[int] = set()
    for owner in owners:
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        with owner.control_lock:
            if not owner.closed or owner.pending is not None or any(owner.control_busy):
                raise RuntimeError("V2 runtime cleanup requires dispatch and output-consumer drain")
            if owner.terminal_cleanup_pending:
                raise RuntimeError("V2 runtime cleanup requires completed terminal native cleanup")
    context = getattr(runner, "_beam_context", None)
    pool = getattr(runner, "_beam_postprocess_workspaces", None)
    if (
        getattr(context, "_gr_execution_owner", None) is not None
        or bool(getattr(getattr(context, "registry", None), "num_active_sessions", 0))
        or (
            pool is not None
            and any(pool.require(key).execution_owner is not None for key in pool.keys)
        )
    ):
        raise RuntimeError("V2 runtime cleanup requires released Beam context and workspace")


def close_v2_runtime(runner: Any) -> None:
    """Invalidate a drained generation before full model load or shutdown.

    Startup profiling and the serialized lifecycle caller must stop dispatches
    and consume receipts first. This does not drain, cancel or add a GPU fence;
    a live borrower is rejected before any runtime reference is changed.
    Weight-only updates keep their existing model-owned cache refresh path.
    """
    from vllm_gr.v1.worker.cuda.v2.gpu_prefill_graph_runner_v2 import (
        uninstall_prefill_graph_manager_v2,
    )

    _require_released_v2_runtime(runner)
    uninstall_prefill_graph_manager_v2(runner)
    graph = getattr(runner, "_beam_decode_graph", None)
    if graph is not None:
        graph.uninstall()
        graph.clear()
        runner._beam_decode_graph = None
    for name in (
        "_gr_v2_runtime",
        "_gr_gpu_stage_runner",
        "_gr_gpu_stage_runner_cache",
        "_gr_gpu_stage_runner_cache_key",
        "_gr_retirement_output",
    ):
        if hasattr(runner, name):
            delattr(runner, name)


def capture_multibatch_forward(worker: Any, *, allow_eager_fallback: bool = True) -> bool:
    """Publish configured V2 Decode shapes, with shape-local fallback."""
    from vllm_gr.v1.worker.cuda.gpu_stage_hooks import validate_stage_hooks_type
    from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2
    from vllm_gr.v1.worker.gr_context import _get_gr_config
    from vllm_gr.v1.worker.gr_startup import _capture_batch, _capture_enabled

    with torch.inference_mode():
        if not _capture_enabled(worker) or str(worker.device).split(":")[0] != "cuda":
            return False
        cfg = _get_gr_config(worker)
        max_reqs = worker.vllm_config.scheduler_config.max_num_seqs
        batches = tuple(b for b in cfg.deployment.beam_batch_sizes if b <= max_reqs)
        if not batches:
            return False
        runtime = get_v2_runtime(worker)
        if runtime.startup_failures is not None:
            owner = getattr(worker, "_gr_gpu_stage_runner", None)
            if owner is None or not owner.closed:
                raise RuntimeError(
                    "Asynchronous scheduling startup encountered an incomplete capture state"
                )
            validate_stage_hooks_type(owner.hooks, CudaStageHooksV2)
            graph = getattr(worker, "_beam_decode_graph", None)
            for batch in batches:
                if batch not in runtime.startup_failures and (
                    graph is None or not graph.select(batch, cfg.beam_max_width).ready
                ):
                    raise RuntimeError("Published Decode graph is not ready")
            return len(runtime.startup_failures) < len(batches)
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
        runtime.startup_failures = failures
        runtime.capture_failed = len(failures) == len(batches)
        runtime.capture_error = next(iter(failures.values()), None)
        return len(failures) < len(batches)
