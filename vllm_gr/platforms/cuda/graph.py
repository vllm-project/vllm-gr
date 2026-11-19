# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA graph execution, diagnostics and native runner resources."""

from __future__ import annotations

from typing import Any

from ..graph import GraphExecution, native_graph_counters


def check_cuda_graphs_idle(runner: Any) -> None:
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


class CudaGraph(GraphExecution):
    implementation_id = "cuda.graph"

    @staticmethod
    def prefill_runner_class() -> Any:
        from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import (
            GPUPrefillGraphRunner,
        )

        return GPUPrefillGraphRunner

    def check_idle(self) -> None:
        super().check_idle()
        if self.closed or self.runner is None:
            return
        check_cuda_graphs_idle(self.runner)

    def stats(self) -> dict[str, Any]:
        report = super().stats()
        stats = native_graph_counters(self.runner)
        prefill = getattr(self.runner, "_prefill_graph_runner", None)
        if prefill is not None:
            stats.update(prefill.capture_stats())
        manager = getattr(self.runner, "cudagraph_manager", None)
        counts = getattr(manager, "dispatch_counts", None)
        if counts is not None:
            assert manager is not None
            stats["prefill_graph_captured_buckets"] = sorted(
                {desc.num_tokens for desc in manager._successful if desc.cg_mode.name == "FULL"}
            )
            stats["prefill_graph_captured_shapes"] = sorted(
                (desc.num_reqs, desc.num_tokens)
                for desc in manager._successful
                if desc.cg_mode.name == "FULL"
            )
            stats["prefill_graph_piecewise_buckets"] = sorted(
                desc.num_tokens for desc in manager._successful if desc.cg_mode.name == "PIECEWISE"
            )
            stats["prefill_graph_full_capture_failures"] = {
                f"{desc.cg_mode.name}:T={desc.num_tokens}:B={desc.num_reqs}": reason
                for desc, reason in manager.full_capture_failures.items()
            }
            stats["prefill_graph_full_dispatches"] = counts["FULL"]
            stats["prefill_graph_piecewise_dispatches"] = counts["PIECEWISE"]
            stats["prefill_graph_none_dispatches"] = counts["NONE"]
        report["runtime_fields"] = stats
        return report

    def _release(self) -> None:
        runner = self.runner
        graph = getattr(runner, "_beam_decode_graph", None)
        if graph is not None:
            graph.uninstall()
            graph.clear()
            runner._beam_decode_graph = None
        manager = getattr(runner, "cudagraph_manager", None)
        if (
            getattr(runner, "_gr_v2_runtime", None) is not None
            or getattr(manager, "graph_service", None) is self
        ):
            from vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_resources import (
                close_v2_runtime,
            )

            close_v2_runtime(runner)
        prefill = getattr(runner, "_prefill_graph_runner", None)
        if prefill is not None:
            prefill.buffers.clear()
        # Legacy V1 capture allocates its own context. Stage V1/V2 borrow the
        # native worker context and prewarmed workspace across full reloads.
        capture_context = getattr(runner, "_gr_cuda_capture_context", None)
        if (
            capture_context is not None
            and getattr(runner, "_beam_context", None) is capture_context
        ):
            runner._beam_context = None
        for name in (
            "_gr_cuda_capture_context",
            "_gr_gpu_stage_runner",
            "_gr_gpu_stage_runner_cache",
            "_gr_gpu_stage_runner_cache_key",
            "_gr_retirement_output",
            "_prefill_graph_runner",
            "_prefill_graphs_captured",
            "_gr_prefill_graph_dispatches",
            "_beam_graph_keys",
            "_beam_graph_full_dispatch_count",
            "_gr_async_decode_graph_capture_failed",
            "_gr_async_decode_graph_capture_error",
        ):
            if hasattr(runner, name):
                delattr(runner, name)
