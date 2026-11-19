# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Ascend graph execution, diagnostics and native runner resources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..graph import PREFILL_OPTIONS, GraphExecution, native_graph_counters

if TYPE_CHECKING:
    from ..plan import BackendPlan


class AscendGraph(GraphExecution):
    implementation_id = "ascend.graph"
    option_names = (*PREFILL_OPTIONS, "VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS")

    def __init__(self, runner: Any, plan: BackendPlan | None) -> None:
        super().__init__(runner, plan)
        if not hasattr(runner, "_beam_graph_fallback_count"):
            runner._beam_graph_fallback_count = 0

    @staticmethod
    def prefill_runner_class() -> Any:
        from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

        return NPUPrefillGraphRunner

    def stats(self) -> dict[str, Any]:
        report = super().stats()
        if self.closed:
            return report
        from vllm_gr.v1.worker.gr_context import _get_gr_config

        runner = self.runner
        config = _get_gr_config(runner)
        stats = {
            **native_graph_counters(runner),
            "beam_graph_batch_size": int(getattr(config, "beam_graph_batch_size", 1)),
            "captured_shapes": sorted(
                tuple(int(value) for value in shape)
                for shape in getattr(runner, "_gr_npu_beam_graph_shapes", ())
            ),
            "fallbacks": int(getattr(runner, "_beam_graph_fallback_count", 0)),
            "fallback_reasons": dict(getattr(runner, "_beam_graph_fallback_reasons", {})),
        }
        prefill = getattr(runner, "_prefill_graph_runner", None)
        if prefill is not None:
            stats.update(prefill.capture_stats())
        report["runtime_fields"] = stats
        return report

    def check_idle(self) -> None:
        super().check_idle()
        if not self.closed:
            from vllm_gr.v1.worker.npu.npu_model_runner_patch import check_graphs_idle

            check_graphs_idle(self.runner)

    def _release(self) -> None:
        from vllm_gr.v1.worker.npu.npu_model_runner_patch import close_graphs

        close_graphs(self.runner)
