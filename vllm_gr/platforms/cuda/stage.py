# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Runner-owned CUDA GR stage and operator bindings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_gr.v1.beam.decision_ops import DecisionOps

if TYPE_CHECKING:
    from vllm_gr.v1.worker.cuda.gpu_stage_hooks import CudaStageHooksType

    from ..plan import BackendPlan


@dataclass(frozen=True)
class CudaStageExecutionService:
    """Execute, retire and pre-capture CUDA GR Beam stages."""

    implementation_id: str
    decision: DecisionOps
    worker_decision: bool
    native_hooks: CudaStageHooksType

    def execute(
        self,
        worker: Any,
        scheduled: Any,
        original: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import get_stage_runner

        stage_runner = get_stage_runner(worker, scheduled, hooks_type=self.native_hooks)
        try:
            return stage_runner.execute(scheduled, original, *args, **kwargs)
        except Exception as error:
            # Native BatchQueue consumes sample first. Stale logits must never
            # hide an execute failure from the corresponding sample Future.
            stage_runner.execution_error = error
            raise

    def retire(self, worker: Any, scheduled: Any) -> Any:
        from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import execute_gr_retirement

        return execute_gr_retirement(worker, scheduled)

    def capture_startup_graph(
        self,
        worker: Any,
        *,
        allow_eager_fallback: bool = False,
    ) -> bool:
        from vllm_gr.v1.worker.gr_startup import capture_cuda_startup_graph

        return capture_cuda_startup_graph(
            worker,
            hooks_type=self.native_hooks,
            allow_eager_fallback=allow_eager_fallback,
        )

    def admit_unconstrained_beam(self, worker: Any) -> None:
        """CUDA serves every unconstrained Beam V1 configuration this release admits."""
        del worker


def _create_stage(
    plan: BackendPlan | None, identity: str, native_hooks: CudaStageHooksType
) -> CudaStageExecutionService:
    import torch

    import vllm_gr.ops.onerec_decode  # noqa: F401  (register the kernels with PyTorch)

    choices = plan.choices.to_dict() if plan is not None else {}
    if choices.get("operator_policy", "cuda.onerec") != "cuda.onerec":
        raise RuntimeError("CUDA runner cannot bind the admitted decision operators")
    if choices.get("stage", identity) != identity:
        raise RuntimeError("CUDA native stage differs from the admitted backend plan")
    operators = DecisionOps(
        "cuda.onerec",
        torch.ops.vllm_gr.beam_search_group.default,
        torch.ops.vllm_gr.beam_search_rec_final_select.default,
        1,
    )
    return CudaStageExecutionService(identity, operators, False, native_hooks)


def create_v1_stage(*, plan: BackendPlan | None, runner: Any) -> CudaStageExecutionService:
    from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1

    return _create_stage(plan, "cuda.v1", CudaStageHooksV1)


def create_v2_stage(*, plan: BackendPlan | None, runner: Any) -> CudaStageExecutionService:
    from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2

    return _create_stage(plan, "cuda.v2", CudaStageHooksV2)
