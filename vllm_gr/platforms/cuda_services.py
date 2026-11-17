# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA stage-execution service.

Shared stage dispatch asks the selected provider for this service instead of
comparing ``worker.device`` against ``"cuda"``. Every CUDA module import stays
inside a method body, so importing this service still loads no CUDA module and
``vllm_gr.platforms`` keeps its import isolation guarantee.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_gr.v1.worker.gpu_stage_hooks import CudaStageHooksType


class CudaStageExecutionService:
    """Execute, retire and pre-capture CUDA GR Beam stages."""

    backend_id = "cuda"

    def execute(
        self,
        worker: Any,
        scheduled: Any,
        original: Callable[..., Any],
        *args: Any,
        hooks_type: CudaStageHooksType | None = None,
        **kwargs: Any,
    ) -> Any:
        from vllm_gr.v1.worker.gpu_beam_stage_runner import get_stage_runner

        if hooks_type is None:
            raise ValueError("CUDA GR execution requires explicit stage hooks")
        stage_runner = get_stage_runner(worker, scheduled, hooks_type=hooks_type)
        try:
            return stage_runner.execute(scheduled, original, *args, **kwargs)
        except Exception as error:
            # Native BatchQueue consumes sample first. Stale logits must never
            # hide an execute failure from the corresponding sample Future.
            stage_runner.execution_error = error
            raise

    def retire(self, worker: Any, scheduled: Any) -> Any:
        from vllm_gr.v1.worker.gpu_beam_stage_runner import execute_gr_retirement

        return execute_gr_retirement(worker, scheduled)

    def capture_startup_graph(
        self,
        worker: Any,
        *,
        hooks_type: CudaStageHooksType | None = None,
        allow_eager_fallback: bool = False,
    ) -> bool:
        from vllm_gr.v1.worker.gr_startup import capture_cuda_startup_graph

        return capture_cuda_startup_graph(
            worker,
            hooks_type=hooks_type,
            allow_eager_fallback=allow_eager_fallback,
        )

    def admit_unconstrained_beam(self, worker: Any) -> None:
        """CUDA serves every unconstrained Beam V1 configuration this release admits."""
        del worker


__all__ = ["CudaStageExecutionService"]
