# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Ascend stage-execution service.

Shared stage dispatch asks the selected provider for this service instead of
comparing ``worker.device`` against ``"npu"``. Every Ascend module import stays
inside a method body, so importing this service still loads no vendor runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class AscendStageExecutionService:
    """Execute, retire and pre-capture Ascend GR Beam stages."""

    backend_id = "ascend"

    def execute(
        self,
        worker: Any,
        scheduled: Any,
        original: Callable[..., Any],
        *args: Any,
        hooks_type: Any = None,
        **kwargs: Any,
    ) -> Any:
        # The Ascend eager owner owns its own stage state, so the CUDA stage-hooks
        # contract does not apply. Accept and ignore the argument rather than
        # requiring shared dispatch to know which backend takes hooks.
        del hooks_type
        from vllm_gr.v1.worker.npu_beam_stage_runner import get_npu_stage_runner

        stage_runner = get_npu_stage_runner(worker, scheduled)
        try:
            return stage_runner.execute(scheduled, original, *args, **kwargs)
        except Exception as error:
            stage_runner.execution_error = error
            raise

    def retire(self, worker: Any, scheduled: Any) -> Any:
        from vllm_gr.v1.worker.npu_beam_stage_runner import (
            execute_npu_gr_retirement,
        )

        return execute_npu_gr_retirement(worker, scheduled)

    def capture_startup_graph(
        self,
        worker: Any,
        *,
        hooks_type: Any = None,
        allow_eager_fallback: bool = False,
    ) -> bool:
        del hooks_type, allow_eager_fallback
        from vllm_gr.v1.worker.npu_model_runner_patch import (
            capture_beam_decode_graph,
        )

        return bool(capture_beam_decode_graph(worker))

    def admit_unconstrained_beam(self, worker: Any) -> None:
        """Apply the Ascend-only device admission rules for unconstrained Beam."""
        from vllm_gr.v1.engine.beam_search_v1 import validate_gr_device_config

        validate_gr_device_config(worker.vllm_config)


__all__ = ["AscendStageExecutionService"]
