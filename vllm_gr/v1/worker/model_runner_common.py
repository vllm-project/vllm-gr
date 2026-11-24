# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Cross-device native ModelRunner V1 execute wrapper.

NPU input remapping and sampler bookkeeping live in their dedicated helper
modules. Configuration/context, GC coordination, dispatch and installation
contracts have separate owners; native V2 does not depend on this module.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

from vllm_gr.v1.beam import BEAM_DATA_VAR
from vllm_gr.v1.worker import gr_execution, gr_gc
from vllm_gr.v1.worker import model_runner_contracts as contracts

if TYPE_CHECKING:
    from vllm_gr.v1.worker.gpu.stage_hooks import CudaStageHooksType


def make_patched_execute_model(
    _orig_execute_model: Callable[..., Any],
    *,
    hooks_type: CudaStageHooksType | None = None,
) -> Callable[..., Any]:
    """Wrap execute_model to publish beam-request context for the run.

    Shared by GPU and NPU runners.
    """
    if getattr(_orig_execute_model, contracts._WORKER_EXECUTE_PATCH_MARKER, False):
        return _orig_execute_model

    @wraps(_orig_execute_model)
    def _patched_execute_model_step(
        self: Any, scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> Any:
        from vllm_gr.config.execution import require_beam_config
        from vllm_gr.v1.beam.dispatch import GR_RETIRE_OUTPUT_ATTR, GR_STAGE_OUTPUT_ATTR

        # Shared stage and retirement code receives an admitted V1 frame.
        if getattr(scheduler_output, GR_RETIRE_OUTPUT_ATTR, None) is not None or getattr(
            scheduler_output, GR_STAGE_OUTPUT_ATTR, None
        ):
            require_beam_config(self.vllm_config)

        retirement = gr_execution._dispatch_retirement_frame(self, scheduler_output)
        if retirement is not None:
            return retirement

        stage = gr_execution._dispatch_stage_frame(
            self, scheduler_output, _orig_execute_model, *args, hooks_type=hooks_type, **kwargs
        )
        if stage is not gr_execution._NOT_A_STAGE_FRAME:
            return stage
        if getattr(scheduler_output, "beam_data", {}):
            raise ValueError(
                "Legacy Beam frames have been removed; use the persistent Beam entrypoint."
            )
        gr_stage = getattr(self, "_gr_gpu_stage_runner", None)
        if gr_stage is not None and not gr_stage.closed:
            if gr_stage.binding.native_request_id in getattr(
                scheduler_output, "finished_req_ids", ()
            ):
                gr_stage.cancel(gr_stage.binding)
        # Ordinary execution must not inherit a completed stage's context.
        self._current_beam_data = {}
        self._current_beam_info = {"beam_data": {}, "req_ids": []}
        token = BEAM_DATA_VAR.set(self._current_beam_info)
        self._current_scheduler_output = scheduler_output
        try:
            return _orig_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            BEAM_DATA_VAR.reset(token)

    @wraps(_orig_execute_model)
    def patched_execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        # Track in-flight model execution so the automatic post-warmup GC
        # freeze finds a quiescent boundary: the teardown signal (and hence
        # the freeze scheduling) happens at the START of this wrapper, and
        # gc.collect() holds the GIL for the whole collection — running it
        # here would stall this request's critical path.  The freeze daemon
        # waits for the depth to reach zero before collecting.
        gr_gc._enter_worker_execute()
        try:
            return _patched_execute_model_step(self, scheduler_output, *args, **kwargs)
        finally:
            gr_gc._exit_worker_execute()

    setattr(patched_execute_model, contracts._WORKER_EXECUTE_PATCH_MARKER, True)
    return patched_execute_model
