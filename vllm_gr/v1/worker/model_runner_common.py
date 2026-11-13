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
from typing import Any

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.worker import gr_execution, gr_gc
from vllm_gr.v1.worker import model_runner_contracts as contracts
from vllm_gr.v1.worker.gpu_stage_hooks import CudaStageHooksType


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
        from vllm_gr.config import BeamExecutionMode
        from vllm_gr.config.execution import require_beam_execution_mode
        from vllm_gr.v1.engine.gr_async_scheduler import (
            GR_RETIRE_OUTPUT_ATTR,
            GR_STAGE_OUTPUT_ATTR,
        )

        # Shared stage and retirement code receives an admitted V1 frame.
        if getattr(scheduler_output, GR_RETIRE_OUTPUT_ATTR, None) is not None or getattr(
            scheduler_output, GR_STAGE_OUTPUT_ATTR, None
        ):
            require_beam_execution_mode(self.vllm_config, BeamExecutionMode.V1)

        retirement = gr_execution._dispatch_retirement_frame(self, scheduler_output)
        if retirement is not None:
            return retirement

        stage = gr_execution._dispatch_stage_frame(
            self, scheduler_output, _orig_execute_model, *args, hooks_type=hooks_type, **kwargs
        )
        if stage is not gr_execution._NOT_A_STAGE_FRAME:
            return stage
        if getattr(scheduler_output, "beam_data", {}):
            raise ValueError("Legacy Beam frames have been removed; use beam.execution_mode='v1'.")
        gr_stage = getattr(self, "_gr_gpu_stage_runner", None)
        if gr_stage is not None and not gr_stage.closed:
            if getattr(scheduler_output, "beam_data", {}):
                raise ValueError("legacy Beam cannot use an active GR execution context")
            if gr_stage.binding.native_request_id in getattr(
                scheduler_output, "finished_req_ids", ()
            ):
                gr_stage.cancel(gr_stage.binding)
        # Clear the previous step's beam context at the START of the next
        # execute_model, not in a finally: with async scheduling the sampled
        # tokens are produced by ``sample_tokens`` AFTER execute_model returns,
        # and the worker-side beam decision needs ``_current_beam_data`` /
        # ``_current_beam_info`` (session + decision buffers) still alive there.
        # A stale previous-step context is overwritten here before the new
        # step's values are installed.
        self._current_beam_data = getattr(scheduler_output, "beam_data", {})
        self._current_beam_info = {"beam_data": self._current_beam_data, "req_ids": []}
        beam_info = self._current_beam_info

        # Worker-side beam decision output (NPU xllm path).  Reset each step so
        # a stale result from a previous run can never leak into this step's
        # ModelRunnerOutput.beam_search_result.
        self._beam_decision_result = None
        self._beam_result_host_ready = False
        token = BEAM_DATA_VAR.set(beam_info)
        self._current_beam_info = beam_info
        # Published for the optional Constraint Tables ``_sample`` hook
        # (constraint_hook): it reads ``_gr_request_row_configs`` to locate the
        # constrained logits rows.  Lives until the next execute_model (which
        # overwrites it at the top); with async scheduling the constraint hook
        # and worker beam decision run in ``sample_tokens`` AFTER
        # execute_model returns, so the context must stay alive past the
        # execute_model frame.
        self._current_scheduler_output = scheduler_output
        try:
            return _orig_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            # Deliberately keep ``_current_beam_data`` / ``_current_beam_info``
            # / ``_current_scheduler_output`` alive for the async sampling
            # phase (sample_tokens runs after execute_model returns).  They are
            # overwritten at the top of the next execute_model call.
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
