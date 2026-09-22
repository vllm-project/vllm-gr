# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native and persistent Beam Decode graph capture for ModelRunner V2."""

from __future__ import annotations

from functools import wraps

from vllm.logger import init_logger

from vllm_gr.v1.worker.model_runner_common import _get_gr_config

logger = init_logger(__name__)


_WORKER_V2_GRAPH_PATCH_MARKER = "_vllm_gr_beam_capture_v2_patch"


def make_patched_capture_model_v2(_orig_capture_model):
    """Capture native graphs followed by the persistent Beam Decode graph."""
    if getattr(_orig_capture_model, _WORKER_V2_GRAPH_PATCH_MARKER, False):
        return _orig_capture_model

    @wraps(_orig_capture_model)
    def patched_capture_model(self, *args, **kwargs):
        from vllm_gr.config import BeamExecutionMode

        # The persistent stage path captures its own Beam Decode forward.
        try:
            execution_mode = _get_gr_config(self).beam_execution_mode
        except AttributeError:
            execution_mode = None
        if execution_mode is BeamExecutionMode.V1:
            res = _orig_capture_model(self, *args, **kwargs)
            from vllm_gr.v1.worker.gr_startup import capture_async_forward

            self._gr_async_decode_graph_capture_failed = False
            try:
                capture_async_forward(self)
            except Exception:
                self._gr_async_decode_graph_capture_failed = True
                logger.exception(
                    "vllm-gr: async scheduling startup capture failed; "
                    "GR decode will fall back to eager"
                )
        else:
            res = _orig_capture_model(self, *args, **kwargs)

        return res

    setattr(patched_capture_model, _WORKER_V2_GRAPH_PATCH_MARKER, True)
    return patched_capture_model


__all__ = ["make_patched_capture_model_v2"]
