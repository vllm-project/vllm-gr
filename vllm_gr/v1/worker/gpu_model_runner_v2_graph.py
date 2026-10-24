# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire instance-local native Prefill capture followed by GR Beam capture."""

from __future__ import annotations

from functools import wraps
from typing import Any

import torch

from vllm_gr.v1.worker.model_runner_common import _get_gr_config

_WORKER_V2_GRAPH_PATCH_MARKER = "_vllm_gr_beam_capture_v2_patch"


def make_patched_capture_model_v2(_orig_capture_model):
    """Install the Prefill plan, run native capture, then capture Beam Decode."""
    if getattr(_orig_capture_model, _WORKER_V2_GRAPH_PATCH_MARKER, False):
        return _orig_capture_model

    @wraps(_orig_capture_model)
    def patched_capture_model(self, *args, **kwargs):
        # Profiling uses temporary KV caches and graph pools. Its native caller
        # owns their teardown; no persistent GR graph/state may escape it.
        if kwargs.get("profile_only", False):
            return _orig_capture_model(self, *args, **kwargs)
        # The Engine admits only beam_search_v1 for V2 Beam engines. The
        # execution layer checks whether Beam is enabled, not its protocol.
        if _get_gr_config(self).beam_max_width is not None:
            from vllm_gr.v1.worker.gpu_prefill_graph_runner_v2 import (
                install_prefill_graph_manager_v2,
            )

            install_prefill_graph_manager_v2(self)
            res = _orig_capture_model(self, *args, **kwargs)

            self._gr_async_decode_graph_capture_failed = False
            self._gr_async_decode_graph_capture_error = None
            capture_multibatch_forward(self)
        else:
            # No Beam resources configured: preserve ordinary native V2.
            res = _orig_capture_model(self, *args, **kwargs)

        # All applicable CUDA graphs are now captured and the serving-time
        # heap is fully built. Freeze it so long-lived post-startup objects
        # are not rescanned by gen-1/gen-2 GC during serving (V1 parity;
        # measured 29-36ms collected=0 pauses when missing).
        from vllm_gr.v1.worker.model_runner_common import (
            freeze_gc_after_graph_capture,
        )

        freeze_gc_after_graph_capture(self)
        return res

    setattr(patched_capture_model, _WORKER_V2_GRAPH_PATCH_MARKER, True)
    return patched_capture_model


__all__ = ["make_patched_capture_model_v2"]


@torch.inference_mode()
def capture_multibatch_forward(worker: Any, *, allow_eager_fallback: bool = True) -> bool:
    """Publish configured V2 Decode shapes, with shape-local fallback."""
    from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import V2StageExecutionHooks
    from vllm_gr.v1.worker.gr_startup import _capture_batch, _capture_enabled

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
    failures = {}
    for batch in batches:
        error = _capture_batch(
            worker,
            batch,
            hooks_type=V2StageExecutionHooks,
            allow_eager_fallback=allow_eager_fallback,
        )
        if error is not None:
            failures[batch] = error
    worker._gr_decode_capture_failures = failures
    worker._gr_async_decode_graph_capture_failed = len(failures) == len(batches)
    worker._gr_async_decode_graph_capture_error = next(iter(failures.values()), None)
    return len(failures) < len(batches)
