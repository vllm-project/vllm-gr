# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire instance-local native Prefill capture followed by GR Beam capture."""

from __future__ import annotations

import gc
from functools import wraps
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_gr.v1.worker.model_runner_common import _get_gr_config

_WORKER_V2_GRAPH_PATCH_MARKER = "_vllm_gr_beam_capture_v2_patch"
_WORKER_V2_KV_PATCH_MARKER = "_vllm_gr_beam_kv_v2_patch"
_WORKER_V2_PROFILE_PATCH_MARKER = "_vllm_gr_beam_profile_v2_patch"
logger = init_logger(__name__)


def make_patched_initialize_kv_cache_v2(original):
    """Publish the GR plan before the native profiler snapshots its manager."""
    if getattr(original, _WORKER_V2_KV_PATCH_MARKER, False):
        return original

    @wraps(original)
    def initialize_kv_cache(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if _get_gr_config(self).beam_max_width is not None:
            from vllm_gr.v1.worker.gpu_prefill_graph_runner_v2 import (
                install_prefill_graph_manager_v2,
            )

            install_prefill_graph_manager_v2(self)
        return result

    setattr(initialize_kv_cache, _WORKER_V2_KV_PATCH_MARKER, True)
    return initialize_kv_cache


def make_patched_profile_cudagraph_memory_v2(original):
    """Keep native temporary KV/pool orchestration and retire GR resources too."""
    if getattr(original, _WORKER_V2_PROFILE_PATCH_MARKER, False):
        return original

    @wraps(original)
    def profile_cudagraph_memory(self, *args, **kwargs):
        if _get_gr_config(self).beam_max_width is None:
            return original(self, *args, **kwargs)
        # Profiling is a startup operation. Never discard a serving owner or
        # a graph bound to the real KV cache if called out of order.
        if (
            getattr(self, "_gr_gpu_stage_runner", None) is not None
            or getattr(self, "_beam_decode_graph", None) is not None
        ):
            raise RuntimeError("V2 graph memory profiling requires an unbound GR runtime")
        context = getattr(self, "_beam_context", None)
        try:
            return original(self, *args, **kwargs)
        finally:
            from vllm_gr.v1.worker.gpu_model_runner_v2_resources import close_v2_runtime

            # Native teardown has dropped its KV/Prefill bindings. GR Decode
            # retains attention metadata and inputs until these owners close.
            close_v2_runtime(self)
            if context is None and hasattr(self, "_beam_context"):
                del self._beam_context
            gc.collect()
            torch.accelerator.empty_cache()

    setattr(profile_cudagraph_memory, _WORKER_V2_PROFILE_PATCH_MARKER, True)
    return profile_cudagraph_memory


def make_patched_capture_model_v2(_orig_capture_model):
    """Install the Prefill plan, run native capture, then capture Beam Decode."""
    if getattr(_orig_capture_model, _WORKER_V2_GRAPH_PATCH_MARKER, False):
        return _orig_capture_model

    @wraps(_orig_capture_model)
    def patched_capture_model(self, *args, **kwargs):
        profile_only = kwargs.get("profile_only", False)
        # The Engine admits only beam_search_v1 for V2 Beam engines. The
        # execution layer checks whether Beam is enabled, not its protocol.
        if _get_gr_config(self).beam_max_width is not None:
            from vllm_gr.v1.worker.gpu_prefill_graph_runner_v2 import (
                GRPrefillCudaGraphManager,
                install_prefill_graph_manager_v2,
            )

            install_prefill_graph_manager_v2(self)
            manager = self.cudagraph_manager
            if profile_only and isinstance(manager, GRPrefillCudaGraphManager):
                # GR's token/request grid has nonuniform per-graph costs.
                # Two native FULL samples underestimated the measured plan by
                # 28% even after accounting for Decode. Measure the full GR
                # plan in the native throwaway pool; leave its sample list
                # empty so native extrapolation returns the measured total.
                manager._max_full_descs_to_capture = None
                manager._capture_mem_samples = None
            res = _orig_capture_model(self, *args, **kwargs)

            self._gr_async_decode_graph_capture_failed = False
            self._gr_async_decode_graph_capture_error = None
            before_decode = torch.accelerator.get_memory_info()[0]
            from vllm.compilation import monitor

            capturing_enabled = monitor.cudagraph_capturing_enabled
            try:
                capture_multibatch_forward(self)
            finally:
                # Beam's local capture closes its window. Profiling runs
                # inside native startup, which still needs to capture later.
                if profile_only:
                    monitor.set_cudagraph_capturing_enabled(capturing_enabled)
            decode_bytes = before_decode - torch.accelerator.get_memory_info()[0]
            res += decode_bytes
            logger.info(
                "GR V2 graph memory (%s): Decode and fixed state %.2f GiB, "
                "Prefill plus Decode %.2f GiB",
                "profile" if profile_only else "actual",
                decode_bytes / (1 << 30),
                res / (1 << 30),
            )
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

        if not profile_only:
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
