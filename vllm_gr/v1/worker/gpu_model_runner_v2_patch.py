# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""ModelRunner V2 GPU wrappers for beam_search_v1 (persistent GR stage path).

The V2 runner (``vllm.v1.worker.gpu.model_runner.GPUModelRunner``) replaces the
V1 ``_prepare_inputs``/``_bookkeeping_sync``/``_sample`` trio with
``prepare_inputs``/``sample``/``sample_tokens`` and a mutable ``InputBatch``
dataclass.  Under V2 the beam surface is the beam_search_v1 persistent GR
stage path (``execution_mode=v1``); the LEGACY ``beam_data`` frame stack has
been removed, so this module installs only the slim hook set:

- ``execute_model``: the local slim wrapper (retirement/stage dispatch and
  legacy beam_data rejection), preserving ordinary native execution.
- ``sample_tokens``: stage bypass (``stage_runner.pending`` → the stage
  runner's own constraint sampling); ordinary frames fall through to the
  native sampler.
- ``prepare_inputs``: guarded single- and multi-request GR Prefill specialization;
  unsupported inputs fall through to the native V2 InputBatch builder.
- ``add_requests``: guarded GR Prefill registration without unused native
  sampler initialization/refresh; other dispatches retain native behavior.
- ``load_model``/``shutdown``: canonical Constraint Backend + shared CUDA
  postprocess workspace wiring.

This module owns graph patch installation and native Prefill profiling/capture
orchestration. V2 resource support owns multi-batch Decode capture; the native
Prefill graph manager retains FULL/PIECEWISE planning and replay.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from functools import wraps
from typing import Any, cast

import torch
from vllm.logger import init_logger

from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES
from vllm_gr.v1.worker import gpu_model_runner_v2_resources as v2_resources
from vllm_gr.v1.worker import gr_execution, gr_gc
from vllm_gr.v1.worker.gpu_postprocess_workspace import (
    _BEAM_POSTPROCESS_LOAD_PATCH_MARKER,
    make_load_model_with_beam_postprocess_workspace,
)
from vllm_gr.v1.worker.gpu_prefill_v2 import (
    try_add_gr_prefill_requests,
    try_prepare_gr_prefill_inputs_v2,
)
from vllm_gr.v1.worker.gr_context import (
    _get_gr_config,
)
from vllm_gr.v1.worker.model_runner_contracts import (
    _CONSTRAINT_LOAD_PATCH_MARKER,
    _CONSTRAINT_SHUTDOWN_PATCH_MARKER,
    _WORKER_EXECUTE_PATCH_MARKER,
    _WORKER_V2_ADD_REQUESTS_PATCH_MARKER,
    _WORKER_V2_PREPARE_PATCH_MARKER,
    _WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER,
    is_worker_runner_v2_patched,
)
from vllm_gr.v1.worker.model_runner_patch import (
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)

logger = init_logger(__name__)

_WORKER_V2_GRAPH_PATCH_MARKER = "_vllm_gr_beam_capture_v2_patch"
_WORKER_V2_KV_PATCH_MARKER = "_vllm_gr_beam_kv_v2_patch"
_WORKER_V2_PROFILE_PATCH_MARKER = "_vllm_gr_beam_profile_v2_patch"


def _warmup_constraint_backend(runner: Any, backend: Any) -> None:
    from vllm_gr.config import ConstraintTableBackendEnum

    if backend.info.backend is ConstraintTableBackendEnum.CUDA:
        backend.warmup_stage_topk(runner.model_config.dtype)
        head_dtype = getattr(runner.model_config, "head_dtype", None)
        if head_dtype is not None and head_dtype != runner.model_config.dtype:
            backend.warmup_stage_topk(head_dtype)


def discover_worker_runners() -> tuple[tuple[str, type[Any]], ...]:
    """Discover the upstream V2 GPU ModelRunner without importing NPU runtime."""
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    except ImportError as exc:
        logger.warning(
            "Failed to import vllm.v1.worker.gpu.model_runner.GPUModelRunner; "
            "skipping vllm-gr V2 beam-search runner patches for this attempt. "
            "ImportError: %s",
            exc,
        )
        logger.debug("GPUModelRunnerV2 import traceback", exc_info=True)
        return ()
    return (("GPUModelRunnerV2", GPUModelRunnerV2),)


def make_patched_execute_model_v2(
    _orig_execute_model: Callable[..., Any],
) -> Callable[..., Any]:
    """Slim execute_model wrapper for the V2 ModelRunner (beam_search_v1 only).

    Refactor design §4.2: under V2 the beam surface is the persistent GR
    stage path; the LEGACY ``beam_data`` frame stack (M1 remap / M2 inline
    sample / session refresh) is not supported.  The
    wrapper therefore:

    - dispatches retirement and stage control frames (shared helpers);
    - FAILS FAST on a ``beam_data`` frame — the historical alternative was a
      silent wrong-beam-results mode when the V1/V2 flag resolved differently
      than the patch layer assumed (ISSUES §1.1); a loud error names both
      remediations (execution_mode=v1, or unset the V2 flag);

    Everything else is the untouched native ``execute_model``.
    """
    if getattr(_orig_execute_model, _WORKER_EXECUTE_PATCH_MARKER, False):
        return _orig_execute_model

    from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import CudaStageHooksV2

    @wraps(_orig_execute_model)
    def _patched_execute_model_step_v2(
        self: Any, scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> Any:
        retirement = gr_execution._dispatch_retirement_frame(self, scheduler_output)
        if retirement is not None:
            return retirement

        stage = gr_execution._dispatch_stage_frame(
            self,
            scheduler_output,
            _orig_execute_model,
            *args,
            hooks_type=CudaStageHooksV2,
            **kwargs,
        )
        if stage is not gr_execution._NOT_A_STAGE_FRAME:
            return stage

        if getattr(scheduler_output, "beam_data", {}):
            raise ValueError(
                "legacy beam_data frames are not supported under ModelRunnerV2; "
                "use beam.execution_mode='v1' or unset VLLM_USE_V2_MODEL_RUNNER"
            )

        return _orig_execute_model(self, scheduler_output, *args, **kwargs)

    @wraps(_orig_execute_model)
    def patched_execute_model_v2(
        self: Any, scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> Any:
        # Same quiescent-boundary tracking as the V1 wrapper: the GC-freeze
        # daemon waits for in-flight executions to drain before collecting.
        gr_gc._enter_worker_execute()
        try:
            return _patched_execute_model_step_v2(self, scheduler_output, *args, **kwargs)
        finally:
            gr_gc._exit_worker_execute()

    setattr(patched_execute_model_v2, _WORKER_EXECUTE_PATCH_MARKER, True)
    return patched_execute_model_v2


def make_patched_sample_tokens_v2(
    _orig_sample_tokens: Callable[..., Any],
) -> Callable[..., Any]:
    """Attach a worker beam decision produced during async sampling (V2)."""
    if getattr(_orig_sample_tokens, _WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, False):
        return _orig_sample_tokens

    @wraps(_orig_sample_tokens)
    def patched_sample_tokens(self: Any, grammar_output: Any, *args: Any, **kwargs: Any) -> Any:
        # Persistent GR stage execution (beam_search_v1 contract): the stage
        # runner produced the dispatch's outputs on the compute stream and
        # owns the async D2H; bypass the native sampler entirely (V1 parity
        # with make_patched_sample_tokens_gpu).
        stage_runner = getattr(self, "_gr_gpu_stage_runner", None)
        if stage_runner is not None and stage_runner.pending is not None:
            return stage_runner.sample(grammar_output)
        # No legacy beam_data frame under V2 (fail-fast in the slim execute
        # wrapper); ordinary frames sample natively.
        return _orig_sample_tokens(self, grammar_output, *args, **kwargs)

    setattr(patched_sample_tokens, _WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, True)
    return patched_sample_tokens


def make_patched_prepare_inputs_v2(
    _orig_prepare_inputs: Callable[..., Any],
) -> Callable[..., Any]:
    """Preserve native V2 input preparation outside the supported GR Prefill."""
    if getattr(_orig_prepare_inputs, _WORKER_V2_PREPARE_PATCH_MARKER, False):
        return _orig_prepare_inputs

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(
        self: Any, scheduler_output: Any, batch_req_state: Any, batch_desc: Any
    ) -> Any:
        batch = try_prepare_gr_prefill_inputs_v2(
            self, scheduler_output, batch_req_state, batch_desc
        )
        if batch is not None:
            return batch
        return _orig_prepare_inputs(self, scheduler_output, batch_req_state, batch_desc)

    setattr(patched_prepare_inputs, _WORKER_V2_PREPARE_PATCH_MARKER, True)
    return patched_prepare_inputs


def make_patched_add_requests_v2(
    _orig_add_requests: Callable[..., Any],
) -> Callable[..., Any]:
    """Keep native registration outside an admitted, supported GR Prefill."""
    if getattr(_orig_add_requests, _WORKER_V2_ADD_REQUESTS_PATCH_MARKER, False):
        return _orig_add_requests

    @wraps(_orig_add_requests)
    def patched_add_requests(self: Any, scheduler_output: Any) -> Any:
        if try_add_gr_prefill_requests(self, scheduler_output):
            return None
        return _orig_add_requests(self, scheduler_output)

    setattr(patched_add_requests, _WORKER_V2_ADD_REQUESTS_PATCH_MARKER, True)
    return patched_add_requests


def make_patched_initialize_kv_cache_v2(original: Any) -> Any:
    """Publish the GR plan before the native profiler snapshots its manager."""
    if getattr(original, _WORKER_V2_KV_PATCH_MARKER, False):
        return original

    @wraps(original)
    def initialize_kv_cache(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        if _get_gr_config(self).beam_max_width is not None:
            from vllm_gr.v1.worker.gpu_prefill_graph_runner_v2 import (
                install_prefill_graph_manager_v2,
            )

            install_prefill_graph_manager_v2(self)
        return result

    setattr(initialize_kv_cache, _WORKER_V2_KV_PATCH_MARKER, True)
    return initialize_kv_cache


def make_patched_profile_cudagraph_memory_v2(original: Any) -> Any:
    """Keep native temporary KV/pool orchestration and retire GR resources too."""
    if getattr(original, _WORKER_V2_PROFILE_PATCH_MARKER, False):
        return original

    @wraps(original)
    def profile_cudagraph_memory(self: Any, *args: Any, **kwargs: Any) -> Any:
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
            # Native teardown has dropped its KV/Prefill bindings. GR Decode
            # retains attention metadata and inputs until these owners close.
            v2_resources.close_v2_runtime(self)
            if context is None and hasattr(self, "_beam_context"):
                del self._beam_context
            gc.collect()
            torch.accelerator.empty_cache()

    setattr(profile_cudagraph_memory, _WORKER_V2_PROFILE_PATCH_MARKER, True)
    return profile_cudagraph_memory


def make_patched_capture_model_v2(_orig_capture_model: Any) -> Any:
    """Install the Prefill plan, run native capture, then capture Beam Decode."""
    if getattr(_orig_capture_model, _WORKER_V2_GRAPH_PATCH_MARKER, False):
        return _orig_capture_model

    @wraps(_orig_capture_model)
    def patched_capture_model(self: Any, *args: Any, **kwargs: Any) -> Any:
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
                # Measure the full GR plan in the native throwaway pool
                # rather than extrapolating two FULL samples. Leave its sample list
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
                v2_resources.capture_multibatch_forward(self)
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
        from vllm_gr.v1.worker.gr_gc import (
            freeze_gc_after_graph_capture,
        )

        if not profile_only:
            freeze_gc_after_graph_capture(self)
        return res

    setattr(patched_capture_model, _WORKER_V2_GRAPH_PATCH_MARKER, True)
    return patched_capture_model


def _apply_v2_runner_patches() -> None:
    """Install grouped-Beam hooks on the V2 GPU ModelRunner class.

    Idempotent: every wrapper is marker-guarded and the whole runner is skipped
    once ``is_worker_runner_v2_patched`` reports success.  The postprocess
    workspace load wrapper is applied first and the canonical Constraint
    Backend load wrapper last, so the final ``load_model`` carries the
    constraint marker that the V2 validation gate checks.
    """
    runners_to_patch = [
        (runner_name, runner_class)
        for runner_name, runner_class in discover_worker_runners()
        if not is_worker_runner_v2_patched(runner_class)
    ]

    for runner_name, runner_type in runners_to_patch:
        # Native classes expose dynamically patched methods outside this boundary.
        runner_class = cast(Any, runner_type)
        load_model = getattr(runner_class, "load_model", None)
        if callable(load_model) and not getattr(
            load_model, _BEAM_POSTPROCESS_LOAD_PATCH_MARKER, False
        ):
            runner_class.load_model = make_load_model_with_beam_postprocess_workspace(load_model)

        load_model = getattr(runner_class, "load_model", None)
        if callable(load_model) and not getattr(load_model, _CONSTRAINT_LOAD_PATCH_MARKER, False):
            runner_class.load_model = make_load_model_with_constraint_backend(
                load_model,
                max_batch=max(SUPPORTED_BEAM_BATCH_SIZES),
                close_runtime=v2_resources.close_v2_runtime,
                prepare_backend=_warmup_constraint_backend,
            )

        shutdown = getattr(runner_class, "shutdown", None)
        if not getattr(shutdown, _CONSTRAINT_SHUTDOWN_PATCH_MARKER, False):
            runner_class.shutdown = make_shutdown_with_constraint_backend(
                shutdown if callable(shutdown) else None,
                close_runtime=v2_resources.close_v2_runtime,
            )

        runner_class.execute_model = make_patched_execute_model_v2(runner_class.execute_model)
        runner_class.sample_tokens = make_patched_sample_tokens_v2(runner_class.sample_tokens)
        runner_class.prepare_inputs = make_patched_prepare_inputs_v2(runner_class.prepare_inputs)
        runner_class.add_requests = make_patched_add_requests_v2(runner_class.add_requests)

        # Native Prefill plan capture + async-forward decode graph capture.
        # These optional graph hooks are NOT part of
        # is_worker_runner_v2_patched's required set.
        capture_model = getattr(runner_class, "capture_model", None)
        if callable(capture_model):
            runner_class.capture_model = make_patched_capture_model_v2(capture_model)

        initialize_kv_cache = getattr(runner_class, "initialize_kv_cache", None)
        if callable(initialize_kv_cache):
            runner_class.initialize_kv_cache = make_patched_initialize_kv_cache_v2(
                initialize_kv_cache
            )
        profile_cudagraph_memory = getattr(runner_class, "profile_cudagraph_memory", None)
        if callable(profile_cudagraph_memory):
            runner_class.profile_cudagraph_memory = make_patched_profile_cudagraph_memory_v2(
                profile_cudagraph_memory
            )

        logger.info("%s V2 hooks patched successfully.", runner_name)


__all__ = [
    "_apply_v2_runner_patches",
    "discover_worker_runners",
    "make_patched_execute_model_v2",
    "make_patched_add_requests_v2",
    "make_patched_capture_model_v2",
    "make_patched_initialize_kv_cache_v2",
    "make_patched_profile_cudagraph_memory_v2",
    "make_patched_prepare_inputs_v2",
    "make_patched_sample_tokens_v2",
]
