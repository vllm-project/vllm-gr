# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""ModelRunner V2 GPU wrappers for beam_search_v1 (persistent GR stage path).

The V2 runner (``vllm.v1.worker.gpu.model_runner.GPUModelRunner``) replaces the
V1 ``_prepare_inputs``/``_bookkeeping_sync``/``_sample`` trio with
``prepare_inputs``/``sample``/``sample_tokens`` and a mutable ``InputBatch``
dataclass.  Under V2 the beam surface is the beam_search_v1 persistent GR
stage path (``execution_mode=v1``); the LEGACY ``beam_data`` frame stack has
been removed, so this module installs only the slim hook set:

- ``execute_model``: the slim wrapper (retirement/stage dispatch + beam_data
  fail-fast), from
  ``model_runner_common.make_patched_execute_model_v2``.
- ``sample_tokens``: stage bypass (``stage_runner.pending`` → the stage
  runner's own constraint sampling); ordinary frames fall through to the
  native sampler.
- ``prepare_inputs``: guarded single- and multi-request GR Prefill specialization;
  unsupported inputs fall through to the native V2 InputBatch builder.
- ``add_requests``: guarded GR Prefill registration without unused native
  sampler initialization/refresh; other dispatches retain native behavior.
- ``load_model``/``shutdown``: canonical Constraint Backend + postprocess
  workspace wiring (reusing the V1 wrappers, which only read V2-present attrs).

Beam CUDA-graph capture (prefill FULL + async-forward decode) lives in
``gpu_model_runner_v2_graph`` and is wired via ``capture_model``.
"""

from __future__ import annotations

from functools import wraps

from vllm.logger import init_logger

from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES
from vllm_gr.v1.worker.gpu_model_runner_patch import (
    _BEAM_POSTPROCESS_LOAD_PATCH_MARKER,
    make_load_model_with_beam_postprocess_workspace,
)
from vllm_gr.v1.worker.gpu_model_runner_v2_resources import close_v2_runtime
from vllm_gr.v1.worker.gpu_prefill_inputs import try_prepare_gr_prefill_inputs_v2
from vllm_gr.v1.worker.model_runner_common import (
    _CONSTRAINT_LOAD_PATCH_MARKER,
    _CONSTRAINT_SHUTDOWN_PATCH_MARKER,
    _WORKER_V2_ADD_REQUESTS_PATCH_MARKER,
    _WORKER_V2_PREPARE_PATCH_MARKER,
    _WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER,
    is_worker_runner_v2_patched,
    make_patched_execute_model_v2,
)
from vllm_gr.v1.worker.model_runner_patch import (
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)

logger = init_logger(__name__)


def discover_worker_runners() -> tuple[tuple[str, type], ...]:
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


def make_patched_sample_tokens_v2(_orig_sample_tokens):
    """Attach a worker beam decision produced during async sampling (V2)."""
    if getattr(_orig_sample_tokens, _WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, False):
        return _orig_sample_tokens

    @wraps(_orig_sample_tokens)
    def patched_sample_tokens(self, grammar_output, *args, **kwargs):
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


def make_patched_prepare_inputs_v2(_orig_prepare_inputs):
    """Preserve native V2 input preparation outside the supported GR Prefill."""
    if getattr(_orig_prepare_inputs, _WORKER_V2_PREPARE_PATCH_MARKER, False):
        return _orig_prepare_inputs

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        batch = try_prepare_gr_prefill_inputs_v2(
            self, scheduler_output, batch_req_state, batch_desc
        )
        if batch is not None:
            return batch
        return _orig_prepare_inputs(self, scheduler_output, batch_req_state, batch_desc)

    setattr(patched_prepare_inputs, _WORKER_V2_PREPARE_PATCH_MARKER, True)
    return patched_prepare_inputs


def make_patched_add_requests_v2(_orig_add_requests):
    """Keep native registration outside an admitted, supported GR Prefill."""
    if getattr(_orig_add_requests, _WORKER_V2_ADD_REQUESTS_PATCH_MARKER, False):
        return _orig_add_requests

    from vllm_gr.v1.worker.gpu_prefill_requests_v2 import try_add_gr_prefill_requests

    @wraps(_orig_add_requests)
    def patched_add_requests(self, scheduler_output):
        if try_add_gr_prefill_requests(self, scheduler_output):
            return None
        return _orig_add_requests(self, scheduler_output)

    setattr(patched_add_requests, _WORKER_V2_ADD_REQUESTS_PATCH_MARKER, True)
    return patched_add_requests


def _apply_v2_runner_patches() -> None:
    """Install grouped-Beam hooks on the V2 GPU ModelRunner class.

    Idempotent: every wrapper is marker-guarded and the whole runner is skipped
    once ``is_worker_runner_v2_patched`` reports success.  The postprocess
    workspace load wrapper is applied first and the canonical Constraint
    Backend load wrapper last, so the final ``load_model`` carries the
    constraint marker that the V2 validation gate checks.
    """
    from vllm_gr.v1.worker.numpy_staging import install_numpy_staging

    install_numpy_staging()
    runners_to_patch = [
        (runner_name, runner_class)
        for runner_name, runner_class in discover_worker_runners()
        if not is_worker_runner_v2_patched(runner_class)
    ]

    for runner_name, runner_class in runners_to_patch:
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
                close_runtime=close_v2_runtime,
            )

        reload_weights = getattr(runner_class, "reload_weights", None)
        if callable(reload_weights):
            from vllm_gr.ops.gpu_stage_logits import make_gpu_stage_reload_weights

            runner_class.reload_weights = make_gpu_stage_reload_weights(reload_weights)

        shutdown = getattr(runner_class, "shutdown", None)
        if not getattr(shutdown, _CONSTRAINT_SHUTDOWN_PATCH_MARKER, False):
            runner_class.shutdown = make_shutdown_with_constraint_backend(
                shutdown if callable(shutdown) else None, close_runtime=close_v2_runtime
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
            from vllm_gr.v1.worker.gpu_model_runner_v2_graph import (
                make_patched_capture_model_v2,
            )

            runner_class.capture_model = make_patched_capture_model_v2(capture_model)

        from vllm_gr.v1.worker.gpu_model_runner_v2_graph import (
            make_patched_initialize_kv_cache_v2,
            make_patched_profile_cudagraph_memory_v2,
        )

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
    "make_patched_add_requests_v2",
    "make_patched_prepare_inputs_v2",
    "make_patched_sample_tokens_v2",
]
