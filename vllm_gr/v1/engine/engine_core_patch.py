# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

from functools import wraps

import torch
from vllm.logger import init_logger

from vllm_gr.v1.engine.scheduler_metadata import build_beam_request_metadata
from vllm_gr.v1.engine.wire import register_engine_core_request_types
from vllm_gr.v1.worker.model_runner_common import BeamReqIdsProxy  # noqa: F401
from vllm_gr.v1.worker.model_runner_common import Mock1DArray  # noqa: F401
from vllm_gr.v1.worker.model_runner_common import Mock2DArray  # noqa: F401
from vllm_gr.v1.worker.model_runner_common import _get_positions_buffers  # noqa: F401
from vllm_gr.v1.worker.model_runner_common import _remux_logprobs_tensors  # noqa: F401
from vllm_gr.v1.worker.model_runner_common import _write_position_segments  # noqa: F401
from vllm_gr.v1.worker.model_runner_common import (  # noqa: F401
    _WORKER_BOOKKEEPING_PATCH_MARKER,
    _WORKER_EXECUTE_PATCH_MARKER,
    _WORKER_PREPARE_PATCH_MARKER,
    InputBatchProxy,
    _beam_collapse_bookkeeping,
    _beam_remap_inputs,
    _process_beam_decode_request,
    is_worker_runner_patched,
    make_patched_execute_model,
)

logger = init_logger(__name__)

_BEAM_STATE_INIT_MARKER = "_vllm_gr_initializes_beam_state"
_SCHEDULER_PATCH_MARKER = "_vllm_gr_beam_scheduler_patch"

# ---------------------------------------------------------------------------
# EngineCore.__init__ wrapper
# ---------------------------------------------------------------------------


def make_patched_engine_core_init(original_init):
    """Wrap EngineCore.__init__ to add beam state."""

    if getattr(original_init, _BEAM_STATE_INIT_MARKER, False):
        return original_init

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        # Initialize before calling upstream. EngineCoreProc starts its input
        # thread from __init__, so beam state must exist throughout startup.
        from vllm_gr.v1.engine.core import _ensure_beam_state

        _ensure_beam_state(self)
        original_init(self, *args, **kwargs)

    setattr(patched_init, _BEAM_STATE_INIT_MARKER, True)

    return patched_init


# ---------------------------------------------------------------------------
# Child-process patch applicator — called from patched run_engine_core
# ---------------------------------------------------------------------------


def apply_engine_core_request_patches() -> None:
    """Install EngineCore request dispatch and beam-state support."""
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    from vllm_gr.v1.engine.core import (
        _cache_beam_request,
        _handle_beam_request_step_update,
        process_input_sockets,
    )

    logger.info("[vllm-gr] apply_engine_core_child_patches called.")

    register_engine_core_request_types()

    # Wrap both constructors. Patching EngineCoreProc explicitly protects
    # against changes in upstream inheritance and plugin loading order.
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)
    EngineCoreProc.__init__ = make_patched_engine_core_init(EngineCoreProc.__init__)

    # Bind request handlers explicitly on both classes.
    EngineCore._cache_beam_request = _cache_beam_request
    EngineCore._handle_beam_request_step_update = _handle_beam_request_step_update
    EngineCoreProc._cache_beam_request = _cache_beam_request
    EngineCoreProc._handle_beam_request_step_update = _handle_beam_request_step_update

    # Replace EngineCoreProc.process_input_sockets
    EngineCoreProc.process_input_sockets = process_input_sockets

    EngineCore._patched_for_beam_requests = True
    EngineCoreProc._patched_for_beam_requests = True
    logger.debug("Engine-core request patches applied.")


def apply_engine_core_child_patches() -> None:
    """Compatibility helper that installs all EngineCore-side capabilities.

    This is called by the process-entry wrapper in a freshly spawned
    interpreter. Each capability has its own idempotent applicator so a
    partially initialized process can be checked and repaired explicitly.
    """
    apply_engine_core_request_patches()
    apply_scheduler_patch()
    apply_worker_patches()


# Re-entry guard to detect recursive calls
_run_engine_core_active = False


def run_engine_core(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    """Wrapper for EngineCoreProc.run_engine_core that applies patches
    before the engine starts.  This is a module-level function so that
    multiprocessing 'spawn' can pickle it by qualified name.

    In the spawned child process, all modules are freshly imported, so
    EngineCoreProc.run_engine_core is the original unpatched version.
    We apply our patches first, then delegate to the original.
    """
    global _run_engine_core_active

    # Re-entry guard: detect if we're already inside run_engine_core
    if _run_engine_core_active:
        raise RuntimeError(
            "run_engine_core re-entry detected. This indicates a recursive "
            "call, likely due to patching being applied multiple times. "
            "Check that patch_batch_and_fork() is only called once."
        )

    from vllm.v1.engine.core import EngineCoreProc

    # Guard against recursion: if vllm_gr/init.py was auto-imported in
    # the child process, run_patch() will have replaced
    # EngineCoreProc.run_engine_core with *this* function again.
    # Use the saved original stashed by patch_batch_and_fork() if
    # available; otherwise fall back to the current class attribute
    # (which is the unpatched original in a fresh child interpreter).
    original_run = getattr(run_engine_core, "_original_run_engine_core", None)

    # Validate the saved original
    if original_run is not None and original_run is not run_engine_core:
        # We have a saved original that isn't ourselves - validate it
        if not callable(original_run):
            raise RuntimeError(
                f"run_engine_core._original_run_engine_core is not callable: "
                f"type={type(original_run)}"
            )
    else:
        # No saved original (or it points to ourselves, which is invalid)
        # Fall back to the current class attribute (should be the original in a fresh process)
        if original_run is run_engine_core:
            logger.warning(
                "run_engine_core._original_run_engine_core points to itself. "
                "This may indicate corrupted patch state from a previous run. "
                "Falling back to EngineCoreProc.run_engine_core."
            )
        original_run = EngineCoreProc.run_engine_core

    # Final validation: ensure we didn't end up with ourselves
    if original_run is run_engine_core:
        raise RuntimeError(
            "run_engine_core recursion detected: EngineCoreProc.run_engine_core "
            "points to this wrapper. This indicates that patching was applied "
            "but the original was not saved, or the child process imported "
            "vllm_gr before starting."
        )

    # Validate that original_run looks like a real function from vllm
    # (Skip validation for test mocks)
    if hasattr(original_run, "__module__"):
        original_module = original_run.__module__
        expected_module = "vllm.v1.engine.core"
        # Skip warning for unittest mocks (used in tests)
        if original_module not in (expected_module, "unittest.mock", "mock"):
            logger.warning(
                "Unexpected module for original run_engine_core: %s (expected %s). "
                "Proceeding anyway, but this may indicate an issue.",
                original_module,
                expected_module,
            )

    # Set re-entry guard and apply patches
    _run_engine_core_active = True
    try:
        # Use the shared initialization mechanism with an EngineCore-only
        # plan. In a spawned child this repairs targets that became importable
        # only after the frontend initialized without importing API modules.
        from vllm_gr.patch import initialize_engine_core_beam_patches

        initialize_engine_core_beam_patches()
        original_run(
            *args,
            dp_rank=dp_rank,
            local_dp_rank=local_dp_rank,
            **kwargs,
        )
    finally:
        _run_engine_core_active = False


def apply_scheduler_patch():
    """Monkey-patch Scheduler.schedule to propagate custom attributes via SchedulerOutput."""
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, False):
        return

    _original_schedule = Scheduler.schedule

    @wraps(_original_schedule)
    def patched_schedule(self):
        _original_get_computed_blocks = self.kv_cache_manager.get_computed_blocks
        block_size = self.cache_config.block_size

        def patch_get_computed_blocks(req):
            bn, tn = _original_get_computed_blocks(req)
            if req and getattr(req, "is_beam_decode", False):
                pl = getattr(req, "prefix_len", 0) // block_size * block_size
                if tn > pl:
                    sliced_blocks = tuple(group[: pl // block_size] for group in bn.blocks)
                    return self.kv_cache_manager.create_kv_cache_blocks(sliced_blocks), pl
            return bn, tn

        self.kv_cache_manager.get_computed_blocks = patch_get_computed_blocks
        try:
            output = _original_schedule(self)

            # Propagate beam request attributes from Request objects to SchedulerOutput
            beam_data = {}
            if hasattr(output, "num_scheduled_tokens"):
                for req_id, num_scheduled_tokens in output.num_scheduled_tokens.items():
                    req = self.requests.get(req_id)
                    if req:
                        metadata = build_beam_request_metadata(req, num_scheduled_tokens)
                        if metadata is not None:
                            beam_data[req_id] = metadata
            output.beam_data = beam_data
            return output
        finally:
            self.kv_cache_manager.get_computed_blocks = _original_get_computed_blocks

    Scheduler.schedule = patched_schedule
    setattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, True)
    Scheduler._patched_for_beam_requests = True
    logger.debug("Scheduler.schedule patched for beam-requests propagation.")


def make_patched_prepare_inputs_gpu(_orig_prepare_inputs):
    """GPU ``_prepare_inputs`` wrapper: ``(scheduler_output, num_scheduled_tokens)``
    returning ``(logits_indices, spec_decode_metadata)``."""

    if getattr(_orig_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False):
        return _orig_prepare_inputs

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        logits_indices, spec_decode_metadata = _orig_prepare_inputs(
            self, scheduler_output, num_scheduled_tokens
        )

        # Job A: the GPU attention builder runs after _prepare_inputs returns, so
        # populating req_ids here (post-orig) is in time.
        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = self.input_batch.req_ids

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data:
            return logits_indices, spec_decode_metadata

        new_logits_indices = _beam_remap_inputs(self, scheduler_output, num_scheduled_tokens)
        return (
            torch.tensor(new_logits_indices, dtype=torch.int32, device=self.device),
            spec_decode_metadata,
        )

    setattr(patched_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, True)
    return patched_prepare_inputs


def make_patched_bookkeeping_sync_gpu(_orig_bookkeeping_sync):
    """GPU ``_bookkeeping_sync`` wrapper: returns the 7-tuple including
    ``num_nans_in_logits``."""

    if getattr(_orig_bookkeeping_sync, _WORKER_BOOKKEEPING_PATCH_MARKER, False):
        return _orig_bookkeeping_sync

    @wraps(_orig_bookkeeping_sync)
    def patched_bookkeeping_sync(self, scheduler_output, *args, **kwargs):
        if hasattr(self, "_beam_logprobs_cache") and self._beam_logprobs_cache:
            if hasattr(scheduler_output, "finished_req_ids"):
                for req_id in scheduler_output.finished_req_ids:
                    self._beam_logprobs_cache.pop(req_id, None)

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data or not hasattr(scheduler_output, "_gr_request_row_configs"):
            return _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)

        request_row_configs = scheduler_output._gr_request_row_configs
        orig_input_batch = object.__getattribute__(self.input_batch, "_obj")

        try:
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
        finally:
            self.input_batch = orig_input_batch

        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = res

        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(
            self, orig_input_batch, request_row_configs, beam_data, sampler_output
        )

        return (
            num_nans_in_logits,
            [[] for _ in range(n)] if logprobs_lists is not None else None,
            [[] for _ in range(n)],
            prompt_logprobs_dict,
            collapsed_req_ids,
            collapsed_idx,
            invalid_req_indices,
        )

    setattr(patched_bookkeeping_sync, _WORKER_BOOKKEEPING_PATCH_MARKER, True)
    return patched_bookkeeping_sync


def make_patched_prepare_inputs_npu(_orig_prepare_inputs):
    """NPU ``_prepare_inputs`` wrapper.

    vLLM-Ascend 0.22 uses the upstream-style signature
    ``(scheduler_output, num_scheduled_tokens)`` and returns
    ``(logits_indices, spec_decode_metadata, total_num_scheduled_tokens,
    num_scheduled_tokens_compressed_list)``.

    Two NPU-specific differences vs the GPU wrapper:

    - The NPU attention metadata builder runs *inside* the original
      ``_prepare_inputs``, so Job A (publishing ``req_ids`` to the contextvar)
      must happen *before* delegating, not after.
    - ``_prepare_inputs`` is called from ``execute_model``, while sampling and
      ``_bookkeeping_sync`` run in a separate ``sample_tokens`` call. If an
      early return skips bookkeeping, a stale ``InputBatchProxy`` could survive
      to the next step, so we defensively un-proxy at entry.

    The per-beam RoPE position overwrite handles both the normal
    ``CpuGpuBuffer`` and the plain device tensor used by 0.22 PCP paths. The
    physical ``slot_mapping`` is materialized from linear positions inside the
    original before the overwrite, so it is unaffected.
    """

    if getattr(_orig_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False):
        return _orig_prepare_inputs

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        # Drop a stale proxy left over from a prior step (see docstring).
        if isinstance(self.input_batch, InputBatchProxy):
            self.input_batch = object.__getattribute__(self.input_batch, "_obj")

        # Job A: publish req_ids BEFORE the original builds attention metadata.
        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = self.input_batch.req_ids

        out = _orig_prepare_inputs(self, scheduler_output, num_scheduled_tokens)

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data:
            return out

        (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
            num_scheduled_tokens_compressed_list,
        ) = out

        new_logits_indices = _beam_remap_inputs(self, scheduler_output, num_scheduled_tokens)
        logits_indices = torch.tensor(new_logits_indices, dtype=torch.int32, device=self.device)

        return (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
            num_scheduled_tokens_compressed_list,
        )

    setattr(patched_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, True)
    return patched_prepare_inputs


def make_patched_bookkeeping_sync_npu(_orig_bookkeeping_sync):
    """NPU ``_bookkeeping_sync`` wrapper: returns the 6-tuple (no
    ``num_nans_in_logits``).

    NPU calls ``_bookkeeping_sync(scheduler_output, sampler_output, logits,
    hidden_states, num_scheduled_tokens, spec_decode_metadata)`` positionally,
    so ``sampler_output`` is ``args[0]``.
    """

    if getattr(_orig_bookkeeping_sync, _WORKER_BOOKKEEPING_PATCH_MARKER, False):
        return _orig_bookkeeping_sync

    @wraps(_orig_bookkeeping_sync)
    def patched_bookkeeping_sync(self, scheduler_output, *args, **kwargs):
        if hasattr(self, "_beam_logprobs_cache") and self._beam_logprobs_cache:
            if hasattr(scheduler_output, "finished_req_ids"):
                for req_id in scheduler_output.finished_req_ids:
                    self._beam_logprobs_cache.pop(req_id, None)

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data or not hasattr(scheduler_output, "_gr_request_row_configs"):
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
            sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
            return res

        request_row_configs = scheduler_output._gr_request_row_configs
        orig_input_batch = object.__getattribute__(self.input_batch, "_obj")

        try:
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
        finally:
            self.input_batch = orig_input_batch

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = res

        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(
            self, orig_input_batch, request_row_configs, beam_data, sampler_output
        )

        return (
            [[] for _ in range(n)] if logprobs_lists is not None else None,
            [[] for _ in range(n)],
            prompt_logprobs_dict,
            collapsed_req_ids,
            collapsed_idx,
            invalid_req_indices,
        )

    setattr(patched_bookkeeping_sync, _WORKER_BOOKKEEPING_PATCH_MARKER, True)
    return patched_bookkeeping_sync


def discover_active_worker_runners() -> tuple[tuple[str, type], ...]:
    """Discover the ModelRunner target for the active platform."""
    from vllm_gr.utils.npu_adapter import is_npu_available

    if is_npu_available():
        try:
            # Import through vllm-ascend's supported worker entry point. Its
            # module initializes vllm_ascend.ops before model_runner_v1, which
            # avoids the device_op -> ops -> fused_moe -> device_op cycle that
            # a direct model_runner_v1 import can trigger.
            from vllm_ascend.worker.worker import NPUModelRunner
        except ImportError as exc:
            logger.warning(
                "Failed to import NPUModelRunner through "
                "vllm_ascend.worker.worker; skipping "
                "vllm-gr NPU beam-search runner patches for this attempt. "
                "The vLLM-Ascend NPU runtime itself is not disabled. "
                "ImportError: %s",
                exc,
            )
            logger.debug("NPUModelRunner import traceback", exc_info=True)
            return ()
        return (("NPUModelRunner", NPUModelRunner),)

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as exc:
        logger.warning(
            "Failed to import vllm.v1.worker.gpu_model_runner.GPUModelRunner; "
            "skipping vllm-gr GPU beam-search runner patches for this attempt. "
            "ImportError: %s",
            exc,
        )
        logger.debug("GPUModelRunner import traceback", exc_info=True)
        return ()
    return (("GPUModelRunner", GPUModelRunner),)


def apply_worker_patches() -> None:
    """Apply Beam hooks to the active-platform ModelRunner target."""
    runners_to_patch = [
        (runner_name, runner_class)
        for runner_name, runner_class in discover_active_worker_runners()
        if not is_worker_runner_patched(runner_class)
    ]

    if not runners_to_patch:
        return

    # Apply patches to each runner
    for runner_name, runner_class in runners_to_patch:
        runner_class.execute_model = make_patched_execute_model(runner_class.execute_model)

        if runner_name == "GPUModelRunner":
            runner_class._prepare_inputs = make_patched_prepare_inputs_gpu(
                runner_class._prepare_inputs
            )
            runner_class._bookkeeping_sync = make_patched_bookkeeping_sync_gpu(
                runner_class._bookkeeping_sync
            )
        elif runner_name == "NPUModelRunner":
            runner_class._prepare_inputs = make_patched_prepare_inputs_npu(
                runner_class._prepare_inputs
            )
            runner_class._bookkeeping_sync = make_patched_bookkeeping_sync_npu(
                runner_class._bookkeeping_sync
            )

        runner_class._patched_for_beam_pos_ids = True
        logger.info("%s hooks patched successfully.", runner_name)
