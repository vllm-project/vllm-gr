# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_gr.v1.engine.scheduler_metadata import build_beam_request_metadata
from vllm_gr.v1.engine.wire import register_engine_core_request_types

if TYPE_CHECKING:
    from vllm_gr.v1.engine.beam_group import BeamGroupingConfig

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
        from vllm_gr.v1.engine.core import (
            _ensure_beam_batch_ingress,
            _ensure_beam_state,
        )

        _ensure_beam_state(self)
        _ensure_beam_batch_ingress(self)
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
        _handle_beam_batch_abort,
        _handle_beam_batch_submit,
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
    EngineCore._handle_beam_batch_submit = _handle_beam_batch_submit
    EngineCore._handle_beam_batch_abort = _handle_beam_batch_abort
    EngineCoreProc._cache_beam_request = _cache_beam_request
    EngineCoreProc._handle_beam_request_step_update = _handle_beam_request_step_update
    EngineCoreProc._handle_beam_batch_submit = _handle_beam_batch_submit
    EngineCoreProc._handle_beam_batch_abort = _handle_beam_batch_abort

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
    from vllm_gr.v1.worker.model_runner_patch import apply_worker_patches

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


def make_patch_get_computed_blocks(
    original_get_computed_blocks, create_kv_cache_blocks, block_size
):
    """Helper to create patched get_computed_blocks function for beam decode requests."""

    def patch_get_computed_blocks(req):
        bn, tn = original_get_computed_blocks(req)
        if req and getattr(req, "is_beam_decode", False):
            pl = getattr(req, "prefix_len", 0)
            if tn > pl:
                # In incremental decode, the FULL prefix (including the trailing
                # partial block) is treated as computed: the paged cache holds
                # all prefix KV written during prefill, and prefix attention
                # reads it with seqused_k = prefix_len (exact), so FA ignores
                # the remainder-block padding. When the remainder block was
                # padded at prefill time (see patch_cache_blocks), the cache
                # hit reaches the next block boundary (tn = ceil(pl/16)*16);
                # we truncate the computed blocks to only the prefix blocks
                # and return pl (exact) so the scheduler schedules only the
                # new beam tokens instead of re-scheduling the trailing
                # partial-block tokens every step.
                num_blocks = (pl + block_size - 1) // block_size
                sliced_blocks = tuple(group[:num_blocks] for group in bn.blocks)
                return create_kv_cache_blocks(sliced_blocks), pl
        return bn, tn

    return patch_get_computed_blocks


def _drive_beam_phase_machine(scheduler):
    """Drive the beam phase machine for one EngineCore step.

    Returns the ``SchedulerOutput`` produced by the phase machine (an empty
    output when no beam group is waiting and none is active).  The caller
    only invokes this when ``beam_waiting`` or ``_beam_active`` is present,
    so ordinary requests are never mixed in here.

    Two-dispatch model:
    - no active group + waiting non-empty → admit first group → single prefill;
    - active PREFILL → single decode dispatch (Worker runs the full decode loop);
    - active DECODE → empty output while waiting for the Worker's final result;
    - resource release happens in ``_finalize_beam_group`` (EngineCore side),
      triggered when the Worker's final result arrives — not here.
    """
    from vllm_gr.v1.engine.beam_group import BeamGroupingConfig
    from vllm_gr.v1.engine.beam_scheduler import (
        peek_prepared_dispatch_group,
        pop_prepared_dispatch_group,
        refresh_prepared_groups,
        schedule,
    )

    active = getattr(scheduler, "_beam_active", None)
    cfg = getattr(scheduler, "_beam_group_cfg", None) or BeamGroupingConfig()
    cfg = _with_gr_max_decode_steps(scheduler, cfg)

    if active is None:
        refresh_prepared_groups(scheduler, cfg)
        dispatch_unit = peek_prepared_dispatch_group(scheduler)
    else:
        dispatch_unit = None

    output, new_state = schedule(
        scheduler,
        active,
        config=cfg,
        dispatch_unit=dispatch_unit,
    )

    if active is None and new_state is not None and dispatch_unit is not None:
        pop_prepared_dispatch_group(scheduler, dispatch_unit)
        scheduler._beam_active = new_state
        return output

    if active is not None:
        # Prefill → decode transition, or decode → empty output while waiting
        # for the Worker's final result.
        scheduler._beam_active = new_state
    return output


def _with_gr_max_decode_steps(scheduler, cfg: BeamGroupingConfig) -> BeamGroupingConfig:
    """Return ``cfg`` with ``max_decode_steps`` from GR config, when available.

    ``max_decode_steps`` is authoritative from ``GRConfig.beam_max_decode_steps``
    (injected via ``additional_config`` by the config PR).  If the scheduler
    carries no ``vllm_config`` (e.g. unit tests) keep the existing value.
    """
    vllm_config = getattr(scheduler, "vllm_config", None)
    if vllm_config is None:
        return cfg
    try:
        from vllm_gr.config import GRConfig

        gr_cfg = GRConfig.from_vllm_config(vllm_config)
    except Exception:  # noqa: BLE001 - keep the existing value on any parse error
        return cfg
    if gr_cfg.beam_max_decode_steps != cfg.max_decode_steps:
        from dataclasses import replace

        return replace(cfg, max_decode_steps=gr_cfg.beam_max_decode_steps)
    return cfg


def apply_scheduler_patch():
    """Monkey-patch Scheduler.schedule to propagate custom attributes via SchedulerOutput."""
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, False):
        return

    _original_schedule = Scheduler.schedule

    @wraps(_original_schedule)
    def patched_schedule(self):
        # Install the beam prefix-cache patches first so both the native path
        # and the two-dispatch beam phase machine use them.
        _original_get_computed_blocks = self.kv_cache_manager.get_computed_blocks
        _original_cache_blocks = self.kv_cache_manager.coordinator.cache_blocks
        block_size = self.cache_config.block_size

        patch_get_computed_blocks = make_patch_get_computed_blocks(
            _original_get_computed_blocks,
            self.kv_cache_manager.create_kv_cache_blocks,
            block_size,
        )

        def patch_cache_blocks(request, num_computed_tokens):
            # Extend the num_computed_tokens passed to cache_blocks to the next
            # block boundary on the FINAL prefill chunk, so the trailing partial
            # block (whose hash exists thanks to the padding in _cache_beam_request)
            # is actually cached. Without this, vLLM would only cache full blocks
            # and the beam-decode request could not prefix-cache-hit the remainder.
            if (
                request
                and not getattr(request, "is_beam_decode", False)
                and hasattr(request, "block_hashes")
                and request.block_hashes
            ):
                num_prompt_tokens = getattr(request, "num_prompt_tokens", request.num_tokens)
                if num_computed_tokens >= num_prompt_tokens:
                    chunk_padded_target = (
                        (num_computed_tokens + block_size - 1) // block_size * block_size
                    )
                    num_hashes = len(request.block_hashes)
                    padded_target = min(chunk_padded_target, num_hashes * block_size)
                    if (
                        num_computed_tokens < padded_target
                        and num_computed_tokens % block_size != 0
                    ):
                        num_computed_tokens = padded_target
            return _original_cache_blocks(request, num_computed_tokens)

        self.kv_cache_manager.get_computed_blocks = patch_get_computed_blocks
        self.kv_cache_manager.coordinator.cache_blocks = patch_cache_blocks
        try:
            # Two-dispatch phase machine: when a beam group is waiting or active,
            # drive it through admission → single prefill → single decode → wait.
            # Ordinary requests (no beam groups) keep the native path untouched.
            beam_waiting: list = getattr(self, "beam_waiting", None)
            beam_active = getattr(self, "_beam_active", None)
            if beam_waiting or beam_active is not None:
                return _drive_beam_phase_machine(self)

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

            # Propagate the authoritative beam-session teardown signal.
            #
            # _handle_beam_request_step_update writes session_ids to
            # self._beam_finished_session_ids when it receives a
            # BeamRequestStepUpdate with beam_width=0 — the beam search
            # loop's explicit "generation finished" message.  This is
            # distinct from finished_req_ids, which fires on every decode
            # step (each step's Request reaches num_computed_tokens >=
            # num_tokens and is freed).  Using finished_req_ids would
            # destroy and recreate the session every step, losing the
            # incremental KV pool.
            beam_finished_session_ids = list(getattr(self, "_beam_finished_session_ids", []))
            self._beam_finished_session_ids = []  # consume after reading

            output.beam_data = beam_data
            output.beam_finished_session_ids = beam_finished_session_ids
            return output
        finally:
            self.kv_cache_manager.get_computed_blocks = _original_get_computed_blocks
            self.kv_cache_manager.coordinator.cache_blocks = _original_cache_blocks

    Scheduler.schedule = patched_schedule
    setattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, True)
    Scheduler._patched_for_beam_requests = True
    logger.debug("Scheduler.schedule patched for beam-requests propagation.")
