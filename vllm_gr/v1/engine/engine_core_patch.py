# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

import time
from collections import OrderedDict
from functools import wraps

from vllm.logger import init_logger

from vllm_gr.v1.engine.scheduler_metadata import build_beam_request_metadata
from vllm_gr.v1.engine.wire import register_engine_core_request_types

logger = init_logger(__name__)

_BEAM_STATE_INIT_MARKER = "_vllm_gr_initializes_beam_state"
_SCHEDULER_PATCH_MARKER = "_vllm_gr_beam_scheduler_patch"
_ENGINE_STEP_PATCH_MARKER = "_vllm_gr_beam_self_advance_step"
_CLIENT_REQUEST_PATCH_MARKER = "_vllm_gr_gr_input_dispatch"
_PIS_PATCH_MARKER = "_vllm_gr_process_input_sockets"
_GR_FINALIZING_IDS_ATTR = "_gr_v1_finalizing_request_ids"
_GR_NATIVE_UPDATE_IDS_ATTR = "_gr_v1_native_update_request_ids"
_GR_RESET_PREFIX_CACHE_PATCH_MARKER = "_vllm_gr_v1_reset_prefix_cache"

# Upstream EngineCoreProc.process_input_sockets captured before the adapter
# is installed.  Parity tests inspect this object against the pinned vLLM.
UPSTREAM_PROCESS_INPUT_SOCKETS = None

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
# EngineCoreProc._process_engine_step wrapper (engine-driven chain hook)
# ---------------------------------------------------------------------------


def make_patched_process_engine_step(original_step):
    """Wrap ``_process_engine_step`` to drive EngineCore engine-driven chaining.

    The wrapper runs between ``step_fn()`` producing the step's
    ``EngineCoreOutputs`` and those outputs being forwarded to the output
    queue: the scheduler patch has already injected the worker's beam
    decision into ``kv_transfer_params['beam_search_result']``, and
    ``_beam_self_advance`` turns a non-terminal decision into the next
    ``BeamRequestStepUpdate`` applied in-process (no frontend RPC).  The
    chained step's ADD lands in ``input_queue`` before this iteration ends,
    so the busy loop continues straight into the next decode step.

    ``_beam_self_advance`` itself is a no-op when ``beam_engine_driven``
    is off, so the wrapper is safe for every engine.
    """

    if getattr(original_step, _ENGINE_STEP_PATCH_MARKER, False):
        return original_step

    @wraps(original_step)
    def patched_process_engine_step(self):
        from vllm_gr.v1.engine.core import _beam_self_advance

        outputs, model_executed = self.step_fn()
        if outputs:
            try:
                _beam_self_advance(self, outputs)
            except Exception:
                logger.exception(
                    "vllm-gr: beam self-advance failed; falling back to the "
                    "frontend step-update loop for this step"
                )
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        self.post_step(model_executed)

        if not model_executed and self.scheduler.has_requests():
            time.sleep(0.001)

        return model_executed

    setattr(patched_process_engine_step, _ENGINE_STEP_PATCH_MARKER, True)
    return patched_process_engine_step


def make_patched_handle_client_request(original):
    """Dispatch GR wire types that reach the EngineCore busy loop.

    Upstream logs unrecognized types.  The socket adapter normally consumes
    GR messages on the input thread; this wrap is the in-process / fallback
    path and a compile-time-shaped extension point for those types.
    """
    if getattr(original, _CLIENT_REQUEST_PATCH_MARKER, False):
        return original

    @wraps(original)
    def patched_handle_client_request(self, request_type, request, *args, **kwargs):
        from vllm_gr.v1.engine.core import dispatch_gr_engine_core_input

        if dispatch_gr_engine_core_input(self, request_type, request):
            return None
        return original(self, request_type, request, *args, **kwargs)

    setattr(patched_handle_client_request, _CLIENT_REQUEST_PATCH_MARKER, True)
    return patched_handle_client_request


# ---------------------------------------------------------------------------
# Child-process patch applicator — called from patched run_engine_core
# ---------------------------------------------------------------------------


def apply_engine_core_request_patches() -> None:
    """Install EngineCore request dispatch and beam-state support."""
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, EngineCoreProc

    from vllm_gr.v1.engine.core import (
        _apply_beam_step,
        _beam_self_advance,
        _cache_beam_request,
        _handle_beam_request_step_update,
        _handle_beam_session_template,
        _handle_submit_gr_session,
        _handle_submit_session,
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
    EngineCore._handle_beam_session_template = _handle_beam_session_template
    EngineCore._handle_submit_session = _handle_submit_session
    EngineCore._handle_submit_gr_session = _handle_submit_gr_session
    EngineCore._apply_beam_step = _apply_beam_step
    EngineCore._beam_self_advance = _beam_self_advance
    EngineCoreProc._cache_beam_request = _cache_beam_request
    EngineCoreProc._handle_beam_request_step_update = _handle_beam_request_step_update
    EngineCoreProc._handle_beam_session_template = _handle_beam_session_template
    EngineCoreProc._handle_submit_session = _handle_submit_session
    EngineCoreProc._handle_submit_gr_session = _handle_submit_gr_session
    EngineCoreProc._apply_beam_step = _apply_beam_step
    EngineCoreProc._beam_self_advance = _beam_self_advance

    # Narrow GR dispatch on the busy-loop request handler so in-process
    # EngineCore and any GR type that misses the socket adapter still land.
    EngineCoreProc._handle_client_request = make_patched_handle_client_request(
        EngineCoreProc._handle_client_request
    )
    DPEngineCoreProc._handle_client_request = make_patched_handle_client_request(
        DPEngineCoreProc._handle_client_request
    )

    # vLLM-version adapter: capture the upstream loop once, then install ours.
    global UPSTREAM_PROCESS_INPUT_SOCKETS
    current_pis = EngineCoreProc.process_input_sockets
    if UPSTREAM_PROCESS_INPUT_SOCKETS is None and current_pis is not process_input_sockets:
        UPSTREAM_PROCESS_INPUT_SOCKETS = current_pis
    EngineCoreProc.process_input_sockets = process_input_sockets
    setattr(EngineCoreProc.process_input_sockets, _PIS_PATCH_MARKER, True)

    # Self-advance hook: _process_engine_step feeds each step's outputs to
    # _beam_self_advance BEFORE they reach the output queue, so the chained
    # BeamRequestStepUpdate (built from the worker decision) can enqueue the
    # next beam-decode ADD in the same busy-loop iteration.
    EngineCoreProc._process_engine_step = make_patched_process_engine_step(
        EngineCoreProc._process_engine_step
    )

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
    apply_kv_hash_memo_patch()
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


def _sync_gr_sessions_after_native_finish(scheduler, finished_requests, finished_status) -> None:
    """Mirror native finish results into GR state using consumed request IDs."""
    from vllm_gr.v1.engine.beam_search_v1 import GRSessionStatus

    status_name = getattr(finished_status, "name", "")
    sessions = getattr(scheduler, "_gr_v1_sessions", {})
    gr_finalizing_ids = getattr(scheduler, _GR_FINALIZING_IDS_ATTR, ())
    gr_native_update_ids = getattr(scheduler, _GR_NATIVE_UPDATE_IDS_ATTR, ())
    for req_id, _client_index in finished_requests:
        session = sessions.get(req_id)
        if session is None:
            continue
        # Native EOS/stop detection may finish a Request while processing a GR
        # dispatch. The update wrapper reconciles that transition after GR has
        # consumed the same output; it is not an external abort/error.
        if req_id in gr_native_update_ids:
            continue
        gr_terminal_delivery = (
            req_id in gr_finalizing_ids
            and session.status == GRSessionStatus.FINISHING
            and session.terminal_result_pending
        )
        if gr_terminal_delivery:
            continue
        if status_name == "FINISHED_ABORTED":
            session.status = GRSessionStatus.ABORTED
        elif status_name == "FINISHED_ERROR":
            session.status = GRSessionStatus.FAILED
        else:
            session.status = GRSessionStatus.FINISHING
        session.terminal_result_pending = False

        # External finish/abort paths need no GR terminal result. Keep state
        # only while already-dispatched device work still has to drain.
        if not session._in_flight_dispatches:
            sessions.pop(req_id, None)


def _patch_scheduler_reset_prefix_cache(scheduler_cls) -> None:
    original_reset = scheduler_cls.reset_prefix_cache
    if getattr(original_reset, _GR_RESET_PREFIX_CACHE_PATCH_MARKER, False):
        return

    @wraps(original_reset)
    def patched_reset_prefix_cache(self, *args, **kwargs):
        reset_running_requests = kwargs.get("reset_running_requests", args[0] if args else False)
        if reset_running_requests and getattr(self, "_gr_v1_sessions", {}):
            logger.warning(
                "Rejecting reset_prefix_cache(reset_running_requests=True) "
                "while GR sessions are active"
            )
            return False
        return original_reset(self, *args, **kwargs)

    setattr(patched_reset_prefix_cache, _GR_RESET_PREFIX_CACHE_PATCH_MARKER, True)
    scheduler_cls.reset_prefix_cache = patched_reset_prefix_cache


def apply_scheduler_patch():
    """Monkey-patch Scheduler.schedule to propagate custom attributes via SchedulerOutput."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_gr.v1.engine.gr_async_scheduler import (
        GR_STAGE_OUTPUT_ATTR,
        install_gr_scheduler_methods,
    )

    install_gr_scheduler_methods(Scheduler)

    if not getattr(Scheduler.add_request, "_vllm_gr_v1_add_request", False):
        original_add_request = Scheduler.add_request

        @wraps(original_add_request)
        def patched_add_request(self, request):
            batch_request = getattr(request, "gr_batch_request", None)
            if batch_request is not None:
                self.add_gr_session(batch_request)
            return original_add_request(self, request)

        setattr(patched_add_request, "_vllm_gr_v1_add_request", True)
        Scheduler.add_request = patched_add_request

    if not getattr(Scheduler.finish_requests, "_vllm_gr_v1_finish_requests", False):
        original_finish_requests = Scheduler.finish_requests

        @wraps(original_finish_requests)
        def patched_finish_requests(self, request_ids, finished_status):
            result = original_finish_requests(self, request_ids, finished_status)
            _sync_gr_sessions_after_native_finish(self, result, finished_status)
            return result

        setattr(patched_finish_requests, "_vllm_gr_v1_finish_requests", True)
        Scheduler.finish_requests = patched_finish_requests

    _patch_scheduler_reset_prefix_cache(Scheduler)

    if getattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, False):
        return

    _original_schedule = Scheduler.schedule

    @wraps(_original_schedule)
    def patched_schedule(self):
        # The two cache-manager wrappers depend only on constants resolved at
        # patch-install time (original bound methods + block_size), so they
        # are built ONCE and stashed on the scheduler instance; rebuilding
        # the closures on every schedule call showed up in prefill-path
        # profiling (~60us/request of pure closure + attribute churn).
        cached_patches = getattr(self, "_beam_schedule_patches", None)
        if (
            cached_patches is None
            # Guard against a swapped/undone cache manager (e.g. engine
            # reset re-installs fresh managers): re-capture the originals.
            or cached_patches[0] is not self.kv_cache_manager.get_computed_blocks
            or cached_patches[1] is not self.kv_cache_manager.coordinator.cache_blocks
        ):
            _original_get_computed_blocks = self.kv_cache_manager.get_computed_blocks
            _original_cache_blocks = self.kv_cache_manager.coordinator.cache_blocks
            block_size = self.cache_config.block_size

            patch_get_computed_blocks = make_patch_get_computed_blocks(
                _original_get_computed_blocks,
                self.kv_cache_manager.create_kv_cache_blocks,
                block_size,
            )

            def patch_cache_blocks(request, num_computed_tokens):
                if getattr(request, "gr_batch_request", None) is not None:
                    # Only complete prompt blocks are cacheable. Placeholder
                    # hashes must never advertise generated Beam KV.
                    return _original_cache_blocks(
                        request, min(num_computed_tokens, request.num_prompt_tokens)
                    )
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

            cached_patches = (
                _original_get_computed_blocks,
                _original_cache_blocks,
                patch_get_computed_blocks,
                patch_cache_blocks,
            )
            self._beam_schedule_patches = cached_patches

        (
            _original_get_computed_blocks,
            _original_cache_blocks,
            patch_get_computed_blocks,
            patch_cache_blocks,
        ) = cached_patches
        self.kv_cache_manager.get_computed_blocks = patch_get_computed_blocks
        self.kv_cache_manager.coordinator.cache_blocks = patch_cache_blocks
        blocked_gr_max_tokens = {}
        try:
            # Keep at most two consecutive stages of one session outstanding.
            # Temporarily making a saturated request length-capped lets native
            # schedule() skip it while still scheduling other ready requests.
            for req_id, session in getattr(self, "_gr_v1_sessions", {}).items():
                request = self.requests.get(req_id)
                if request is None or self.can_issue_gr_stage(session):
                    continue
                blocked_gr_max_tokens[req_id] = request.max_tokens
                request.max_tokens = 0
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
            gr_stages = {}
            for req_id in output.num_scheduled_tokens:
                session = getattr(self, "_gr_v1_sessions", {}).get(req_id)
                if session is None:
                    continue
                metadata = self.attach_gr_stage_metadata(output, session)
                if metadata is not None:
                    gr_stages[req_id] = metadata
            setattr(output, GR_STAGE_OUTPUT_ATTR, gr_stages)
            return output
        finally:
            for req_id, max_tokens in blocked_gr_max_tokens.items():
                request = self.requests.get(req_id)
                if request is not None:
                    request.max_tokens = max_tokens
            self.kv_cache_manager.get_computed_blocks = _original_get_computed_blocks
            self.kv_cache_manager.coordinator.cache_blocks = _original_cache_blocks

    Scheduler.schedule = patched_schedule
    setattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, True)
    Scheduler._patched_for_beam_requests = True
    logger.debug("Scheduler.schedule patched for beam-requests propagation.")

    # Also wrap update_from_output so the worker's beam decision
    # (ModelRunnerOutput.beam_search_result) reaches the frontend via
    # EngineCoreOutput.kv_transfer_params (ACS-aligned save flow).
    _patch_scheduler_update_from_output(Scheduler)


def _find_gr_output(result, request_id):
    for engine_outputs in result.values():
        for engine_output in engine_outputs.outputs:
            if engine_output.request_id == request_id:
                return engine_outputs, engine_output
    return None, None


def _native_request_terminated(scheduler, result, request_id) -> bool:
    _, target_output = _find_gr_output(result, request_id)
    if target_output is not None and target_output.finish_reason is not None:
        return True
    for engine_outputs in result.values():
        if request_id in (engine_outputs.finished_requests or ()):
            return True
    return request_id not in scheduler.requests


def _reconcile_native_gr_termination(scheduler, result, session) -> None:
    from vllm_gr.v1.engine.beam_search_v1 import GRSessionStatus
    from vllm_gr.v1.engine.gr_async_scheduler import _make_failure_result

    if session.status not in (GRSessionStatus.WAITING, GRSessionStatus.RUNNING):
        return
    if not _native_request_terminated(scheduler, result, session.session_id):
        return

    session.status = GRSessionStatus.FINISHING
    session.terminal_result_pending = True
    if session._pending_terminal_result is None:
        session._pending_terminal_result = _make_failure_result(
            session,
            "native_termination_without_gr_result",
            "Native request terminated before GR produced a terminal result",
        )


def _suppress_native_terminal(result, request_id, target_output) -> None:
    if target_output is not None:
        target_output.finish_reason = None
    for engine_outputs in result.values():
        if engine_outputs.finished_requests is not None:
            engine_outputs.finished_requests.discard(request_id)


def _finish_native_gr_request(scheduler, request, request_id, request_status) -> None:
    if request is None or request.is_finished():
        return

    # Every physical dispatch has drained. Any remaining placeholders are
    # speculative reservations and no longer represent real outputs.
    request.num_output_placeholders = 0
    finalizing_ids = getattr(scheduler, _GR_FINALIZING_IDS_ATTR, None)
    if finalizing_ids is None:
        finalizing_ids = set()
        setattr(scheduler, _GR_FINALIZING_IDS_ATTR, finalizing_ids)
    finalizing_ids.add(request_id)
    try:
        scheduler.finish_requests(request_id, request_status)
    finally:
        finalizing_ids.discard(request_id)
        if not finalizing_ids:
            delattr(scheduler, _GR_FINALIZING_IDS_ATTR)


def _discard_pending_native_finish(scheduler, client_index, request_id) -> None:
    finished_by_client = getattr(scheduler, "finished_req_ids_dict", None)
    if finished_by_client is None:
        return
    pending_finished = finished_by_client.get(client_index)
    if pending_finished is None:
        return
    pending_finished.discard(request_id)
    if not pending_finished:
        finished_by_client.pop(client_index, None)


def _finalize_ready_gr_session(scheduler, result, session) -> None:
    """Publish one terminal native output after speculative work drains."""
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
    from vllm.v1.request import RequestStatus

    from vllm_gr.v1.engine.beam_search_v1 import GRSessionStatus
    from vllm_gr.v1.engine.gr_async_scheduler import (
        _close_gr_session,
        _gr_session_ready_to_finalize,
        _gr_terminal_result,
    )

    request_id = session.session_id
    target_outputs, target_output = _find_gr_output(result, request_id)

    if not _gr_session_ready_to_finalize(session):
        # Native stop detection can race ahead of a queued GR successor. Keep
        # the request open at the frontend until the GR session has drained;
        # the finalizer will synthesize the one authoritative terminal frame.
        if session.status == GRSessionStatus.FINISHING and session.terminal_result_pending:
            _suppress_native_terminal(result, request_id, target_output)
        return

    terminal_result = _gr_terminal_result(session)
    native_request = scheduler.requests.get(request_id)
    client_index = (
        native_request.client_index
        if native_request is not None
        else session.request.items[0].request.client_index
    )

    request_status = (
        RequestStatus.FINISHED_ERROR
        if terminal_result.failure is not None
        else RequestStatus.FINISHED_STOPPED
    )
    finish_reason = FinishReason.ERROR if terminal_result.failure is not None else FinishReason.STOP

    _finish_native_gr_request(scheduler, native_request, request_id, request_status)

    if target_outputs is None:
        target_outputs = result.get(client_index)
        if target_outputs is None:
            target_outputs = EngineCoreOutputs(outputs=[])
            result[client_index] = target_outputs
    if target_output is None:
        target_output = EngineCoreOutput(
            request_id=request_id,
            new_token_ids=[],
            finish_reason=finish_reason,
        )
        target_outputs.outputs.append(target_output)
    else:
        # The GR result, rather than the native placeholder token, owns the
        # externally visible terminal status.
        target_output.finish_reason = finish_reason

    kv_params = dict(target_output.kv_transfer_params or {})
    kv_params["gr_batch_result"] = terminal_result
    target_output.kv_transfer_params = kv_params
    finished_requests = set(target_outputs.finished_requests or ())
    finished_requests.add(request_id)
    target_outputs.finished_requests = finished_requests

    # finish_requests() ran after upstream assembled this step's outputs. Avoid
    # publishing the same finished ID again on the next update.
    _discard_pending_native_finish(scheduler, client_index, request_id)

    _close_gr_session(scheduler, session)


def _patch_scheduler_update_from_output(scheduler_cls) -> None:
    """Carry ``ModelRunnerOutput.beam_search_result`` through kv_transfer_params.

    The NPU worker attaches the beam decision dict to the ModelRunnerOutput
    it returns (see ``_attach_beam_search_result`` in model_runner_common).
    This wrapper reads it and injects it into the first EngineCoreOutput's
    ``kv_transfer_params`` so the frontend can consume it from the streamed
    RequestOutput without an extra round trip.  Pure beam-decode batches carry
    exactly one logical session request by design (max_batch=1), so attaching
    to the first output of a beam session is unambiguous.
    """
    marker = "_vllm_gr_update_from_output_patch"
    if getattr(scheduler_cls.update_from_output, marker, False):
        return

    _original_update = scheduler_cls.update_from_output

    @wraps(_original_update)
    def patched_update_from_output(self, scheduler_output, model_runner_output, *args, **kwargs):
        from vllm_gr.v1.engine.gr_async_scheduler import GR_STAGE_OUTPUT_ATTR

        gr_stages = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, {})
        sessions = getattr(self, "_gr_v1_sessions", {})
        dispatched_sessions = {
            session_id: sessions[session_id] for session_id in gr_stages if session_id in sessions
        }
        native_update_ids = getattr(self, _GR_NATIVE_UPDATE_IDS_ATTR, None)
        if native_update_ids is None:
            native_update_ids = set()
            setattr(self, _GR_NATIVE_UPDATE_IDS_ATTR, native_update_ids)
        added_update_ids = set(dispatched_sessions).difference(native_update_ids)
        native_update_ids.update(added_update_ids)
        try:
            result = _original_update(self, scheduler_output, model_runner_output, *args, **kwargs)
        finally:
            native_update_ids.difference_update(added_update_ids)
            if not native_update_ids:
                delattr(self, _GR_NATIVE_UPDATE_IDS_ATTR)

        updated_sessions = []
        for session in dispatched_sessions.values():
            self.update_gr_from_output(session, scheduler_output, model_runner_output)
            _reconcile_native_gr_termination(self, result, session)
            updated_sessions.append(session)

        # A terminal result may arrive before a look-ahead successor. Buffer it
        # in GRSessionState and publish only after every physical dispatch has
        # drained, together with a genuinely finished native output.
        for session in updated_sessions:
            _finalize_ready_gr_session(self, result, session)

        beam_search_result = getattr(model_runner_output, "beam_search_result", None)
        if beam_search_result is None:
            return result

        # Attach to exactly one EngineCoreOutput.  Prefer the output whose
        # request_id owns the decision (worker decisions carry session_id —
        # review P5: a mixed batch of session A's decode and session B's
        # constrained prefill must never cross-attach); fall back to the first
        # beam-session output for legacy payloads without the marker.  The
        # dict payload carries host tensors; the IPC envelope only serialises
        # the kv_transfer_params mapping, so host tensors are safe here (they
        # were D2H'd by the worker).
        beam_data = getattr(scheduler_output, "beam_data", {})
        beam_ids = {
            req_id
            for req_id, mdata in beam_data.items()
            if mdata.get("is_beam_decode", False) or mdata.get("is_constraint_prefill", False)
        }
        decision_owner = (
            beam_search_result.get("session_id") if isinstance(beam_search_result, dict) else None
        )

        def _attach(target_out) -> bool:
            kv_params = dict(target_out.kv_transfer_params or {})
            kv_params["beam_search_result"] = beam_search_result
            target_out.kv_transfer_params = kv_params
            return True

        attached = False
        fallback = None
        for ec_outputs in result.values():
            for out in getattr(ec_outputs, "outputs", []):
                if out.request_id not in beam_ids:
                    continue
                if decision_owner is not None and out.request_id == decision_owner:
                    attached = _attach(out)
                    break
                if fallback is None:
                    fallback = out
            if attached:
                break
        if not attached and fallback is not None:
            _attach(fallback)
        if not attached:
            logger.debug(
                "vllm-gr: beam_search_result present but no matching EngineCoreOutput "
                "in this step; the decision was dropped (num_beam_ids=%d)",
                len(beam_ids),
            )
        return result

    setattr(patched_update_from_output, marker, True)
    scheduler_cls.update_from_output = patched_update_from_output


_KV_HASH_MEMO_LIMIT = 65536

# True while the whole-chain memo's fallback hashes a from-scratch request
# that MISSED the chain memo.  Such traffic is (statistically) never
# repeated: inserting its ~64 blocks per request into the per-block memo
# costs key builds, big-dict probes, and FIFO evictions that evict hot
# entries, with no hits in return — the "varied prompts after memo
# saturation" regression.  The per-block wrapper checks this flag first and
# stays transparent, so chain-miss requests hash at stock vLLM speed while
# the per-block memo keeps serving the paths where repeats are real:
# incremental (block_hashes-seeded) beam requests and the mm / lora /
# salt / prompt-embeds fallbacks.  The two memos' recency windows coincide
# (65,536 blocks ~= 1,024 x 64-block prompts = the chain-memo limit), so a
# from-scratch prompt the chain memo cannot serve is beyond the per-block
# memo's reach anyway.  EngineCore is single-threaded; even if a concurrent
# caller ever observed the flag, bypassing the memo is semantically
# transparent (it only memoizes a pure function).
_CHAIN_FALLBACK_ACTIVE = False


def apply_kv_hash_memo_patch() -> None:
    """Memoize vLLM's per-block prefix-cache hashing (EngineCore process).

    ``vllm.v1.core.kv_cache_utils.hash_block_tokens`` is a pure function of
    (hash_function, parent_block_hash, block token ids, extra keys): the same
    inputs always produce the same ``BlockHash``.  Repeated blocks
    (re-sent beam suffixes, popular prefixes) re-hash identical blocks on
    every request — ~1.3ms of sha256 work per 1024-token prompt — so a
    bounded FIFO memo turns those hits into dict lookups.  First-seen blocks
    pay one extra tuple() key build (microseconds).  The wrapper is
    idempotent and installs on the module attribute
    (``request_block_hasher`` resolves it as a module global at call time,
    so every existing and future hasher picks it up).

    From-scratch requests that miss the whole-chain memo set
    ``_CHAIN_FALLBACK_ACTIVE`` and bypass this wrapper entirely (see there);
    the per-block memo serves incremental beam requests and the mm / lora /
    salt / prompt-embeds fallbacks, where block-level repeats are real.
    """
    global _CHAIN_FALLBACK_ACTIVE
    from vllm.v1.core import kv_cache_utils

    if getattr(kv_cache_utils.hash_block_tokens, "_gr_hash_memo", False):
        return

    _CHAIN_FALLBACK_ACTIVE = False
    original = kv_cache_utils.hash_block_tokens
    # OrderedDict (not plain dict): after the memo saturates, every miss
    # evicts the oldest entry.  ``dict.pop(next(iter(...)))`` is NOT
    # constant-time — the fresh iterator rescans the leading run of deleted
    # (dummy) slots and CPython only compacts the table on resize, so varied
    # traffic pays an ever-longer scan per eviction (measured 14-32x slowdown
    # on the saturated hashing path).  ``OrderedDict.popitem(last=False)``
    # unlinks the head of its doubly-linked list in O(1).
    memo: OrderedDict[tuple, object] = OrderedDict()

    def hash_block_tokens_memo(
        hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None
    ):
        if _CHAIN_FALLBACK_ACTIVE:
            # Chain-memo miss in flight: hash at stock speed (see
            # _CHAIN_FALLBACK_ACTIVE).
            return original(hash_function, parent_block_hash, curr_block_token_ids, extra_keys)
        key = (
            hash_function,
            parent_block_hash,
            tuple(curr_block_token_ids),
            extra_keys,
        )
        hit = memo.get(key)
        if hit is not None:
            return hit
        hit = original(hash_function, parent_block_hash, curr_block_token_ids, extra_keys)
        if len(memo) >= _KV_HASH_MEMO_LIMIT:
            # Constant-time FIFO eviction keeps the memo bounded in
            # long-running servers under varied (never-repeating) traffic.
            memo.popitem(last=False)
        memo[key] = hit
        return hit

    hash_block_tokens_memo._gr_hash_memo = True  # type: ignore[attr-defined]
    # Exposed for tests / diagnostics: the live memo store behind this wrapper.
    hash_block_tokens_memo._gr_hash_memo_entries = memo  # type: ignore[attr-defined]
    kv_cache_utils.hash_block_tokens = hash_block_tokens_memo
    _apply_block_hash_chain_memo_patch(kv_cache_utils)
    logger.info("[vllm-gr] kv block-hash memo patch applied.")


_CHAIN_MEMO_LIMIT = 1024


def _apply_block_hash_chain_memo_patch(kv_cache_utils) -> None:
    """Whole-chain memo on top of the per-block memo (from-scratch requests).

    The per-block memo removes the sha256 work but a repeated prompt still
    walks the ``request_block_hasher`` loop (64 iterations of slicing,
    ``generate_block_hash_extra_keys``, tuple-key building — ~0.8ms for a
    1024-token prompt).  For a request hashing from scratch with no mm /
    lora / salt variation, the full hash chain is a pure function of
    (hash_function, block_size, salt, the full-block token prefix), so a
    hit can replay the whole list in one dict lookup.

    A MISS hashes at stock vLLM speed: ``_CHAIN_FALLBACK_ACTIVE`` is set
    around the fallback so the per-block memo stays transparent and a
    never-repeating prompt pays zero memo bookkeeping (no key builds, no
    probes, no evictions of hot entries).  The chain entry is still
    recorded, so a repeat of this exact prompt hits next time.

    Fast-path eligibility is conservative: any incremental call
    (``block_hashes`` non-empty), mm_features, lora_request, a non-``None``
    cache_salt, or a non-``None`` prompt_embeds falls back to the original
    hasher.  prompt_embeds requests MUST bypass the chain memo: their
    ``all_token_ids`` are ``[0] * num_prompt_tokens`` placeholders (vLLM
    distinguishes their contents via per-block embeds hashes), so the token
    key alone would collide across distinct embeddings of equal length.
    """
    global _CHAIN_FALLBACK_ACTIVE
    if getattr(kv_cache_utils.get_request_block_hasher, "_gr_chain_memo", False):
        return

    original_factory = kv_cache_utils.get_request_block_hasher
    # Same constant-time FIFO rationale as the per-block memo above.
    chain_memo: OrderedDict[tuple, list] = OrderedDict()

    def get_request_block_hasher_memo(block_size, caching_hash_fn):
        original_hasher = original_factory(block_size, caching_hash_fn)

        def request_block_hasher_chain_memo(request):
            global _CHAIN_FALLBACK_ACTIVE
            if (
                request.block_hashes
                or request.mm_features
                or request.lora_request is not None
                or request.cache_salt is not None
                or getattr(request, "prompt_embeds", None) is not None
            ):
                return original_hasher(request)

            num_tokens = request.num_tokens
            num_full_blocks = num_tokens // block_size
            if num_full_blocks == 0:
                return original_hasher(request)

            # Only the full-block prefix participates in hashing; trailing
            # partial-block tokens never reach hash_block_tokens.
            tokens = tuple(request.all_token_ids[: num_full_blocks * block_size])
            key = (caching_hash_fn, block_size, tokens)
            hit = chain_memo.get(key)
            if hit is not None:
                return list(hit)

            _CHAIN_FALLBACK_ACTIVE = True
            try:
                # Stock-speed fallback (see _CHAIN_FALLBACK_ACTIVE): a miss
                # means this prompt is new to the chain memo, and the
                # per-block memo's recency window cannot reach it either.
                result = original_hasher(request)
            finally:
                _CHAIN_FALLBACK_ACTIVE = False
            if len(chain_memo) >= _CHAIN_MEMO_LIMIT:
                chain_memo.popitem(last=False)
            chain_memo[key] = list(result)
            return result

        return request_block_hasher_chain_memo

    get_request_block_hasher_memo._gr_chain_memo = True  # type: ignore[attr-defined]
    # Exposed for tests / diagnostics: the live chain-memo store.
    get_request_block_hasher_memo._gr_chain_memo_entries = chain_memo  # type: ignore[attr-defined]
    kv_cache_utils.get_request_block_hasher = get_request_block_hasher_memo
    # ``vllm.v1.engine.core`` binds this factory via ``from ... import`` at
    # module load, so it holds its OWN reference — patching only the
    # kv_cache_utils attribute never reaches ``core.get_request_block_hasher``
    # and Requests keep using the original per-block loop.  Install the same
    # wrapper object on both module attributes so a single shared chain memo
    # serves every path.  Import the submodule explicitly: relying on the
    # ``vllm.v1.engine.core`` attribute of the parent package would silently
    # no-op whenever nothing else has bound it yet.
    try:
        import vllm.v1.engine.core as _vllm_engine_core

        _vllm_engine_core.get_request_block_hasher = get_request_block_hasher_memo
    except ImportError:
        # Fallback: the core module may not be importable in exotic
        # processes; the kv_cache_utils binding alone still helps any
        # caller that resolves it as a module global.
        pass
