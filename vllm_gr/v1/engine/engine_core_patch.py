# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

import time
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
_GR_ABORT_PATCH_MARKER = "_vllm_gr_releases_runner_state_on_abort"

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


def make_patched_abort_requests(original_abort):
    """Release Worker-owned GR state when its logical request is aborted."""

    if getattr(original_abort, _GR_ABORT_PATCH_MARKER, False) is True:
        return original_abort

    @wraps(original_abort)
    def patched_abort_requests(self, request_ids):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        requests = getattr(getattr(self, "scheduler", None), "requests", {})
        gr_ids = [
            request_id
            for request_id in ids
            if bool(
                getattr(requests.get(request_id), "gr_persistent_runner_state", False)
            )
        ]
        result = original_abort(self, ids)
        if gr_ids:
            self.model_executor.collective_rpc(
                "release_gr_model_runner_states",
                args=(gr_ids,),
                non_block=True,
            )
        return result

    setattr(patched_abort_requests, _GR_ABORT_PATCH_MARKER, True)
    return patched_abort_requests


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

    from vllm_gr.v1.core.sched.async_scheduler import GRAsyncScheduler

    @wraps(original_step)
    def patched_process_engine_step(self):
        from vllm_gr.v1.engine.core import _beam_self_advance

        outputs, model_executed = self.step_fn()
        persistent_gr = isinstance(self.scheduler, GRAsyncScheduler)
        if outputs and not persistent_gr:
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

        # Match the upstream idle policy for every scheduler. Merely selecting
        # GRAsyncScheduler is not evidence of BatchQueue/Worker progress and
        # must not turn an idle engine into a busy spin.
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
        _handle_submit_session,
        process_input_sockets,
    )

    logger.info("[vllm-gr] apply_engine_core_child_patches called.")

    register_engine_core_request_types()

    # Wrap both constructors. Patching EngineCoreProc explicitly protects
    # against changes in upstream inheritance and plugin loading order.
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)
    EngineCoreProc.__init__ = make_patched_engine_core_init(EngineCoreProc.__init__)
    EngineCore.abort_requests = make_patched_abort_requests(EngineCore.abort_requests)
    EngineCoreProc.abort_requests = make_patched_abort_requests(
        EngineCoreProc.abort_requests
    )

    # Bind request handlers explicitly on both classes.
    EngineCore._cache_beam_request = _cache_beam_request
    EngineCore._handle_beam_request_step_update = _handle_beam_request_step_update
    EngineCore._handle_beam_session_template = _handle_beam_session_template
    EngineCore._handle_submit_session = _handle_submit_session
    EngineCore._apply_beam_step = _apply_beam_step
    EngineCore._beam_self_advance = _beam_self_advance
    EngineCoreProc._cache_beam_request = _cache_beam_request
    EngineCoreProc._handle_beam_request_step_update = _handle_beam_request_step_update
    EngineCoreProc._handle_beam_session_template = _handle_beam_session_template
    EngineCoreProc._handle_submit_session = _handle_submit_session
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


def apply_scheduler_patch():
    """Monkey-patch Scheduler.schedule to propagate custom attributes via SchedulerOutput."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_gr.v1.core.sched.async_scheduler import GRAsyncScheduler

    if getattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, False):
        return

    _original_schedule = Scheduler.schedule

    @wraps(_original_schedule)
    def patched_schedule(self):
        # The persistent-request scheduler owns its KV-domain rules directly;
        # never install the legacy per-step prefix-cache monkey patches on it.
        if isinstance(self, GRAsyncScheduler):
            return _original_schedule(self)
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

    # Also wrap update_from_output so the worker's beam decision
    # (ModelRunnerOutput.beam_search_result) reaches the frontend via
    # EngineCoreOutput.kv_transfer_params (ACS-aligned save flow).
    _patch_scheduler_update_from_output(Scheduler)


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
        result = _original_update(self, scheduler_output, model_runner_output, *args, **kwargs)

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
