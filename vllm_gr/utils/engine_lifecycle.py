# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sync engine shutdown helpers for the vLLM LLM passthrough path."""

from __future__ import annotations

import functools
import gc
import inspect
import threading
import time
from typing import Any

from vllm_gr.logger import init_logger

logger = init_logger(__name__)

# GC threshold applied to ALL generations (0/1/2) while serving, matching the
# worker-side freeze (model_runner_common._freeze_serving_gc).  Tracer
# evidence: every collection during serving measured ``collected=0`` — the
# old-gen scans were 17-36 ms pauses, and gen-0 scans (the in-flight
# request's beam bookkeeping) 3.7-8 ms pauses landing inside the request
# window on a fixed cadence.  Beam temporaries die by refcount and never
# form cycles, so raising all three thresholds removes pure pauses without
# losing reclamation.
_SERVING_GC_THRESHOLD = 50_000

# Bounded wait for a quiescent serving boundary (no beam request in flight)
# before the one-shot frontend collection runs.  A frontend that stays busy
# past this deadline runs the collection anyway (once), degrading to the
# pre-barrier placement rather than never freezing the heap.
_QUIESCENT_WAIT_TIMEOUT_S = 30.0

# Process-level guard for the one-time frontend freeze.  Under the default
# multi-process engine the frontend (GRLLM / AsyncLLM driver) is a *separate*
# process from the EngineCore/worker, so the worker's freeze never touches its
# heap.  A module-level flag dedupes across calls; in-proc mode it is a harmless
# no-op duplicate of the worker-side freeze (gc.freeze is idempotent).
_frontend_gc_freeze_done = False
_frontend_gc_freeze_lock = threading.Lock()

# Beam requests currently in flight on this frontend process.  gc.collect()
# holds the GIL for the whole collection, so the one-shot freeze must not
# START while a request is in flight: it would stall that request's response
# path (a real-GC CPU probe observed the collection beginning while the
# scheduling thread was still inside the Thread.start() call).  The freeze
# thread therefore waits for this counter to reach zero — a quiescent serving
# boundary — before collecting.  A request admitted while the collection is
# already running can still see the tail of that one-shot pause.
_frontend_inflight = 0
_frontend_inflight_cond = threading.Condition()


# Offline V1 lacks the HTTP server's startup lifespan, which freezes the
# frontend's static heap. Keep this ownership separate from the legacy
# post-request freeze: startup work must finish before requests are admitted.
_startup_gc_lock = threading.RLock()
_startup_gc_users = 0


def acquire_frontend_startup_heap() -> None:
    """Freeze static objects before serving, retaining normal automatic GC.

    Construct engines during application startup. GC freeze/unfreeze affects
    the entire process, just as in native EngineCore/server startup. A lease
    avoids thawing another live GRLLM's heap. Later engines do not freeze
    serving objects; the last close ends this process's offline serving phase.
    """
    global _startup_gc_users
    with _startup_gc_lock:
        if _startup_gc_users == 0:
            from vllm.utils.gc_utils import freeze_gc_heap

            # CPython itself can already have a small permanent generation;
            # get_freeze_count() cannot tell whether model startup is covered.
            freeze_gc_heap()
        _startup_gc_users += 1


def release_frontend_startup_heap() -> None:
    """Restore collection of startup cycles when the last offline owner closes."""
    global _startup_gc_users
    with _startup_gc_lock:
        if _startup_gc_users <= 0:
            raise RuntimeError("frontend startup heap has no active owner")
        _startup_gc_users -= 1
        if _startup_gc_users == 0:
            gc.unfreeze()


def _enter_frontend_request() -> None:
    global _frontend_inflight
    with _frontend_inflight_cond:
        _frontend_inflight += 1


def _exit_frontend_request() -> None:
    global _frontend_inflight
    with _frontend_inflight_cond:
        _frontend_inflight -= 1
        if _frontend_inflight <= 0:
            _frontend_inflight_cond.notify_all()


def frontend_request_scope(fn):
    """Mark a beam-search entrypoint as an in-flight frontend request.

    The one-shot frontend GC freeze (``schedule_frontend_gc_freeze``) waits
    for a quiescent serving boundary before collecting; entrypoints wrapped
    here are counted as in-flight for their whole lifetime, so the freeze
    never starts inside a request's response path.  For an async generator
    the scope spans until the generator is exhausted or closed (a request
    streaming intermediate outputs stays in flight).  Supports plain
    functions/methods and async-generator functions.
    """
    if inspect.isasyncgenfunction(fn):

        @functools.wraps(fn)
        async def _scoped_async_generator(*args, **kwargs):
            _enter_frontend_request()
            try:
                async for output in fn(*args, **kwargs):
                    yield output
            finally:
                _exit_frontend_request()

        return _scoped_async_generator

    @functools.wraps(fn)
    def _scoped_call(*args, **kwargs):
        _enter_frontend_request()
        try:
            return fn(*args, **kwargs)
        finally:
            _exit_frontend_request()

    return _scoped_call


def schedule_frontend_gc_freeze() -> bool:
    """Freeze the frontend process heap once, at a quiescent boundary.

    The beam-search frontend loop allocates heavily per request (candidate
    bookkeeping, ``materialize_selected_beams``, ``complete_eos_candidates``,
    batched logprob reconstruction and detokenization across 128 beams x
    SID steps).  Long-lived objects (constraint trie/strategy, tokenizer,
    engine bookkeeping) stay in gen-1/gen-2; with CPython's default
    thresholds ``(700, 10, 10)`` a gen-1 full scan fires every few requests,
    pauses the driver for tens of ms (``collected=0``), and lands inside the
    ``beam_search()`` E2E wall time.  A gc-callbacks tracer aligned to
    request boundaries further showed the *gen-0* scans themselves pause the
    driver 3.7-8 ms on a fixed cadence inside the request window (they scan
    the in-flight request's own ~1.0-1.5k beam-bookkeeping objects yet
    collect nothing), so the freeze raises all three generation thresholds.

    The worker-side freeze (``schedule_warmup_gc_freeze``) only protects the
    EngineCore/worker process.  In multi-process deployments the frontend is a
    distinct process and must freeze its own heap.  The collection runs on a
    one-shot daemon thread that first waits until no beam request is in
    flight (``frontend_request_scope`` maintains the counter): the pause sits
    in the inter-request gap instead of on a response path.  The wait is
    bounded by ``_QUIESCENT_WAIT_TIMEOUT_S``; a frontend that never goes
    quiescent falls back to collecting once.  Failures roll back the guard
    and never affect serving.
    """
    global _frontend_gc_freeze_done
    with _frontend_gc_freeze_lock:
        if _frontend_gc_freeze_done:
            return False
        _frontend_gc_freeze_done = True

    def _run() -> None:
        global _frontend_gc_freeze_done
        try:
            with _frontend_inflight_cond:
                deadline = time.monotonic() + _QUIESCENT_WAIT_TIMEOUT_S
                while _frontend_inflight > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.info(
                            "vllm-gr: frontend stayed busy for %.0fs; "
                            "running the one-shot GC freeze without a "
                            "quiescent boundary",
                            _QUIESCENT_WAIT_TIMEOUT_S,
                        )
                        break
                    _frontend_inflight_cond.wait(remaining)
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            gc.freeze()
            gc.set_threshold(
                _SERVING_GC_THRESHOLD,
                _SERVING_GC_THRESHOLD,
                _SERVING_GC_THRESHOLD,
            )
            logger.info(
                "vllm-gr: auto-froze frontend GC heap at a quiescent "
                "boundary after the first beam request (gen thresholds now %s)",
                list(gc.get_threshold()),
            )
        except Exception:
            with _frontend_gc_freeze_lock:
                _frontend_gc_freeze_done = False
            logger.exception(
                "vllm-gr: automatic frontend GC freeze failed; continuing with default GC"
            )

    threading.Thread(
        target=_run,
        name="vllm-gr-frontend-gc-freeze",
        daemon=True,
    ).start()
    return True


def shutdown_sync_engine(engine: Any) -> None:
    """Shut down a sync engine (vLLM LLMEngine)."""
    if engine is None:
        return

    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return

    engine_core = getattr(engine, "engine_core", None)
    if engine_core is not None:
        core_shutdown = getattr(engine_core, "shutdown", None)
        if callable(core_shutdown):
            core_shutdown()
            return

    logger.warning(
        "Engine %s has no shutdown method; resources may not be released",
        type(engine).__name__,
    )
