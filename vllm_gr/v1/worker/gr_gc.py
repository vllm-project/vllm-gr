# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Process-local Worker GC coordination shared by execute and RPC entrypoints."""

import gc
import threading
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_GC_FROZEN_AFTER_CAPTURE_MARKER = "_vllm_gr_gc_frozen_after_capture"
# Process-level guard for the one-time post-warmup freeze.  GC state is
# per-process (not per-runner), so a module-level flag — rather than a runner
# attribute — correctly deduplicates across the engine/worker layers and
# across in-proc vs. spawned EngineCore deployments.
#
# Two states are distinguished so that a freeze is only ever *acknowledged*
# after it actually completed:
# - _GC_WARMUP_FREEZE_DONE: the one-shot has been claimed (scheduled by the
#   automatic daemon or executed/attempted by the RPC).  Blocks re-scheduling.
# - _GC_WARMUP_FREEZE_COMPLETE: a collect+freeze actually finished
#   successfully (via the daemon or the RPC).  Gates dedupe acknowledgement.
# A claim whose attempt failed rolls DONE back so a later path can retry.
_GC_WARMUP_FREEZE_DONE = False
_GC_WARMUP_FREEZE_COMPLETE = False
_GC_WARMUP_FREEZE_LOCK = threading.Lock()

# Bounded wait for a quiescent boundary (no execute_model in flight) before
# the automatic one-shot collection runs.  An engine that stays busy past
# this deadline gets the collection anyway (once), degrading to the
# pre-barrier placement rather than never freezing the heap.
_QUIESCENT_WAIT_TIMEOUT_S = 30.0

# In-flight (patched) execute_model calls in this worker process.
# gc.collect() holds the GIL for the whole collection, so the automatic
# warmup freeze — scheduled from the teardown branch that runs at the START
# of the next execute_model — must not run its collection there: it would
# stall that request's critical path for the whole pause.  The freeze daemon
# waits for this depth to reach zero and starts the collection at that
# quiescent boundary instead.  A step admitted while the collection is
# already running can still see the tail of that one-shot pause.
_WORKER_EXECUTE_DEPTH = 0
_GC_FREEZE_IN_PROGRESS = False
_WORKER_EXECUTE_COND = threading.Condition()

# GC threshold applied to ALL generations (0/1/2) while serving, after the
# warmup freeze.  Every collection during serving measured ``collected=0``:
# gen-1/gen-2 pauses were 17-36 ms (old-gen scans of the persistent beam
# heap), gen-0 pauses 3.7-8 ms (scans of the in-flight request's ~1.0-1.5k
# beam-bookkeeping objects).  None of them ever reclaimed a single object —
# beam temporaries die by refcount, they never form cycles — so the scan is
# a pure pause landing on a fixed request cadence inside the E2E window.
# The finite (not 0) value keeps the door open for real cross-generation
# garbage to still be collected eventually / at shutdown.
_SERVING_GC_THRESHOLD = 50_000


def _enter_worker_execute() -> None:
    global _WORKER_EXECUTE_DEPTH
    with _WORKER_EXECUTE_COND:
        while _GC_FREEZE_IN_PROGRESS:
            _WORKER_EXECUTE_COND.wait()
        _WORKER_EXECUTE_DEPTH += 1


def _exit_worker_execute() -> None:
    global _WORKER_EXECUTE_DEPTH
    with _WORKER_EXECUTE_COND:
        _WORKER_EXECUTE_DEPTH -= 1
        if _WORKER_EXECUTE_DEPTH <= 0:
            _WORKER_EXECUTE_COND.notify_all()


def _freeze_serving_gc(*, raise_thresholds: bool) -> list[int]:
    """Collect all generations, freeze the live heap into the permanent
    generation, and (optionally) raise the gen-0/gen-1/gen-2 thresholds.

    Process-level operation shared by the capture-time freeze, the warmup
    RPC freeze, and the automatic post-first-request freeze.  ``gc.freeze()``
    moves every currently-tracked live object into the permanent generation
    so that gen-0/gen-1 collections no longer scan the persistent beam
    infrastructure (beam-session pools, captured graph state, scheduler beam
    bookkeeping).  When ``raise_thresholds`` is set, all generation scans
    are additionally disabled during serving:

    - gen-1/gen-2 measured ``collected=0`` (pure 17-36 ms pauses on a fixed
      4-request cadence);
    - gen-0 also measured ``collected=0`` with 3.7-8 ms pauses landing inside
      the request window on a fixed cadence (gc-callbacks tracer, aligned to
      request boundaries: every spiked request contained exactly one gen-0
      STOP, non-spiked requests none).  The young generation holds the
      in-flight request's own beam bookkeeping (~1.0-1.5k live objects:
      BeamSearchSequence x128, lists/dicts, wrapped tensors), so each scan
      walks them all yet reclaims nothing — temporaries die by refcount the
      moment their owner is dropped, they never form cycles.  Raising the
      gen-0 threshold therefore trades nothing measurable for removing the
      per-request pause.
    """
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()
    if raise_thresholds:
        _g0, _g1, _g2 = gc.get_threshold()
        gc.set_threshold(_SERVING_GC_THRESHOLD, _SERVING_GC_THRESHOLD, _SERVING_GC_THRESHOLD)
    return list(gc.get_threshold())


def schedule_warmup_gc_freeze() -> bool:
    """Schedule the one-time post-warmup GC freeze at a quiescent boundary.

    Called from the beam-session teardown path after the first beam-search
    request finishes in the EngineCore/worker process.  The teardown signal
    is consumed at the START of the next ``execute_model``, so a plain
    background thread would begin its collection while that request is in
    flight — and ``gc.collect()`` holds the GIL for the whole collection,
    stalling the request's critical path.  Instead the daemon thread first
    waits (bounded by ``_QUIESCENT_WAIT_TIMEOUT_S``) until no
    ``execute_model`` is running, then runs the ``collect + freeze`` (a few
    tens of ms) at that quiescent boundary — never inside a model step.
    Idempotent and thread-safe: only the first call schedules the freeze;
    subsequent calls are no-ops.  If the explicit RPC freeze has already
    completed the heap by then, the daemon collects nothing.

    This is the production equivalent of the benchmark's
    ``collective_rpc("gr_freeze_gc_after_warmup")``: the cross-request beam
    infrastructure is built lazily by the first beam request, so freezing
    once it has been torn down captures the whole persistent set.
    """
    global _GC_WARMUP_FREEZE_DONE
    with _GC_WARMUP_FREEZE_LOCK:
        if _GC_WARMUP_FREEZE_DONE:
            return False
        _GC_WARMUP_FREEZE_DONE = True

    def _run() -> None:
        global _GC_FREEZE_IN_PROGRESS
        global _GC_WARMUP_FREEZE_DONE, _GC_WARMUP_FREEZE_COMPLETE
        claimed_freeze_boundary = False
        try:
            with _WORKER_EXECUTE_COND:
                deadline = time.monotonic() + _QUIESCENT_WAIT_TIMEOUT_S
                while _WORKER_EXECUTE_DEPTH > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.info(
                            "vllm-gr: engine stayed busy for %.0fs; running "
                            "the one-shot GC freeze without a quiescent "
                            "boundary",
                            _QUIESCENT_WAIT_TIMEOUT_S,
                        )
                        break
                    _WORKER_EXECUTE_COND.wait(remaining)
                with _GC_WARMUP_FREEZE_LOCK:
                    if _GC_WARMUP_FREEZE_COMPLETE:
                        # The explicit warmup RPC already froze this heap.
                        return
                # Claim the admission boundary while still holding the same
                # condition lock used by _enter_worker_execute. No new model
                # step can enter between the zero-depth observation and GC.
                _GC_FREEZE_IN_PROGRESS = True
                claimed_freeze_boundary = True
            thresholds = _freeze_serving_gc(raise_thresholds=True)
            with _GC_WARMUP_FREEZE_LOCK:
                _GC_WARMUP_FREEZE_COMPLETE = True
            logger.info(
                "vllm-gr: auto-froze GC heap at a quiescent boundary after "
                "the first beam request (gen thresholds now %s)",
                thresholds,
            )
        except Exception:
            # Roll back the guard so a later teardown/RPC can retry.
            with _GC_WARMUP_FREEZE_LOCK:
                _GC_WARMUP_FREEZE_DONE = False
            # Never let a GC hint failure surface into the serving process.
            logger.exception(
                "vllm-gr: automatic warmup GC freeze failed; continuing with default GC"
            )
        finally:
            if claimed_freeze_boundary:
                with _WORKER_EXECUTE_COND:
                    _GC_FREEZE_IN_PROGRESS = False
                    _WORKER_EXECUTE_COND.notify_all()

    threading.Thread(
        target=_run,
        name="vllm-gr-gc-freeze",
        daemon=True,
    ).start()
    return True


def freeze_gc_after_graph_capture(runner: Any) -> None:
    """Freeze the serving-time heap once CUDA-graph capture has finished.

    GC state is process-local. vLLM freezes the EngineCore startup heap, but
    that does not reliably cover graph objects created in a separate Worker
    process, nor vllm-gr's native/beam/prefill capture objects. During serving,
    gen-1 collections otherwise scan this long-lived graph heap every few
    requests for cycles that never exist (measured 29-36 ms pauses with
    ``collected=0``, landing on a fixed 4-request cadence and spiking that
    request's latency). Running a full collection and ``gc.freeze()`` here
    moves the completed capture heap into the permanent generation so later
    collections scan only serving-time allocations.

    Idempotent: only the first invocation after capture runs the collection.
    Runs in the EngineCore/worker process (``capture_model`` executes there).
    """
    if getattr(runner, _GC_FROZEN_AFTER_CAPTURE_MARKER, False):
        return
    try:
        _freeze_serving_gc(raise_thresholds=False)
        setattr(runner, _GC_FROZEN_AFTER_CAPTURE_MARKER, True)
        logger.info(
            "vllm-gr: froze GC heap after CUDA-graph capture "
            "(gen thresholds now scan serving allocations only)"
        )
    except Exception:
        logger.exception(
            "vllm-gr: freeze_gc_after_graph_capture failed; continuing with default GC"
        )


def freeze_gc_after_warmup(worker: Any) -> dict[str, Any]:
    """Collect + freeze the heap in the EngineCore process after warmup.

    Reachable via ``collective_rpc("gr_freeze_gc_after_warmup")``.  The
    cross-request beam infrastructure (beam-session pools, beam graph runner
    state, scheduler beam bookkeeping) is built lazily by the first
    beam-search request — i.e. *after* the startup/capture freeze — and then
    stays live for the whole process.  Without a second freeze here, every
    gen-1 collection during serving rescans that persistent set (measured
    29-36 ms pauses with ``collected=0`` on a fixed 4-request cadence).

    This RPC is a *deterministic warmup barrier*: it reports ``frozen=True``
    only after a freeze has actually completed — either one that already
    finished (``deduped: True``, e.g. the automatic daemon ran first) or the
    synchronous collect+freeze performed here.  It never acknowledges a
    merely-scheduled (still pending) collection, so a benchmark using it as
    a barrier cannot start timing during the GC pause, and a collector that
    subsequently fails cannot invalidate an acknowledgement.  If the
    automatic daemon happens to be collecting concurrently, the two GIL-
    serialized idempotent freezes at worst double the one-shot pause.

    ``gc.freeze()`` is safe to call repeatedly.
    """
    runner = getattr(worker, "model_runner", None)
    global _GC_WARMUP_FREEZE_DONE, _GC_WARMUP_FREEZE_COMPLETE
    try:
        with _GC_WARMUP_FREEZE_LOCK:
            if _GC_WARMUP_FREEZE_COMPLETE:
                # A freeze (automatic daemon or an earlier RPC) already
                # completed; nothing left to do.
                return {"frozen": True, "thresholds": list(gc.get_threshold()), "deduped": True}
            # Claim the one-shot even if the automatic daemon has not fired
            # yet, so it does not schedule a second collection afterwards.
            _GC_WARMUP_FREEZE_DONE = True
        # After freeze, the persistent heap (weights, KV caches, captured
        # graphs, beam-session pools) lives in the permanent generation and is
        # never scanned.  Every collection that still fires during serving
        # is measured to be collected=0 (no cyclic garbage in the beam path):
        # gen-1/gen-2 scans were pure 17-36ms pauses landing mid-request on a
        # fixed cadence, and gen-0 scans (the in-flight request's beam
        # bookkeeping, ~1.0-1.5k objects) pure 3.7-8ms pauses on their own
        # fixed cadence.  Push all generation thresholds high enough that
        # these scans do not fire during serving; they can still run
        # eventually / at shutdown if real cross-generation garbage ever
        # accumulates.
        thresholds = _freeze_serving_gc(raise_thresholds=True)
        with _GC_WARMUP_FREEZE_LOCK:
            _GC_WARMUP_FREEZE_COMPLETE = True
        if runner is not None:
            setattr(runner, _GC_FROZEN_AFTER_CAPTURE_MARKER, True)
        logger.info(
            "vllm-gr: froze GC heap after warmup and raised all gen thresholds to %d",
            _SERVING_GC_THRESHOLD,
        )
        return {"frozen": True, "thresholds": thresholds}
    except Exception as e:  # never fail the request over a GC hint
        # Roll back the one-shot claim so the automatic path (or a later RPC)
        # can retry instead of leaving the heap permanently unfrozen.
        with _GC_WARMUP_FREEZE_LOCK:
            _GC_WARMUP_FREEZE_DONE = False
        logger.exception("vllm-gr: freeze_gc_after_warmup failed")
        return {"frozen": False, "error": str(e)}
