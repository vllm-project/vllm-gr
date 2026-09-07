# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)

logger = init_logger(__name__)

_BEAM_STATE_INIT_LOCK = threading.Lock()


def _engine_constraint_table_active(engine: Any) -> bool:
    """Whether the EngineCore's own config selects Constraint Tables.

    The EngineCore holds ``self.vllm_config``; reading the constraint backend
    from it is authoritative and does not depend on the process-global config
    hook (which may be None in the EngineCore subprocess).
    """
    from vllm_gr.config import ConstraintBackend, GRConfig

    gr_config = GRConfig.from_vllm_config(getattr(engine, "vllm_config", None))
    canonical_table = gr_config.constraint_table
    return gr_config.constraint_backend == ConstraintBackend.CONSTRAINT_TABLE or (
        canonical_table is not None and canonical_table.enabled
    )


def _ensure_beam_state(self) -> None:
    """Initialize per-engine beam state, including for late plugin loading."""
    if (
        getattr(self, "beam_cache", None) is not None
        and getattr(self, "beam_cache_lock", None) is not None
    ):
        return

    # General plugins may first load from inside EngineCore.__init__. In that
    # case, wrapping the class cannot affect the constructor already in flight,
    # so the request handlers also need a thread-safe lazy fallback.
    with _BEAM_STATE_INIT_LOCK:
        if getattr(self, "beam_cache", None) is None:
            self.beam_cache: dict[str, dict] = {}
        if getattr(self, "beam_cache_lock", None) is None:
            self.beam_cache_lock = threading.Lock()


# ------------------------------------------------------------------------------------
# _cache_beam_request / _handle_beam_request_step_update  (ported from source core.py)
# ------------------------------------------------------------------------------------


def _cache_beam_request(self, request) -> None:
    """Cache Request state for later beam forking.

    Thread-safe: uses beam_cache_lock to prevent data races.

    When the prompt length is not a multiple of the KV-cache block size,
    the last partial block would normally remain unhashed and therefore
    uncacheable.  We temporarily extend ``request._all_token_ids`` in-place
    to the next block boundary so that ``update_block_hashes()`` (which reads
    via ConstantList wrapper ``request.all_token_ids``) sees a complete last
    block and produces its hash.  ``patch_cache_blocks`` (see
    engine_core_patch.py) later caches that padded block after prefill, so
    the beam-decode request can prefix-cache-hit the remainder block and
    inherit its KV from the paged cache.  We then revert ``request._all_token_ids``
    in-place so prefill execution runs strictly on original tokens.
    """
    _ensure_beam_state(self)

    # Make a clean, un-mutable copy of the original prompt token IDs
    raw_token_ids = list(request._all_token_ids)
    full_block_hashes = list(request.block_hashes)
    block_hasher = getattr(self, "request_block_hasher", None)
    vllm_config = getattr(self, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None) if vllm_config else None
    block_size = getattr(cache_config, "block_size", 16) if cache_config else 16
    pad_len = (-len(raw_token_ids)) % block_size

    partial_block_hash = None
    if pad_len > 0 and block_hasher is not None:
        # Extend in-place with sentinel token ID (-1) to avoid collision with real prompt tokens.
        # Set cache_salt to request.request_id so block_hasher hashes the full string request ID
        # directly into the partial tail block hash pipeline, guaranteeing 0% cross-request collision.
        orig_salt = request.cache_salt
        request._all_token_ids.extend([-1] * pad_len)
        request.cache_salt = request.request_id
        request._block_hasher = block_hasher
        try:
            request.update_block_hashes()
            if len(request.block_hashes) > len(full_block_hashes):
                partial_block_hash = request.block_hashes[-1]
        finally:
            request.cache_salt = orig_salt
            # Remove pad tokens in-place before prefill execution
            del request._all_token_ids[len(raw_token_ids) :]

    with self.beam_cache_lock:
        # Preserve an out-of-order BEAM_SESSION_TEMPLATE (stub entry created
        # by _handle_beam_session_template before ADD_BATCH was cached).
        preserved_template = self.beam_cache.get(request.request_id, {}).get("step_template")
        self.beam_cache[request.request_id] = {
            "all_token_ids": list(raw_token_ids),
            "full_block_hashes": full_block_hashes,
            "partial_block_hash": partial_block_hash,
            "block_hashes": list(request.block_hashes),
            "num_pad_tokens": pad_len,
            "eos_token_id": request.sampling_params.eos_token_id
            if request.sampling_params is not None
            else None,
            "lora_request": request.lora_request,
            "cache_salt": request.cache_salt,
            "mm_features": request.mm_features,
            "prompt_embeds": request.prompt_embeds,
            "prompt_is_token_ids": request.prompt_is_token_ids,
            "num_prompt_tokens": request.num_prompt_tokens,
        }
        if preserved_template is not None:
            self.beam_cache[request.request_id]["step_template"] = preserved_template

    if _engine_constraint_table_active(self):
        # Constraint Tables generate the step-0 candidates in the Worker from
        # the raw prefill logits.  Only beam-session prefills reach this cache
        # (ADD_BATCH), so any such request is the session's constrained prefill.
        # The Worker derives top_k from beam_width, carried on the prefill's
        # sampling params (logprobs == beam_width in the beam-search loop).
        params = getattr(request, "sampling_params", None)
        beam_width = getattr(params, "logprobs", None) if params is not None else None
        request.is_constraint_prefill = True
        request.beam_width = beam_width


def _handle_beam_request_step_update(self, update) -> None:
    """Apply a grouped step update using only the prefill cache entry."""
    _ensure_beam_state(self)
    _apply_beam_step(self, update)


def _add_preprocessed_request(self, preprocessed) -> None:
    if hasattr(self, "input_queue"):
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, preprocessed))
    else:
        self.add_request(*preprocessed)


def _handle_beam_session_template(self, template, *, via: str | None = None) -> None:
    """Cache a session's step-update template for EngineCore engine-driven chaining.

    The frontend sends BEAM_SESSION_TEMPLATE right after the session's
    ADD_BATCH (same socket, FIFO), so the template is cached before the
    prefill output iteration runs.  The prefill branch of
    ``_beam_advance_from_decision`` consumes it to construct the step-0
    BeamRequestStepUpdate in-process — it carries the fields EngineCore
    cannot derive itself (decode sampling_params, max_decode_steps,
    constraint_step, prefix_len, routing stamps).

    A ``beam_width == 0`` template is ignored (cleanup semantics).  The
    template never enqueues anything by itself.  ``via`` tags the wire
    message the template arrived on (G-nit2: the merged SUBMIT_SESSION
    handler reuses this function, and an unattributed
    "BEAM_SESSION_TEMPLATE received" line would mislead log-based
    triage; the prefix substring is kept stable for existing greps).
    """
    _ensure_beam_state(self)
    if template.beam_width == 0:
        return
    logger.info(
        "vllm-gr: BEAM_SESSION_TEMPLATE received for session %s (beam_width=%d, "
        "max_decode_steps=%d, constraint_step=%d%s)",
        template.session_id,
        template.beam_width,
        template.max_decode_steps,
        template.constraint_step,
        f", via {via}" if via else "",
    )
    with self.beam_cache_lock:
        entry = self.beam_cache.get(template.session_id)
        if entry is None:
            # Out-of-order template (ADD_BATCH not cached yet): store a stub
            # so the template is not lost; _cache_beam_request preserves the
            # "step_template" key when it (re)creates the entry.
            self.beam_cache[template.session_id] = {"step_template": template}
        else:
            entry["step_template"] = template


def _handle_submit_session(self, requests, template) -> None:
    """SUBMIT_SESSION: the session's ADD_BATCH and template in one message.

    Cache the template *before* enqueueing ADDs so the busy-loop thread
    cannot run prefill and invoke the engine-driven hook while the
    template is still absent.  Stub-preservation in ``_cache_beam_request``
    keeps an out-of-order template when ADD later (re)creates the entry.
    """
    if template is not None:
        self._handle_beam_session_template(template, via="SUBMIT_SESSION")
    for req in requests:
        try:
            request = self.preprocess_add_request(req)
        except Exception:
            self._handle_request_preproc_error(req)
            continue
        self._cache_beam_request(request[0])
        _add_preprocessed_request(self, request)


def _publish_beam_session_error(self, update) -> None:
    """Terminate the grouped stream registered under the session ID."""
    self.output_queue.put_nowait(
        (
            update.client_index,
            EngineCoreOutputs(
                engine_index=self.engine_index,
                finished_requests={update.session_id},
                outputs=[
                    EngineCoreOutput(
                        request_id=update.session_id,
                        new_token_ids=[],
                        finish_reason=FinishReason.ERROR,
                    )
                ],
            ),
        )
    )


def _apply_beam_step(self, update) -> bool:
    """Reusable body of the BEAM_REQUEST_STEP_UPDATE handler.

    Validates the parallel arrays on ``update``, rebuilds a single beam
    decode ``Request`` from the session's prefill cache entry, enqueues it,
    and drains pruned children.  Split from
    :func:`_handle_beam_request_step_update` so future EngineCore-internal
    callers (e.g. self-resume) can drive a step without going through the
    ZMQ input socket.

    Returns True only when a beam-decode Request was actually enqueued
    (ADD landed in ``input_queue``); every validation / cleanup path that
    chains nothing returns False so callers can tell whether the engine is
    now running the next step.
    """
    from vllm.v1.request import Request

    session_id = update.session_id
    beam_width = update.beam_width

    # Clear the session completely upon generation termination / cleanup message
    if beam_width == 0:
        with self.beam_cache_lock:
            self.beam_cache.pop(session_id, None)

        # Record the teardown signal on the Scheduler so that
        # patched_schedule can propagate it to the worker via
        # SchedulerOutput.beam_finished_session_ids.  This is the
        # authoritative "beam search finished" signal — it is sent by the
        # beam search loop (GRLLM offline / OpenAI serving) via
        # _beam_request_cleanup(beam_width=0), and is distinct from
        # finished_req_ids which fires on every decode step (because each
        # step's Request reaches num_computed_tokens >= num_tokens).
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            if not hasattr(scheduler, "_beam_finished_session_ids"):
                scheduler._beam_finished_session_ids = []
            scheduler._beam_finished_session_ids.append(session_id)

        # We MUST abort the running beam-request session_id itself!
        self.aborts_queue.put_nowait(session_id)
        for abort_id in update.pruned_ids:
            self.aborts_queue.put_nowait(abort_id)
        return False

    # Validate parallel arrays
    if (
        len(update.beam_tokens) != beam_width
        or len(update.parent_beam_ids) != beam_width
        or len(update.child_beam_ids) != beam_width
    ):
        logger.error("BEAM_REQUEST_STEP_UPDATE: session=%s length mismatch", session_id)
        _publish_beam_session_error(self, update)
        return False

    parent_scores = update.parent_cumulative_logprobs
    if parent_scores is not None and (
        len(parent_scores) != beam_width
        or any(math.isnan(float(score)) or float(score) == math.inf for score in parent_scores)
    ):
        logger.error(
            "BEAM_REQUEST_STEP_UPDATE: session=%s invalid parent cumulative scores",
            session_id,
        )
        # EOS-masked beams legitimately carry -inf cumulative scores and are
        # accepted above; only NaN/+inf are protocol errors.  Either way a
        # bare rejection would leave the frontend waiting on this session
        # forever (serving: 120s timeout; offline: hang), so unblock the
        # waiter with an ERROR result for the registered session.
        _publish_beam_session_error(self, update)
        return False

    # Fetch the shared prefill cache entry
    with self.beam_cache_lock:
        session_cached = self.beam_cache.get(session_id)

    if session_cached is None and beam_width > 0:
        logger.error("BEAM_REQUEST_STEP_UPDATE: session %s not in cache", session_id)
        _publish_beam_session_error(self, update)
        return False

    arrival = time.time()
    prefill_tokens = session_cached["all_token_ids"]
    prefill_block_hashes = list(session_cached["block_hashes"])
    lora_request = session_cached["lora_request"]
    cache_salt = session_cached["cache_salt"]
    mm_features = session_cached["mm_features"]
    prompt_embeds = session_cached["prompt_embeds"]
    prompt_is_token_ids = session_cached["prompt_is_token_ids"]
    eos = session_cached["eos_token_id"]

    eos_token_id = update.eos_token_id if update.eos_token_id is not None else eos
    if eos_token_id is not None:
        update.sampling_params.update_from_generation_config({}, eos_token_id)

    # Pack the shared prefill and all beam suffixes into a single contiguous Beam Request
    # In incremental decode mode, only send the new token for each beam
    # (not all previous tokens — those are already in the KV pool)
    # decode_step is 0-indexed: first decode step = 0, second = 1, etc.
    decode_steps = len(update.beam_tokens[0]) if update.beam_tokens else 0
    assert all(len(t) == decode_steps for t in update.beam_tokens)
    beam_decode_step = decode_steps - 1 if decode_steps > 0 else 0

    # For incremental decode, only send the new token per beam
    # (prefix KV is in paged cache, previous decode KV is in pool)
    # But we need to include the prefix tokens in prompt_token_ids so the
    # scheduler can correctly compute the number of already-computed tokens.
    # The key insight: the scheduler will skip prefix tokens (because
    # get_computed_blocks returns prefix_len), and only schedule the
    # new beam tokens.
    beam_token_ids = list(prefill_tokens)
    for gen_tokens in update.beam_tokens:
        if decode_steps > 0:
            # Incremental path: only take the last token (the new one for this step)
            beam_token_ids.append(gen_tokens[-1])
        else:
            beam_token_ids.extend(gen_tokens)

    req = Request(
        request_id=session_id,  # Single request ID representing all beams
        prompt_token_ids=beam_token_ids,
        sampling_params=update.sampling_params,
        pooling_params=None,
        client_index=update.client_index,
        arrival_time=arrival,
        lora_request=(update.lora_request or lora_request),
        cache_salt=update.cache_salt or cache_salt,
        priority=update.priority,
        trace_headers=update.trace_headers,
        prompt_embeds=prompt_embeds,
        prompt_is_token_ids=prompt_is_token_ids,
        mm_features=mm_features,
        block_hasher=None,
    )

    # Start with the prefill block hashes; block_hasher extends it incrementally
    req.block_hashes = prefill_block_hashes
    if self.request_block_hasher is not None:
        req._block_hasher = self.request_block_hasher
        req.update_block_hashes()

    # Metadata Enrichment
    req.is_beam = True
    req.beam_width = beam_width
    req.is_beam_decode = True  # Flag indicating it's the decode stage
    req.beam_decode_steps = decode_steps  # total decode steps so far
    req.beam_decode_step = beam_decode_step  # current decode step (0-indexed)
    req.prefix_len = update.prefix_len
    # Store parent beam indices for incremental decode KV reordering
    req.parent_beam_indices = update.parent_beam_indices
    req.parent_cumulative_logprobs = (
        [float(score) for score in parent_scores] if parent_scores is not None else None
    )
    # Store max decode steps for KV pool pre-allocation
    req.max_decode_steps = update.max_decode_steps

    # Parent beam token sequences ride along for every beam decode request:
    # the Worker seeds its beam-decision state from them when a session is
    # (re)created (_seed_beam_decision_from_metadata), which rejects parent
    # scores without suffixes.  Constraint Table decode additionally uses
    # them to rebuild the operator's sequence_group.
    req.beam_suffix = [list(t) for t in update.beam_tokens]

    if _engine_constraint_table_active(self):
        # Constraint Table decode: the Worker runs the constrained top-k
        # operator at raw-logits time.  constraint_step equals the number of
        # generated tokens so far (1 or 2 under the 3-step cap).
        # constraint_step 0 is reserved for the constrained prefill, so this
        # decode request clears is_constraint_prefill.
        req.is_constrained = True
        req.constraint_step = update.constraint_step if update.constraint_step > 0 else decode_steps
        req.is_constraint_prefill = False

    _add_preprocessed_request(self, (req, update.current_wave))

    for abort_id in update.pruned_ids:
        self.aborts_queue.put_nowait(abort_id)

    # Record the running step state so EngineCore engine-driven chaining
    # can construct the NEXT BeamRequestStepUpdate from the worker's beam
    # decision without a frontend round-trip: the new beam suffixes extend
    # the recorded ones, and the routing fields (client, wave, priority,
    # sampling params) are replayed unchanged.
    with self.beam_cache_lock:
        cached = self.beam_cache.get(session_id)
        if cached is not None:
            cached["running_suffix"] = {
                "beam_tokens": [list(t) for t in update.beam_tokens],
                "parent_cumulative_logprobs": (
                    [float(score) for score in parent_scores] if parent_scores is not None else None
                ),
                "prefix_len": update.prefix_len,
                "beam_width": beam_width,
                "sampling_params": update.sampling_params,
                "eos_token_id": eos_token_id,
                "client_index": update.client_index,
                "current_wave": update.current_wave,
                "priority": update.priority,
                "data_parallel_rank": update.data_parallel_rank,
                "lora_request": (update.lora_request or lora_request),
                "cache_salt": (update.cache_salt or cache_salt),
                "trace_headers": update.trace_headers,
                "max_decode_steps": update.max_decode_steps,
                "is_constrained": bool(getattr(req, "is_constrained", False)),
            }
    return True


def _engine_self_advance_active(engine: Any) -> bool:
    """Whether engine-driven in-process chaining is enabled.

    The result is cached on the engine instance so the per-step hook stays
    cheap.  Resolved lazily at the first hook call: several enabling facts
    (worker decision auto-default) are only visible in the runtime/engine
    context — a Driver-side default alone can silently stay unset.
    """
    cached = getattr(engine, "_vllm_gr_self_advance", None)
    if cached is not None:
        return cached
    from vllm_gr.config import GRConfig

    cfg = GRConfig.from_vllm_config(getattr(engine, "vllm_config", None))
    engine_driven = cfg.beam_engine_driven
    engine_driven_runtime_default = False
    if engine_driven is None:
        engine_driven = _engine_driven_runtime_default(cfg)
        engine_driven_runtime_default = engine_driven
    logger.info(
        "vllm-gr: beam engine-driven = %s (beam_engine_driven=%s%s) "
        "[lazy gates: a boundary chains only when its worker beam decision "
        "(and the step template / running_suffix seed) actually exists — "
        "trie-catalog and other decision-less deployments never chain "
        "regardless of this flag; unconstrained no-catalog deployments "
        "WITH active worker decisions do produce decode decisions and chain "
        "from step 1 — prefill stays frontend-driven]",
        bool(engine_driven),
        cfg.beam_engine_driven,
        " [runtime default]" if engine_driven_runtime_default else "",
    )
    engine._vllm_gr_self_advance = bool(engine_driven)
    return bool(engine_driven)


def _engine_driven_runtime_default(cfg: Any) -> bool:
    """Accelerator beam deployments with worker decisions run engine-driven.

    Mirrors the Driver-side rule (no async condition, no constraint-backend
    condition — the backend promotion is invisible to the config read and
    the constraint requirement is enforced by the data: worker beam
    decisions only exist for constrained beam sessions).  The caller only
    consults this default when ``beam_engine_driven`` is unset.
    """
    return _accelerator_beam_with_worker_decision(cfg)


def _accelerator_beam_with_worker_decision(cfg: Any) -> bool:
    """(cuda|npu) AND beam_worker_decision AND beam_max_width configured."""
    try:
        from vllm.platforms import current_platform

        device_type = str(getattr(current_platform, "device_type", ""))
        if device_type not in ("cuda", "npu"):
            return False
    except Exception:  # pragma: no cover - platform introspection is best-effort
        return False
    return bool(cfg.beam_worker_decision and cfg.beam_max_width is not None)


def _beam_self_advance(self, engine_core_outputs) -> None:
    """Chain the next beam-decode step from the worker's beam decision.

    Called after ``scheduler.update_from_output`` produced the step's
    outputs (the scheduler patch already injected the worker decision into
    ``EngineCoreOutput.kv_transfer_params['beam_search_result']``).  When
    engine-driven chaining is enabled and the decision is non-terminal,
    assemble the next :class:`BeamRequestStepUpdate` in-process and apply
    it directly — no frontend RPC round-trip.  Terminal decisions stop the
    chain; the frontend still consumes each step's output (EOS / materialize)
    and sends cleanup.

    Every independent session decision in this iteration is drained; a
    session that cannot chain (no template, already chained, ownership
    mismatch) leaves the marker unset so the frontend falls back.
    """
    if not engine_core_outputs:
        return
    if not _engine_self_advance_active(self):
        return
    for ec_outputs in engine_core_outputs.values():
        for out in getattr(ec_outputs, "outputs", []):
            kv_params = getattr(out, "kv_transfer_params", None) or {}
            decision = kv_params.get("beam_search_result")
            if decision is None:
                continue
            _beam_advance_from_decision(self, out.request_id, decision)


def _beam_advance_from_decision(self, session_id: str, decision: Any) -> None:
    """Build and apply the next step update from a worker beam decision.

    Both chained transitions are gated by ``beam_engine_driven``:
    - decode step -> next decode step, seeded by ``running_suffix``;
    - prefill -> step0, seeded by the cached ``step_template``.
    The prefill decision's parents are all zero (``_run_cuda_prefill_decision``
    contract), so the mapping is the identity the frontend would have sent.
    """
    from vllm_gr.v1.engine.types import BeamRequestStepUpdate

    # A non-Mapping payload cannot be a worker decision; leave the frontend
    # path untouched rather than failing the step.
    if not isinstance(decision, dict):
        return

    # Idempotence guard (engine-driven decode chaining): a decision that
    # already chained its successor (marker set on success below, re-read
    # here on any re-delivery / double hook invocation) must never advance
    # twice — the second apply would re-ADD the session's still-running
    # request and trip scheduler.add_request's duplicate-id assert.
    if decision.get("beam_self_advance_chained"):
        return

    if decision.get("is_last_step"):
        # Terminal decision: stop chaining.  The frontend consumes the final
        # output (out_sequence + cumulative scores) and sends the cleanup
        # update (beam_width=0) which pops the beam_cache entry.
        return

    # Ownership guard (mixed batches): a decision produced for another
    # session (e.g. session B's constrained prefill batched with session A's
    # decode) must never chain onto this session's step update.
    decision_owner = decision.get("session_id")
    if isinstance(decision_owner, str) and decision_owner != session_id:
        logger.debug(
            "vllm-gr: beam self-advance skipped (decision owner %s != session %s)",
            decision_owner,
            session_id,
        )
        return

    from vllm_gr.v1.beam.beam_decision import (
        flatten_decision_value,
        unpack_beam_result_buffer,
    )

    compact = unpack_beam_result_buffer(decision)
    if compact is not None:
        tokens_flat, scores_flat, parents_flat = compact
    else:
        out_token_ids = decision.get("out_token_ids")
        out_token_index = decision.get("out_token_index")
        out_log_probs = decision.get("out_log_probs")
        if out_token_ids is None or out_token_index is None:
            logger.debug(
                "vllm-gr: beam self-advance skipped (no candidate tensors) for %s",
                session_id,
            )
            return
        tokens_flat = flatten_decision_value(out_token_ids)
        parents_flat = flatten_decision_value(out_token_index)
        scores_flat = flatten_decision_value(out_log_probs) if out_log_probs is not None else None
    if tokens_flat is None or parents_flat is None:
        logger.debug(
            "vllm-gr: beam self-advance skipped (unflattenable candidate payload) for %s",
            session_id,
        )
        return

    if not _engine_self_advance_active(self):
        return

    _ensure_beam_state(self)
    with self.beam_cache_lock:
        session_cached = self.beam_cache.get(session_id)
        running_raw = session_cached.get("running_suffix") if session_cached else None
        running = dict(running_raw) if running_raw is not None else None
        template = session_cached.get("step_template") if session_cached else None

    # Worker decision row 0 is this session's row; the request offset is 0
    # so the indices are already request-local parents.  The payload may be
    # host tensors (in-process) or nested lists (post-IPC normalization).
    parents = [int(p) for p in parents_flat]
    new_tokens = [int(t) for t in tokens_flat]
    parent_scores = [float(score) for score in scores_flat] if scores_flat is not None else None
    if (
        len(parents) != len(new_tokens)
        or (parent_scores is not None and len(parent_scores) != len(new_tokens))
        or not parents
    ):
        logger.warning(
            "vllm-gr: beam self-advance skipped (malformed decision width %d) for %s",
            len(parents),
            session_id,
        )
        return

    stage = decision.get("stage")
    is_prefill = stage == "prefill" or (stage is None and running is None and template is not None)

    if is_prefill:
        # ----- prefill -> step0 -----
        if template is None:
            logger.debug(
                "vllm-gr: prefill self-advance skipped (no step_template) for %s",
                session_id,
            )
            return
        beam_width = template.beam_width
        if beam_width < 1 or len(new_tokens) != beam_width:
            logger.warning(
                "vllm-gr: prefill self-advance skipped (decision width %d != template %d) for %s",
                len(new_tokens),
                beam_width,
                session_id,
            )
            return
        if template.max_decode_steps < 2:
            # Orphan-chain guard: a single-decode-step session's frontend
            # loop ends right after the prefill candidates; chaining step0
            # would only produce work that cleanup must then reclaim.
            logger.debug(
                "vllm-gr: prefill self-advance skipped (max_decode_steps %d < 2) for %s",
                template.max_decode_steps,
                session_id,
            )
            return
        routing = template
        prev_suffixes: list[list[int]] = [[] for _ in range(beam_width)]
        step = 1
    else:
        # ----- decode step -> next decode step -----
        if running is None:
            # The session was never seeded with running-suffix state (e.g.
            # the flag was enabled mid-session); nothing to chain from.
            logger.debug(
                "vllm-gr: beam self-advance skipped (no running_suffix) for %s",
                session_id,
            )
            return
        # running_suffix is a plain dict; give both branches one attribute-
        # accessor view (the template branch already is a struct).
        from types import SimpleNamespace

        routing = SimpleNamespace(**running)
        beam_width = routing.beam_width
        prev_suffixes = routing.beam_tokens
        step = None  # derived from the extended suffixes below

    # A worker decision that contains EOS needs frontend lifecycle handling:
    # the frontend records the completed candidates, removes them from the
    # active set, and may continue with fewer than ``beam_width`` survivors.
    # Chaining the fixed-width worker decision here would instead advance the
    # EOS candidate before the frontend can split it from the live beams.
    eos_token_id = getattr(routing, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id in new_tokens:
        logger.debug(
            "vllm-gr: beam self-advance skipped (decision contains EOS) for %s",
            session_id,
        )
        return

    if any(p < 0 or p >= len(prev_suffixes) for p in parents):
        logger.warning(
            "vllm-gr: beam self-advance skipped (parent index out of range) for %s",
            session_id,
        )
        return

    new_suffixes = [prev_suffixes[p] + [t] for p, t in zip(parents, new_tokens)]
    if step is None:
        step = len(new_suffixes[0])

    update = BeamRequestStepUpdate(
        session_id=session_id,
        # Tracking-only ids: EngineCore requests run under the single
        # session_id, so these are never scheduler request ids.  pruned_ids
        # are empty for the same reason (aborting unknown ids is a no-op).
        parent_beam_ids=[f"{session_id}-sa-{step}-{p}" for p in parents],
        child_beam_ids=[f"{session_id}-sa-{step}-{i}" for i in range(len(new_suffixes))],
        beam_tokens=new_suffixes,
        pruned_ids=[],
        prefix_len=routing.prefix_len,
        beam_width=beam_width,
        sampling_params=routing.sampling_params,
        eos_token_id=routing.eos_token_id,
        client_index=routing.client_index,
        current_wave=routing.current_wave,
        priority=routing.priority,
        data_parallel_rank=routing.data_parallel_rank,
        lora_request=routing.lora_request,
        cache_salt=routing.cache_salt,
        trace_headers=routing.trace_headers,
        parent_beam_indices=parents,
        parent_cumulative_logprobs=parent_scores,
        max_decode_steps=routing.max_decode_steps,
        # Prefill: the template carries the frontend's step-0 value directly
        # (=1 for constrained sessions, 0 otherwise).  Decode: derived from
        # the running suffix length as before.
        constraint_step=(
            routing.constraint_step if is_prefill else (step if routing.is_constrained else 0)
        ),
    )
    if _apply_beam_step(self, update):
        if is_prefill:
            logger.info(
                "vllm-gr: prefill self-advance chained step 0 for session %s "
                "(beam_width=%d, prefix_len=%d)",
                session_id,
                beam_width,
                routing.prefix_len,
            )
        # Signal the frontend that this step was already chained in-process:
        # the beam_search loop must skip its own beam_request_step_update
        # for the next iteration, or scheduler.add_request would assert
        # "duplicate request id" (the chained request is still running
        # under session_id).  Mutating the shared decision dict is safe: it
        # is the same object referenced by
        # EngineCoreOutput.kv_transfer_params, which is forwarded to the
        # frontend right after this hook returns.  Every skip path above
        # (and _apply_beam_step's validation failures) leaves the flag
        # unset, so the frontend-driven loop remains the automatic fallback.
        decision["beam_self_advance_chained"] = True


# ---------------------------------------------------------------------------
# Narrow GR input dispatch (socket adapter + busy-loop fallback)
# ---------------------------------------------------------------------------


def dispatch_gr_engine_core_input(self, request_type, payload) -> bool:
    """Handle vLLM-GR EngineCore request types.

    Returns True when the type is owned by GR and has been consumed, so
    callers skip upstream ADD/ABORT/generic enqueue.  Used by the socket
    adapter and by the ``_handle_client_request`` wrap so GR messages are
    never logged as unrecognized.
    """
    from vllm.v1.engine import EngineCoreRequestType

    if request_type == EngineCoreRequestType.ADD_BATCH:
        for req in payload:
            try:
                request = self.preprocess_add_request(req)
            except Exception:
                self._handle_request_preproc_error(req)
                continue
            self._cache_beam_request(request[0])
            _add_preprocessed_request(self, request)
        return True
    if request_type == EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE:
        self._handle_beam_request_step_update(payload)
        return True
    if request_type == EngineCoreRequestType.BEAM_SESSION_TEMPLATE:
        self._handle_beam_session_template(payload)
        return True
    if request_type == EngineCoreRequestType.SUBMIT_SESSION:
        submit_requests, submit_template = payload
        self._handle_submit_session(submit_requests, submit_template)
        return True
    return False


# ---------------------------------------------------------------------------
# EngineCoreProc.process_input_sockets vLLM-version adapter
# ---------------------------------------------------------------------------


def process_input_sockets(
    self,
    input_addresses,
    coord_input_address,
    identity,
    ready_event,
):
    """vLLM-version adapter for EngineCoreProc.process_input_sockets.

    Handshake, READY, ABORT, and generic enqueue stay aligned with the
    pinned upstream loop; GR types are decoded here and handed to
    :func:`dispatch_gr_engine_core_input` instead of copying that dispatch
    into ``_handle_client_request``.
    """
    from contextlib import ExitStack

    import msgspec
    import zmq
    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine import (
        EngineCoreReadyResponse,
        EngineCoreRequest,
    )

    from vllm_gr.v1.engine.codec import (
        build_engine_core_decoders,
        decode_engine_core_payload,
    )

    decoders = build_engine_core_decoders(self.tensor_ipc_receiver)

    with ExitStack() as stack, zmq.Context() as ctx:
        input_sockets = [
            stack.enter_context(
                make_zmq_socket(ctx, input_address, zmq.DEALER, identity=identity, bind=False)
            )
            for input_address in input_addresses
        ]
        if coord_input_address is None:
            coord_socket = None
        else:
            coord_socket = stack.enter_context(
                make_zmq_socket(
                    ctx,
                    coord_input_address,
                    zmq.XSUB,
                    identity=identity,
                    bind=False,
                )
            )
            # Send subscription message to coordinator.
            coord_socket.send(b"\x01")

        # Register sockets with poller.
        poller = zmq.Poller()
        ready_response = EngineCoreReadyResponse(
            max_model_len=self.vllm_config.model_config.max_model_len,
            num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
            dp_stats_address=self.frontend_stats_publish_address,
            dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
        )
        ready_payload = msgspec.msgpack.encode(ready_response)
        for input_socket in input_sockets:
            input_socket.send(ready_payload)
            poller.register(input_socket, zmq.POLLIN)

        if coord_socket is not None:
            assert coord_socket.recv() == b"READY"
            poller.register(coord_socket, zmq.POLLIN)

        ready_event.set()
        del ready_event
        while True:
            for input_socket, _ in poller.poll():
                type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                if type_frame.buffer == b"READY":
                    assert input_socket == coord_socket
                    continue
                request_type = EngineCoreRequestType(bytes(type_frame.buffer))
                decoded_payload = decode_engine_core_payload(
                    decoders,
                    request_type,
                    data_frames,
                )

                request: Any
                if request_type == EngineCoreRequestType.ADD:
                    req: EngineCoreRequest = decoded_payload
                    try:
                        request = self.preprocess_add_request(req)
                    except Exception:
                        self._handle_request_preproc_error(req)
                        continue

                elif dispatch_gr_engine_core_input(self, request_type, decoded_payload):
                    continue

                else:
                    request = decoded_payload

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
