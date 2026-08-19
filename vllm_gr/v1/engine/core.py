# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

import threading
import time
from copy import copy
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


def _ensure_beam_state(self) -> None:
    """Initialize per-engine beam state, including for late plugin loading."""
    state_attributes = (
        "beam_cache",
        "beam_cache_lock",
        "beam_controllers",
        "beam_controller_contexts",
        "beam_seen_session_ids",
        "beam_cancelled_session_ids",
        "beam_constraint_resources",
        "persistent_beam_states",
        "persistent_pending_starts",
        "persistent_pending_session_ids",
    )
    if all(getattr(self, name, None) is not None for name in state_attributes):
        return

    # General plugins may first load from inside EngineCore.__init__. In that
    # case, wrapping the class cannot affect the constructor already in flight,
    # so the request handlers also need a thread-safe lazy fallback.
    with _BEAM_STATE_INIT_LOCK:
        if getattr(self, "beam_cache", None) is None:
            self.beam_cache: dict[str, dict] = {}
        if getattr(self, "beam_cache_lock", None) is None:
            self.beam_cache_lock = threading.Lock()
        if getattr(self, "beam_controllers", None) is None:
            self.beam_controllers = {}
        if getattr(self, "beam_controller_contexts", None) is None:
            self.beam_controller_contexts = {}
        if getattr(self, "beam_seen_session_ids", None) is None:
            self.beam_seen_session_ids = set()
        if getattr(self, "beam_cancelled_session_ids", None) is None:
            self.beam_cancelled_session_ids = set()
        if getattr(self, "beam_constraint_resources", None) is None:
            from vllm_gr.constraints.trie_resource import (
                BeamConstraintResourceRegistry,
            )

            self.beam_constraint_resources = BeamConstraintResourceRegistry()
        from vllm_gr.v1.engine.persistent_beam import initialize_persistent_engine_state

        initialize_persistent_engine_state(self)


def _controller_failure_result(
    start, error: str, *, error_code: str | None = None
) -> dict[str, Any]:
    from vllm_gr.v1.beam.controller_types import (
        BeamControllerFinalResult,
        BeamControllerFinalResultLite,
    )

    result_type = (
        BeamControllerFinalResultLite
        if start.controller.output_spec is not None
        and start.controller.output_spec.logprobs_mode == "none"
        else BeamControllerFinalResult
    )
    return result_type(
        beam_session_id=start.controller.beam_session_id,
        external_request_id=start.controller.external_request_id,
        beams=[],
        metrics=None,
        finish_reason="error",
        error=error,
        error_code=error_code,
    ).to_wire()


def _put_controller_result(self, client_index: int, call_id: int, result: dict) -> None:
    """Resolve the frontend's one long-lived beam submission future."""
    from vllm.v1.engine import EngineCoreOutputs, UtilityOutput
    from vllm.v1.serial_utils import UtilityResult

    output = UtilityOutput(call_id=call_id, result=UtilityResult(result))
    if not hasattr(self, "output_queue"):
        raise RuntimeError("EngineCore beam controller output requires an output queue")
    self.output_queue.put_nowait(
        (
            client_index,
            EngineCoreOutputs(engine_index=getattr(self, "engine_index", 0), utility_output=output),
        )
    )


def register_constraint_resource(self, request) -> dict:
    """Validate and register one Trie resource, with an O(1) cache hit path."""
    from vllm_gr.constraints.trie_resource import trie_resource_from_wire

    _ensure_beam_state(self)
    resource_id = request.resource_id
    if not isinstance(resource_id, str) or len(resource_id) != 64:
        raise ValueError("Trie resource ID must be a SHA-256 hex digest")
    try:
        int(resource_id, 16)
    except ValueError as exc:
        raise ValueError("Trie resource ID must be a SHA-256 hex digest") from exc

    # This check deliberately precedes from_wire(): a duplicate registration
    # must not deserialize, copy, validate, or hash the large payload again.
    if resource_id in self.beam_constraint_resources:
        return {
            "resource_id": resource_id,
            "status": "already_registered",
            "error_code": None,
            "error": None,
        }

    resource = trie_resource_from_wire(request.constraint_resource)
    if resource.digest != resource_id:
        raise ValueError("Trie resource ID does not match payload digest")
    self.beam_constraint_resources.ensure(resource)
    return {
        "resource_id": resource_id,
        "status": "registered",
        "error_code": None,
        "error": None,
    }


def create_controller(self, start, request) -> Any:
    """Create and admit one EngineCore-owned beam controller."""
    from vllm_gr.config import BeamControlMode, GRConfig
    from vllm_gr.constraints.trie_resource import trie_resource_from_wire
    from vllm_gr.v1.beam.beam_search_controller import BeamSearchController

    _ensure_beam_state(self)
    session_id = start.controller.beam_session_id
    if session_id in self.beam_seen_session_ids:
        raise ValueError(f"duplicate beam_session_id {session_id!r}")
    configured_mode = GRConfig.from_vllm_config(self.vllm_config).beam_control_mode
    engine_modes = (
        BeamControlMode.ENGINE_PER_STEP_REQUEST,
        BeamControlMode.ENGINE_PERSISTENT_REQUEST,
    )
    if configured_mode not in engine_modes:
        raise ValueError(
            "EngineCore beam controller request rejected because configured "
            f"beam_control_mode is {configured_mode.value!r}"
        )
    if request.request_id != session_id:
        raise ValueError(
            f"beam engine request id {request.request_id!r} does not match "
            f"session id {session_id!r}"
        )
    if configured_mode is BeamControlMode.ENGINE_PERSISTENT_REQUEST:
        from vllm_gr.v1.engine.persistent_beam import defer_persistent_start

        if defer_persistent_start(self, start, request):
            return None

    # Backward-compatible fallback for old clients. New clients register the
    # resource separately and Start carries only constraint_resource_id.
    if (
        start.constraint_resource is not None
        and start.controller.constraint_resource_id not in self.beam_constraint_resources
    ):
        resource = trie_resource_from_wire(start.constraint_resource)
        self.beam_constraint_resources.ensure(resource)

    constraint_resource = None
    resource_id = start.controller.constraint_resource_id
    if start.controller.constraint_backend == "trie" and resource_id is not None:
        constraint_resource = self.beam_constraint_resources.require(resource_id)

    controller = BeamSearchController.from_request(
        start.controller,
        constraint_resource=constraint_resource,
    )
    self.beam_seen_session_ids.add(session_id)
    self.beam_controllers[session_id] = controller
    context = {
        "client_index": start.client_index,
        "call_id": start.call_id,
        "current_wave": start.current_wave,
        "sampling_params": request.sampling_params,
        "eos_token_id": start.controller.eos_token_id,
        "lora_request": request.lora_request,
        "cache_salt": request.cache_salt,
        "trace_headers": request.trace_headers,
        "priority": request.priority,
        "data_parallel_rank": start.data_parallel_rank,
        "beam_control_mode": configured_mode,
    }
    self.beam_controller_contexts[session_id] = context

    try:
        if not controller.requires_step:
            _finish_controller(self, session_id)
            return controller

        if configured_mode is BeamControlMode.ENGINE_PERSISTENT_REQUEST:
            from vllm_gr.v1.engine.persistent_beam import admit_persistent_request

            admit_persistent_request(self, request, controller)
            context["persistent_request"] = request
        else:
            self._cache_beam_request(request)
        controller.mark_step_dispatched()
        self.add_request(request, start.current_wave)
    except Exception:
        # Admission is transactional with respect to live controller state.
        # Keep the session ID in ``seen`` so a failed start cannot be replayed
        # as a different logical session.
        remove_controller(self, session_id)
        with self.beam_cache_lock:
            self.beam_cache.pop(session_id, None)
        if configured_mode is BeamControlMode.ENGINE_PERSISTENT_REQUEST:
            from vllm_gr.v1.engine.persistent_beam import rollback_persistent_admission

            rollback_persistent_admission(self, session_id)
        raise
    return controller


def get_controller(self, session_id: str) -> Any:
    _ensure_beam_state(self)
    try:
        return self.beam_controllers[session_id]
    except KeyError as exc:
        raise KeyError(f"unknown beam controller session {session_id!r}") from exc


def consume_step_output(self, session_id: str, output) -> Any:
    """Consume one raw EngineCore output and issue the next internal request."""
    from vllm.v1.engine import FinishReason

    from vllm_gr.v1.beam.controller_types import BeamControllerStepOutput

    controller = get_controller(self, session_id)
    old_internal_ids = list(controller.previous_internal_ids)
    error = None
    if output.finish_reason == FinishReason.ERROR:
        error = "worker returned an error for beam execution"

    token_ids: list[int] = []
    logprobs: list[float] = []
    ranks: list[int | None] | None = None
    if output.new_logprobs is not None:
        token_ids = output.new_logprobs.logprob_token_ids.reshape(-1).tolist()
        logprobs = output.new_logprobs.logprobs.reshape(-1).tolist()
        parent_count = max(1, len(controller.active_beams))
        if len(token_ids) % parent_count == 0:
            stride = len(token_ids) // parent_count
            sampled_ranks = output.new_logprobs.sampled_token_ranks.reshape(-1).tolist()
            ranks = []
            for parent_index in range(parent_count):
                sampled_rank = (
                    int(sampled_ranks[parent_index])
                    if parent_index < len(sampled_ranks)
                    else (int(sampled_ranks[0]) if sampled_ranks else None)
                )
                ranks.extend([sampled_rank, *range(1, stride)])

    decision = controller.consume_output(
        BeamControllerStepOutput(
            token_ids=token_ids,
            logprobs=logprobs,
            ranks=ranks,
            decoded_tokens=[None] * len(token_ids),
            error=error,
        )
    )
    if decision.finished:
        _finish_controller(self, session_id)
        return decision

    context = self.beam_controller_contexts[session_id]
    controller.mark_step_dispatched()
    from vllm_gr.config import BeamControlMode

    if context["beam_control_mode"] is BeamControlMode.ENGINE_PERSISTENT_REQUEST:
        from vllm_gr.v1.engine.persistent_beam import continue_persistent_request

        continue_persistent_request(
            self,
            session_id,
            decision,
            controller,
            context["sampling_params"],
        )
    else:
        from vllm_gr.v1.engine.types import BeamRequestStepUpdate

        used_parents = {
            old_internal_ids[parent_index] for parent_index in decision.parent_beam_indices
        }
        pruned_ids = [
            internal_id for internal_id in old_internal_ids if internal_id not in used_parents
        ]
        update = BeamRequestStepUpdate(
            session_id=session_id,
            parent_beam_ids=[
                old_internal_ids[parent_index] for parent_index in decision.parent_beam_indices
            ],
            child_beam_ids=list(controller.previous_internal_ids),
            beam_tokens=[[token_id] for token_id in decision.next_tokens],
            pruned_ids=pruned_ids,
            prefix_len=len(controller.initial_tokens),
            beam_width=decision.active_width,
            sampling_params=copy(context["sampling_params"]),
            eos_token_id=context["eos_token_id"],
            client_index=context["client_index"],
            current_wave=context["current_wave"],
            priority=context["priority"],
            data_parallel_rank=context["data_parallel_rank"],
            lora_request=context["lora_request"],
            cache_salt=context["cache_salt"],
            trace_headers=context["trace_headers"],
            parent_beam_indices=decision.parent_beam_indices,
            max_decode_steps=controller.total_execution_steps,
            decode_step=decision.completed_step,
            controller_owned=True,
        )
        self._handle_beam_request_step_update(update)
    return decision


def cancel_controller(self, session_id: str) -> bool:
    """Cancel a controller; repeated cleanup is idempotent."""
    _ensure_beam_state(self)
    controller = self.beam_controllers.get(session_id)
    if controller is None:
        if session_id in self.beam_cancelled_session_ids:
            return False
        raise KeyError(f"unknown beam controller session {session_id!r}")
    controller.cancel()
    self.beam_cancelled_session_ids.add(session_id)
    _finish_controller(self, session_id)
    return True


def finalize_controller(self, session_id: str) -> Any:
    return get_controller(self, session_id).finalize()


def remove_controller(self, session_id: str) -> bool:
    _ensure_beam_state(self)
    removed = self.beam_controllers.pop(session_id, None) is not None
    self.beam_controller_contexts.pop(session_id, None)
    return removed


def _finish_controller(self, session_id: str) -> None:
    from vllm.sampling_params import SamplingParams

    from vllm_gr.v1.engine.types import BeamRequestStepUpdate

    controller = get_controller(self, session_id)
    context = self.beam_controller_contexts[session_id]
    result = controller.finalize()

    from vllm_gr.config import BeamControlMode

    if context["beam_control_mode"] is BeamControlMode.ENGINE_PERSISTENT_REQUEST:
        # A zero-step controller never admitted its Request, so there is no
        # parked Scheduler/Worker state to tear down.
        if "persistent_request" in context:
            from vllm_gr.v1.engine.persistent_beam import begin_persistent_finalization

            begin_persistent_finalization(self, session_id)
    else:
        # Reuse the authoritative legacy teardown signal for Scheduler/Worker KV.
        self._handle_beam_request_step_update(
            BeamRequestStepUpdate(
                session_id=session_id,
                parent_beam_ids=[],
                child_beam_ids=[],
                beam_tokens=[],
                pruned_ids=list(controller.previous_internal_ids),
                prefix_len=0,
                beam_width=0,
                sampling_params=SamplingParams(),
                client_index=context["client_index"],
                current_wave=context["current_wave"],
                data_parallel_rank=context["data_parallel_rank"],
                controller_owned=True,
            )
        )
    _put_controller_result(
        self,
        context["client_index"],
        context["call_id"],
        result.to_wire(),
    )
    if context["beam_control_mode"] is BeamControlMode.ENGINE_PERSISTENT_REQUEST and (
        "persistent_request" in context
    ):
        from vllm_gr.v1.engine.persistent_beam import mark_persistent_final_result_sent

        mark_persistent_final_result_sent(self, session_id)
    remove_controller(self, session_id)


def _fail_controller(self, session_id: str, error: str) -> None:
    controller = get_controller(self, session_id)
    controller._fail(error)
    _finish_controller(self, session_id)


def route_beam_controller_outputs(self, outputs):
    """Consume controller step outputs and suppress them from the frontend."""
    if not outputs:
        return outputs
    _ensure_beam_state(self)
    for client_index, batch in list(outputs.items()):
        remaining = []
        consumed_ids: set[str] = set()
        for output in batch.outputs:
            session_id = output.request_id
            if session_id in self.beam_controllers:
                consumed_ids.add(session_id)
                try:
                    consume_step_output(self, session_id, output)
                except Exception as exc:
                    logger.exception("Beam controller %s failed while consuming output", session_id)
                    if session_id in self.beam_controllers:
                        _fail_controller(self, session_id, str(exc))
                continue
            if session_id in self.beam_seen_session_ids:
                logger.error("Received output for removed beam controller session %s", session_id)
                consumed_ids.add(session_id)
                continue
            remaining.append(output)
        batch.outputs = remaining
        if batch.finished_requests is not None and consumed_ids:
            batch.finished_requests = set(batch.finished_requests) - consumed_ids
    return outputs


def clear_controllers(self) -> None:
    """Best-effort idempotent EngineCore shutdown cleanup."""
    _ensure_beam_state(self)
    for session_id in list(self.beam_controllers):
        try:
            cancel_controller(self, session_id)
        except Exception:
            logger.exception("Failed to cancel beam controller %s during shutdown", session_id)
            remove_controller(self, session_id)
    self.beam_constraint_resources.clear()
    self.beam_seen_session_ids.clear()
    self.beam_cancelled_session_ids.clear()


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


def _handle_beam_request_step_update(self, update) -> None:
    """Apply a grouped step update using only the prefill cache entry."""
    from vllm.v1.request import Request

    _ensure_beam_state(self)
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
        return

    # Validate parallel arrays
    if (
        len(update.beam_tokens) != beam_width
        or len(update.parent_beam_ids) != beam_width
        or len(update.child_beam_ids) != beam_width
    ):
        logger.error("BEAM_REQUEST_STEP_UPDATE: session=%s length mismatch", session_id)
        if getattr(update, "controller_owned", False):
            _fail_controller(self, session_id, "beam step update length mismatch")
            return
        for child_id in update.child_beam_ids:
            self.output_queue.put_nowait(
                (
                    update.client_index,
                    EngineCoreOutputs(
                        engine_index=self.engine_index,
                        finished_requests={child_id},
                        outputs=[
                            EngineCoreOutput(
                                request_id=child_id,
                                new_token_ids=[],
                                finish_reason=FinishReason.ERROR,
                            )
                        ],
                    ),
                )
            )
        return

    # Fetch the shared prefill cache entry
    with self.beam_cache_lock:
        session_cached = self.beam_cache.get(session_id)

    if session_cached is None and beam_width > 0:
        logger.error("BEAM_REQUEST_STEP_UPDATE: session %s not in cache", session_id)
        if getattr(update, "controller_owned", False):
            _fail_controller(self, session_id, "beam prefill cache entry is missing")
            return
        for child_id in update.child_beam_ids:
            self.output_queue.put_nowait(
                (
                    update.client_index,
                    EngineCoreOutputs(
                        engine_index=self.engine_index,
                        finished_requests={child_id},
                        outputs=[
                            EngineCoreOutput(
                                request_id=child_id,
                                new_token_ids=[],
                                finish_reason=FinishReason.ERROR,
                            )
                        ],
                    ),
                )
            )
        return

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
    if update.decode_step is None:
        decode_steps = len(update.beam_tokens[0]) if update.beam_tokens else 0
        assert all(len(t) == decode_steps for t in update.beam_tokens)
        beam_decode_step = decode_steps - 1 if decode_steps > 0 else 0
    else:
        beam_decode_step = update.decode_step
        decode_steps = beam_decode_step + 1
        if any(len(tokens) != 1 for tokens in update.beam_tokens):
            if getattr(update, "controller_owned", False):
                _fail_controller(
                    self,
                    session_id,
                    "EngineCore-controlled beam updates must carry one token per beam",
                )
                return
            raise ValueError("explicit beam decode_step requires one token per beam")

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
    # Store max decode steps for KV pool pre-allocation
    req.max_decode_steps = update.max_decode_steps

    if hasattr(self, "input_queue"):
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, (req, update.current_wave)))
    else:
        self.add_request(req, update.current_wave)

    for abort_id in update.pruned_ids:
        self.aborts_queue.put_nowait(abort_id)


# ---------------------------------------------------------------------------
# EngineCoreProc.process_input_sockets replacement
# ---------------------------------------------------------------------------


def process_input_sockets(
    self,
    input_addresses,
    coord_input_address,
    identity,
    ready_event,
):
    """Replacement for EngineCoreProc.process_input_sockets with
    ADD_BATCH and BEAM_REQUEST_STEP_UPDATE decoders."""
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
    from vllm_gr.v1.engine.types import BeamRequestStepUpdate

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

                elif request_type == EngineCoreRequestType.ADD_BATCH:
                    batch: list[EngineCoreRequest] = decoded_payload
                    for req in batch:
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                        self._cache_beam_request(request[0])
                        self.input_queue.put_nowait((EngineCoreRequestType.ADD, request))
                    continue

                elif request_type == EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE:
                    beam_update: BeamRequestStepUpdate = decoded_payload
                    self._handle_beam_request_step_update(beam_update)
                    continue

                elif request_type == EngineCoreRequestType.BEAM_CONTROLLER_START:
                    start = decoded_payload
                    try:
                        preprocessed = self.preprocess_add_request(start.engine_request)
                    except Exception as exc:
                        logger.exception(
                            "Failed to preprocess beam controller session %s",
                            start.controller.beam_session_id,
                        )
                        _put_controller_result(
                            self,
                            start.client_index,
                            start.call_id,
                            _controller_failure_result(start, str(exc)),
                        )
                        continue
                    request = (start, preprocessed[0])

                elif request_type == EngineCoreRequestType.BEAM_CONTROLLER_CANCEL:
                    request = decoded_payload

                elif request_type == EngineCoreRequestType.BEAM_CONSTRAINT_RESOURCE_REGISTER:
                    request = decoded_payload

                else:
                    request = decoded_payload

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
