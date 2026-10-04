# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH, upstream grouped beam updates, and internal BeamProc plans."""

from __future__ import annotations

import copy
import json
import os
import queue
import threading
import time
from collections import deque
from functools import partial
from typing import Any

import msgspec
import numpy as np
from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)

from vllm_gr.utils.env import is_truthy_env

logger = init_logger(__name__)

_BEAMPROC_MSG_ABORT = b"ABORT"
_BEAMPROC_MSG_BEAM_PLAN = b"BEAM_PLAN"
_BEAMPROC_MSG_BEAM_FINISHED = b"BEAM_FINISHED"
_BEAMPROC_MSG_ENGINE_OUTPUT_BATCH = b"ENGINE_OUTPUT_BATCH"
_BEAMPROC_MSG_BEAM_STATE_UPDATE = b"BEAM_STATE_UPDATE"
_BEAMPROC_ENABLED_ENV = "VLLM_GR_BEAM_PROC_ENABLED"
_BEAMPROC_ZMQ_BUFFER_BYTES = 2_147_483_647
_BEAMPROC_ZMQ_BACKPRESSURE_POLL_MS = 1
_BEAMPROC_QUEUE_PUT_TIMEOUT_S = 5.0
_BEAMPROC_DUAL_QUEUE_POLL_SLEEP_S = 0.000005
_BEAMPROC_F32_ITEMSIZE = np.dtype("<f4").itemsize
_BEAMPROC_U64_ITEMSIZE = np.dtype("<u8").itemsize
_BEAMPROC_CONTEXT_TRACE_HEADER = "x-vllm-gr-beam-context"
_BEAMPROC_WAKE_REQUEST = object()
_BEAMPROC_CHILD_BATCH_REQUEST = object()
_BEAM_STATE_INIT_LOCK = threading.Lock()


def _positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero, got {value}")
    return value


def _beamproc_queue_put(target_queue, item, queue_name: str) -> None:
    try:
        target_queue.put(item, timeout=_BEAMPROC_QUEUE_PUT_TIMEOUT_S)
    except queue.Full as exc:
        raise RuntimeError(
            f"BeamProc {queue_name} remained full for {_BEAMPROC_QUEUE_PUT_TIMEOUT_S:g}s"
        ) from exc


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


# ---------------------------------------------------------------------------
# _cache_beam_request / _handle_beam_fork  (ported from source core.py)
# ---------------------------------------------------------------------------


def _beamproc_model_prefix_token_count(
    request,
    trace_context: dict[str, Any],
    token_count: int,
) -> int:
    """Return the prefix boundary used by grouped beam attention.

    The root request's trace context retains the API prompt length so response
    token stripping excludes only the user's prompt. The model prompt can be
    longer because serving appends a fixed sid-begin token. Grouped requests,
    however, explicitly store their shared model-prefix boundary in the trace
    context because ``num_prompt_tokens`` includes all packed beam rows.
    """
    if getattr(request, "is_beam_decode", False):
        prefix_token_count = getattr(
            request,
            "_beamproc_prefix_token_count",
            None,
        )
        if not isinstance(prefix_token_count, int):
            prefix_token_count = trace_context.get("beam_prefix_token_count")
    else:
        prefix_token_count = getattr(request, "num_prompt_tokens", token_count)
    if not isinstance(prefix_token_count, int):
        prefix_token_count = getattr(request, "num_prompt_tokens", token_count)
    return max(0, min(int(prefix_token_count), token_count))


def _beamproc_prepare_grouped_parent_state(
    request,
    all_token_ids: list[int],
    block_hashes: list[Any],
) -> dict[str, Any]:
    trace_context = (
        _beamproc_context_from_trace_headers(getattr(request, "trace_headers", None)) or {}
    )
    prompt_token_count = _beamproc_model_prefix_token_count(
        request, trace_context, len(all_token_ids)
    )
    api_prompt_token_count = trace_context.get("prompt_token_count")
    if not isinstance(api_prompt_token_count, int):
        api_prompt_token_count = prompt_token_count
    api_prompt_token_count = max(0, min(api_prompt_token_count, len(all_token_ids)))
    prefix_token_ids = (
        all_token_ids
        if prompt_token_count == len(all_token_ids)
        else all_token_ids[:prompt_token_count]
    )
    generated_token_ids = (
        ()
        if prompt_token_count == len(all_token_ids)
        else tuple(all_token_ids[prompt_token_count:])
    )
    return {
        "prefix_token_ids": prefix_token_ids,
        "generated_token_ids": generated_token_ids,
        "response_token_ids": tuple(all_token_ids[api_prompt_token_count:]),
        "block_hashes": block_hashes,
        "request": request,
        "trace_context": trace_context,
        "trace_headers": _beamproc_shared_trace_headers(getattr(request, "trace_headers", None)),
        "step_index": int(trace_context.get("step_index") or 0),
    }


def _beam_cache_entry(self, request) -> dict[str, Any]:
    """Build a partial-block-aware snapshot plus BeamProc's live handle."""
    all_token_ids = list(request._all_token_ids)
    full_block_hashes = list(request.block_hashes)
    block_hasher = getattr(self, "request_block_hasher", None)
    vllm_config = getattr(self, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None) if vllm_config is not None else None
    block_size = getattr(cache_config, "block_size", 16) if cache_config else 16
    pad_len = (-len(all_token_ids)) % block_size

    partial_block_hash = None
    if pad_len > 0 and block_hasher is not None:
        original_salt = request.cache_salt
        request._all_token_ids.extend([-1] * pad_len)
        request.cache_salt = request.request_id
        request._block_hasher = block_hasher
        try:
            request.update_block_hashes()
            if len(request.block_hashes) > len(full_block_hashes):
                partial_block_hash = request.block_hashes[-1]
        finally:
            request.cache_salt = original_salt
            del request._all_token_ids[len(all_token_ids) :]

    block_hashes = list(request.block_hashes)
    entry = {
        "all_token_ids": all_token_ids,
        "full_block_hashes": full_block_hashes,
        "partial_block_hash": partial_block_hash,
        "block_hashes": block_hashes,
        "num_pad_tokens": pad_len,
        "eos_token_id": _request_eos_token_id(request),
        "lora_request": request.lora_request,
        "cache_salt": request.cache_salt,
        "mm_features": request.mm_features,
        "prompt_embeds": request.prompt_embeds,
        "prompt_is_token_ids": getattr(request, "prompt_is_token_ids", True),
        "num_prompt_tokens": request.num_prompt_tokens,
        "_request": request,
        "_block_hash_count": len(block_hashes),
    }
    if _beamproc_should_apply_plan():
        entry["_grouped_parent_state"] = _beamproc_prepare_grouped_parent_state(
            request,
            all_token_ids,
            block_hashes,
        )
    return entry


def _request_eos_token_id(request) -> int | None:
    sampling_params = getattr(request, "sampling_params", None)
    if sampling_params is not None:
        return getattr(sampling_params, "eos_token_id", None)
    return getattr(request, "eos_token_id", None)


def _clone_grouped_sampling_params(sampling_params):
    """Normalize legacy per-beam ``n`` into one grouped scheduler request."""
    from vllm_gr.sampling_params import clone_beam_decode_sampling_params

    normalized = copy.copy(sampling_params)
    normalized.n = 1
    return clone_beam_decode_sampling_params(normalized)


def _beamproc_cached_request(entry):
    """Return the live Request from current or legacy BeamProc cache entries."""
    if isinstance(entry, dict):
        return entry.get("_request")
    return entry[0]


def _cache_beam_request(self, request) -> None:
    """Cache upstream's immutable prefill snapshot and BeamProc live state."""
    _ensure_beam_state(self)
    entry = _beam_cache_entry(self, request)
    with self.beam_cache_lock:
        self.beam_cache[request.request_id] = entry
    prepared_state = entry.get("_grouped_parent_state")
    if prepared_state is not None:
        states = getattr(self, "beamproc_grouped_parent_states", None)
        if states is None:
            states = {}
            self.beamproc_grouped_parent_states = states
        states[request.request_id] = prepared_state


def _publish_beam_children(self, requests, remove_ids) -> None:
    """Atomically replace cached parents with internal BeamProc children.

    Grouped ``BeamRequestStepUpdate`` sessions need immutable snapshots, but
    Rust plans still operate on live per-child ``Request`` objects.  Retaining
    the legacy tuple here avoids copying every child's full prompt-token and
    block-hash lists after each plan step.
    """
    _ensure_beam_state(self)
    with self.beam_cache_lock:
        for request_id in remove_ids:
            self.beam_cache.pop(request_id, None)
        for request in requests:
            self.beam_cache[request.request_id] = (
                request,
                len(request.block_hashes),
            )


def _build_grouped_beam_request(
    self,
    *,
    request_id: str,
    prefix_token_ids,
    prefix_block_hashes,
    beam_tokens=None,
    flat_beam_tokens=None,
    beam_decode_steps: int | None = None,
    beam_width: int,
    sampling_params,
    eos_token_id,
    client_index: int,
    lora_request,
    cache_salt,
    priority: int,
    trace_headers,
    prompt_embeds,
    mm_features,
    prefix_len: int,
    timings: dict[str, float] | None = None,
):
    """Build the single scheduler Request shared by Python and Rust beams."""
    from vllm.v1.request import Request

    request_sampling_params = _clone_grouped_sampling_params(sampling_params)
    if eos_token_id is not None:
        request_sampling_params.update_from_generation_config({}, eos_token_id)

    request_started_at = time.perf_counter() if timings is not None else None
    if flat_beam_tokens is not None:
        decode_steps = int(beam_decode_steps or 0)
    else:
        decode_steps = len(beam_tokens[0]) if beam_tokens else 0
    packed_token_ids = list(prefix_token_ids)
    if flat_beam_tokens is not None:
        packed_token_ids.extend(flat_beam_tokens)
    else:
        for generated_tokens in beam_tokens or ():
            packed_token_ids.extend(generated_tokens)

    req = Request(
        request_id=request_id,
        prompt_token_ids=packed_token_ids,
        sampling_params=request_sampling_params,
        pooling_params=None,
        client_index=client_index,
        arrival_time=time.time(),
        lora_request=lora_request,
        cache_salt=cache_salt,
        priority=priority,
        trace_headers=trace_headers,
        prompt_embeds=prompt_embeds,
        mm_features=mm_features,
        block_hasher=None,
    )
    if request_started_at is not None:
        timings["request_ms"] = (time.perf_counter() - request_started_at) * 1000.0

    hash_started_at = time.perf_counter() if timings is not None else None
    req.block_hashes = list(prefix_block_hashes)
    if self.request_block_hasher is not None:
        req.get_hash_new_full_blocks = partial(self.request_block_hasher, req)
        req.block_hashes.extend(req.get_hash_new_full_blocks())
    else:
        req.get_hash_new_full_blocks = None
    req.is_beam = True
    req.beam_width = beam_width
    req.is_beam_decode = True
    req.beam_decode_steps = decode_steps
    req.prefix_len = prefix_len
    if hash_started_at is not None:
        timings["hash_ms"] = (time.perf_counter() - hash_started_at) * 1000.0
    return req


def _handle_beam_request_step_update(self, update) -> None:
    """Apply upstream's grouped beam update using the cached prefill state."""
    _ensure_beam_state(self)
    session_id = update.session_id
    beam_width = update.beam_width
    if beam_width == 0:
        with self.beam_cache_lock:
            self.beam_cache.pop(session_id, None)

        # We MUST abort the running beam-request session_id itself!
        self.aborts_queue.put_nowait(session_id)
        for request_id in update.pruned_ids:
            self.aborts_queue.put_nowait(request_id)
        return

    # Validate parallel arrays
    input_beam_count = len(update.beam_tokens)
    if (
        len(update.parent_beam_ids) != input_beam_count
        or len(update.child_beam_ids) != input_beam_count
    ):
        logger.error("BEAM_REQUEST_STEP_UPDATE: session=%s length mismatch", session_id)
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

    with self.beam_cache_lock:
        cached = self.beam_cache.get(session_id)
    if cached is None:
        logger.error("BEAM_REQUEST_STEP_UPDATE: session %s not in cache", session_id)
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

    if isinstance(cached, dict):
        prefill_token_ids = list(cached["all_token_ids"])
        prefill_block_hashes = list(cached["block_hashes"])
        cached_eos_token_id = cached["eos_token_id"]
        cached_lora_request = cached["lora_request"]
        cached_cache_salt = cached["cache_salt"]
        cached_prompt_embeds = cached["prompt_embeds"]
        cached_mm_features = cached["mm_features"]
    else:
        cached_req, cached_block_hash_count = cached
        prefill_token_ids = list(cached_req.prompt_token_ids)
        prefill_block_hashes = list(cached_req.block_hashes[:cached_block_hash_count])
        cached_eos_token_id = _request_eos_token_id(cached_req)
        cached_lora_request = cached_req.lora_request
        cached_cache_salt = cached_req.cache_salt
        cached_prompt_embeds = cached_req.prompt_embeds
        cached_mm_features = cached_req.mm_features
    effective_eos_token_id = (
        update.eos_token_id if update.eos_token_id is not None else cached_eos_token_id
    )
    if effective_eos_token_id is not None:
        update.sampling_params.update_from_generation_config(
            {},
            effective_eos_token_id,
        )
    decode_steps = len(update.beam_tokens[0]) if update.beam_tokens else 0
    if any(len(tokens) != decode_steps for tokens in update.beam_tokens):
        logger.error("BEAM_REQUEST_STEP_UPDATE: session=%s ragged beam tokens", session_id)
        return
    total_gen_tokens = sum(len(tokens) for tokens in update.beam_tokens)

    if decode_steps > 0:
        calculated_beam_width = total_gen_tokens // decode_steps
        if calculated_beam_width == 0:
            calculated_beam_width = beam_width
    else:
        calculated_beam_width = beam_width

    req = _build_grouped_beam_request(
        self,
        request_id=session_id,
        prefix_token_ids=prefill_token_ids,
        prefix_block_hashes=prefill_block_hashes,
        beam_tokens=update.beam_tokens,
        beam_width=calculated_beam_width,
        sampling_params=update.sampling_params,
        eos_token_id=effective_eos_token_id,
        client_index=update.client_index,
        lora_request=update.lora_request or cached_lora_request,
        cache_salt=update.cache_salt or cached_cache_salt,
        priority=update.priority,
        trace_headers=update.trace_headers,
        prompt_embeds=cached_prompt_embeds,
        mm_features=cached_mm_features,
        prefix_len=update.prefix_len,
    )
    req.log_probs_num = beam_width
    req.parent_beam_indices = update.parent_beam_indices
    req.max_decode_steps = update.max_decode_steps
    req.parent_cum_logprobs = update.parent_cum_logprobs
    req.target_beam_width = getattr(update, "target_beam_width", 0) or update.beam_width
    req.is_last_step = getattr(update, "is_last_step", False)

    if getattr(update, "is_last_step", False):
        with self.beam_cache_lock:
            self.beam_cache.pop(session_id, None)
        # Deliberately do NOT push session_id onto self.aborts_queue here
        # (unlike the beam_width == 0 path): EngineCoreProc.step() drains
        # aborts_queue and calls abort_requests() BEFORE
        # scheduler.update_from_output() runs for that same step, so an
        # abort queued alongside this final step's ADD can race ahead of --
        # and silently discard -- the output this very step is about to
        # produce, hanging the client forever waiting on it. The
        # beam_width == 0 cleanup call carries no output of its own (the
        # real final output was already delivered by a prior step), so that
        # path's abort can't race against anything. The request still
        # completes and is removed from the scheduler on its own once this
        # final step's output is processed, same as any normal request
        # reaching its stop condition.

    self.input_queue.put_nowait((EngineCoreRequestType.ADD, (req, update.current_wave)))
    for request_id in update.pruned_ids:
        self.aborts_queue.put_nowait(request_id)


# Fields that differ per child in a BEAM_FORK batch.  Must be kept in sync
# with vllm.v1.request.Request — if upstream adds new mutable per-request
# state, add the corresponding key here to avoid silent sharing.
_PER_CHILD_FIELDS = frozenset(
    {
        "request_id",
        "prompt_token_ids",
        "num_prompt_tokens",
        "_all_token_ids",
        "_output_token_ids",
        "output_token_ids",
        "all_token_ids",
        "events",
        "spec_token_ids",
        "block_hashes",
        "get_hash_new_full_blocks",
    }
)


def _beamproc_context_from_trace_headers(trace_headers) -> dict[str, Any] | None:
    if not trace_headers:
        return None
    raw = trace_headers.get(_BEAMPROC_CONTEXT_TRACE_HEADER)
    if not raw:
        return None
    try:
        context = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        logger.warning("Invalid BeamProc context trace header; ignoring")
        return None
    return context if isinstance(context, dict) else None


def _beamproc_request_context(request) -> dict[str, Any] | None:
    context = getattr(request, "_beamproc_trace_context", None)
    if isinstance(context, dict):
        return context
    return _beamproc_context_from_trace_headers(getattr(request, "trace_headers", None))


def _beamproc_shared_trace_headers(trace_headers) -> dict[str, str] | None:
    if not trace_headers:
        return None
    if _BEAMPROC_CONTEXT_TRACE_HEADER not in trace_headers:
        return trace_headers
    headers = dict(trace_headers)
    headers.pop(_BEAMPROC_CONTEXT_TRACE_HEADER, None)
    return headers or None


def _beamproc_trace_headers_with_context(trace_headers, context) -> dict[str, str] | None:
    if not context:
        return dict(trace_headers) if trace_headers else None
    headers = dict(trace_headers or {})
    headers[_BEAMPROC_CONTEXT_TRACE_HEADER] = json.dumps(
        context,
        separators=(",", ":"),
        sort_keys=True,
    )
    return headers


def _beamproc_child_context_from_parent(parent_req) -> dict[str, Any] | None:
    context = _beamproc_context_from_trace_headers(getattr(parent_req, "trace_headers", None))
    if context is None:
        return None

    next_context = dict(context)
    step_index = next_context.get("step_index")
    if isinstance(step_index, int):
        next_context["step_index"] = step_index + 1
    total_decode_steps = next_context.get("total_decode_steps")
    if isinstance(next_context.get("step_index"), int) and isinstance(total_decode_steps, int):
        next_context["is_final_step"] = next_context["step_index"] >= total_decode_steps - 1
    return next_context


def _create_beam_fork_children(
    self,
    *,
    parent_ids,
    child_ids,
    token_ids,
    abort_ids,
    sampling_params,
    eos_token_id,
    client_index,
    current_wave,
    priority,
    lora_request,
    cache_salt,
    trace_headers,
    cached_parent_by_id=None,
    require_live_parents: bool = False,
    enqueue_child_batch: bool = False,
    wake_child_batch: bool = True,
    direct_add_request: bool = False,
) -> bool:
    from vllm.v1.request import Request
    from vllm.v1.utils import ConstantList

    child_sampling_params = _clone_grouped_sampling_params(sampling_params)
    if eos_token_id is not None:
        child_sampling_params.update_from_generation_config({}, eos_token_id)

    shared_fields: dict[str, Any] | None = None
    now = time.time()
    has_block_hasher = self.request_block_hasher is not None
    block_size = getattr(self, "beamproc_block_size", None)
    created_requests = []
    parent_work_by_id = {}
    if cached_parent_by_id is None:
        unique_parent_ids = list(dict.fromkeys(parent_ids))
        with self.beam_cache_lock:
            cached_parent_by_id = {
                parent_id: self.beam_cache.get(parent_id) for parent_id in unique_parent_ids
            }

    for parent_id, child_id, token_id in zip(parent_ids, child_ids, token_ids):
        cached = cached_parent_by_id.get(parent_id)

        if cached is None:
            if require_live_parents:
                logger.info(
                    "BEAM_FORK skipped: parent no longer live parent=%s client_index=%s",
                    parent_id,
                    client_index,
                )
                return False
            logger.error("BEAM_FORK: parent %s not in cache", parent_id)
            self.output_queue.put_nowait(
                (
                    client_index,
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
            continue

        parent_work = parent_work_by_id.get(parent_id)
        if parent_work is None:
            cached_req = _beamproc_cached_request(cached)
            if cached_req is None:
                logger.error("BEAM_FORK: parent %s has no live request", parent_id)
                continue
            cached_block_hashes = (
                cached.get("_block_hash_count", len(cached["block_hashes"]))
                if isinstance(cached, dict)
                else cached[1]
            )
            child_beam_context = _beamproc_child_context_from_parent(cached_req)
            child_trace_headers = _beamproc_trace_headers_with_context(
                trace_headers or getattr(cached_req, "trace_headers", None),
                child_beam_context,
            )
            parent_work = (
                cached_req,
                cached_block_hashes,
                child_beam_context,
                child_trace_headers,
            )
            parent_work_by_id[parent_id] = parent_work

        (
            cached_req,
            cached_block_hashes,
            child_beam_context,
            child_trace_headers,
        ) = parent_work
        child_token_ids = cached_req.prompt_token_ids + [token_id]
        if shared_fields is None:
            req = Request(
                request_id=child_id,
                prompt_token_ids=child_token_ids,
                sampling_params=child_sampling_params,
                pooling_params=None,
                client_index=client_index,
                arrival_time=now,
                lora_request=(lora_request or cached_req.lora_request),
                cache_salt=cache_salt or cached_req.cache_salt,
                priority=priority,
                trace_headers=child_trace_headers,
                prompt_embeds=cached_req.prompt_embeds,
                mm_features=cached_req.mm_features,
                block_hasher=None,
            )
            shared_fields = {k: v for k, v in req.__dict__.items() if k not in _PER_CHILD_FIELDS}
        else:
            req = object.__new__(Request)
            child_dict = shared_fields.copy()
            all_token_ids = list(child_token_ids)
            output_token_ids: list[int] = []
            child_dict["request_id"] = child_id
            child_dict["prompt_token_ids"] = child_token_ids
            child_dict["num_prompt_tokens"] = len(child_token_ids)
            child_dict["_all_token_ids"] = all_token_ids
            child_dict["_output_token_ids"] = output_token_ids
            child_dict["output_token_ids"] = ConstantList(output_token_ids)
            child_dict["all_token_ids"] = ConstantList(all_token_ids)
            child_dict["events"] = []
            child_dict["spec_token_ids"] = []
            req.__dict__ = child_dict

        if isinstance(cached_block_hashes, int):
            req.block_hashes = cached_req.block_hashes[:cached_block_hashes]
        else:
            # Accept list snapshots left by an older cache entry during tests
            # or a rolling process transition.
            req.block_hashes = list(cached_block_hashes)
        if has_block_hasher:
            req.get_hash_new_full_blocks = partial(self.request_block_hasher, req)
            if not isinstance(block_size, int) or (
                (len(req.block_hashes) + 1) * block_size <= len(child_token_ids)
            ):
                req.block_hashes.extend(req.get_hash_new_full_blocks())
        else:
            req.get_hash_new_full_blocks = None
        created_requests.append(req)

    _publish_beam_children(
        self,
        created_requests,
        set(parent_ids) | set(abort_ids),
    )

    if enqueue_child_batch and created_requests:
        pb_queue = getattr(self, "pb_queue", None)
        if pb_queue is not None:
            _publish_beamproc_item(
                self,
                (
                    _BEAMPROC_CHILD_BATCH_REQUEST,
                    created_requests,
                    current_wave,
                ),
                "plan queue",
            )
        else:
            # Fallback for tests and configurations without the BeamProc I/O
            # threads.
            child_batch = [(req, current_wave) for req in created_requests]
            self.input_queue.put_nowait((_BEAMPROC_CHILD_BATCH_REQUEST, child_batch))
        if wake_child_batch:
            _wake_beamproc_engine_loop(self)
    elif direct_add_request:
        for req in created_requests:
            self.add_request(req, current_wave)
    else:
        for req in created_requests:
            self.input_queue.put_nowait((EngineCoreRequestType.ADD, (req, current_wave)))
    return True


def _handle_beam_fork(
    self,
    fork_req,
    *,
    require_live_parents: bool = False,
    enqueue_child_batch: bool = False,
    wake_child_batch: bool = True,
    enqueue_state_update: bool = True,
) -> bool:
    """Create child Requests from cached parent state.

    Uses a template-clone optimisation: the first child is built with the
    normal ``Request()`` constructor, then a lean shared-fields dict is
    snapshotted (excluding per-child fields).  Subsequent children are
    created via ``object.__new__(Request)`` + shared dict copy + per-child
    field insertion, avoiding both the full constructor cost and the
    copy-then-overwrite overhead of a full ``__dict__.copy()``.

    Thread-safe: uses beam_cache_lock to prevent data races.
    """
    if enqueue_state_update:
        _enqueue_beamproc_state_update(self, fork_req)

    cached_parent_by_id = None
    if require_live_parents and fork_req.parent_ids:
        unique_parent_ids = list(dict.fromkeys(fork_req.parent_ids))
        with self.beam_cache_lock:
            cached_parent_by_id = {
                parent_id: self.beam_cache.get(parent_id) for parent_id in unique_parent_ids
            }
        missing_parent_ids = [
            parent_id for parent_id, cached in cached_parent_by_id.items() if cached is None
        ]
        if missing_parent_ids:
            logger.warning(
                "BEAM_FORK skipped: parent no longer live missing=%s parents=%s client_index=%s",
                missing_parent_ids[:4],
                len(unique_parent_ids),
                getattr(fork_req, "client_index", None),
            )
            return False

    return _create_beam_fork_children(
        self,
        parent_ids=fork_req.parent_ids,
        child_ids=fork_req.child_ids,
        token_ids=fork_req.token_ids,
        abort_ids=fork_req.abort_ids,
        sampling_params=fork_req.sampling_params,
        eos_token_id=fork_req.eos_token_id,
        client_index=fork_req.client_index,
        current_wave=fork_req.current_wave,
        priority=fork_req.priority,
        lora_request=fork_req.lora_request,
        cache_salt=fork_req.cache_salt,
        trace_headers=fork_req.trace_headers,
        cached_parent_by_id=cached_parent_by_id,
        require_live_parents=require_live_parents,
        enqueue_child_batch=enqueue_child_batch,
        wake_child_batch=wake_child_batch,
    )


def _beamproc_should_apply_plan() -> bool:
    return is_truthy_env(_BEAMPROC_ENABLED_ENV)


def _beamproc_cached_request_for_plan(self, plan: dict[str, Any]):
    for parent_id in plan.get("parent_ids", ()) or ():
        with self.beam_cache_lock:
            cached = self.beam_cache.get(parent_id)
        if cached is not None:
            return _beamproc_cached_request(cached)

    for parent_id in plan.get("abort_ids", ()) or ():
        with self.beam_cache_lock:
            cached = self.beam_cache.get(parent_id)
        if cached is not None:
            return _beamproc_cached_request(cached)

    return None


def _beamproc_plan_to_fork_request(self, plan: dict[str, Any]):
    from vllm_gr.v1.engine.types import BeamForkRequest

    parent_ids = [str(parent_id) for parent_id in plan.get("parent_ids", ()) or ()]
    child_ids = [str(child_id) for child_id in plan.get("child_ids", ()) or ()]
    token_ids = [int(token_id) for token_id in plan.get("token_ids", ()) or ()]
    abort_ids = [str(abort_id) for abort_id in plan.get("abort_ids", ()) or ()]
    child_cum_logprobs = [float(score) for score in plan.get("scores", ()) or ()]
    child_flat_indices = [int(index) for index in plan.get("flat_indices", ()) or ()]

    if not parent_ids and not abort_ids:
        return None
    if len(parent_ids) != len(child_ids) or len(parent_ids) != len(token_ids):
        logger.warning(
            "BeamProc plan cannot be applied: mismatched fork arrays root=%s "
            "parents=%s children=%s tokens=%s",
            plan.get("root_request_id"),
            len(parent_ids),
            len(child_ids),
            len(token_ids),
        )
        return None

    cached_req = None
    missing_parent_ids: list[str] = []
    if parent_ids:
        unique_parent_ids = list(dict.fromkeys(parent_ids))
        with self.beam_cache_lock:
            for parent_id in unique_parent_ids:
                cached = self.beam_cache.get(parent_id)
                if cached is None:
                    missing_parent_ids.append(parent_id)
                elif cached_req is None:
                    cached_req = _beamproc_cached_request(cached)
        if missing_parent_ids:
            logger.info(
                "BeamProc plan skipped: parent no longer live root=%s "
                "correlation_id=%s missing=%s total_parents=%s",
                plan.get("root_request_id"),
                plan.get("correlation_id"),
                missing_parent_ids[:4],
                len(unique_parent_ids),
            )
            return None
    else:
        cached_req = _beamproc_cached_request_for_plan(self, plan)

    if cached_req is None:
        logger.warning(
            "BeamProc plan cannot be applied: no cached parent root=%s parents=%s aborts=%s",
            plan.get("root_request_id"),
            parent_ids[:4],
            abort_ids[:4],
        )
        return None

    return BeamForkRequest(
        parent_ids=parent_ids,
        child_ids=child_ids,
        token_ids=token_ids,
        abort_ids=abort_ids,
        sampling_params=cached_req.sampling_params,
        eos_token_id=_request_eos_token_id(cached_req),
        client_index=int(plan.get("client_index") or getattr(cached_req, "client_index", 0)),
        current_wave=int(getattr(cached_req, "current_wave", 0) or 0),
        priority=int(getattr(cached_req, "priority", 0) or 0),
        data_parallel_rank=getattr(cached_req, "data_parallel_rank", None),
        beam_width=int(plan.get("beam_width") or len(parent_ids)),
        child_cum_logprobs=child_cum_logprobs,
        child_flat_indices=child_flat_indices,
        lora_request=getattr(cached_req, "lora_request", None),
        cache_salt=getattr(cached_req, "cache_salt", None),
        trace_headers=getattr(cached_req, "trace_headers", None),
    )


def _beamproc_grouped_parent_state(self, parent_id: str) -> dict[str, Any] | None:
    states = getattr(self, "beamproc_grouped_parent_states", None)
    if states is not None:
        prepared_state = states.get(parent_id)
        if prepared_state is not None:
            return prepared_state

    with self.beam_cache_lock:
        cached = self.beam_cache.get(parent_id)
    if cached is None:
        return None
    if isinstance(cached, dict):
        prepared_state = cached.get("_grouped_parent_state")
        if prepared_state is not None:
            return prepared_state
    cached_req = _beamproc_cached_request(cached)
    if cached_req is None:
        return None

    trace_context = _beamproc_request_context(cached_req) or {}
    full_token_ids = list(getattr(cached_req, "prompt_token_ids", ()) or ())
    prompt_token_count = _beamproc_model_prefix_token_count(
        cached_req, trace_context, len(full_token_ids)
    )
    api_prompt_token_count = trace_context.get("prompt_token_count")
    if not isinstance(api_prompt_token_count, int):
        api_prompt_token_count = prompt_token_count
    api_prompt_token_count = max(0, min(api_prompt_token_count, len(full_token_ids)))

    if isinstance(cached, dict):
        block_hashes = list(cached.get("block_hashes", ()))
    else:
        block_hash_count = int(cached[1])
        block_hashes = list(cached_req.block_hashes[:block_hash_count])

    return {
        "prefix_token_ids": full_token_ids[:prompt_token_count],
        "generated_token_ids": tuple(full_token_ids[prompt_token_count:]),
        "response_token_ids": tuple(full_token_ids[api_prompt_token_count:]),
        "block_hashes": block_hashes,
        "request": cached_req,
        "trace_context": trace_context,
        "trace_headers": _beamproc_shared_trace_headers(getattr(cached_req, "trace_headers", None)),
        "step_index": int(trace_context.get("step_index") or 0),
    }


def _apply_beamproc_grouped_plan(self, plan: dict[str, Any]) -> bool:
    """Apply a Rust plan through upstream's single grouped beam request."""
    parent_ids = [str(value) for value in plan.get("parent_ids", ()) or ()]
    child_ids = [str(value) for value in plan.get("child_ids", ()) or ()]
    token_ids = [int(value) for value in plan.get("token_ids", ()) or ()]
    beam_tokens_flat = plan.get("beam_tokens_flat")
    beam_token_stride = int(plan.get("beam_token_stride") or 0)
    if beam_tokens_flat is not None:
        beam_tokens_flat = np.asarray(beam_tokens_flat, dtype="<i4")
        beam_history_count = (
            beam_tokens_flat.size // beam_token_stride if beam_token_stride > 0 else 0
        )
        beam_tokens = None
        decode_steps = beam_token_stride
    else:
        beam_tokens = [
            [int(token_id) for token_id in tokens] for tokens in plan.get("beam_tokens", ()) or ()
        ]
        beam_history_count = len(beam_tokens)
        decode_steps = len(beam_tokens[0]) if beam_tokens else 0
    abort_ids = [str(value) for value in plan.get("abort_ids", ()) or ()]
    if not parent_ids or not (
        len(parent_ids) == len(child_ids) == len(token_ids) == beam_history_count
    ):
        return False
    if beam_tokens is not None and any(len(tokens) != decode_steps for tokens in beam_tokens):
        logger.warning(
            "BeamProc grouped plan has ragged beam histories root=%s",
            plan.get("root_request_id"),
        )
        return False
    root_request_id = str(plan.get("root_request_id") or _beamproc_root_request_id(parent_ids[0]))
    parent_state_map = getattr(self, "beamproc_grouped_parent_states", None)
    root_state = parent_state_map.get(root_request_id) if parent_state_map is not None else None
    current_by_root = getattr(self, "beamproc_grouped_request_by_root", None)
    current_grouped_request_id = (
        current_by_root.get(root_request_id) if current_by_root is not None else None
    )
    active_by_root = getattr(self, "beamproc_grouped_active_ids_by_root", None)
    expected_parent_ids = (
        active_by_root.get(root_request_id) if active_by_root is not None else None
    )
    if expected_parent_ids is not None and any(
        parent_id not in expected_parent_ids for parent_id in parent_ids
    ):
        logger.info(
            "BeamProc grouped plan skipped stale parents root=%s",
            root_request_id,
        )
        return False

    if current_grouped_request_id is not None:
        with self.beam_cache_lock:
            cached_source = self.beam_cache.get(current_grouped_request_id)
        if cached_source is None:
            return False
        if root_state is not None:
            source_request = root_state["request"]
            prefix_token_ids = root_state["prefix_token_ids"]
            prefix_block_hashes = root_state["block_hashes"]
            current_request = _beamproc_cached_request(cached_source)
            if current_request is None:
                return False
            trace_context = root_state["trace_context"]
            response_prefix_by_root = getattr(
                self,
                "beamproc_grouped_response_prefix_by_root",
                {},
            )
            response_prefix_token_count = int(response_prefix_by_root.get(root_request_id, 0))
            prefix_len = len(prefix_token_ids)
        else:
            source_request = _beamproc_cached_request(cached_source)
            if source_request is None:
                return False
            prefix_len = int(getattr(source_request, "prefix_len", 0) or 0)
            prefix_token_ids = list(source_request.prompt_token_ids[:prefix_len])
            if isinstance(cached_source, dict):
                prefix_block_hashes = list(cached_source.get("block_hashes", ()))
            else:
                prefix_block_hashes = list(source_request.block_hashes[: int(cached_source[1])])
            trace_context = (
                _beamproc_context_from_trace_headers(getattr(source_request, "trace_headers", None))
                or {}
            )
            response_prefix_by_root = getattr(
                self,
                "beamproc_grouped_response_prefix_by_root",
                {},
            )
            response_prefix_token_count = int(response_prefix_by_root.get(root_request_id, 0))
    else:
        first_state = _beamproc_grouped_parent_state(self, parent_ids[0])
        if first_state is None:
            return False
        for parent_id in dict.fromkeys(parent_ids[1:]):
            if parent_id == parent_ids[0]:
                continue
            parent_state = _beamproc_grouped_parent_state(self, parent_id)
            if parent_state is None:
                return False
            other_prefix = parent_state["prefix_token_ids"]
            first_prefix = first_state["prefix_token_ids"]
            if other_prefix is not first_prefix and other_prefix != first_prefix:
                logger.warning(
                    "BeamProc grouped plan has distinct parent prefixes root=%s",
                    root_request_id,
                )
                return False
        if parent_state_map is None:
            parent_state_map = {}
            self.beamproc_grouped_parent_states = parent_state_map
        root_state = parent_state_map.setdefault(root_request_id, first_state)
        source_request = root_state["request"]
        prefix_token_ids = root_state["prefix_token_ids"]
        prefix_block_hashes = root_state["block_hashes"]
        trace_context = root_state["trace_context"]
        response_prefix_token_count = len(root_state["response_token_ids"]) - len(
            root_state["generated_token_ids"]
        )
        prefix_len = len(prefix_token_ids)

    if (
        response_prefix_token_count < 0
        or (beam_tokens_flat is not None and decode_steps < response_prefix_token_count)
        or (
            beam_tokens is not None
            and any(len(tokens) < response_prefix_token_count for tokens in beam_tokens)
        )
    ):
        logger.warning(
            "BeamProc grouped plan has invalid response prefix root=%s offset=%s",
            plan.get("root_request_id"),
            response_prefix_token_count,
        )
        return False
    flat_generated_beam_tokens = None
    if beam_tokens_flat is not None:
        response_matrix = beam_tokens_flat.reshape(
            len(child_ids),
            decode_steps,
        )
        if not np.array_equal(
            response_matrix[:, -1],
            np.asarray(token_ids, dtype=np.int32),
        ):
            logger.warning(
                "BeamProc grouped plan history/token mismatch root=%s",
                plan.get("root_request_id"),
            )
            return False
        generated_decode_steps = decode_steps - response_prefix_token_count
        flat_generated_beam_tokens = (
            np.ascontiguousarray(response_matrix[:, response_prefix_token_count:])
            .reshape(-1)
            .tolist()
        )
        generated_beam_tokens = None
    else:
        generated_decode_steps = decode_steps - response_prefix_token_count
        generated_beam_tokens = [tokens[response_prefix_token_count:] for tokens in beam_tokens]
        for child_id, token_id, response_tokens in zip(
            child_ids,
            token_ids,
            beam_tokens,
        ):
            if not response_tokens or response_tokens[-1] != token_id:
                logger.warning(
                    "BeamProc grouped plan history/token mismatch root=%s child=%s",
                    plan.get("root_request_id"),
                    child_id,
                )
                return False
    child_trace_context = dict(trace_context)
    child_trace_context["root_request_id"] = root_request_id
    step_index = (
        int(root_state["step_index"]) + 1
        if root_state is not None
        else int(child_trace_context.get("step_index") or 0) + 1
    )
    child_trace_context["step_index"] = step_index
    total_decode_steps = child_trace_context.get("total_decode_steps")
    child_trace_context["is_final_step"] = bool(
        isinstance(total_decode_steps, int) and step_index >= total_decode_steps - 1
    )
    grouped_request_id = f"{root_request_id}-beamgroup-{step_index}"
    sampling_params = copy.copy(source_request.sampling_params)
    sampling_params.n = 1
    if root_state is not None:
        grouped_trace_headers = root_state["trace_headers"]
    else:
        grouped_trace_context = dict(child_trace_context)
        grouped_trace_context["beam_prefix_token_count"] = len(prefix_token_ids)
        grouped_trace_headers = _beamproc_trace_headers_with_context(
            getattr(source_request, "trace_headers", None),
            grouped_trace_context,
        )
    req = _build_grouped_beam_request(
        self,
        request_id=grouped_request_id,
        prefix_token_ids=prefix_token_ids,
        prefix_block_hashes=prefix_block_hashes,
        beam_tokens=generated_beam_tokens,
        flat_beam_tokens=flat_generated_beam_tokens,
        beam_decode_steps=generated_decode_steps,
        beam_width=len(child_ids),
        sampling_params=sampling_params,
        eos_token_id=_request_eos_token_id(source_request),
        client_index=int(plan.get("client_index") or getattr(source_request, "client_index", 0)),
        lora_request=getattr(source_request, "lora_request", None),
        cache_salt=getattr(source_request, "cache_salt", None),
        priority=int(getattr(source_request, "priority", 0) or 0),
        trace_headers=grouped_trace_headers,
        prompt_embeds=getattr(source_request, "prompt_embeds", None),
        mm_features=getattr(source_request, "mm_features", None),
        prefix_len=prefix_len,
        timings=None,
    )
    req._beamproc_trace_context = child_trace_context
    req._beamproc_prefix_token_count = prefix_len
    if parent_state_map is not None:
        for parent_id in set(parent_ids) | set(abort_ids):
            if parent_id != root_request_id:
                parent_state_map.pop(parent_id, None)
    if root_state is not None:
        root_state["step_index"] = step_index

    output_states = getattr(self, "beamproc_grouped_output_states", None)
    if output_states is None:
        output_states = {}
        self.beamproc_grouped_output_states = output_states
    output_states[grouped_request_id] = {
        "root_request_id": root_request_id,
        "child_ids": list(child_ids),
        "trace_context": child_trace_context,
        "is_final_step": bool(child_trace_context.get("is_final_step")),
    }

    if current_by_root is None:
        current_by_root = {}
        self.beamproc_grouped_request_by_root = current_by_root
    old_grouped_request_id = current_grouped_request_id
    current_by_root[root_request_id] = grouped_request_id
    if active_by_root is None:
        active_by_root = {}
        self.beamproc_grouped_active_ids_by_root = active_by_root
    active_by_root[root_request_id] = frozenset(child_ids)
    response_prefix_by_root = getattr(
        self,
        "beamproc_grouped_response_prefix_by_root",
        None,
    )
    if response_prefix_by_root is None:
        response_prefix_by_root = {}
        self.beamproc_grouped_response_prefix_by_root = response_prefix_by_root
    response_prefix_by_root[root_request_id] = response_prefix_token_count
    with self.beam_cache_lock:
        if old_grouped_request_id is not None:
            self.beam_cache.pop(old_grouped_request_id, None)
        for parent_id in set(parent_ids) | set(abort_ids):
            self.beam_cache.pop(parent_id, None)
        self.beam_cache[grouped_request_id] = (req, len(req.block_hashes))
    pb_queue = getattr(self, "pb_queue", None)
    if pb_queue is not None:
        _publish_beamproc_item(
            self,
            (_BEAMPROC_CHILD_BATCH_REQUEST, [req], getattr(source_request, "current_wave", 0)),
            "plan queue",
        )
    else:
        self.input_queue.put_nowait(
            (EngineCoreRequestType.ADD, (req, getattr(source_request, "current_wave", 0)))
        )
    return True


def _apply_beamproc_plan(self, plan: dict[str, Any]) -> bool:
    if _apply_beamproc_grouped_plan(self, plan):
        return True

    parent_ids = [str(parent_id) for parent_id in plan.get("parent_ids", ()) or ()]
    child_ids = [str(child_id) for child_id in plan.get("child_ids", ()) or ()]
    token_ids = [int(token_id) for token_id in plan.get("token_ids", ()) or ()]
    abort_ids = [str(abort_id) for abort_id in plan.get("abort_ids", ()) or ()]

    if not parent_ids and not abort_ids:
        return False
    if len(parent_ids) != len(child_ids) or len(parent_ids) != len(token_ids):
        logger.warning(
            "BeamProc plan cannot be applied: mismatched fork arrays root=%s "
            "parents=%s children=%s tokens=%s",
            plan.get("root_request_id"),
            len(parent_ids),
            len(child_ids),
            len(token_ids),
        )
        return False

    cached_req = None
    cached_parent_by_id = None
    if parent_ids:
        unique_parent_ids = list(dict.fromkeys(parent_ids))
        missing_parent_ids: list[str] = []
        with self.beam_cache_lock:
            cached_parent_by_id = {
                parent_id: self.beam_cache.get(parent_id) for parent_id in unique_parent_ids
            }
        for parent_id, cached in cached_parent_by_id.items():
            if cached is None:
                missing_parent_ids.append(parent_id)
            elif cached_req is None:
                cached_req = _beamproc_cached_request(cached)
        if missing_parent_ids:
            logger.info(
                "BeamProc plan skipped: parent no longer live root=%s "
                "correlation_id=%s missing=%s total_parents=%s",
                plan.get("root_request_id"),
                plan.get("correlation_id"),
                missing_parent_ids[:4],
                len(unique_parent_ids),
            )
            return False
    else:
        cached_req = _beamproc_cached_request_for_plan(self, plan)

    if cached_req is None:
        logger.warning(
            "BeamProc plan cannot be applied: no cached parent root=%s parents=%s aborts=%s",
            plan.get("root_request_id"),
            parent_ids[:4],
            abort_ids[:4],
        )
        return False

    if not _create_beam_fork_children(
        self,
        parent_ids=parent_ids,
        child_ids=child_ids,
        token_ids=token_ids,
        abort_ids=abort_ids,
        sampling_params=cached_req.sampling_params,
        eos_token_id=_request_eos_token_id(cached_req),
        client_index=int(plan.get("client_index") or getattr(cached_req, "client_index", 0)),
        current_wave=int(getattr(cached_req, "current_wave", 0) or 0),
        priority=int(getattr(cached_req, "priority", 0) or 0),
        lora_request=getattr(cached_req, "lora_request", None),
        cache_salt=getattr(cached_req, "cache_salt", None),
        trace_headers=getattr(cached_req, "trace_headers", None),
        cached_parent_by_id=cached_parent_by_id,
        require_live_parents=bool(parent_ids),
        enqueue_child_batch=True,
        wake_child_batch=False,
        direct_add_request=False,
    ):
        logger.debug(
            "Skipped BeamProc plan root=%s correlation_id=%s parents=%s children=%s "
            "tokens=%s aborts=%s finished=%s",
            plan.get("root_request_id"),
            plan.get("correlation_id"),
            len(parent_ids),
            len(child_ids),
            len(token_ids),
            len(abort_ids),
            plan.get("finished"),
        )
        return False
    logger.debug(
        "Applied BeamProc plan root=%s correlation_id=%s parents=%s children=%s "
        "tokens=%s aborts=%s finished=%s",
        plan.get("root_request_id"),
        plan.get("correlation_id"),
        len(parent_ids),
        len(child_ids),
        len(token_ids),
        len(abort_ids),
        plan.get("finished"),
    )
    return True


def _adjust_beamproc_pending_plans(self, delta: int) -> int:
    pending = max(
        0,
        int(getattr(self, "beamproc_pending_plan_count", 0)) + int(delta),
    )
    self.beamproc_pending_plan_count = pending
    scheduler = getattr(self, "scheduler", None)
    if scheduler is not None:
        scheduler._gr_beamproc_pending_plans = pending
    return pending


def _apply_beamproc_plan_and_track_pending(self, plan: dict[str, Any]) -> bool:
    applied = _apply_beamproc_plan(self, plan)
    _adjust_beamproc_pending_plans(self, -1)
    return applied


def _start_beamproc_io_thread(self) -> None:
    """Start the EngineCore-side BeamProc I/O owner threads."""
    beam_proc_tx_addr = getattr(
        self,
        "BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR",
        getattr(self, "BEAM_PROC_ENGINECORE_ZMQ_ADDR", None),
    )
    beam_proc_rx_addr = getattr(
        self,
        "BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR",
        getattr(self, "BEAM_PROC_ENGINECORE_ZMQ_ADDR", None),
    )
    if not beam_proc_tx_addr:
        self.beamproc_outbound_queue = None
        self.pb_queue = None
        self.beamproc_wake_lock = None
        self.beamproc_wake_pending = False
        self.beamproc_io_thread = None
        self.beamproc_io_bp_tx_thread = None
        self.beamproc_io_bp_rx_thread = None
        return
    if not beam_proc_rx_addr:
        beam_proc_rx_addr = beam_proc_tx_addr

    queue_capacity = _positive_env_int("VLLM_GR_BEAMPROC_QUEUE_CAPACITY", 64)
    self.beamproc_outbound_queue = None
    self.pb_queue = queue.Queue(maxsize=queue_capacity)
    self.beamproc_wake_lock = threading.Lock()
    self.beamproc_wake_pending = False
    self.beamproc_correlation_id = 0
    self.beamproc_pending_plan_count = 0
    self.scheduler._gr_beamproc_pending_plans = 0
    ce_to_bp = os.environ.get("VLLM_GR_CE_BP_TRANSPORT", "zmq")
    if ce_to_bp not in ("zmq", "sharedmem"):
        raise ValueError("VLLM_GR_CE_BP_TRANSPORT must be zmq or sharedmem")
    if ce_to_bp == "sharedmem":
        _start_beamproc_native_io(self, beam_proc_tx_addr, beam_proc_rx_addr, queue_capacity)
        return
    self.beamproc_outbound_queue = queue.Queue(maxsize=queue_capacity)
    self.beamproc_io_bp_tx_thread = threading.Thread(
        target=_beamproc_io_bp_tx_run,
        args=(self, beam_proc_tx_addr),
        daemon=True,
        name="beamproc-io-bp-tx",
    )
    self.beamproc_io_bp_rx_thread = threading.Thread(
        target=_beamproc_io_bp_rx_run,
        args=(self, beam_proc_rx_addr),
        daemon=True,
        name="beamproc-io-bp-rx",
    )
    self.beamproc_io_thread = self.beamproc_io_bp_tx_thread
    self.beamproc_io_bp_rx_thread.start()
    self.beamproc_io_bp_tx_thread.start()
    logger.info(
        "Started EngineCore BeamProc I/O threads tx_addr=%s rx_addr=%s",
        beam_proc_tx_addr,
        beam_proc_rx_addr,
    )


def _start_beamproc_native_io(self, tx_addr: str, rx_addr: str, capacity: int) -> None:
    from vllm_gr.v1.engine.native_ce_transport import NativeBridge

    self.beamproc_native_stop = threading.Event()
    self.beamproc_native_error = None
    self.beamproc_io_bp_tx_thread = None
    self.beamproc_io_bp_rx_thread = None
    self.beamproc_io_thread = None
    self.beamproc_native_bridge = NativeBridge(
        tx_addr,
        rx_addr,
        capacity,
        _positive_env_int("VLLM_GR_BEAMPROC_ZMQ_HWM", 64),
    )
    try:
        # Pull delivery stays entirely native until the scheduler drains it.
        self.beamproc_native_bridge.start_receive()
    except BaseException:
        _stop_beamproc_native_io(self)
        raise
    logger.info(
        "Started native Rust CE %s RX/TX with direct TX and scheduler-owned RX decoding",
        os.environ.get("VLLM_GR_CE_BP_TRANSPORT", "zmq"),
    )


def _start_beamproc_python_shm_io(self, tx_addr: str, rx_addr: str, capacity: int) -> None:
    from vllm_gr.v1.engine.native_ce_transport import NativeBridge

    self.beamproc_native_stop = threading.Event()
    self.beamproc_native_error = None
    self.beamproc_python_bridge = NativeBridge(
        tx_addr,
        rx_addr,
        capacity,
        _positive_env_int("VLLM_GR_BEAMPROC_ZMQ_HWM", 64),
        caller_owned=True,
    )
    self.beamproc_io_bp_tx_thread = threading.Thread(
        target=_beamproc_python_shm_run,
        args=(self, True),
        daemon=True,
        name="beamproc-io-bp-tx",
    )
    self.beamproc_io_bp_rx_thread = threading.Thread(
        target=_beamproc_python_shm_run,
        args=(self, False),
        daemon=True,
        name="beamproc-io-bp-rx",
    )
    self.beamproc_io_thread = self.beamproc_io_bp_tx_thread
    try:
        self.beamproc_io_bp_rx_thread.start()
        self.beamproc_io_bp_tx_thread.start()
    except BaseException:
        _stop_beamproc_native_io(self)
        raise
    logger.info("Started Python CE TX/RX threads with synchronous iceoryx2 ports")


def _beamproc_python_shm_run(self, transmit: bool) -> None:
    bridge = self.beamproc_python_bridge
    stop = self.beamproc_native_stop
    pending = deque()
    output_buffer_pool: list[dict[str, Any]] = []
    try:
        while not stop.is_set():
            if transmit:
                try:
                    message = self.beamproc_outbound_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                _send_beamproc_outbound_message(
                    self,
                    bridge,
                    message,
                    pending,
                    output_buffer_pool,
                )
            else:
                frames = bridge.receive(100)
                if frames is not None:
                    _process_beamproc_plan_frames(self, frames)
    except Exception as exc:
        if not stop.is_set():
            self.beamproc_native_error = exc
            stop.set()
            logger.exception(
                "EngineCore Python shared-memory %s failed", "TX" if transmit else "RX"
            )


def _stop_beamproc_native_io(self) -> None:
    python_bridge = getattr(self, "beamproc_python_bridge", None)
    if python_bridge is not None:
        self.beamproc_native_stop.set()
        python_bridge.close()
        for worker in (self.beamproc_io_bp_tx_thread, self.beamproc_io_bp_rx_thread):
            if worker is not None and worker.ident is not None:
                worker.join()
        self.beamproc_python_bridge = None
    bridge = getattr(self, "beamproc_native_bridge", None)
    if bridge is None:
        return
    self.beamproc_native_stop.set()
    # Native close wakes blocked callers and joins socket owners without the GIL.
    bridge.close()
    self.beamproc_native_bridge = None


def _check_beamproc_native_io(self) -> None:
    bridge = getattr(self, "beamproc_native_bridge", None) or getattr(
        self, "beamproc_python_bridge", None
    )
    if bridge is not None:
        error = getattr(self, "beamproc_native_error", None)
        if error is not None:
            raise RuntimeError("CE transport adapter failed") from error
        bridge.check()


def _beamproc_io_bp_tx_run(self, beam_proc_tx_addr: str) -> None:
    """Run the EngineCore-to-BeamProc TX thread."""
    import zmq
    from vllm.utils.network_utils import make_zmq_socket

    context = zmq.Context()
    socket = make_zmq_socket(
        context,
        beam_proc_tx_addr,
        zmq.PUSH,
        bind=False,
        linger=0,
    )
    _configure_beamproc_tx_socket(socket)

    outbound_queue = getattr(self, "beamproc_outbound_queue", None)
    if outbound_queue is None:
        socket.close(linger=0)
        context.term()
        return
    pending = deque()
    output_buffer_pool: list[dict[str, Any]] = []

    try:
        while True:
            message = outbound_queue.get()
            _free_done_beamproc_sends(pending, output_buffer_pool)
            _send_beamproc_outbound_message(
                self,
                socket,
                message,
                pending,
                output_buffer_pool,
            )
    except Exception:
        logger.exception("EngineCore BeamProc TX thread failed.")
    finally:
        socket.close(linger=0)
        pending.clear()
        context.term()


def _beamproc_io_bp_rx_run(self, beam_proc_rx_addr: str) -> None:
    """Run the BeamProc-to-EngineCore RX thread."""
    import zmq
    from vllm.utils.network_utils import make_zmq_socket

    context = zmq.Context()
    socket = make_zmq_socket(
        context,
        beam_proc_rx_addr,
        zmq.PULL,
        bind=False,
        linger=0,
    )
    socket.setsockopt(
        zmq.RCVHWM,
        _positive_env_int("VLLM_GR_BEAMPROC_ZMQ_HWM", 64),
    )

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    try:
        while True:
            for readable_socket, events in poller.poll():
                if readable_socket is socket and events & zmq.POLLIN:
                    _receive_beamproc_plan(self, socket)
    except Exception:
        logger.exception("EngineCore BeamProc RX thread failed.")
    finally:
        try:
            poller.unregister(socket)
        except (KeyError, ValueError, zmq.ZMQError):
            pass
        socket.close(linger=0)
        context.term()


def _release_beamproc_tx_buffer(refs, output_buffer_pool) -> None:
    if (
        output_buffer_pool is not None
        and isinstance(refs, dict)
        and refs.get("_beamproc_tx_buffer_set")
    ):
        output_buffer_pool.append(refs)


def _free_done_beamproc_sends(pending, output_buffer_pool=None) -> None:
    while pending and pending[-1][0].done:
        entry = pending.pop()
        refs = entry[2] if len(entry) > 2 else None
        _release_beamproc_tx_buffer(refs, output_buffer_pool)


def _configure_beamproc_tx_socket(socket) -> None:
    import zmq

    socket.setsockopt(
        zmq.SNDHWM,
        _positive_env_int("VLLM_GR_BEAMPROC_ZMQ_HWM", 64),
    )
    socket.setsockopt(zmq.SNDBUF, _BEAMPROC_ZMQ_BUFFER_BYTES)
    socket.setsockopt(zmq.SNDTIMEO, 0)


def _send_beamproc_outbound_message(
    self,
    socket,
    message,
    pending,
    output_buffer_pool=None,
) -> None:
    if len(message) == 3:
        message_type, payload, _ = message
    else:
        message_type, payload = message
    _send_beamproc_message(
        self,
        socket,
        (message_type, payload),
        pending,
        output_buffer_pool,
    )


def _send_beamproc_message(self, socket, message, pending, output_buffer_pool=None) -> None:
    message_type, payload = message
    if message_type == _BEAMPROC_MSG_ENGINE_OUTPUT_BATCH:
        _send_beamproc_engine_output_batches(
            self,
            socket,
            payload,
            pending,
            output_buffer_pool,
        )
        return

    frames = [_message_type_bytes(message_type)]
    if payload is not None:
        frames.append(msgspec.msgpack.encode(payload))
    _send_beamproc_frames(socket, frames, pending)


def _send_beamproc_engine_output_batches(
    self,
    socket,
    payload,
    pending,
    output_buffer_pool=None,
) -> None:
    batches = payload.get("batches")
    if batches is None:
        outputs = payload.get("outputs")
        if not outputs:
            return
        batches = _prepare_beamproc_engine_output_batches(
            self,
            outputs,
            int(payload.get("engine_index", self.engine_index)),
            int(payload.get("correlation_base", 0)),
            payload.get("output_contexts"),
            output_buffer_pool,
        )
        payload["batches"] = batches
        payload["outputs"] = None
        payload["output_contexts"] = None
    if not batches:
        return
    for batch in batches:
        _send_beamproc_frames(
            socket,
            batch["frames"],
            pending,
            refs=batch.get("buffer_set"),
            output_buffer_pool=output_buffer_pool,
        )
        logger.debug(
            "Sent BeamProc engine output batch client_index=%s correlation_base=%s outputs=%s",
            batch["client_index"],
            batch["correlation_base"],
            len(batch["batch_outputs"]),
        )


def _send_beamproc_frames(
    socket,
    frames,
    pending,
    *,
    refs=None,
    output_buffer_pool=None,
) -> None:
    import zmq

    zmq_frames = [_as_zmq_frame(frame) for frame in frames]
    # Rust copies all submitted buffers before returning; pooled Python arrays
    # are immediately reusable, independently of ZMQ's asynchronous completion.
    if not isinstance(socket, zmq.Socket):
        from vllm_gr.v1.engine.native_ce_transport import NativeBridge

        if isinstance(socket, NativeBridge):
            socket.send(zmq_frames)
            _release_beamproc_tx_buffer(refs, output_buffer_pool)
            return
    while True:
        try:
            tracker = socket.send_multipart(
                zmq_frames,
                flags=zmq.DONTWAIT,
                copy=False,
                track=True,
            )
            break
        except zmq.Again:
            socket.poll(_BEAMPROC_ZMQ_BACKPRESSURE_POLL_MS, zmq.POLLOUT)
    if not tracker.done:
        pending.appendleft((tracker, zmq_frames, refs))
    else:
        _release_beamproc_tx_buffer(refs, output_buffer_pool)


def _message_type_bytes(message_type) -> bytes:
    if isinstance(message_type, bytes):
        return message_type
    if isinstance(message_type, bytearray):
        return bytes(message_type)
    if isinstance(message_type, memoryview):
        return message_type.tobytes()
    if isinstance(message_type, str):
        return message_type.encode()
    raise TypeError(f"BeamProc message type must be bytes or str, got {type(message_type)!r}")


def _beamproc_frame_nbytes(frame) -> int:
    if isinstance(frame, memoryview):
        return frame.nbytes
    if hasattr(frame, "nbytes"):
        return int(frame.nbytes)
    return len(frame)


def _decode_beamproc_plan_v2_frames(
    payload: dict[str, Any],
    frame_bytes,
    *,
    materialize_auxiliary: bool = True,
) -> bool:
    if len(frame_bytes) < 5:
        logger.warning(
            "BeamProc BEAM_PLAN v2 missing binary frames frames=%s",
            len(frame_bytes),
        )
        return False

    token_ids = np.frombuffer(frame_bytes[2], dtype="<i4")
    scores_nbytes = _beamproc_frame_nbytes(frame_bytes[3])
    flat_indices_nbytes = _beamproc_frame_nbytes(frame_bytes[4])
    expected_token_ids = int(payload.get("token_ids_len") or 0)
    expected_scores = int(payload.get("scores_len") or 0)
    expected_flat_indices = int(payload.get("flat_indices_len") or 0)
    if (
        token_ids.size != expected_token_ids
        or scores_nbytes != expected_scores * _BEAMPROC_F32_ITEMSIZE
        or flat_indices_nbytes != expected_flat_indices * _BEAMPROC_U64_ITEMSIZE
    ):
        logger.warning(
            "BeamProc BEAM_PLAN v2 binary frame length mismatch "
            "token_ids=%s/%s scores=%s/%s flat_indices=%s/%s",
            token_ids.size,
            expected_token_ids,
            scores_nbytes // _BEAMPROC_F32_ITEMSIZE,
            expected_scores,
            flat_indices_nbytes // _BEAMPROC_U64_ITEMSIZE,
            expected_flat_indices,
        )
        return False

    payload["token_ids"] = token_ids.tolist()
    if materialize_auxiliary:
        payload["scores"] = np.frombuffer(frame_bytes[3], dtype="<f4").tolist()
        payload["flat_indices"] = np.frombuffer(frame_bytes[4], dtype="<u8").tolist()
    return True


def _decode_beamproc_plan_v3_frames(
    payload: dict[str, Any],
    frame_bytes,
    *,
    materialize_auxiliary: bool = True,
) -> bool:
    if len(frame_bytes) < 7:
        logger.warning(
            "BeamProc BEAM_PLAN v3 missing binary frames frames=%s",
            len(frame_bytes),
        )
        return False
    if not _decode_beamproc_plan_v2_frames(
        payload,
        frame_bytes,
        materialize_auxiliary=materialize_auxiliary,
    ):
        return False

    beam_tokens = np.frombuffer(frame_bytes[5], dtype="<i4")
    beam_token_offsets = np.frombuffer(frame_bytes[6], dtype="<u8")
    expected_beam_tokens = int(payload.get("beam_tokens_len") or 0)
    expected_offsets = int(payload.get("beam_token_offsets_len") or 0)
    if (
        beam_tokens.size != expected_beam_tokens
        or beam_token_offsets.size != expected_offsets
        or expected_offsets != int(payload.get("beam_width") or 0) + 1
        or not beam_token_offsets.size
        or int(beam_token_offsets[0]) != 0
        or int(beam_token_offsets[-1]) != beam_tokens.size
        or np.any(beam_token_offsets[1:] < beam_token_offsets[:-1])
    ):
        logger.warning(
            "BeamProc BEAM_PLAN v3 history frame mismatch "
            "beam_tokens=%s/%s offsets=%s/%s beam_width=%s",
            beam_tokens.size,
            expected_beam_tokens,
            beam_token_offsets.size,
            expected_offsets,
            payload.get("beam_width"),
        )
        return False

    payload["beam_tokens"] = [
        beam_tokens[int(start) : int(end)].tolist()
        for start, end in zip(
            beam_token_offsets[:-1],
            beam_token_offsets[1:],
        )
    ]
    return True


def _decode_beamproc_plan_v4_frames(
    payload: dict[str, Any],
    frame_bytes,
    *,
    materialize_auxiliary: bool = True,
) -> bool:
    if len(frame_bytes) < 6:
        logger.warning(
            "BeamProc BEAM_PLAN v4 missing binary frames frames=%s",
            len(frame_bytes),
        )
        return False
    if not _decode_beamproc_plan_v2_frames(
        payload,
        frame_bytes,
        materialize_auxiliary=materialize_auxiliary,
    ):
        return False

    beam_tokens_flat = np.frombuffer(frame_bytes[5], dtype="<i4")
    expected_beam_tokens = int(payload.get("beam_tokens_len") or 0)
    beam_token_stride = int(payload.get("beam_token_stride") or 0)
    beam_width = int(payload.get("beam_width") or 0)
    if (
        beam_token_stride <= 0
        or beam_tokens_flat.size != expected_beam_tokens
        or expected_beam_tokens != beam_width * beam_token_stride
    ):
        logger.warning(
            "BeamProc BEAM_PLAN v4 history frame mismatch "
            "beam_tokens=%s/%s beam_width=%s stride=%s",
            beam_tokens_flat.size,
            expected_beam_tokens,
            beam_width,
            beam_token_stride,
        )
        return False

    payload["beam_tokens_flat"] = beam_tokens_flat
    return True


def _decode_beamproc_plan_v5_frames(payload: dict[str, Any], frame_bytes) -> bool:
    if len(frame_bytes) < 4:
        logger.warning(
            "BeamProc BEAM_PLAN v5 missing binary frames frames=%s",
            len(frame_bytes),
        )
        return False

    token_ids = np.frombuffer(frame_bytes[2], dtype="<i4")
    beam_tokens_flat = np.frombuffer(frame_bytes[3], dtype="<i4")
    expected_token_ids = int(payload.get("token_ids_len") or 0)
    expected_beam_tokens = int(payload.get("beam_tokens_len") or 0)
    beam_token_stride = int(payload.get("beam_token_stride") or 0)
    beam_width = int(payload.get("beam_width") or 0)
    if (
        token_ids.size != expected_token_ids
        or beam_token_stride <= 0
        or beam_tokens_flat.size != expected_beam_tokens
        or expected_beam_tokens != beam_width * beam_token_stride
    ):
        logger.warning(
            "BeamProc BEAM_PLAN v5 binary frame mismatch "
            "token_ids=%s/%s beam_tokens=%s/%s beam_width=%s stride=%s",
            token_ids.size,
            expected_token_ids,
            beam_tokens_flat.size,
            expected_beam_tokens,
            beam_width,
            beam_token_stride,
        )
        return False

    payload["token_ids"] = token_ids.tolist()
    payload["beam_tokens_flat"] = beam_tokens_flat
    return True


def _beamproc_payload_root_request_id(output: Any) -> Any:
    if isinstance(output, dict):
        return output.get("root_request_id")
    if isinstance(output, (list, tuple)) and len(output) > 12:
        return output[1]
    if isinstance(output, (list, tuple)) and len(output) > 0:
        return _beamproc_root_request_id(str(output[0]))
    return None


def _beamproc_payload_context(output: Any, contexts: list[Any] | None = None) -> Any:
    if isinstance(output, dict):
        return output.get("beam_context")
    if isinstance(output, (list, tuple)) and len(output) > 12:
        return output[12]
    if isinstance(output, (list, tuple)) and len(output) == 12 and contexts is not None:
        context_index = output[11]
        if context_index is None:
            return None
        try:
            return contexts[int(context_index)]
        except (IndexError, TypeError, ValueError):
            return None
    if isinstance(output, (list, tuple)) and len(output) > 10 and contexts is not None:
        context_index = output[10]
        if context_index is None:
            return None
        try:
            return contexts[int(context_index)]
        except (IndexError, TypeError, ValueError):
            return None
    return None


def _as_zmq_frame(frame):
    if isinstance(frame, bytes):
        return frame
    if isinstance(frame, bytearray):
        return frame
    if isinstance(frame, memoryview):
        return frame
    if isinstance(frame, str):
        return frame.encode()
    return msgspec.msgpack.encode(frame)


def _receive_beamproc_plan(self, socket) -> None:
    frames = socket.recv_multipart(copy=False)
    _process_beamproc_plan_frames(self, frames)


def _process_beamproc_plan_frames(self, frames) -> None:
    pb_queue = getattr(self, "pb_queue", None)
    if pb_queue is None:
        logger.warning("Dropping BeamProc message because PB queue is not initialized")
        return

    frame_bytes = [
        bytes(frame.buffer) if index < 2 else memoryview(frame.buffer)
        for index, frame in enumerate(frames)
    ]
    message_type = frame_bytes[0] if frame_bytes else b"<missing>"
    if message_type == _BEAMPROC_MSG_BEAM_FINISHED and len(frame_bytes) == 2:
        try:
            payload = msgspec.msgpack.decode(frame_bytes[1])
        except Exception:
            logger.exception(
                "Failed to decode BeamProc BEAM_FINISHED payload bytes=%s",
                len(frame_bytes[1]),
            )
            return
        if payload.get("version") != 1 or not payload.get("root_request_id"):
            logger.warning("Dropping invalid BeamProc BEAM_FINISHED payload: %s", payload)
            return
        payload["action"] = "beamproc_finished"
        _publish_beamproc_item(self, payload, "finished queue")
        return
    if message_type == _BEAMPROC_MSG_BEAM_PLAN and len(frame_bytes) >= 2:
        try:
            payload = msgspec.msgpack.decode(frame_bytes[1])
        except Exception:
            logger.exception(
                "Failed to decode BeamProc BEAM_PLAN payload bytes=%s",
                len(frame_bytes[1]),
            )
            return
        version = payload.get("version")
        apply_live_plan = (
            _beamproc_should_apply_plan()
            and payload.get("action") == "observed_full"
            and not payload.get("finished")
        )
        materialize_auxiliary = not apply_live_plan
        if version == 2:
            valid = _decode_beamproc_plan_v2_frames(
                payload,
                frame_bytes,
                materialize_auxiliary=materialize_auxiliary,
            )
        elif version == 3:
            valid = _decode_beamproc_plan_v3_frames(
                payload,
                frame_bytes,
                materialize_auxiliary=materialize_auxiliary,
            )
        elif version == 4:
            valid = _decode_beamproc_plan_v4_frames(
                payload,
                frame_bytes,
                materialize_auxiliary=materialize_auxiliary,
            )
        elif version == 5:
            valid = _decode_beamproc_plan_v5_frames(payload, frame_bytes)
        else:
            valid = True
        if not valid:
            return
        if apply_live_plan:
            applied = _apply_beamproc_plan_and_track_pending(self, payload)
            logger.debug(
                "Received BeamProc live BEAM_PLAN correlation_id=%s request_id=%s "
                "action=%s parents=%s children=%s tokens=%s aborts=%s "
                "completed=%s finished=%s applied=%s sample_child=%s sample_token=%s",
                payload.get("correlation_id"),
                payload.get("request_id"),
                payload.get("action"),
                len(payload.get("parent_ids", ())),
                len(payload.get("child_ids", ())),
                len(payload.get("token_ids", ())),
                len(payload.get("abort_ids", ())),
                payload.get("completed_count"),
                payload.get("finished"),
                applied,
                next(iter(payload.get("child_ids", ()) or ()), None),
                next(iter(payload.get("token_ids", ()) or ()), None),
            )
            return
        _publish_beamproc_item(self, payload, "plan queue")
        logger.debug(
            "Received BeamProc BEAM_PLAN correlation_id=%s request_id=%s action=%s "
            "parents=%s children=%s tokens=%s aborts=%s completed=%s finished=%s "
            "sample_child=%s sample_token=%s",
            payload.get("correlation_id"),
            payload.get("request_id"),
            payload.get("action"),
            len(payload.get("parent_ids", ())),
            len(payload.get("child_ids", ())),
            len(payload.get("token_ids", ())),
            len(payload.get("abort_ids", ())),
            payload.get("completed_count"),
            payload.get("finished"),
            next(iter(payload.get("child_ids", ()) or ()), None),
            next(iter(payload.get("token_ids", ()) or ()), None),
        )
        return

    _publish_beamproc_item(
        self,
        {"message_type": message_type, "frames": frame_bytes},
        "plan queue",
    )
    logger.debug(
        "Received BeamProc message type=%s frames=%s frame_bytes=%s",
        message_type,
        len(frame_bytes),
        [len(frame) for frame in frame_bytes],
    )


def _wake_beamproc_engine_loop(self) -> None:
    input_queue = getattr(self, "input_queue", None)
    if input_queue is None:
        return
    wake_lock = getattr(self, "beamproc_wake_lock", None)
    if wake_lock is not None:
        with wake_lock:
            if getattr(self, "beamproc_wake_pending", False):
                return
            self.beamproc_wake_pending = True
    input_queue.put_nowait((_BEAMPROC_WAKE_REQUEST, None))


def _clear_beamproc_engine_wake(self) -> None:
    wake_lock = getattr(self, "beamproc_wake_lock", None)
    if wake_lock is None:
        return
    with wake_lock:
        self.beamproc_wake_pending = False


def _publish_beamproc_item(self, item, queue_name: str) -> None:
    # While pulling native replies, decoding and admission share the scheduler
    # thread. Deliver immediately rather than enqueueing work for that same thread.
    delivery = getattr(self, "beamproc_scheduler_delivery", None)
    if delivery is not None:
        delivery(item)
    elif getattr(self, "beamproc_python_bridge", None) is not None:
        # The RX owner must be able to stop even if the scheduler has exited
        # while its admission queue is full.
        deadline = time.monotonic() + _BEAMPROC_QUEUE_PUT_TIMEOUT_S
        while not self.beamproc_native_stop.is_set():
            try:
                self.pb_queue.put(item, timeout=0.1)
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"BeamProc {queue_name} remained full") from None
    else:
        _beamproc_queue_put(self.pb_queue, item, queue_name)


def _consume_beamproc_item(self, item) -> int:
    admitted = 0
    if isinstance(item, tuple) and len(item) == 3 and item[0] is _BEAMPROC_CHILD_BATCH_REQUEST:
        _, requests, current_wave = item
        for req in requests or ():
            self.add_request(req, current_wave)
            admitted += 1
    elif isinstance(item, tuple) and len(item) == 2 and item[0] is _BEAMPROC_CHILD_BATCH_REQUEST:
        _, request_batch = item
        for req, current_wave in request_batch or ():
            self.add_request(req, current_wave)
            admitted += 1
    elif isinstance(item, tuple) and len(item) == 2:
        req, current_wave = item
        self.add_request(req, current_wave)
        admitted += 1
    else:
        _handle_beamproc_plan(self, item)
    return admitted


def _drain_pb_queue(self) -> int:
    """Decode/admit native replies on the scheduler before newer AS input."""
    _check_beamproc_native_io(self)
    pb_queue = getattr(self, "pb_queue", None)
    if pb_queue is None or getattr(self, "beamproc_scheduler_delivery", None) is not None:
        return 0

    admitted = 0
    while True:
        try:
            item = pb_queue.get_nowait()
        except queue.Empty:
            break
        admitted += _consume_beamproc_item(self, item)

    bridge = getattr(self, "beamproc_native_bridge", None)
    if bridge is None:
        return admitted

    def deliver(item):
        nonlocal admitted
        admitted += _consume_beamproc_item(self, item)

    self.beamproc_scheduler_delivery = deliver
    try:
        while (frames := bridge.receive(0)) is not None:
            _process_beamproc_plan_frames(self, frames)
    finally:
        # Do not retain the engine through the local function, including on errors.
        self.beamproc_scheduler_delivery = None
    return admitted


def _enqueue_beamproc_message(self, message_type, payload=None) -> None:
    bridge = getattr(self, "beamproc_native_bridge", None)
    if bridge is not None:
        _check_beamproc_native_io(self)
        if message_type == _BEAMPROC_MSG_ENGINE_OUTPUT_BATCH and payload.get("batches") is None:
            from vllm_gr.v1.engine.native_ce_batch import send_engine_outputs

            send_engine_outputs(self, bridge, payload)
        else:
            # Controls and already prepared frames share the same native FIFO as
            # raw batches, preserving state-update/output/abort ordering.
            _send_beamproc_message(self, bridge, (message_type, payload), deque())
        return
    outbound_queue = getattr(self, "beamproc_outbound_queue", None)
    if outbound_queue is not None:
        _beamproc_queue_put(
            outbound_queue,
            (message_type, payload),
            "outbound queue",
        )


def _enqueue_beamproc_abort(self, request_ids) -> None:
    if isinstance(request_ids, str):
        request_ids = [request_ids]
    root_request_ids = sorted(
        {
            root_request_id
            for request_id in request_ids
            if (root_request_id := _beamproc_parent_root(request_id)) is not None
        }
    )
    for root_request_id in root_request_ids:
        related_request_ids = [
            request_id
            for request_id in request_ids
            if _beamproc_parent_root(request_id) == root_request_id
        ]
        _cleanup_beamproc_root_state(
            self,
            root_request_id,
            *related_request_ids,
        )
    _enqueue_beamproc_message(
        self,
        _BEAMPROC_MSG_ABORT,
        {
            "request_ids": list(request_ids),
            "root_request_ids": root_request_ids,
        },
    )


def _beamproc_has_transport(self) -> bool:
    return (
        getattr(self, "beamproc_native_bridge", None) is not None
        or getattr(self, "beamproc_outbound_queue", None) is not None
    )


def _enqueue_beamproc_state_update_values(
    self,
    *,
    child_ids,
    child_cum_logprobs,
    child_valid_token_ids,
    beam_width: int,
) -> None:
    if not _beamproc_has_transport(self):
        return
    if not child_ids:
        return
    if not child_cum_logprobs:
        return

    root_request_id = _beamproc_parent_root(child_ids[0])
    if root_request_id is None:
        return

    _enqueue_beamproc_message(
        self,
        _BEAMPROC_MSG_BEAM_STATE_UPDATE,
        {
            "version": 1,
            "root_request_id": root_request_id,
            "beam_width": beam_width or len(child_ids),
            "active_parent_ids": list(child_ids),
            "active_parent_cum_logprobs": list(child_cum_logprobs),
            "active_parent_valid_token_ids": list(child_valid_token_ids or []),
        },
    )


def _enqueue_beamproc_state_update(self, fork_req) -> None:
    _enqueue_beamproc_state_update_values(
        self,
        child_ids=fork_req.child_ids,
        child_cum_logprobs=getattr(fork_req, "child_cum_logprobs", None),
        child_valid_token_ids=getattr(fork_req, "child_valid_token_ids", None),
        beam_width=fork_req.beam_width,
    )


def _enqueue_beamproc_engine_outputs(self, outputs) -> None:
    if not outputs or not _beamproc_has_transport(self):
        return

    correlation_base = getattr(self, "beamproc_correlation_id", 0)
    self.beamproc_correlation_id = correlation_base + 1
    output_contexts = {}
    for engine_core_outputs in outputs.values():
        for output in engine_core_outputs.outputs:
            if output.request_id:
                context = _beamproc_context_for_output(self, output)
                if context is not None:
                    output_contexts[output.request_id] = context
    live_plan_count = sum(
        not bool(context.get("is_final_step")) for context in output_contexts.values()
    )
    if live_plan_count:
        _adjust_beamproc_pending_plans(self, live_plan_count)

    outputs_count = len(output_contexts)
    if outputs_count <= 0:
        return
    _enqueue_beamproc_message(
        self,
        _BEAMPROC_MSG_ENGINE_OUTPUT_BATCH,
        {
            "engine_index": self.engine_index,
            "correlation_base": correlation_base,
            "outputs": outputs,
            "output_contexts": output_contexts,
            "outputs_count": outputs_count,
            "byte_count": 0,
        },
    )


def _beamproc_engine_outputs_count(outputs) -> int:
    count = 0
    for engine_core_outputs in outputs.values():
        count += sum(1 for output in engine_core_outputs.outputs if output.request_id)
    return count


def _prepare_beamproc_engine_output_batches(
    self,
    outputs,
    engine_index: int,
    correlation_base: int,
    output_contexts: dict[str, Any] | None = None,
    output_buffer_pool=None,
) -> list[dict[str, Any]]:
    batches = []
    for client_index, engine_core_outputs in outputs.items():
        buffer_set = _acquire_beamproc_tx_buffer(output_buffer_pool)
        (
            batch_outputs,
            contexts,
            token_ids,
            logprobs,
            ranks,
            new_token_ids,
        ) = _beamproc_batch_payload(
            self,
            engine_core_outputs.outputs,
            correlation_base,
            output_contexts,
            buffer_set,
        )
        if not batch_outputs:
            _release_beamproc_tx_buffer(buffer_set, output_buffer_pool)
            continue

        batch_payload = msgspec.msgpack.encode(
            {
                "version": 6,
                "engine_index": engine_index,
                "client_index": client_index,
                "correlation_base": correlation_base,
                "contexts": contexts,
                "outputs": batch_outputs,
            }
        )
        frames = [
            _BEAMPROC_MSG_ENGINE_OUTPUT_BATCH,
            batch_payload,
            token_ids,
            logprobs,
            ranks,
            new_token_ids,
        ]
        batches.append(
            {
                "client_index": client_index,
                "correlation_base": correlation_base,
                "frames": frames,
                "batch_outputs": batch_outputs,
                "buffer_set": buffer_set,
            }
        )

    return batches


def _acquire_beamproc_tx_buffer(output_buffer_pool) -> dict[str, Any]:
    if output_buffer_pool:
        return output_buffer_pool.pop()
    return {"_beamproc_tx_buffer_set": True}


def _beamproc_batch_payload(
    self,
    outputs,
    correlation_base: int,
    output_contexts: dict[str, Any] | None = None,
    buffer_set: dict[str, Any] | None = None,
) -> tuple[
    list[list[Any]],
    list[Any],
    memoryview,
    memoryview,
    memoryview,
    memoryview,
]:
    batch_outputs = []
    contexts = []
    context_index_by_key = {}
    token_arrays = []
    logprob_arrays = []
    rank_arrays = []
    new_token_values = []
    token_offset = 0
    rank_offset = 0
    new_token_offset = 0

    for output in outputs:
        if not output.request_id:
            continue
        if output_contexts is not None and output.request_id not in output_contexts:
            continue

        grouped_states = getattr(self, "beamproc_grouped_output_states", {})
        grouped_state = grouped_states.get(output.request_id)
        if grouped_state is not None and output.new_logprobs is not None:
            child_ids = grouped_state["child_ids"]
            beam_width = len(child_ids)
            raw_token_ids = np.ascontiguousarray(
                output.new_logprobs.logprob_token_ids,
                dtype=np.int32,
            )
            raw_logprobs = np.ascontiguousarray(
                output.new_logprobs.logprobs,
                dtype=np.float32,
            )
            if raw_token_ids.ndim == 1:
                raw_token_ids = raw_token_ids.reshape(1, -1)
                raw_logprobs = raw_logprobs.reshape(1, -1)
            if raw_token_ids.shape[0] == 1 and beam_width > 1:
                total_cols = raw_token_ids.shape[1]
                if total_cols % beam_width != 0:
                    logger.error(
                        "Grouped BeamProc output cannot split candidates "
                        "request=%s shape=%s beam_width=%s",
                        output.request_id,
                        raw_token_ids.shape,
                        beam_width,
                    )
                    grouped_state = None
                else:
                    row_cols = total_cols // beam_width
                    raw_token_ids = raw_token_ids.reshape(beam_width, row_cols)
                    raw_logprobs = raw_logprobs.reshape(beam_width, row_cols)
            if grouped_state is not None and raw_token_ids.shape[0] != beam_width:
                logger.error(
                    "Grouped BeamProc output row mismatch request=%s shape=%s beam_width=%s",
                    output.request_id,
                    raw_token_ids.shape,
                    beam_width,
                )
                grouped_state = None

        if grouped_state is not None and output.new_logprobs is not None:
            trace_context = dict(grouped_state["trace_context"])
            trace_context.pop("parent_token_ids", None)
            trace_context["prompt_token_count"] = 0
            trace_context["use_shadow_parent_tokens"] = True
            context_index = len(contexts)
            contexts.append(trace_context)
            for row_index, child_id in enumerate(grouped_state["child_ids"]):
                token_row = raw_token_ids[row_index : row_index + 1]
                logprob_row = raw_logprobs[row_index : row_index + 1]
                cols = int(token_row.shape[1])
                payload = [
                    child_id,
                    new_token_offset,
                    0,
                    _beamproc_value(output.finish_reason),
                    _beamproc_value(output.stop_reason),
                    _beamproc_num_cached_tokens(output),
                    token_offset,
                    1,
                    cols,
                    rank_offset,
                    1,
                    context_index,
                ]
                token_arrays.append(token_row)
                logprob_arrays.append(logprob_row)
                rank_arrays.append(np.zeros(1, dtype=np.int32))
                token_offset += cols
                rank_offset += 1
                batch_outputs.append(payload)

            grouped_states.pop(output.request_id, None)
            if grouped_state.get("is_final_step"):
                _cleanup_beamproc_root_state(
                    self,
                    grouped_state["root_request_id"],
                    output.request_id,
                )
            continue

        context = None
        if output_contexts is not None:
            context = output_contexts.get(output.request_id)
        if context is None:
            context = _beamproc_context_for_output(self, output)
            context_index = None
        if context is not None:
            context_key = msgspec.msgpack.encode(context)
            context_index = context_index_by_key.get(context_key)
            if context_index is None:
                context_index = len(contexts)
                context_index_by_key[context_key] = context_index
                contexts.append(context)

        payload = _beamproc_output_payload(self, output, context_index)
        new_tokens = output.new_token_ids
        new_token_len = len(new_tokens)
        if new_token_len:
            new_token_values.extend(new_tokens)
        payload[1] = new_token_offset
        payload.insert(2, new_token_len)
        new_token_offset += new_token_len
        if output.new_logprobs is not None:
            token_ids, logprobs, ranks = _beamproc_logprobs_arrays(
                output.new_logprobs,
                restore_global_ids=False,
            )
            token_len = token_ids.size
            rank_len = ranks.size
            rows, cols = token_ids.shape
            payload[6] = token_offset
            payload[7] = rows
            payload[8] = cols
            payload[9] = rank_offset
            payload[10] = rank_len
            token_arrays.append(token_ids)
            logprob_arrays.append(logprobs)
            rank_arrays.append(ranks)
            token_offset += token_len
            rank_offset += rank_len

        batch_outputs.append(payload)

    flat_token_ids = _concat_or_empty(token_arrays, np.int32, buffer_set, "token_ids")
    flat_token_ids = _beamproc_global_candidate_token_ids_buffered(
        flat_token_ids,
        buffer_set,
    )
    flat_logprobs = _concat_or_empty(logprob_arrays, np.float32, buffer_set, "logprobs")
    flat_ranks = _concat_or_empty(rank_arrays, np.int32, buffer_set, "ranks")
    flat_new_token_ids = _array_or_empty(new_token_values, np.int32, buffer_set, "new_token_ids")
    return (
        batch_outputs,
        contexts,
        memoryview(flat_token_ids),
        memoryview(flat_logprobs),
        memoryview(flat_ranks),
        memoryview(flat_new_token_ids),
    )


def _beamproc_index_nullable_value(
    values: list[Any], index_by_key: dict[bytes, int], value: Any
) -> int:
    if value is None:
        return -1
    key = msgspec.msgpack.encode(value)
    index = index_by_key.get(key)
    if index is None:
        index = len(values)
        index_by_key[key] = index
        values.append(value)
    return index


def _beamproc_root_request_id(request_id: str) -> str:
    beam_index = request_id.find("-beam-")
    beamproc_index = request_id.find("-beamproc-")
    split_indices = [index for index in (beam_index, beamproc_index) if index >= 0]
    if not split_indices:
        return request_id
    return request_id[: min(split_indices)]


def _beamproc_num_cached_tokens(output) -> int | None:
    """Read prefix-cache accounting from current or legacy EngineCoreOutput."""
    prefill_stats = getattr(output, "prefill_stats", None)
    if prefill_stats is not None:
        return prefill_stats.num_cached_tokens
    return getattr(output, "num_cached_tokens", None)


def _beamproc_output_payload(self, output, context_index: int | None) -> list[Any]:
    request_id = output.request_id
    return [
        request_id,
        list(output.new_token_ids),
        _beamproc_value(output.finish_reason),
        _beamproc_value(output.stop_reason),
        _beamproc_num_cached_tokens(output),
        0,
        0,
        0,
        0,
        0,
        context_index,
    ]


def _beamproc_wire_parent_tokens(cached_req, trace_context) -> tuple[list[int], int | None]:
    """Return the generated-only parent history when its prompt size is known."""
    parent_token_ids = list(getattr(cached_req, "prompt_token_ids", []) or [])
    prompt_token_count = trace_context.get("prompt_token_count")
    if type(prompt_token_count) is int and 0 <= prompt_token_count <= len(parent_token_ids):
        return parent_token_ids[prompt_token_count:], 0
    return parent_token_ids, prompt_token_count


def _beamproc_context_for_output(self, output) -> dict[str, Any] | None:
    request_id = output.request_id
    if not request_id:
        return None

    with self.beam_cache_lock:
        cached = self.beam_cache.get(request_id)
    if cached is None:
        return None

    cached_req = _beamproc_cached_request(cached)
    if cached_req is None:
        return None
    sampling_params = getattr(cached_req, "sampling_params", None)
    if sampling_params is None:
        return None

    trace_context = _beamproc_request_context(cached_req) or {}
    if "-beamproc-" in request_id:
        minimal_context = {
            key: trace_context[key]
            for key in (
                "root_request_id",
                "total_decode_steps",
                "step_index",
                "is_final_step",
            )
            if key in trace_context
        }
        prompt_token_count = trace_context.get("prompt_token_count")
        if prompt_token_count is not None:
            minimal_context["prompt_token_count"] = (
                0 if type(prompt_token_count) is int else prompt_token_count
            )
        return minimal_context or None

    parent_token_ids, wire_prompt_token_count = _beamproc_wire_parent_tokens(
        cached_req,
        trace_context,
    )
    context = {
        "max_tokens": getattr(cached_req, "max_tokens", None),
        "sampling_max_tokens": getattr(sampling_params, "max_tokens", None),
        "ignore_eos": bool(getattr(sampling_params, "ignore_eos", False)),
        "include_stop_token": False,
        "eos_token_id": _request_eos_token_id(cached_req),
        "client_index": getattr(cached_req, "client_index", 0),
        "priority": getattr(cached_req, "priority", 0),
    }
    context.update(trace_context)
    context["parent_token_ids"] = parent_token_ids
    if wire_prompt_token_count is not None:
        context["prompt_token_count"] = wire_prompt_token_count
    return context


def _filter_beamproc_outputs_for_apiserver(self, engine_core_outputs):
    if not _beamproc_should_apply_plan():
        return engine_core_outputs

    hidden_request_ids = {
        output.request_id
        for output in engine_core_outputs.outputs
        if _beamproc_context_for_output(self, output) is not None
    }
    if not hidden_request_ids:
        return engine_core_outputs

    visible_outputs = [
        output
        for output in engine_core_outputs.outputs
        if output.request_id not in hidden_request_ids
    ]
    visible_finished = None
    if engine_core_outputs.finished_requests is not None:
        visible_finished = set(engine_core_outputs.finished_requests) - hidden_request_ids

    if not visible_outputs and not visible_finished:
        return None

    return EngineCoreOutputs(
        engine_index=engine_core_outputs.engine_index,
        outputs=visible_outputs,
        scheduler_stats=engine_core_outputs.scheduler_stats,
        timestamp=engine_core_outputs.timestamp,
        utility_output=engine_core_outputs.utility_output,
        finished_requests=visible_finished,
        wave_complete=engine_core_outputs.wave_complete,
        start_wave=engine_core_outputs.start_wave,
    )


def _beamproc_logprobs_arrays(new_logprobs, *, restore_global_ids: bool = True):
    # Same numeric data vLLM later formats for the APIServer. Position 0 is the
    # sampled token; following positions are top-k logprob entries.
    token_ids = np.ascontiguousarray(new_logprobs.logprob_token_ids, dtype=np.int32)
    if restore_global_ids:
        token_ids = _beamproc_global_candidate_token_ids(token_ids)
    logprobs = np.ascontiguousarray(new_logprobs.logprobs, dtype=np.float32)
    ranks = np.ascontiguousarray(new_logprobs.sampled_token_ranks, dtype=np.int32)
    return token_ids, logprobs, ranks


def _beamproc_global_candidate_token_ids(token_ids: np.ndarray) -> np.ndarray:
    """Translate sliced-logits candidate IDs back to tokenizer-global IDs."""
    from vllm_gr.ops.sliced_logits import get_vocab_slice_config

    vocab_offset, vocab_slice_size, _ = get_vocab_slice_config()
    if vocab_offset is None or vocab_slice_size is None or vocab_slice_size <= 0:
        return token_ids

    local_ids = (token_ids >= 0) & (token_ids < vocab_slice_size)
    if not np.any(local_ids):
        return token_ids

    global_token_ids = token_ids.copy()
    global_token_ids[local_ids] += vocab_offset
    return global_token_ids


def _beamproc_global_candidate_token_ids_buffered(
    token_ids: np.ndarray,
    buffer_set: dict[str, Any] | None,
) -> np.ndarray:
    """Translate IDs in an owned TX buffer without another full-sized copy."""
    if buffer_set is None:
        return _beamproc_global_candidate_token_ids(token_ids)

    from vllm_gr.ops.sliced_logits import get_vocab_slice_config

    vocab_offset, vocab_slice_size, _ = get_vocab_slice_config()
    if (
        vocab_offset is None
        or vocab_offset == 0
        or vocab_slice_size is None
        or vocab_slice_size <= 0
        or token_ids.size == 0
    ):
        return token_ids

    local_mask = _beamproc_tx_buffer_array(
        buffer_set,
        "token_id_local_mask",
        token_ids.size,
        np.bool_,
    )
    # Viewing signed IDs as unsigned makes one comparison cover both bounds:
    # negative sentinels become large unsigned values and remain unchanged.
    np.less(
        token_ids.view(np.uint32),
        np.uint32(vocab_slice_size),
        out=local_mask,
    )
    # Boolean indexing is faster here than a conditional ufunc for the mixed
    # local/global candidate rows produced by sliced-logits sampling.
    token_ids[local_mask] += np.int32(vocab_offset)
    return token_ids


def _beamproc_tx_buffer_array(
    buffer_set: dict[str, Any],
    key: str,
    size: int,
    dtype,
):
    array = buffer_set.get(key)
    if array is None or array.size < size or array.dtype != dtype:
        array = np.empty(size, dtype=dtype)
        buffer_set[key] = array
    return array[:size]


def _concat_or_empty(arrays, dtype, buffer_set=None, buffer_key=None):
    if not arrays:
        return np.empty(0, dtype=dtype)
    if buffer_set is not None and buffer_key is not None:
        total = sum(array.size for array in arrays)
        target = _beamproc_tx_buffer_array(buffer_set, buffer_key, total, dtype)
        offset = 0
        for array in arrays:
            flat = array.reshape(-1)
            size = flat.size
            target[offset : offset + size] = flat
            offset += size
        return target
    if len(arrays) == 1:
        return arrays[0].reshape(-1)
    return np.concatenate([array.reshape(-1) for array in arrays]).astype(dtype, copy=False)


def _array_or_empty(values, dtype, buffer_set=None, buffer_key=None):
    if not values:
        return np.empty(0, dtype=dtype)
    if buffer_set is not None and buffer_key is not None:
        target = _beamproc_tx_buffer_array(buffer_set, buffer_key, len(values), dtype)
        target[:] = values
        return target
    return np.asarray(values, dtype=dtype)


def _beamproc_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _handle_beamproc_plan(self, plan: dict[str, Any]) -> str | None:
    if isinstance(plan, dict) and plan.get("action") == "beamproc_finished":
        _cleanup_beamproc_root_state(
            self,
            plan["root_request_id"],
            plan.get("request_id"),
            *(plan.get("request_ids") or ()),
        )
    elif isinstance(plan, dict) and plan.get("action") == "observed_full":
        root_request_id = plan.get("root_request_id") or _beamproc_parent_root(
            plan.get("request_id")
        )
        if _beamproc_should_apply_plan() and not plan.get("finished"):
            logger.warning(
                "Dropping queued live BeamProc plan on EngineCore main thread "
                "root=%s correlation_id=%s parents=%s children=%s",
                root_request_id,
                plan.get("correlation_id"),
                len(plan.get("parent_ids") or ()),
                len(plan.get("child_ids") or ()),
            )
            return None
        if root_request_id is not None:
            if plan.get("finished"):
                _cleanup_beamproc_root_state(
                    self,
                    root_request_id,
                    plan.get("request_id"),
                    *(plan.get("scored_parent_ids") or ()),
                    *(plan.get("parent_ids") or ()),
                    *(plan.get("abort_ids") or ()),
                )
            logger.debug(
                "Handled BeamProc full plan root=%s correlation_id=%s parents=%s tokens=%s "
                "beam_width=%s",
                root_request_id,
                plan.get("correlation_id"),
                len(plan.get("parent_ids") or ()),
                len(plan.get("token_ids") or ()),
                plan.get("beam_width"),
            )
    elif isinstance(plan, dict) and plan.get("action") == "observed":
        logger.debug(
            "Handled BeamProc plan correlation_id=%s request_id=%s parents=%s tokens=%s",
            plan.get("correlation_id"),
            plan.get("request_id"),
            len(plan.get("parent_ids") or ()),
            len(plan.get("token_ids") or ()),
        )
    else:
        logger.debug("Handled BeamProc non-plan message: %s", plan)

    return None


def _beamproc_parent_root(parent_id: str | None) -> str | None:
    if not parent_id:
        return None
    return _beamproc_root_request_id(parent_id)


def _cleanup_beamproc_root_state(
    self,
    root_request_id: str,
    *request_ids: str | None,
) -> None:
    """Release all EngineCore metadata owned by a completed BeamProc root."""
    current_by_root = getattr(self, "beamproc_grouped_request_by_root", {})
    current_request_id = current_by_root.pop(root_request_id, None)
    active_by_root = getattr(self, "beamproc_grouped_active_ids_by_root", {})
    active_request_ids = active_by_root.pop(root_request_id, ())
    for attr_name in (
        "beamproc_grouped_response_prefix_by_root",
        "beamproc_grouped_parent_states",
        "beamproc_full_plans_by_root",
        "beamproc_full_plan_received_at_by_root",
    ):
        getattr(self, attr_name, {}).pop(root_request_id, None)

    output_states = getattr(self, "beamproc_grouped_output_states", {})
    for grouped_request_id, state in list(output_states.items()):
        if state.get("root_request_id") == root_request_id:
            output_states.pop(grouped_request_id, None)

    cache_ids = {
        root_request_id,
        current_request_id,
        *request_ids,
        *active_request_ids,
    }
    cache_ids.discard(None)
    with self.beam_cache_lock:
        for request_id in cache_ids:
            self.beam_cache.pop(request_id, None)


def make_patched_process_input_queue(original_process_input_queue):
    def patched_process_input_queue(self, *args, **kwargs):
        # AS RX publishes to input_queue; native PB RX publishes to its Rust inbox.
        # Always drain PB work first so ready children precede newer AS input.
        if getattr(self, "pb_queue", None) is None:
            return original_process_input_queue(self, *args, **kwargs)

        while True:
            if not getattr(self, "is_running", lambda: True)():
                return None
            _drain_pb_queue(self)
            if (
                self.engines_running
                or self.scheduler.has_requests()
                or self.batch_queue
                or not self.input_queue.empty()
            ):
                return original_process_input_queue(self, *args, **kwargs)
            if getattr(self, "beamproc_native_bridge", None) is not None:
                os.sched_yield()
            else:
                time.sleep(_BEAMPROC_DUAL_QUEUE_POLL_SLEEP_S)

    return patched_process_input_queue


def make_patched_process_engine_step(original_process_engine_step):
    def patched_process_engine_step(self) -> bool:
        outputs, model_executed = self.step_fn()
        _enqueue_beamproc_engine_outputs(self, outputs)

        for client_index, engine_core_outputs in outputs.items() if outputs else ():
            visible_outputs = _filter_beamproc_outputs_for_apiserver(
                self,
                engine_core_outputs,
            )
            if visible_outputs is not None:
                self.output_queue.put_nowait((client_index, visible_outputs))

        self.post_step(model_executed)
        return model_executed

    return patched_process_engine_step


def make_patched_handle_client_request(original_handle_client_request):
    def patched_handle_client_request(self, request_type, request):
        # Give PB RX work priority over newer AS RX input items.
        _drain_pb_queue(self)
        if request_type is _BEAMPROC_WAKE_REQUEST:
            _clear_beamproc_engine_wake(self)
            _drain_pb_queue(self)
            return None
        if request_type is _BEAMPROC_CHILD_BATCH_REQUEST:
            if request:
                for req, current_wave in request:
                    self.add_request(req, current_wave)
            return None
        if _beamproc_should_apply_plan():
            _drain_pb_queue(self)
        if request_type == EngineCoreRequestType.ABORT:
            _enqueue_beamproc_abort(self, request)
        return original_handle_client_request(self, request_type, request)

    return patched_handle_client_request


def process_output_sockets(
    self,
    output_paths: list[str],
    coord_output_path: str | None,
    engine_index: int,
):
    """Replacement supporting reusable zero-copy output buffers."""
    from contextlib import ExitStack

    import zmq
    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.serial_utils import MsgpackEncoder

    encoder = MsgpackEncoder()
    reuse_buffers: list[bytearray] = []
    pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

    with ExitStack() as stack, zmq.Context() as ctx:
        sockets = [
            stack.enter_context(make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))
            for output_path in output_paths
        ]
        coord_socket = (
            stack.enter_context(
                make_zmq_socket(
                    ctx,
                    coord_output_path,
                    zmq.PUSH,
                    bind=False,
                    linger=4000,
                )
            )
            if coord_output_path is not None
            else None
        )
        max_reuse_bufs = len(sockets) + 1

        while True:
            output = self.output_queue.get()
            if output == EngineCoreProc.ENGINE_CORE_DEAD:
                for socket in sockets:
                    socket.send(output)
                break
            assert not isinstance(output, bytes)
            client_index, outputs = output
            outputs.engine_index = engine_index

            if client_index == -1:
                assert coord_socket is not None
                coord_socket.send_multipart(encoder.encode(outputs))
                continue

            while pending and pending[-1][0].done:
                reuse_buffers.append(pending.pop()[2])

            buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
            buffers = encoder.encode_into(outputs, buffer)
            tracker = sockets[client_index].send_multipart(
                buffers,
                copy=False,
                track=True,
            )
            if not tracker.done:
                ref = outputs if len(buffers) > 1 else None
                pending.appendleft((tracker, ref, buffer))
            elif len(reuse_buffers) < max_reuse_bufs:
                reuse_buffers.append(buffer)


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

                else:
                    request = decoded_payload

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
