# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

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


def _constraint_table_backend_active() -> bool:
    """True when the serialized Worker constraint configuration is enabled.

    Reads the process-global vLLM config (same source as ``ascend_compat``) so
    both the driver and the EngineCore process agree without passing the flag
    around the request path.
    """
    from vllm.config import get_current_vllm_config_or_none

    from vllm_gr.config import GRConfig

    return GRConfig.from_vllm_config(get_current_vllm_config_or_none()).constraint_table.enabled


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

    if _constraint_table_backend_active():
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
    # Store max decode steps for KV pool pre-allocation
    req.max_decode_steps = update.max_decode_steps

    if _constraint_table_backend_active():
        # Constraint Table decode: the Worker runs the constrained top-k
        # operator at raw-logits time.  constraint_step equals the number of
        # generated tokens so far (1 or 2 under the 3-step cap); beam_suffix
        # lets the Worker rebuild the sequence_group the operator reads.
        # constraint_step 0 is reserved for the constrained prefill, so this
        # decode request clears is_constraint_prefill.
        req.is_constrained = True
        req.constraint_step = update.constraint_step if update.constraint_step > 0 else decode_steps
        req.beam_suffix = [list(t) for t in update.beam_tokens]
        req.is_constraint_prefill = False

    self.input_queue.put_nowait((EngineCoreRequestType.ADD, (req, update.current_wave)))

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

                else:
                    request = decoded_payload

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
