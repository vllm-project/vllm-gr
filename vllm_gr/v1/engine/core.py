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
    """
    _ensure_beam_state(self)
    with self.beam_cache_lock:
        self.beam_cache[request.request_id] = {
            "all_token_ids": list(request._all_token_ids),
            "block_hashes": list(request.block_hashes),
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


def _emit_beam_request_error(self, update, message: str) -> None:
    """Finish the logical beam request with an error visible to its caller."""
    logger.error(
        "BEAM_REQUEST_STEP_UPDATE rejected: session=%s: %s",
        update.session_id,
        message,
    )
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
                        stop_reason=message,
                    )
                ],
            ),
        )
    )


def _validate_beam_request_step_update(update) -> str | None:
    """Return a user-facing validation error for a decoded beam update."""
    if not isinstance(update.session_id, str) or not update.session_id:
        return "session_id must be a non-empty string"

    beam_width = update.beam_width
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 0:
        return f"beam_width must be a non-negative integer, got {beam_width!r}"

    # beam_width=0 is the existing cleanup message. Its pruned_ids are the
    # branches that the frontend wants to forget; the other arrays are unused.
    if beam_width == 0:
        return None

    if not isinstance(update.prefix_len, int) or isinstance(update.prefix_len, bool):
        return f"prefix_len must be an integer, got {update.prefix_len!r}"
    if update.prefix_len < 0:
        return f"prefix_len must be non-negative, got {update.prefix_len}"

    parallel_lengths = {
        "beam_tokens": len(update.beam_tokens),
        "parent_beam_ids": len(update.parent_beam_ids),
        "child_beam_ids": len(update.child_beam_ids),
    }
    mismatched = {name: size for name, size in parallel_lengths.items() if size != beam_width}
    if mismatched:
        return f"parallel array lengths must equal beam_width={beam_width}, got {mismatched}"

    decode_steps = len(update.beam_tokens[0])
    if decode_steps == 0:
        return "beam_tokens entries must contain at least one decode token"
    if any(len(tokens) != decode_steps for tokens in update.beam_tokens):
        return "all beam_tokens entries must have the same decode length"

    if len(set(update.child_beam_ids)) != len(update.child_beam_ids):
        return "child_beam_ids must be unique"

    return None


def _handle_beam_request_step_update(self, update) -> None:
    """Apply a grouped step update using only the prefill cache entry."""
    from vllm.v1.request import Request

    _ensure_beam_state(self)
    session_id = update.session_id
    beam_width = update.beam_width

    if validation_error := _validate_beam_request_step_update(update):
        _emit_beam_request_error(self, update, validation_error)
        return

    # Clear the session completely upon generation termination / cleanup message
    if beam_width == 0:
        with self.beam_cache_lock:
            self.beam_cache.pop(session_id, None)

        # We MUST abort the running beam-request session_id itself!
        self.aborts_queue.put_nowait(session_id)
        if update.pruned_ids:
            for abort_id in update.pruned_ids:
                self.aborts_queue.put_nowait(abort_id)
        return

    # Fetch the shared prefill cache entry
    with self.beam_cache_lock:
        session_cached = self.beam_cache.get(session_id)

    if session_cached is None and update.beam_width > 0:
        _emit_beam_request_error(self, update, "beam session was not found in the prefill cache")
        return

    arrival = time.time()
    inputs_to_push = []

    prefill_tokens = session_cached["all_token_ids"]
    prefill_block_hashes = list(session_cached["block_hashes"])
    lora_request = session_cached["lora_request"]
    cache_salt = session_cached["cache_salt"]
    mm_features = session_cached["mm_features"]
    prompt_embeds = session_cached["prompt_embeds"]
    prompt_is_token_ids = session_cached["prompt_is_token_ids"]
    eos = session_cached["eos_token_id"]

    if update.prefix_len > len(prefill_tokens):
        _emit_beam_request_error(
            self,
            update,
            f"prefix_len={update.prefix_len} exceeds cached prefill length={len(prefill_tokens)}",
        )
        return

    eos_token_id = update.eos_token_id if update.eos_token_id is not None else eos
    if eos_token_id is not None:
        update.sampling_params.update_from_generation_config({}, eos_token_id)

    # Pack the shared prefill and all beam suffixes into a single contiguous Beam Request
    beam_token_ids = list(prefill_tokens)
    decode_steps = len(update.beam_tokens[0])
    for gen_tokens in update.beam_tokens:
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
    req.beam_decode_steps = decode_steps
    req.prefix_len = update.prefix_len

    inputs_to_push.append((EngineCoreRequestType.ADD, (req, update.current_wave)))

    # Non-blocking concurrent queue pushes
    for item in inputs_to_push:
        self.input_queue.put_nowait(item)

    if update.pruned_ids:
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
        EngineCoreRequestType,
    )
    from vllm.v1.serial_utils import MsgpackDecoder

    from vllm_gr.v1.engine.types import BeamRequestStepUpdate

    # Msgpack serialization decoding.
    decoder_kwargs = {"oob_tensor_provider": self.tensor_ipc_receiver}
    add_request_decoder = MsgpackDecoder(EngineCoreRequest, **decoder_kwargs)
    add_batch_decoder = MsgpackDecoder(list[EngineCoreRequest], **decoder_kwargs)
    beam_request_decoder = MsgpackDecoder(BeamRequestStepUpdate, **decoder_kwargs)
    generic_decoder = MsgpackDecoder(**decoder_kwargs)

    def process_message(type_frame, data_frames) -> None:
        """Decode and dispatch one message from the input socket thread."""
        request_type = EngineCoreRequestType(bytes(type_frame.buffer))

        request: Any
        if request_type == EngineCoreRequestType.ADD:
            req: EngineCoreRequest = add_request_decoder.decode(data_frames)
            try:
                request = self.preprocess_add_request(req)
            except Exception:
                self._handle_request_preproc_error(req)
                return

        elif request_type == EngineCoreRequestType.ADD_BATCH:
            batch: list[EngineCoreRequest] = add_batch_decoder.decode(data_frames)
            for req in batch:
                try:
                    request = self.preprocess_add_request(req)
                except Exception:
                    self._handle_request_preproc_error(req)
                    continue
                self._cache_beam_request(request[0])
                self.input_queue.put_nowait((EngineCoreRequestType.ADD, request))
            return

        elif request_type == EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE:
            beam_update: BeamRequestStepUpdate = beam_request_decoder.decode(data_frames)
            self._handle_beam_request_step_update(beam_update)
            return

        else:
            request = generic_decoder.decode(data_frames)

            if request_type == EngineCoreRequestType.ABORT:
                self.aborts_queue.put_nowait(request)

        self.input_queue.put_nowait((request_type, request))

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
                try:
                    process_message(type_frame, data_frames)
                except Exception:
                    # Once the input thread dies, the EngineCore can otherwise
                    # remain alive but permanently stop consuming requests.
                    # Escalate undecodable/unknown wire messages to the core
                    # loop so the normal ENGINE_CORE_DEAD path wakes callers.
                    logger.exception("Fatal EngineCore input protocol error")
                    self.input_queue.put_nowait((EngineCoreRequestType.EXECUTOR_FAILED, b""))
                    return
