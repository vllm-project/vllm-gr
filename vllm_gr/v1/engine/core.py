# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

import threading
import time
from functools import partial
from typing import Any

import msgspec
from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)

logger = init_logger(__name__)


# ------------------------------------------------------------------------------------
# _cache_beam_request / _handle_beam_request_step_update  (ported from source core.py)
# ------------------------------------------------------------------------------------


def _ensure_beam_state(self) -> None:
    if not hasattr(self, "beam_cache"):
        self.beam_cache = {}
    if not hasattr(self, "beam_cache_lock"):
        self.beam_cache_lock = threading.Lock()


def _cache_beam_request(self, request) -> None:
    """Cache Request state for later beam forking.

    Thread-safe: uses beam_cache_lock to prevent data races.
    """
    _ensure_beam_state(self)
    sampling_params = request.sampling_params
    if sampling_params is None:
        raise ValueError(
            "Beam requests must have sampling_params; pooling requests are not supported."
        )
    with self.beam_cache_lock:
        self.beam_cache[request.request_id] = {
            "all_token_ids": list(request._all_token_ids),
            "block_hashes": list(request.block_hashes),
            "eos_token_id": sampling_params.eos_token_id,
            "lora_request": request.lora_request,
            "cache_salt": request.cache_salt,
            "mm_features": request.mm_features,
            "prompt_embeds": request.prompt_embeds,
            "num_prompt_tokens": request.num_prompt_tokens,
        }


def _emit_beam_session_error(self, update, message: str) -> None:
    """Finish a failed grouped update on the session ID the client awaits.

    The grouped V2 runner represents all fork children with one EngineCore
    request named ``session_id``. Emitting failures for child IDs leaves the
    registered session output collector waiting forever.
    """
    session_id = update.session_id
    logger.error("BEAM_REQUEST_STEP_UPDATE: session=%s %s", session_id, message)
    with self.beam_cache_lock:
        self.beam_cache.pop(session_id, None)

    def _safe_abort_put(item):
        try:
            self.aborts_queue.put_nowait(item)
        except Exception:
            # Queue may be full or closed on shutdown; the error path should not
            # mask the original beam failure.
            logger.debug(
                "BEAM_REQUEST_STEP_UPDATE: failed to enqueue abort for %s",
                item,
                exc_info=True,
            )

    _safe_abort_put(session_id)
    for abort_id in update.pruned_ids:
        _safe_abort_put(abort_id)
    self.output_queue.put_nowait(
        (
            update.client_index,
            EngineCoreOutputs(
                engine_index=self.engine_index,
                finished_requests={session_id},
                outputs=[
                    EngineCoreOutput(
                        request_id=session_id,
                        new_token_ids=[],
                        finish_reason=FinishReason.ERROR,
                    )
                ],
            ),
        )
    )


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

        # We MUST abort the running beam-request session_id itself!
        self.aborts_queue.put_nowait(session_id)
        if update.pruned_ids:
            for abort_id in update.pruned_ids:
                self.aborts_queue.put_nowait(abort_id)
        return

    # Validate parallel arrays
    if (
        len(update.beam_tokens) != beam_width
        or len(update.parent_beam_ids) != beam_width
        or len(update.child_beam_ids) != beam_width
    ):
        _emit_beam_session_error(self, update, "length mismatch")
        return

    # Fetch the shared prefill cache entry
    with self.beam_cache_lock:
        session_cached = self.beam_cache.get(session_id)

    if session_cached is None and update.beam_width > 0:
        _emit_beam_session_error(self, update, "not in cache")
        return

    arrival = time.time()
    inputs_to_push = []

    prefill_tokens = session_cached["all_token_ids"]
    prefill_block_hashes = list(session_cached["block_hashes"])
    lora_request = session_cached["lora_request"]
    cache_salt = session_cached["cache_salt"]
    mm_features = session_cached["mm_features"]
    prompt_embeds = session_cached["prompt_embeds"]
    eos = session_cached["eos_token_id"]

    sampling_params = update.sampling_params.clone()
    if eos is not None or update.eos_token_id is not None:
        sampling_params.update_from_generation_config(
            {}, update.eos_token_id if update.eos_token_id is not None else eos
        )

    # Pack the shared prefill and all beam suffixes into a single contiguous Beam Request
    beam_token_ids = list(prefill_tokens)
    decode_steps = len(update.beam_tokens[0]) if update.beam_tokens else 0
    if not all(len(t) == decode_steps for t in update.beam_tokens):
        _emit_beam_session_error(self, update, "beam token lengths differ")
        return
    for gen_tokens in update.beam_tokens:
        beam_token_ids.extend(gen_tokens)

    req = Request(
        request_id=session_id,  # Single request ID representing all beams
        prompt_token_ids=beam_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        client_index=update.client_index,
        arrival_time=arrival,
        lora_request=(update.lora_request or lora_request),
        cache_salt=update.cache_salt or cache_salt,
        priority=update.priority,
        trace_headers=update.trace_headers,
        prompt_embeds=prompt_embeds,
        mm_features=mm_features,
        block_hasher=None,
    )

    # Start with the prefill block hashes; block_hasher extends it incrementally
    req.block_hashes = prefill_block_hashes
    if self.request_block_hasher is not None:
        req.get_hash_new_full_blocks = partial(self.request_block_hasher, req)
        req.block_hashes.extend(req.get_hash_new_full_blocks())

    # Metadata Enrichment
    req.is_beam = True
    req.beam_width = beam_width
    req.is_beam_decode = True  # Flag indicating it's the decode stage
    req.beam_decode_steps = decode_steps
    req.prefix_len = update.prefix_len

    logger.info(
        "[vllm-gr-beam] enqueue grouped update: session=%s beam_width=%d "
        "decode_steps=%d prefix_len=%d packed_prompt_tokens=%d pruned=%d",
        session_id,
        beam_width,
        decode_steps,
        update.prefix_len,
        len(beam_token_ids),
        len(update.pruned_ids),
    )

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

    import zmq
    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine import (
        EngineCoreReadyResponse,
        EngineCoreRequest,
        EngineCoreRequestType,
    )
    from vllm.v1.serial_utils import MsgpackDecoder

    from vllm_gr.v1.engine.engine_core_patch import VLLM_GR_PROTOCOL_VERSION
    from vllm_gr.v1.engine.types import BeamRequestStepUpdate

    # Msgpack serialization decoding with optional tensor IPC receiver.
    add_request_decoder = MsgpackDecoder(
        EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
    )
    add_batch_decoder = MsgpackDecoder(
        list[EngineCoreRequest], oob_tensor_provider=self.tensor_ipc_receiver
    )
    beam_request_decoder = MsgpackDecoder(
        BeamRequestStepUpdate, oob_tensor_provider=self.tensor_ipc_receiver
    )
    generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

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
        # Encode as a dict so we can attach the vllm-gr protocol version.
        # The upstream dataclass ignores unknown fields on decode, so older
        # clients remain compatible; vllm-gr clients verify the version below.
        ready_payload = msgspec.msgpack.encode(
            {
                "max_model_len": ready_response.max_model_len,
                "num_gpu_blocks": ready_response.num_gpu_blocks,
                "dp_stats_address": ready_response.dp_stats_address,
                "dtype": ready_response.dtype,
                "vllm_gr_protocol_version": VLLM_GR_PROTOCOL_VERSION,
            }
        )
        for input_socket in input_sockets:
            input_socket.send(ready_payload)
            poller.register(input_socket, zmq.POLLIN)

        if coord_socket is not None:
            coord_ready = coord_socket.recv()
            if coord_ready != b"READY":
                raise RuntimeError(
                    f"Expected READY handshake from coordinator, but received {coord_ready!r}."
                )
            poller.register(coord_socket, zmq.POLLIN)

        ready_event.set()
        del ready_event
        while True:
            for input_socket, _ in poller.poll():
                type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                if type_frame.buffer == b"READY":
                    if input_socket != coord_socket:
                        raise RuntimeError("Received READY frame from a non-coordinator socket.")
                    continue
                request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                request: Any
                if request_type == EngineCoreRequestType.ADD:
                    req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                    try:
                        request = self.preprocess_add_request(req)
                    except Exception:
                        self._handle_request_preproc_error(req)
                        continue

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
                    continue

                elif request_type == EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE:
                    beam_update: BeamRequestStepUpdate = beam_request_decoder.decode(data_frames)
                    self._handle_beam_request_step_update(beam_update)
                    continue

                else:
                    request = generic_decoder.decode(data_frames)

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
