# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm
"""Atomic GR session admission and EngineCore socket transport."""

from __future__ import annotations

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


def loaded_constraint_trie_count(worker: Any) -> int:
    """Read the validated, live table once through the executor startup RPC."""
    backend = getattr(worker.model_runner, "constraint_backend", None)
    return backend.info.table_metadata.trie_count if backend is not None else 1


def is_gr_session_retired(self: Any, session_id: str) -> bool:
    """Ordered Core-thread query; never block the thread that drains BatchQueue."""
    return session_id not in getattr(self.scheduler, "_gr_v1_sessions", {})


def send_gr_rejection(engine: Any, engine_request: Any, message: str) -> None:
    """Answer a refused GR session with the terminal payload it never produced.

    Upstream's error output carries no ``gr_batch_result``, and a frontend can
    only read that absence as a broken engine. Sending the failure the session
    would have reported keeps the reason, and its status class, intact.
    """
    from vllm_gr.v1.beam.batch_contracts import admission_rejected

    request_id = engine_request.request_id
    output = EngineCoreOutput(request_id, [], finish_reason=FinishReason.ERROR)
    output.kv_transfer_params = {"gr_batch_result": admission_rejected(request_id, message)}
    engine.output_queue.put_nowait(
        (
            engine_request.client_index,
            EngineCoreOutputs(outputs=[output], finished_requests=[request_id]),
        )
    )


def _handle_submit_gr_session(self, batch_request) -> None:
    """Keep a caller-formed batch atomic across the input-thread boundary."""
    if not batch_request.items:
        logger.error("Rejecting an empty beam_search_v1 batch")
        return
    try:
        from vllm_gr.v1.engine.beam_search_v1 import (
            _GRRequestValidation,
            validate_gr_request,
        )

        config = getattr(self, "vllm_config", None)
        trie_count = getattr(self, "_gr_trie_count", None)
        validate_gr_request(
            batch_request,
            config=config,
            trie_count=trie_count,
        )
    except ValueError as error:
        # A malformed batch is a request rejection, not a preprocessing fault.
        if getattr(self, "output_queue", None) is None:
            from vllm_gr.v1.engine.gr_async_scheduler import GRAdmissionError

            raise GRAdmissionError(str(error)) from error
        send_gr_rejection(self, batch_request.items[0].request, str(error))
        return
    except Exception:
        handler = getattr(self, "_handle_request_preproc_error", None)
        if handler is None:
            raise
        handler(batch_request.items[0].request)
        return
    if hasattr(self, "input_queue"):
        self.input_queue.put_nowait((EngineCoreRequestType.SUBMIT_GR_SESSION, batch_request))
        return
    _admit_gr_session(
        self, batch_request, _validation=_GRRequestValidation(batch_request, config, trie_count)
    )


def _admit_gr_session(self, batch_request, *, _validation=None) -> None:
    """Preprocess and admit one complete GR session on the core thread."""
    if not batch_request.items:
        logger.error("Rejecting an empty beam_search_v1 batch")
        return
    engine_requests = tuple(item.request for item in batch_request.items)
    try:
        from vllm_gr.v1.engine.beam_search_v1 import (
            GRDeviceRequestError,
            _GRRequestValidation,
            validate_gr_device_config,
            validate_gr_execution_config,
            validate_gr_request,
        )

        config = getattr(self, "vllm_config", None)
        trie_count = getattr(self, "_gr_trie_count", None)
        if not isinstance(_validation, _GRRequestValidation) or not _validation.matches(
            batch_request, config, trie_count
        ):
            validate_gr_request(batch_request, config=config, trie_count=trie_count)
    except ValueError as error:
        # The request is answerable, so it must not take the preprocessing
        # error path: that one logs a traceback and sends a bare
        # FinishReason.ERROR, which the frontend can only read as a fault.
        # In-process callers have no output socket, so the rejection travels
        # to them as the exception the multiprocess handler encodes.
        if getattr(self, "output_queue", None) is None:
            from vllm_gr.v1.engine.gr_async_scheduler import GRAdmissionError

            raise GRAdmissionError(str(error)) from error
        send_gr_rejection(self, engine_requests[0], str(error))
        return
    try:
        # Deployment configuration stays on the preprocessing error path: an
        # engine started with the wrong settings is a server fault, not
        # something the caller can fix by resending.
        validate_gr_execution_config(self.vllm_config)
        device = getattr(getattr(self.vllm_config, "device_config", None), "device", None)
        if str(device).split(":")[0] == "npu":
            validate_gr_device_config(self.vllm_config, batch_request)
        else:
            validate_gr_device_config(self.vllm_config)
        if self.batch_queue is None or self.batch_queue_size < 2:
            raise RuntimeError("beam_search_v1 requires a non-blocking EngineCore batch queue")
        preprocessed_batch = []
        for engine_request in engine_requests:
            owned_request = copy(engine_request)
            # Native Request construction only reads nested sampling fields. GR
            # replaces stop_token_ids below; it never mutates the original list.
            # The Worker separately detaches every mutable field it retains.
            owned_request.sampling_params = copy(engine_request.sampling_params)
            preprocessed_batch.append(self.preprocess_add_request(owned_request))
    except GRDeviceRequestError as error:
        # Device-specific request limits have the same answerable rejection
        # contract as shared validation. Deployment faults still use the
        # preprocessing handler below.
        if getattr(self, "output_queue", None) is None:
            from vllm_gr.v1.engine.gr_async_scheduler import GRAdmissionError

            raise GRAdmissionError(str(error)) from error
        send_gr_rejection(self, engine_requests[0], str(error))
        return
    except Exception:
        handler = getattr(self, "_handle_request_preproc_error", None)
        if handler is None:
            raise  # InprocClient has no output-socket error handler.
        # The batch has one external result channel, owned by the first item.
        handler(engine_requests[0])
        return
    for member_index, preprocessed in enumerate(preprocessed_batch):
        native_request = preprocessed[0]
        # GR owns stop-token/EOS handling. Native tokens are placeholders for
        # the device-owned batch state and cannot terminate members alone.
        native_request.sampling_params.ignore_eos = True
        native_request.sampling_params.stop_token_ids = []
        native_request.gr_batch_request = batch_request
        native_request.gr_batch_member_index = member_index
        # SUBMIT_GR_SESSION reaches this handler as one busy-loop message, so
        # native scheduling cannot observe a partially inserted batch.
        # Native preprocessing can mutate values shared by the shallow copy.
        # The scheduler must validate the final batch at ADD; the entry proof
        # cannot certify data across that extensible boundary.
        self.add_request(*preprocessed)


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

    if request_type == EngineCoreRequestType.SUBMIT_GR_SESSION:
        _admit_gr_session(self, payload)
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
    pinned upstream loop. GR batches are decoded and queued intact; the
    busy-loop handler admits them on the core thread.
    """
    from contextlib import ExitStack

    import msgspec
    import zmq
    from vllm.multimodal.cache import MultiModalCacheMissError
    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.fault_tolerance.engine_core_sentinel import FT_UTILITY_METHOD

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
        # Upstream owns the complete post-initialization configuration schema.
        ready_response = self._make_ready_response()
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
                    except MultiModalCacheMissError as error:
                        self._handle_mm_cache_miss(req, error)
                        continue
                    except Exception:
                        self._handle_request_preproc_error(req)
                        continue

                elif request_type == EngineCoreRequestType.SUBMIT_GR_SESSION:
                    # Preserve the caller-formed batch as one queue item. The
                    # busy-loop handler preprocesses and admits every member
                    # before native scheduling can run.
                    request = decoded_payload

                elif request_type == EngineCoreRequestType.UTILITY:
                    request = decoded_payload
                    client_idx, call_id, method, args = request
                    if method == FT_UTILITY_METHOD:
                        self.ft_sentinel.handle_command(client_idx, call_id, args[0])
                        continue

                else:
                    request = decoded_payload

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
