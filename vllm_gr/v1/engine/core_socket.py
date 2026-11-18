# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm
"""Pinned vLLM socket loop for atomic GR session transport."""

from __future__ import annotations

from typing import Any

from vllm.v1.engine import EngineCoreRequestType


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
