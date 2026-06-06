# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_FORK."""

from __future__ import annotations

import time
from functools import partial
from typing import Any

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# _cache_beam_request / _handle_beam_fork  (ported from source core.py)
# ---------------------------------------------------------------------------


def _cache_beam_request(self, request) -> None:
    """Cache Request state for later beam forking.

    Stores ``(request, block_hashes_snapshot)`` — the Request object itself
    (immutable fields read via attribute access) plus a snapshot of its
    ``block_hashes`` (the only scheduler-mutated list we need).

    Thread-safe: uses beam_cache_lock to prevent data races.
    """
    with self.beam_cache_lock:
        self.beam_cache[request.request_id] = (
            request,
            list(request.block_hashes),
        )


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


def _handle_beam_fork(self, fork_req) -> None:
    """Create child Requests from cached parent state.

    Uses a template-clone optimisation: the first child is built with the
    normal ``Request()`` constructor, then a lean shared-fields dict is
    snapshotted (excluding per-child fields).  Subsequent children are
    created via ``object.__new__(Request)`` + shared dict copy + per-child
    field insertion, avoiding both the full constructor cost and the
    copy-then-overwrite overhead of a full ``__dict__.copy()``.

    Thread-safe: uses beam_cache_lock to prevent data races.
    """
    # Deferred imports to avoid circular dependency (engine ↔ request).
    from vllm.v1.request import Request
    from vllm.v1.utils import ConstantList

    shared_fields: dict[str, Any] | None = None
    # All children share the same arrival_time — acceptable because every
    # child in a BEAM_FORK batch originates from the same prompt at the
    # same logical point in time.
    now = time.time()
    has_block_hasher = self.request_block_hasher is not None

    for parent_id, child_id, token_id in zip(
        fork_req.parent_ids, fork_req.child_ids, fork_req.token_ids
    ):
        with self.beam_cache_lock:
            cached = self.beam_cache.get(parent_id)

        if cached is None:
            logger.error("BEAM_FORK: parent %s not in cache", parent_id)
            self.output_queue.put_nowait(
                (
                    fork_req.client_index,
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

        cached_req, cached_block_hashes = cached
        child_token_ids = cached_req.prompt_token_ids + [token_id]

        if shared_fields is None:
            # First valid child — use normal constructor
            req = Request(
                request_id=child_id,
                prompt_token_ids=child_token_ids,
                sampling_params=fork_req.sampling_params,
                pooling_params=None,
                eos_token_id=(
                    fork_req.eos_token_id
                    if fork_req.eos_token_id is not None
                    else cached_req.eos_token_id
                ),
                client_index=fork_req.client_index,
                arrival_time=now,
                lora_request=(fork_req.lora_request or cached_req.lora_request),
                cache_salt=fork_req.cache_salt or cached_req.cache_salt,
                priority=fork_req.priority,
                trace_headers=fork_req.trace_headers,
                prompt_embeds=cached_req.prompt_embeds,
                mm_features=cached_req.mm_features,
                block_hasher=None,
            )
            # Snapshot shared fields BEFORE enqueuing (scheduler may mutate).
            #
            # SHALLOW COPY — mutable objects (sampling_params, prompt_embeds,
            # mm_features, etc.) are shared by reference across all children.
            # This is safe because all children in a BEAM_FORK originate from
            # the same prompt and the scheduler does not mutate these objects
            # in-place.  If that invariant ever changes, the values below must
            # be deep-copied.
            shared_fields = {k: v for k, v in req.__dict__.items() if k not in _PER_CHILD_FIELDS}
        else:
            # Children 2..N — lean clone from shared dict.
            # We use object.__new__ + __dict__ assignment instead of the full
            # Request() constructor for performance (avoids re-validating
            # identical fields).  Request is an upstream vLLM class that uses
            # __dict__ (not __slots__), so this is safe.
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

        # Clone parent's block hashes and incrementally compute new ones
        req.block_hashes = list(cached_block_hashes)
        if has_block_hasher:
            req.get_hash_new_full_blocks = partial(self.request_block_hasher, req)
            req.block_hashes.extend(req.get_hash_new_full_blocks())
        else:
            req.get_hash_new_full_blocks = None

        self._cache_beam_request(req)
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, (req, fork_req.current_wave)))

    # Clean up parents + aborted beams from cache
    with self.beam_cache_lock:
        for pid in set(fork_req.parent_ids) | set(fork_req.abort_ids):
            self.beam_cache.pop(pid, None)


# ---------------------------------------------------------------------------
# EngineCoreProc.process_output_sockets replacement
# ---------------------------------------------------------------------------


def process_output_sockets(
    self,
    output_paths,
    coord_output_path,
    engine_index,
):
    """Replacement for EngineCoreProc.process_output_sockets that fixes a
    ZMQ zero-copy race condition (issue #91).

    The original upstream implementation reuses a bytearray pool and sends
    frames with ``copy=False``, relying on ``MessageTracker.done`` to know
    when a buffer is safe to recycle.  Under high concurrency the tracker
    can report completion before ZMQ's I/O thread has fully released the
    memory, leading to garbled msgpack frames.

    Fix: allocate a fresh buffer each iteration so no buffer is ever shared
    with a prior in-flight send.  Zero-copy (``copy=False``) is preserved —
    the ZMQ Frame holds a Python reference to the buffer, preventing GC
    until transmission completes.  Tracker allocation is skipped for the
    common single-frame case (no tensor backing buffers) since there is
    nothing to track.
    """
    from contextlib import ExitStack

    import zmq
    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.serial_utils import MsgpackEncoder

    encoder = MsgpackEncoder()
    # Keep references to outputs until ZMQ is finished with their tensor
    # backing buffers (buffers[1:] are zero-copy views into tensor memory).
    # Only used when len(buffers) > 1 (tensor/numpy data in outputs),
    # which does not occur for beam-search outputs (token IDs / scalars).
    pending: list[tuple[zmq.MessageTracker, object]] = []

    with ExitStack() as stack, zmq.Context() as ctx:
        sockets = [
            stack.enter_context(make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))
            for output_path in output_paths
        ]
        coord_socket = (
            stack.enter_context(
                make_zmq_socket(ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000)
            )
            if coord_output_path is not None
            else None
        )

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

            # Release outputs whose tensor backing buffers have been sent.
            pending = [p for p in pending if not p[0].done]

            buffer = bytearray()
            buffers = encoder.encode_into(outputs, buffer)

            if len(buffers) > 1:
                # Tensor backing buffers present — need tracking to
                # prevent GC of outputs while ZMQ reads tensor memory.
                tracker = sockets[client_index].send_multipart(buffers, copy=False, track=True)
                if not tracker.done:
                    pending.append((tracker, outputs))
            else:
                # Common path: single msgpack frame, no tensors.
                # copy=False with a unique buffer is safe — the ZMQ
                # Frame prevents the buffer from being freed until
                # transmission completes.  No tracker needed.
                sockets[client_index].send(buffers[0], copy=False)


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
    ADD_BATCH and BEAM_FORK decoders."""
    from contextlib import ExitStack

    import zmq
    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
    from vllm.v1.serial_utils import MsgpackDecoder

    from vllm_gr.v1.engine.types import BeamForkRequest

    # Msgpack serialization decoding.
    add_request_decoder = MsgpackDecoder(EngineCoreRequest)
    add_batch_decoder = MsgpackDecoder(list[EngineCoreRequest])
    beam_fork_decoder = MsgpackDecoder(BeamForkRequest)
    generic_decoder = MsgpackDecoder()

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
        for input_socket in input_sockets:
            input_socket.send(b"")
            poller.register(input_socket, zmq.POLLIN)

        if coord_socket is not None:
            assert coord_socket.recv() == b"READY"
            poller.register(coord_socket, zmq.POLLIN)

        ready_event.set()
        del ready_event
        while True:
            for input_socket, _ in poller.poll():
                type_frame, *data_frames = input_socket.recv_multipart(copy=False)
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

                elif request_type == EngineCoreRequestType.BEAM_FORK:
                    fork_req: BeamForkRequest = beam_fork_decoder.decode(data_frames)
                    self._handle_beam_fork(fork_req)
                    continue

                else:
                    request = generic_decoder.decode(data_frames)

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
