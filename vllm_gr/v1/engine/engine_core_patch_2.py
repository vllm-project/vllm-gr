# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_FORK."""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import vllm.v1.engine as _engine_mod

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers: add an enum member to EngineCoreRequestType at runtime
# ---------------------------------------------------------------------------

def _add_enum_member(name: str, value: bytes) -> None:
    """Dynamically add a member to EngineCoreRequestType if absent."""
    if hasattr(EngineCoreRequestType, name):
        return
    member = object.__new__(EngineCoreRequestType)
    member._name_ = name
    member._value_ = value
    EngineCoreRequestType._value2member_map_[value] = member
    EngineCoreRequestType._member_map_[name] = member
    EngineCoreRequestType._member_names_.append(name)
    type.__setattr__(EngineCoreRequestType, name, member)


# ---------------------------------------------------------------------------
# Child-process patch applicator — called from patched run_engine_core
# ---------------------------------------------------------------------------

def apply_engine_core_child_patches():
    """Apply all patches inside the EngineCore child process.

    This must be called at the start of run_engine_core (the child process
    entry point) because multiprocessing 'spawn' starts a fresh Python
    interpreter that does not inherit monkey patches from the parent.
    """
    import vllm.v1.engine as engine_mod
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    from vllm_gr.v1.engine.types import (
        BeamForkRequest,
    )

    # Inject types into vllm.v1.engine module
    engine_mod.BeamForkRequest = BeamForkRequest

    # Add new enum members: ADD_BATCH, BEAM_FORK
    _add_enum_member("ADD_BATCH", b"\x05")
    _add_enum_member("BEAM_FORK", b"\x06")

    # Wrap EngineCore.__init__ to add beam state
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)

    # Bind _cache_beam_request and _handle_beam_fork on both classes
    EngineCore._cache_beam_request = _cache_beam_request
    EngineCore._handle_beam_fork = _handle_beam_fork
    EngineCoreProc._cache_beam_request = _cache_beam_request
    EngineCoreProc._handle_beam_fork = _handle_beam_fork

    # Replace EngineCoreProc.process_input_sockets
    EngineCoreProc.process_input_sockets = patched_process_input_sockets

    logger.info("Engine-core patches (ADD_BATCH, BEAM_FORK) "
                "applied in child process.")


def patched_run_engine_core(*args, **kwargs):
    """Wrapper for EngineCoreProc.run_engine_core that applies patches
    before the engine starts.  This is a module-level function so that
    multiprocessing 'spawn' can pickle it by qualified name.

    In the spawned child process, all modules are freshly imported, so
    EngineCoreProc.run_engine_core is the original unpatched version.
    We apply our patches first, then delegate to the original.
    """
    from vllm.v1.engine.core import EngineCoreProc
    # Save reference to original before applying patches
    original_run = EngineCoreProc.run_engine_core
    apply_engine_core_child_patches()
    original_run(*args, **kwargs)


# ---------------------------------------------------------------------------
# EngineCore.__init__ wrapper
# ---------------------------------------------------------------------------

def make_patched_engine_core_init(original_init):
    """Wrap EngineCore.__init__ to add beam state."""

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.beam_cache: dict[str, dict] = {}
        
    return patched_init


# ---------------------------------------------------------------------------
# _cache_beam_request / _handle_beam_fork  (ported from source core.py)
# ---------------------------------------------------------------------------

def _cache_beam_request(self, request) -> None:
    """Cache Request state for later beam forking (input thread only)."""
    self.beam_cache[request.request_id] = {
        'all_token_ids': list(request._all_token_ids),
        'block_hashes': list(request.block_hashes),
        'eos_token_id': request.eos_token_id,
        'lora_request': request.lora_request,
        'cache_salt': request.cache_salt,
        'mm_features': request.mm_features,
        'prompt_embeds': request.prompt_embeds,
        'num_prompt_tokens': request.num_prompt_tokens,
    }


def _handle_beam_fork(self, fork_req) -> None:
    """Create child Requests from cached parent state (input thread)."""
    from vllm.v1.request import Request

    for parent_id, child_id, token_id in zip(
        fork_req.parent_ids, fork_req.child_ids, fork_req.token_ids
    ):
        cached = self.beam_cache.get(parent_id)
        if cached is None:
            logger.error("BEAM_FORK: parent %s not in cache", parent_id)
            continue

        child_token_ids = cached['all_token_ids'] + [token_id]

        req = Request(
            request_id=child_id,
            prompt_token_ids=child_token_ids,
            sampling_params=fork_req.sampling_params,
            pooling_params=None,
            eos_token_id=(fork_req.eos_token_id
                          if fork_req.eos_token_id is not None
                          else cached['eos_token_id']),
            client_index=fork_req.client_index,
            arrival_time=time.time(),
            lora_request=(fork_req.lora_request
                          or cached['lora_request']),
            cache_salt=fork_req.cache_salt or cached['cache_salt'],
            priority=fork_req.priority,
            trace_headers=fork_req.trace_headers,
            prompt_embeds=cached['prompt_embeds'],
            mm_features=cached['mm_features'],
            block_hasher=None,  # Skip O(N) hash computation
        )

        # Clone parent's block hashes and incrementally compute new ones
        req.block_hashes = list(cached['block_hashes'])
        if self.request_block_hasher is not None:
            req.get_hash_new_full_blocks = partial(
                self.request_block_hasher, req)
            req.block_hashes.extend(req.get_hash_new_full_blocks())

        self._cache_beam_request(req)
        self.input_queue.put_nowait(
            (EngineCoreRequestType.ADD, (req, fork_req.current_wave)))

    # Clean up parents + aborted beams from cache
    for pid in set(fork_req.parent_ids) | set(fork_req.abort_ids):
        self.beam_cache.pop(pid, None)


# ---------------------------------------------------------------------------
# EngineCoreProc.process_input_sockets replacement
# ---------------------------------------------------------------------------

def patched_process_input_sockets(
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
                make_zmq_socket(
                    ctx, input_address, zmq.DEALER, identity=identity,
                    bind=False
                )
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
                type_frame, *data_frames = input_socket.recv_multipart(
                    copy=False
                )
                request_type = EngineCoreRequestType(
                    bytes(type_frame.buffer)
                )

                request: Any
                if request_type == EngineCoreRequestType.ADD:
                    req: EngineCoreRequest = (
                        add_request_decoder.decode(data_frames)
                    )
                    try:
                        request = self.preprocess_add_request(req)
                    except Exception:
                        self._handle_request_preproc_error(req)
                        continue

                elif request_type == EngineCoreRequestType.ADD_BATCH:
                    batch: list[EngineCoreRequest] = (
                        add_batch_decoder.decode(data_frames))
                    for req in batch:
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                        self._cache_beam_request(request[0])
                        self.input_queue.put_nowait(
                            (EngineCoreRequestType.ADD, request))
                    continue

                elif request_type == EngineCoreRequestType.BEAM_FORK:
                    fork_req: BeamForkRequest = (
                        beam_fork_decoder.decode(data_frames))
                    self._handle_beam_fork(fork_req)
                    continue

                else:
                    request = generic_decoder.decode(data_frames)

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
