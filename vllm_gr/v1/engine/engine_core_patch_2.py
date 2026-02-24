# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH, BEAM_FORK, and BEAM_SEARCH."""

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
        BeamSearchRequest,
        BeamSearchResult,
        BeamSearchSequenceResult,
        PatchedEngineCoreOutputs,
    )

    # Inject types into vllm.v1.engine module
    engine_mod.BeamForkRequest = BeamForkRequest
    engine_mod.BeamSearchRequest = BeamSearchRequest
    engine_mod.BeamSearchSequenceResult = BeamSearchSequenceResult
    engine_mod.BeamSearchResult = BeamSearchResult

    # Add new enum members: ADD_BATCH, BEAM_FORK, BEAM_SEARCH
    _add_enum_member("ADD_BATCH", b"\x05")
    _add_enum_member("BEAM_FORK", b"\x06")
    _add_enum_member("BEAM_SEARCH", b"\x07")

    # Replace EngineCoreOutputs with patched version
    engine_mod.EngineCoreOutputs = PatchedEngineCoreOutputs

    # Wrap EngineCore.__init__ to add beam state
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)

    # Bind _cache_beam_request and _handle_beam_fork on both classes
    EngineCore._cache_beam_request = _cache_beam_request
    EngineCore._handle_beam_fork = _handle_beam_fork
    EngineCoreProc._cache_beam_request = _cache_beam_request
    EngineCoreProc._handle_beam_fork = _handle_beam_fork

    # Replace methods on BOTH EngineCore and EngineCoreProc to ensure
    # the patched version is found regardless of MRO resolution.
    EngineCore._process_engine_step = patched_process_engine_step
    EngineCore._handle_client_request = patched_handle_client_request
    EngineCoreProc._process_engine_step = patched_process_engine_step
    EngineCoreProc._handle_client_request = patched_handle_client_request

    # Replace EngineCoreProc.process_input_sockets
    EngineCoreProc.process_input_sockets = patched_process_input_sockets

    logger.info("Engine-core patches (ADD_BATCH, BEAM_FORK, BEAM_SEARCH) "
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
        # For BEAM_SEARCH-in-engine (lazy import to avoid missing module)
        self.active_beam_searches: list = []

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
# EngineCore._process_engine_step replacement
# ---------------------------------------------------------------------------

def patched_process_engine_step(self) -> bool:
    """Replacement for EngineCore._process_engine_step with beam search."""

    # Step the engine core.
    outputs, model_executed = self.step_fn()

    # Feed outputs to active beam search controllers and filter
    # beam sub-request outputs before sending to client.
    if self.active_beam_searches and outputs:
        # Collect all pending beam request IDs to filter them out.
        beam_rids: set[str] = set()
        for ctrl in self.active_beam_searches:
            beam_rids.update(ctrl._pending_rids)

        for ctrl in self.active_beam_searches:
            ctrl.on_outputs(outputs)

        # Filter beam sub-request outputs from client-bound outputs.
        if beam_rids:
            filtered_outputs: dict[int, _engine_mod.EngineCoreOutputs] = {}
            for client_idx, engine_outputs in outputs.items():
                non_beam = [
                    o for o in engine_outputs.outputs
                    if o.request_id not in beam_rids
                ]
                if non_beam or engine_outputs.scheduler_stats:
                    filtered = _engine_mod.EngineCoreOutputs(
                        engine_index=engine_outputs.engine_index,
                        outputs=non_beam,
                        scheduler_stats=engine_outputs.scheduler_stats,
                        timestamp=engine_outputs.timestamp,
                    )
                    filtered_outputs[client_idx] = filtered
            outputs = filtered_outputs

        # Collect finished controllers and emit their results.
        finished = [c for c in self.active_beam_searches if c.is_done]
        for c in finished:
            self.output_queue.put_nowait(
                (c.client_index,
                 _engine_mod.EngineCoreOutputs(beam_search_result=c.result))
            )
            self.active_beam_searches.remove(c)

    # Put EngineCoreOutputs into the output queue.
    for output in outputs.items() if outputs else ():
        self.output_queue.put_nowait(output)

    # Post-step hook.
    self.post_step(model_executed)

    # If no model execution happened but there are waiting requests
    # (e.g., WAITING_FOR_REMOTE_KVS), yield the GIL briefly to allow
    # background threads (like NIXL handshake) to make progress.
    # Without this, the tight polling loop can starve background threads.
    if not model_executed and self.scheduler.has_unfinished_requests():
        time.sleep(0.001)

    return model_executed


# ---------------------------------------------------------------------------
# EngineCore._handle_client_request replacement
# ---------------------------------------------------------------------------

def patched_handle_client_request(
    self, request_type: EngineCoreRequestType, request: Any
) -> None:
    """Replacement for EngineCore._handle_client_request with beam search."""
    from vllm.v1.engine import UtilityOutput, UtilityResult

    if request_type == EngineCoreRequestType.ADD:
        req, request_wave = request
        self.add_request(req, request_wave)
    elif request_type == EngineCoreRequestType.ABORT:
        self.abort_requests(request)
    elif request_type == EngineCoreRequestType.UTILITY:
        client_idx, call_id, method_name, args = request
        output = UtilityOutput(call_id)
        try:
            method = getattr(self, method_name)
            result = method(*self._convert_msgspec_args(method, args))
            output.result = UtilityResult(result)
        except BaseException as e:
            logger.exception("Invocation of %s method failed", method_name)
            output.failure_message = (
                f"Call to {method_name} method failed: {str(e)}"
            )
        self.output_queue.put_nowait(
            (client_idx, _engine_mod.EngineCoreOutputs(utility_output=output))
        )
    elif request_type == EngineCoreRequestType.BEAM_SEARCH:
        from vllm_gr.v1.engine.beam_search_2 import BeamSearchController
        controller = BeamSearchController(self, request)
        controller.submit_step()
        self.active_beam_searches.append(controller)
    elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
        raise RuntimeError("Executor failed.")
    else:
        logger.error(
            "Unrecognized input request type encountered: %s", request_type
        )


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
    ADD_BATCH, BEAM_FORK, and BEAM_SEARCH decoders."""
    from contextlib import ExitStack

    import zmq

    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
    from vllm.v1.serial_utils import MsgpackDecoder

    from vllm_gr.v1.engine.types import BeamForkRequest, BeamSearchRequest

    # Msgpack serialization decoding.
    add_request_decoder = MsgpackDecoder(EngineCoreRequest)
    add_batch_decoder = MsgpackDecoder(list[EngineCoreRequest])
    beam_fork_decoder = MsgpackDecoder(BeamForkRequest)
    beam_search_decoder = MsgpackDecoder(BeamSearchRequest)
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

                elif request_type == EngineCoreRequestType.BEAM_SEARCH:
                    request = beam_search_decoder.decode(data_frames)

                else:
                    request = generic_decoder.decode(data_frames)

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
