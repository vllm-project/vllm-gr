# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support beam search."""

from __future__ import annotations

import time
from typing import Any

import vllm.v1.engine as _engine_mod

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.engine.beam_search import BeamSearchController
from vllm_gr.v1.engine.types import BeamSearchResult

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Child-process patch applicator — called from patched run_engine_core
# ---------------------------------------------------------------------------

def apply_engine_core_child_patches():
    """Apply all beam search patches inside the EngineCore child process.

    This must be called at the start of run_engine_core (the child process
    entry point) because multiprocessing 'spawn' starts a fresh Python
    interpreter that does not inherit monkey patches from the parent.
    """
    import vllm.v1.engine as engine_mod
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    from vllm_gr.v1.engine.types import (
        BeamSearchRequest,
        BeamSearchResult,
        BeamSearchSequenceResult,
        PatchedEngineCoreOutputs,
    )

    # Inject beam search types into vllm.v1.engine module
    engine_mod.BeamSearchRequest = BeamSearchRequest
    engine_mod.BeamSearchSequenceResult = BeamSearchSequenceResult
    engine_mod.BeamSearchResult = BeamSearchResult

    # Add BEAM_SEARCH to EngineCoreRequestType enum
    if not hasattr(EngineCoreRequestType, "BEAM_SEARCH"):
        beam_val = b"\x06"
        beam_member = object.__new__(EngineCoreRequestType)
        beam_member._name_ = "BEAM_SEARCH"
        beam_member._value_ = beam_val
        EngineCoreRequestType._value2member_map_[beam_val] = beam_member
        EngineCoreRequestType._member_map_["BEAM_SEARCH"] = beam_member
        EngineCoreRequestType._member_names_.append("BEAM_SEARCH")
        type.__setattr__(EngineCoreRequestType, "BEAM_SEARCH", beam_member)

    # Replace EngineCoreOutputs with patched version
    engine_mod.EngineCoreOutputs = PatchedEngineCoreOutputs

    # Wrap EngineCore.__init__ to add active_beam_searches
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)

    # Replace methods on BOTH EngineCore and EngineCoreProc to ensure
    # the patched version is found regardless of MRO resolution.
    EngineCore._process_engine_step = patched_process_engine_step
    EngineCore._handle_client_request = patched_handle_client_request
    EngineCoreProc._process_engine_step = patched_process_engine_step
    EngineCoreProc._handle_client_request = patched_handle_client_request

    # Replace EngineCoreProc.process_input_sockets
    EngineCoreProc.process_input_sockets = patched_process_input_sockets

    logger.debug("Beam search engine-core patches applied in child process.")


def patched_run_engine_core(*args, **kwargs):
    """Wrapper for EngineCoreProc.run_engine_core that applies beam search
    patches before the engine starts.  This is a module-level function so
    that multiprocessing 'spawn' can pickle it by qualified name.

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
    """Wrap EngineCore.__init__ to add active_beam_searches list."""

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.active_beam_searches: list[BeamSearchController] = []

    return patched_init


# ---------------------------------------------------------------------------
# EngineCore._process_engine_step replacement
# ---------------------------------------------------------------------------

def patched_process_engine_step(self) -> bool:
    """Replacement for EngineCore._process_engine_step with beam search."""

    # Step the engine core.
    outputs, model_executed = self.step_fn()

    if self.active_beam_searches:
        n_pending = sum(len(c._pending_rids) for c in self.active_beam_searches)
        n_outputs = sum(
            len(eo.outputs) for eo in outputs.values()
        ) if outputs else 0
        logger.debug(
            "patched_step: active_ctrls=%d, pending_rids=%d, "
            "outputs=%d, model_executed=%s, has_requests=%s",
            len(self.active_beam_searches),
            n_pending,
            n_outputs,
            model_executed,
            self.scheduler.has_requests(),
        )

    # Feed outputs to active beam search controllers and filter
    # beam sub-request outputs before sending to client.
    if self.active_beam_searches and outputs:
        # Snapshot all beam request IDs for the current step BEFORE
        # on_outputs(), which mutates _pending_rids and may trigger
        # submit_step() for the next step.  We need the full set of
        # current-step IDs to filter them from client-bound outputs.
        beam_rids: set[str] = set()
        for ctrl in self.active_beam_searches:
            beam_rids.update(ctrl._beam_req_ids)

        failed_ctrls: list[BeamSearchController] = []
        for ctrl in self.active_beam_searches:
            try:
                ctrl.on_outputs(outputs)
            except Exception:
                logger.exception(
                    "beam_search controller failed: req_id=%s",
                    ctrl.request_id,
                )
                failed_ctrls.append(ctrl)

        # Send error results for failed controllers and remove them
        # so they don't stay stuck in active_beam_searches.
        for ctrl in failed_ctrls:
            error_result = BeamSearchResult(
                request_id=ctrl.request_id,
                sequences=[],
                error="internal beam search error",
            )
            self.output_queue.put_nowait(
                (ctrl.client_index,
                 _engine_mod.EngineCoreOutputs(
                     beam_search_result=error_result))
            )
            self.active_beam_searches.remove(ctrl)

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
            logger.debug(
                "beam_search FINISHED: req_id=%s, n_sequences=%d, "
                "client_index=%d, queueing result",
                c.request_id,
                len(c.result.sequences),
                c.client_index,
            )
            self.output_queue.put_nowait(
                (c.client_index,
                 _engine_mod.EngineCoreOutputs(beam_search_result=c.result))
            )
            self.active_beam_searches.remove(c)

    # Check for timed-out controllers (scheduler never produced output).
    if self.active_beam_searches:
        timed_out = [c for c in self.active_beam_searches if c.is_timed_out]
        for ctrl in timed_out:
            logger.error(
                "beam_search controller timed out: req_id=%s, step=%d, "
                "pending_rids=%d",
                ctrl.request_id, ctrl._step, len(ctrl._pending_rids),
            )
            error_result = BeamSearchResult(
                request_id=ctrl.request_id,
                sequences=[],
                error="beam search step timed out waiting for scheduler",
            )
            self.output_queue.put_nowait(
                (ctrl.client_index,
                 _engine_mod.EngineCoreOutputs(
                     beam_search_result=error_result))
            )
            self.active_beam_searches.remove(ctrl)

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
        try:
            controller = BeamSearchController(self, request)
            controller.submit_step()
        except BaseException as e:
            logger.exception("Beam search request handling failed")
            error_result = BeamSearchResult(
                request_id=request.request_id,
                sequences=[],
                error=str(e),
            )
            self.output_queue.put_nowait(
                (request.client_index,
                 _engine_mod.EngineCoreOutputs(
                     beam_search_result=error_result))
            )
            return
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
    """Replacement for EngineCoreProc.process_input_sockets with beam search
    decoder."""
    from contextlib import ExitStack

    import zmq

    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
    from vllm.v1.serial_utils import MsgpackDecoder

    from vllm_gr.v1.engine.types import BeamSearchRequest

    # Msgpack serialization decoding.
    add_request_decoder = MsgpackDecoder(EngineCoreRequest)
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
                elif request_type == EngineCoreRequestType.BEAM_SEARCH:
                    request = beam_search_decoder.decode(data_frames)
                else:
                    request = generic_decoder.decode(data_frames)

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
