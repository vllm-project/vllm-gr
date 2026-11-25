# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm
"""GR session admission, terminal consumption, and EngineCore socket transport."""

from __future__ import annotations

from concurrent.futures import Future
from copy import copy
from functools import wraps
from typing import Any

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)

from vllm_gr.v1.beam.batch_contracts import BeamBatchResult, GRAdmissionError
from vllm_gr.v1.beam.dispatch import (
    GR_RETIRE_OUTPUT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    GRStageMetadata,
    GRWorkerResult,
)
from vllm_gr.v1.core.sched.beam_session import GRSessionStatus
from vllm_gr.v1.engine.core_socket import process_input_sockets as process_input_sockets

logger = init_logger(__name__)


def loaded_constraint_trie_count(worker: Any) -> int:
    """Read the validated, live table once through the executor startup RPC."""
    backend = getattr(worker.model_runner, "constraint_backend", None)
    return backend.info.table_metadata.trie_count if backend is not None else 1


def is_gr_session_retired(self: Any, session_id: str) -> bool:
    """Ordered Core-thread query; never block the thread that drains BatchQueue."""
    return session_id not in getattr(self.scheduler, "_gr_v1_sessions", {})


def _try_consume_ready_gr_terminal(core: Any) -> dict[int, EngineCoreOutputs] | None:
    """Consume the sole ready terminal GR receipt without scheduling new work."""
    queue = core.batch_queue
    if queue is None or len(queue) != 1:
        return None
    future, scheduler_output, _ = queue[-1]
    if (
        not isinstance(future, Future)
        or not future.done()
        or future.cancelled()
        or future.exception() is not None
    ):
        return None

    stages = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None)
    if (
        not isinstance(stages, dict)
        or len(stages) != 1
        or getattr(scheduler_output, GR_RETIRE_OUTPUT_ATTR, None) is not None
    ):
        return None
    session_id, stage = next(iter(stages.items()))
    if (
        not isinstance(stage, GRStageMetadata)
        or stage.session_id != session_id
        or not stage.is_last_stage
        or stage.decode_step is None
    ):
        return None

    scheduler = core.scheduler
    sessions = getattr(scheduler, "_gr_v1_sessions", {})
    session = sessions.get(session_id)
    if (
        len(sessions) != 1
        or session is None
        or getattr(scheduler, "_gr_owner", None) != session_id
        or session.status is not GRSessionStatus.RUNNING
        or stage.owner_generation != session.owner_generation
        or len(session._in_flight_dispatches) != 1
        or stage.dispatch_id not in session._in_flight_dispatches
        or session._dispatch_records.get(stage.dispatch_id) != stage
        or not core.aborts_queue.empty()
        or any(
            getattr(request, "gr_batch_request", None) is not session.request
            for request in scheduler.requests.values()
        )
    ):
        return None

    model_output = future.result()
    receipts = getattr(model_output, GR_WORKER_RESULT_ATTR, None)
    receipt = receipts.get(session_id) if isinstance(receipts, dict) else None
    if (
        not isinstance(receipt, GRWorkerResult)
        or not isinstance(receipt.terminal_result, BeamBatchResult)
        or receipt.terminal_result.failure is not None
        or receipt.terminal_result.output is None
        or receipt.terminal_result.batch_id != session_id
        or receipt.finished is not True
        or receipt.resources_released is not True
        or receipt.session_id != session_id
        or receipt.dispatch_id != stage.dispatch_id
        or receipt.stage_index != stage.stage_index
        or receipt.owner_generation != stage.owner_generation
        or receipt.produced_token_count != 1
        or receipt.device_state != stage.output_device_state
    ):
        return None

    # Earlier stages have already been applied in FIFO order.
    queue.pop()
    with (
        core.capture_iteration_details(scheduler_output) as iteration_details,
        core.log_error_detail(scheduler_output),
    ):
        core._process_aborts_queue()
        outputs = scheduler.update_from_output(scheduler_output, model_output)
    core._attach_iteration_details(outputs, iteration_details)
    return outputs or {}


def install_inproc_gr_terminal_step(core: Any) -> None:
    """Let the native in-process batch loop consume a ready terminal result first."""
    from vllm.v1.engine.core import EngineCore

    if type(core) is not EngineCore:
        return
    native_step = core.step_fn
    if getattr(native_step, "__func__", None) is not EngineCore.step_with_batch_queue:
        return

    @wraps(native_step)
    def step_with_ready_terminal() -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        outputs = _try_consume_ready_gr_terminal(core)
        if outputs is None:
            return native_step()
        return outputs, False

    core.step_fn = step_with_ready_terminal


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
        logger.error("Rejecting an empty beam_search batch")
        return
    config = getattr(self, "vllm_config", None)
    trie_count = getattr(self, "_gr_trie_count", None)
    try:
        from vllm_gr.v1.engine.beam_admission import validate_gr_request

        validate_gr_request(batch_request, config=config, trie_count=trie_count)
    except ValueError as error:
        # A malformed batch is a request rejection, not a preprocessing fault.
        if getattr(self, "output_queue", None) is None:
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
    device = getattr(getattr(config, "device_config", None), "device", None)
    if getattr(device, "type", None) == "npu":
        # Synchronous admission sees the same request that just passed the
        # frontend/Core boundary. The queued path above must validate again.
        _admit_gr_session(self, batch_request, validation_context=(config, trie_count))
    else:
        _admit_gr_session(self, batch_request)


def _admit_gr_session(self, batch_request, *, validation_context=None) -> None:
    """Preprocess and admit one complete GR session on the core thread."""
    if not batch_request.items:
        logger.error("Rejecting an empty beam_search batch")
        return
    engine_requests = tuple(item.request for item in batch_request.items)
    try:
        from vllm_gr.config.execution import validate_gr_execution_config
        from vllm_gr.platforms.errors import GRDeviceRequestError
        from vllm_gr.v1.engine.beam_admission import (
            _sync_validation,
            validate_gr_device_config,
            validate_gr_request,
        )

        config = getattr(self, "vllm_config", None)
        trie_count = getattr(self, "_gr_trie_count", None)
        if (
            validation_context is None
            or config is not validation_context[0]
            or trie_count != validation_context[1]
        ):
            validate_gr_request(batch_request, config=config, trie_count=trie_count)
    except ValueError as error:
        # The request is answerable, so it must not take the preprocessing
        # error path: that one logs a traceback and sends a bare
        # FinishReason.ERROR, which the frontend can only read as a fault.
        # In-process callers have no output socket, so the rejection travels
        # to them as the exception the multiprocess handler encodes.
        if getattr(self, "output_queue", None) is None:
            raise GRAdmissionError(str(error)) from error
        send_gr_rejection(self, engine_requests[0], str(error))
        return
    try:
        # Deployment configuration stays on the preprocessing error path: an
        # engine started with the wrong settings is a server fault, not
        # something the caller can fix by resending.
        validate_gr_execution_config(self.vllm_config)
        validate_gr_device_config(self.vllm_config, batch_request)
        if self.batch_queue is None or self.batch_queue_size < 2:
            raise RuntimeError("beam_search requires a non-blocking EngineCore batch queue")
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
    scheduler = getattr(self, "scheduler", None)
    proof = (
        _sync_validation.set((scheduler, batch_request, config, trie_count))
        if validation_context is not None and scheduler is not None
        else None
    )
    try:
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
            self.add_request(*preprocessed)
    finally:
        if proof is not None:
            _sync_validation.reset(proof)


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
