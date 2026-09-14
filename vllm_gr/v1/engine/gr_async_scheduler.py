# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam Search V1 state layered on vLLM's native asynchronous scheduler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchFailure,
    BeamBatchRequest,
    BeamBatchResult,
)
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRSessionState,
    GRSessionStatus,
    GRStageCompletion,
    GRStageKind,
    GRStageMetadata,
    GRWorkerResult,
)

GR_STAGE_OUTPUT_ATTR = "gr_stage_metadata"
GR_WORKER_RESULT_ATTR = "gr_worker_results"
GR_BATCH_RESULT_ATTR = "gr_batch_results"
GR_LOOKAHEAD_DEPTH = 2
_DRAINING_STATUSES = frozenset(
    {
        GRSessionStatus.FINISHING,
        GRSessionStatus.ABORTED,
        GRSessionStatus.FAILED,
    }
)


class _GRWorkerProtocolError(ValueError):
    """A malformed per-request Worker result; never escapes EngineCore."""


def _sessions(scheduler: Any) -> dict[str, GRSessionState]:
    sessions = getattr(scheduler, "_gr_v1_sessions", None)
    if sessions is None:
        sessions = {}
        scheduler._gr_v1_sessions = sessions
    return sessions


def _install_gr_prompt_cache_guard(scheduler: Any) -> None:
    """Cap native async output caching as well as schedule-time caching."""
    manager = getattr(scheduler, "kv_cache_manager", None)
    original = getattr(manager, "cache_blocks", None)
    if (
        manager is None
        or not callable(original)
        or getattr(manager, "_gr_prompt_cache_guard", False)
    ):
        return

    def cache_prompt(request: Any, num_computed_tokens: int) -> Any:
        if getattr(request, "gr_batch_request", None) is not None:
            num_computed_tokens = min(num_computed_tokens, request.num_prompt_tokens)
        return original(request, num_computed_tokens)

    manager.cache_blocks = cache_prompt
    manager._gr_prompt_cache_guard = True


def add_gr_session(scheduler: Any, request: BeamBatchRequest) -> GRSessionState:
    """Register one B=1 session; EngineCore adds its native Request separately."""
    if not scheduler.scheduler_config.async_scheduling:
        raise RuntimeError("beam_search_v1 requires vLLM async_scheduling")
    if len(request.items) != 1:
        raise ValueError("Beam Search V1 currently requires batch size B=1")
    session_id = request.batch_id
    engine_request = request.items[0].request
    if engine_request.request_id != session_id:
        raise ValueError("batch_id and the EngineCore request_id must be identical")
    sessions = _sessions(scheduler)
    if session_id in sessions:
        raise ValueError(f"GR session already exists: {session_id}")
    session = GRSessionState(
        session_id=session_id,
        request=request,
        # The final Prefill invocation produces output token 0; the remaining
        # max_tokens - 1 invocations are Decode stages.
        max_decode_steps=max(0, request.beam_params.max_tokens - 1),
    )
    _install_gr_prompt_cache_guard(scheduler)
    sessions[session_id] = session
    return session


def can_issue_gr_stage(scheduler: Any, session: GRSessionState) -> bool:
    return (
        session.status in (GRSessionStatus.WAITING, GRSessionStatus.RUNNING)
        and session.next_stage_kind is not None
        and len(session._in_flight_dispatches) < GR_LOOKAHEAD_DEPTH
    )


def reserve_gr_output_placeholder(scheduler: Any, session: GRSessionState, num_tokens: int) -> None:
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    session.output_placeholders += num_tokens
    session.pending_device_token_count += num_tokens


def attach_gr_stage_metadata(
    scheduler: Any, scheduler_output: Any, session: GRSessionState
) -> GRStageMetadata | None:
    req_id = session.session_id
    if req_id not in scheduler_output.num_scheduled_tokens:
        return None
    native_request = scheduler.requests.get(req_id)
    if native_request is None:
        return None

    # Initial chunked Prefill and post-preemption recomputation do not produce
    # a token. Physical dispatch accounting still tracks them, but logical
    # stage, placeholder, and device-state ownership must not advance.
    prefill_chunk = bool(native_request.is_prefill_chunk)
    dispatch_id = session._next_dispatch_id
    stage_index = session.issued_stages
    kind = GRStageKind.PREFILL if prefill_chunk or stage_index == 0 else GRStageKind.DECODE
    output_state = GRDeviceStateRef(req_id, stage_index)
    metadata = GRStageMetadata(
        session_id=req_id,
        dispatch_id=dispatch_id,
        stage_index=stage_index,
        kind=kind,
        decode_step=None if kind == GRStageKind.PREFILL else stage_index - 1,
        is_last_stage=not prefill_chunk and stage_index == session.total_stages - 1,
        input_device_state=session.latest_device_state,
        output_device_state=output_state,
        beam_params=session.request.beam_params,
        output_options=session.request.output_options,
        num_prompt_tokens=int(native_request.num_prompt_tokens),
        produces_output=not prefill_chunk,
    )
    manager = getattr(scheduler, "kv_cache_manager", None)
    if manager is not None:
        # Scheduler owns the native block refcounts. Python references in a
        # Worker cannot prevent block reuse after abort or preemption.
        groups = tuple(tuple(group) for group in manager.get_blocks(req_id).blocks)
        blocks = tuple(block for group in groups for block in group)
        manager.block_pool.touch(blocks)
        # The last reference determines eviction order. Match native free():
        # preserve group order and release each group's prompt tail first.
        session._kv_dispatch_leases[dispatch_id] = tuple(
            block for group in groups for block in reversed(group)
        )
    session._next_dispatch_id += 1
    session._in_flight_dispatches.add(dispatch_id)
    if not prefill_chunk:
        session.issued_stages += 1
        session.status = GRSessionStatus.RUNNING
        session.latest_device_state = output_state
        native_placeholders = int(getattr(native_request, "num_output_placeholders", 0))
        session.output_placeholders = native_placeholders
        session.pending_device_token_count = native_placeholders
    return metadata


def _stage_completion(
    session: GRSessionState,
    stage_index: int,
    model_output: Any,
    *,
    finished: bool = False,
    device_state: GRDeviceStateRef | None = None,
    error: str | None = None,
) -> GRStageCompletion:
    return GRStageCompletion(
        session_id=session.session_id,
        stage_index=stage_index,
        finished=finished,
        confirmed_token_count=session.confirmed_token_count,
        device_state=device_state,
        model_output=model_output,
        error=error,
    )


def _result_mapping(model_output: Any, attribute: str) -> Mapping[Any, Any] | None:
    value = getattr(model_output, attribute, None)
    if value is not None and not isinstance(value, Mapping):
        raise _GRWorkerProtocolError(f"Worker returned an invalid {attribute} mapping")
    return value


def _read_worker_payload(
    session: GRSessionState,
    metadata: GRStageMetadata,
    model_output: Any,
) -> tuple[GRWorkerResult | None, BeamBatchResult | None, int, bool]:
    """Read and validate one output-producing dispatch response.

    A missing ``gr_worker_results`` attribute is retained as the A3-only
    compatibility path. Once A4 attaches the mapping, an entry for every GR
    dispatch is mandatory and must echo its identity.
    """
    worker_results = _result_mapping(model_output, GR_WORKER_RESULT_ATTR)
    worker_result = worker_results.get(session.session_id) if worker_results is not None else None
    if worker_results is not None and worker_result is None:
        raise _GRWorkerProtocolError("Worker omitted GRWorkerResult for a dispatched GR stage")
    if worker_result is not None and not isinstance(worker_result, GRWorkerResult):
        raise _GRWorkerProtocolError("Worker returned an invalid GRWorkerResult")
    if worker_result is not None and (
        worker_result.session_id != session.session_id
        or worker_result.dispatch_id != metadata.dispatch_id
        or worker_result.stage_index != metadata.stage_index
    ):
        raise _GRWorkerProtocolError("Worker returned a GR result for a different dispatch")

    terminal_result = worker_result.terminal_result if worker_result is not None else None
    if terminal_result is None:
        terminal_results = _result_mapping(model_output, GR_BATCH_RESULT_ATTR)
        if terminal_results is not None:
            terminal_result = terminal_results.get(session.session_id)
    if terminal_result is not None:
        if not isinstance(terminal_result, BeamBatchResult):
            raise _GRWorkerProtocolError("Worker returned an invalid BeamBatchResult")
        if terminal_result.batch_id != session.session_id:
            raise _GRWorkerProtocolError("Worker returned a terminal result for another session")

    produced = (
        worker_result.produced_token_count
        if worker_result is not None
        else int(metadata.produces_output)
    )
    worker_finished = bool(worker_result.finished) if worker_result is not None else False
    if not isinstance(produced, int) or isinstance(produced, bool) or produced not in (0, 1):
        raise _GRWorkerProtocolError("Worker returned an invalid produced_token_count")
    if (
        metadata.produces_output
        and produced == 0
        and not worker_finished
        and terminal_result is None
    ):
        raise _GRWorkerProtocolError("Worker returned no token for a non-terminal logical stage")
    if not metadata.produces_output and produced != 0:
        raise _GRWorkerProtocolError("Worker produced a token for a non-output Prefill chunk")
    if (
        worker_result is not None
        and produced == 1
        and worker_result.device_state != metadata.output_device_state
    ):
        raise _GRWorkerProtocolError("Worker returned a device state for a different logical stage")
    return worker_result, terminal_result, produced, worker_finished


def _sync_output_placeholders(scheduler: Any, session: GRSessionState) -> None:
    native_request = scheduler.requests.get(session.session_id)
    session.output_placeholders = (
        int(native_request.num_output_placeholders)
        if native_request is not None
        else session.pending_device_token_count
    )


def update_gr_from_output(
    scheduler: Any,
    session: GRSessionState,
    scheduler_output: Any,
    model_output: Any,
) -> GRStageCompletion:
    metadata = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, {}).get(session.session_id)
    if metadata is None:
        return _stage_completion(session, session.completed_stages, model_output)

    # Completion is keyed by the physical dispatch rather than logical stage.
    # Recompute chunks can reuse a logical stage, while each dispatched
    # SchedulerOutput must still be consumed exactly once.
    if metadata.dispatch_id not in session._in_flight_dispatches:
        return _stage_completion(session, metadata.stage_index, model_output)
    session._in_flight_dispatches.remove(metadata.dispatch_id)
    blocks = session._kv_dispatch_leases.pop(metadata.dispatch_id, None)
    if blocks is not None:
        scheduler.kv_cache_manager.block_pool.free_blocks(blocks)

    # Once termination starts, successors that were already queued are only
    # drained. They must not advance logical tokens or replace the result that
    # selected the terminal path.
    if session.status in _DRAINING_STATUSES:
        if metadata.produces_output:
            # The logical stage was issued before termination and is now
            # retired, but its speculative output must not be confirmed.
            session.completed_stages += 1
        if not session._in_flight_dispatches and (
            session.status in (GRSessionStatus.ABORTED, GRSessionStatus.FAILED)
            or not session.terminal_result_pending
        ):
            _sessions(scheduler).pop(session.session_id, None)
        return _stage_completion(session, metadata.stage_index, model_output)

    try:
        worker_result, terminal_result, produced, worker_finished = _read_worker_payload(
            session, metadata, model_output
        )
    except _GRWorkerProtocolError as error:
        return _fail_gr_dispatch(scheduler, session, metadata, model_output, str(error))

    if not metadata.produces_output:
        if worker_finished or terminal_result is not None:
            session.status = GRSessionStatus.FINISHING
            session.terminal_result_pending = True
            session._pending_terminal_result = terminal_result
        return _stage_completion(
            session, metadata.stage_index, model_output, finished=worker_finished
        )

    session.completed_stages += 1
    session.confirmed_token_count += produced
    session.pending_device_token_count = max(0, session.pending_device_token_count - produced)
    _sync_output_placeholders(scheduler, session)
    finished = worker_finished or terminal_result is not None
    if finished or session.completed_stages >= session.total_stages:
        session.status = GRSessionStatus.FINISHING
        session.terminal_result_pending = True
        session._pending_terminal_result = terminal_result
    return _stage_completion(
        session,
        metadata.stage_index,
        model_output,
        finished=finished,
        device_state=(
            worker_result.device_state
            if worker_result is not None
            else metadata.output_device_state
        ),
    )


def _make_failure_result(session: GRSessionState, code: str, message: str) -> BeamBatchResult:
    return BeamBatchResult(
        protocol_version=session.request.protocol_version,
        batch_id=session.session_id,
        output=None,
        failure=BeamBatchFailure(code=code, message=message, item_index=None),
    )


def _fail_gr_dispatch(
    scheduler: Any,
    session: GRSessionState,
    metadata: GRStageMetadata,
    model_output: Any,
    message: str,
) -> GRStageCompletion:
    """Turn a malformed Worker response into one request-scoped failure."""
    if metadata.produces_output:
        session.completed_stages += 1
    session.status = GRSessionStatus.FINISHING
    session.terminal_result_pending = True
    if session._pending_terminal_result is None:
        session._pending_terminal_result = _make_failure_result(
            session, "worker_protocol_error", message
        )
    _sync_output_placeholders(scheduler, session)
    return _stage_completion(
        session,
        metadata.stage_index,
        model_output,
        finished=True,
        error=message,
    )


def _gr_session_ready_to_finalize(session: GRSessionState) -> bool:
    return (
        session.status == GRSessionStatus.FINISHING
        and session.terminal_result_pending
        and not session._terminal_emitted
        and not session._in_flight_dispatches
    )


def _gr_terminal_result(session: GRSessionState) -> BeamBatchResult:
    result = session._pending_terminal_result
    if result is not None:
        return result
    return _make_failure_result(
        session,
        "worker_result_unavailable",
        "Worker did not return a terminal BeamBatchResult",
    )


def _close_gr_session(scheduler: Any, session: GRSessionState) -> None:
    if session._terminal_emitted:
        return
    session._terminal_emitted = True
    session.terminal_result_pending = False
    session.output_placeholders = 0
    session.pending_device_token_count = 0
    session.status = GRSessionStatus.FINISHED
    _sessions(scheduler).pop(session.session_id, None)


def finish_gr_session(
    scheduler: Any, session: GRSessionState, status: GRSessionStatus
) -> BeamBatchResult | None:
    session.status = status
    session.terminal_result_pending = False
    if status == GRSessionStatus.FAILED:
        return _make_failure_result(session, "engine_failure", "GR session failed in EngineCore")
    return None


def abort_gr_session(scheduler: Any, session_id: str) -> None:
    from vllm.v1.request import RequestStatus

    session = _sessions(scheduler).get(session_id)
    if session is not None:
        session.status = GRSessionStatus.ABORTED
        session.terminal_result_pending = False
    scheduler.finish_requests(session_id, RequestStatus.FINISHED_ABORTED)
    if session is not None and not session._in_flight_dispatches:
        _sessions(scheduler).pop(session_id, None)


def release_gr_session(scheduler: Any, session_id: str) -> None:
    session = _sessions(scheduler).get(session_id)
    if session is None:
        return
    if session._in_flight_dispatches:
        raise RuntimeError("cannot release a GR session with in-flight stages")
    del _sessions(scheduler)[session_id]


def install_gr_scheduler_methods(scheduler_cls: type[Any]) -> None:
    scheduler_cls.add_gr_session = add_gr_session
    scheduler_cls.can_issue_gr_stage = can_issue_gr_stage
    scheduler_cls.reserve_gr_output_placeholder = reserve_gr_output_placeholder
    scheduler_cls.attach_gr_stage_metadata = attach_gr_stage_metadata
    scheduler_cls.update_gr_from_output = update_gr_from_output
    scheduler_cls.finish_gr_session = finish_gr_session
    scheduler_cls.abort_gr_session = abort_gr_session
    scheduler_cls.release_gr_session = release_gr_session
