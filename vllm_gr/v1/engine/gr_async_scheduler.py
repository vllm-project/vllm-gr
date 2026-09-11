"""Beam Search V1 state layered on vLLM's native asynchronous scheduler."""

from __future__ import annotations

from typing import Any

from vllm_gr.v1.beam.batch_contracts import BeamBatchFailure, BeamBatchRequest, BeamBatchResult
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


def _sessions(scheduler: Any) -> dict[str, GRSessionState]:
    sessions = getattr(scheduler, "_gr_v1_sessions", None)
    if sessions is None:
        sessions = {}
        scheduler._gr_v1_sessions = sessions
    return sessions


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
    sessions[session_id] = session
    return session


def can_issue_gr_stage(scheduler: Any, session: GRSessionState) -> bool:
    return (
        session.status in (GRSessionStatus.WAITING, GRSessionStatus.RUNNING)
        and session.next_stage_kind is not None
        and session.in_flight_stages < GR_LOOKAHEAD_DEPTH
    )


def reserve_gr_output_placeholder(
    scheduler: Any, session: GRSessionState, num_tokens: int
) -> None:
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

    # A chunked prefill does not produce a token and therefore does not consume
    # a logical stage. It still carries metadata so A4 can prepare device state.
    prefill_chunk = session.issued_stages == 0 and native_request.is_prefill_chunk
    stage_index = session.issued_stages
    kind = GRStageKind.PREFILL if stage_index == 0 else GRStageKind.DECODE
    output_state = GRDeviceStateRef(req_id, stage_index)
    metadata = GRStageMetadata(
        session_id=req_id,
        stage_index=stage_index,
        kind=kind,
        decode_step=None if kind == GRStageKind.PREFILL else stage_index - 1,
        is_last_stage=stage_index == session.total_stages - 1,
        input_device_state=session.latest_device_state,
        output_device_state=output_state,
        beam_params=session.request.beam_params,
        output_options=session.request.output_options,
        num_prompt_tokens=int(native_request.num_prompt_tokens),
        produces_output=not prefill_chunk,
    )
    if not prefill_chunk:
        session.issued_stages += 1
        session.status = GRSessionStatus.RUNNING
        session.latest_device_state = output_state
        native_placeholders = int(getattr(native_request, "num_output_placeholders", 0))
        session.output_placeholders = native_placeholders
        session.pending_device_token_count = native_placeholders
    return metadata


def update_gr_from_output(
    scheduler: Any,
    session: GRSessionState,
    scheduler_output: Any,
    model_output: Any,
) -> GRStageCompletion:
    metadata = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, {}).get(session.session_id)
    if metadata is None or not metadata.produces_output:
        return GRStageCompletion(
            session_id=session.session_id,
            stage_index=session.completed_stages,
            finished=False,
            confirmed_token_count=session.confirmed_token_count,
            model_output=model_output,
        )

    worker_results = getattr(model_output, GR_WORKER_RESULT_ATTR, None) or {}
    worker_result = worker_results.get(session.session_id)
    if worker_result is not None and not isinstance(worker_result, GRWorkerResult):
        raise TypeError("Worker returned an invalid GRWorkerResult")
    produced = worker_result.produced_token_count if worker_result is not None else 1
    session.completed_stages += 1
    session.confirmed_token_count += produced
    session.pending_device_token_count = max(0, session.pending_device_token_count - produced)
    native_request = scheduler.requests.get(session.session_id)
    if native_request is not None:
        session.output_placeholders = int(native_request.num_output_placeholders)
    else:
        session.output_placeholders = session.pending_device_token_count
    finished = bool(worker_result.finished) if worker_result is not None else False
    if finished or session.completed_stages >= session.total_stages:
        session.status = GRSessionStatus.FINISHING
        session.terminal_result_pending = True
    if session.completed_stages >= session.issued_stages and session.status in (
        GRSessionStatus.FINISHING,
        GRSessionStatus.ABORTED,
        GRSessionStatus.FAILED,
    ):
        if session.status == GRSessionStatus.FINISHING:
            session.status = GRSessionStatus.FINISHED
        session.terminal_result_pending = False
        _sessions(scheduler).pop(session.session_id, None)
    return GRStageCompletion(
        session_id=session.session_id,
        stage_index=metadata.stage_index,
        finished=finished,
        confirmed_token_count=session.confirmed_token_count,
        device_state=(worker_result.device_state if worker_result is not None else metadata.output_device_state),
        model_output=model_output,
    )


def finish_gr_session(
    scheduler: Any, session: GRSessionState, status: GRSessionStatus
) -> BeamBatchResult | None:
    session.status = status
    session.terminal_result_pending = False
    if status == GRSessionStatus.FAILED:
        return BeamBatchResult(
            protocol_version=session.request.protocol_version,
            batch_id=session.session_id,
            output=None,
            failure=BeamBatchFailure(
                code="engine_failure", message="GR session failed in EngineCore", item_index=None
            ),
        )
    return None


def abort_gr_session(scheduler: Any, session_id: str) -> None:
    from vllm.v1.request import RequestStatus

    session = _sessions(scheduler).get(session_id)
    if session is not None:
        session.status = GRSessionStatus.ABORTED
        session.terminal_result_pending = False
    scheduler.finish_requests(session_id, RequestStatus.FINISHED_ABORTED)


def release_gr_session(scheduler: Any, session_id: str) -> None:
    session = _sessions(scheduler).get(session_id)
    if session is None:
        return
    if session.in_flight_stages:
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
