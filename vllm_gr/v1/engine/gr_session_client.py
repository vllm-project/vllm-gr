"""Online and offline frontend adapters for one-shot GR session submission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchFailure,
    BeamBatchRequest,
    BeamBatchResult,
)


@dataclass(slots=True)
class SyncGRSessionHandle:
    request: BeamBatchRequest


@dataclass(slots=True)
class AsyncGRSessionHandle:
    request: BeamBatchRequest
    output_collector: Any


def _terminal_result(request: BeamBatchRequest, output: Any) -> BeamBatchResult:
    params = getattr(output, "kv_transfer_params", None) or {}
    result = params.get("gr_batch_result")
    if isinstance(result, BeamBatchResult):
        return result
    return BeamBatchResult(
        protocol_version=request.protocol_version,
        batch_id=request.batch_id,
        output=None,
        failure=BeamBatchFailure(
            code="worker_result_unavailable",
            message="A4/A5 did not attach gr_batch_result to the terminal output",
            item_index=None,
        ),
    )


def submit_gr_session_sync(engine: Any, request: BeamBatchRequest) -> SyncGRSessionHandle:
    native_request = request.items[0].request
    engine.output_processor.add_request(native_request, None)
    engine.engine_core.submit_gr_session(request)
    return SyncGRSessionHandle(request)


def wait_gr_session_sync(
    engine: Any, session_id: str, handle: SyncGRSessionHandle
) -> BeamBatchResult:
    if handle.request.batch_id != session_id:
        raise ValueError("GR session handle does not match session_id")
    while True:
        for output in engine.step():
            if output.request_id == session_id and output.finished:
                return _terminal_result(handle.request, output)


def abort_gr_session_sync(engine: Any, session_id: str) -> None:
    engine.abort_request([session_id])


def release_gr_session_sync(engine: Any, session_id: str) -> None:
    # EngineCore releases the scheduler state after all queued stages drain.
    return None


async def submit_gr_session_async(
    engine: Any, request: BeamBatchRequest
) -> AsyncGRSessionHandle:
    from vllm.v1.engine.output_processor import RequestOutputCollector

    native_request = request.items[0].request
    collector = RequestOutputCollector(
        native_request.sampling_params.output_kind, request.batch_id
    )
    engine.output_processor.add_request(native_request, None, queue=collector)
    engine._run_output_handler()
    await engine.engine_core.submit_gr_session_async(request)
    return AsyncGRSessionHandle(request, collector)


async def wait_gr_session_async(
    engine: Any, session_id: str, handle: AsyncGRSessionHandle
) -> BeamBatchResult:
    if handle.request.batch_id != session_id:
        raise ValueError("GR session handle does not match session_id")
    while True:
        output = await handle.output_collector.get()
        if output.finished:
            return _terminal_result(handle.request, output)


async def abort_gr_session_async(engine: Any, session_id: str) -> None:
    await engine.abort(session_id)


async def release_gr_session_async(engine: Any, session_id: str) -> None:
    # EngineCore releases the scheduler state after all queued stages drain.
    return None


def install_gr_frontend_methods(llm_engine_cls: type[Any], async_llm_cls: type[Any]) -> None:
    llm_engine_cls.submit_gr_session = submit_gr_session_sync
    llm_engine_cls.wait_gr_session = wait_gr_session_sync
    llm_engine_cls.abort_gr_session = abort_gr_session_sync
    llm_engine_cls.release_gr_session = release_gr_session_sync
    async_llm_cls.submit_gr_session = submit_gr_session_async
    async_llm_cls.wait_gr_session = wait_gr_session_async
    async_llm_cls.abort_gr_session = abort_gr_session_async
    async_llm_cls.release_gr_session = release_gr_session_async
