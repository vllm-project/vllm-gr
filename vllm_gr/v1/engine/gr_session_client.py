# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online and offline frontend adapters for one-shot GR session submission."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import msgspec

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchFailure,
    BeamBatchRequest,
    BeamBatchResult,
)
from vllm_gr.v1.engine.gr_result_validation import decode_terminal_result


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
    # kv_transfer_params is an untyped IPC field, so array-like msgspec
    # structs arrive as plain lists after a multiprocess round trip.
    if isinstance(result, BeamBatchResult):
        # Struct construction does not validate fields, and convert() trusts
        # existing structs. Rebuild the four wire fields to discard local
        # validation metadata, then use the same typed boundary as IPC values.
        result = msgspec.to_builtins(
            BeamBatchResult(result.protocol_version, result.batch_id, result.output, result.failure)
        )
    if isinstance(result, (list, tuple)):
        result = decode_terminal_result(result, request.beam_params.max_tokens)
    if isinstance(result, BeamBatchResult):
        # Fence transport identity here. Public conversion checks the request's
        # result envelope and reuses any sequence checks done during decoding.
        if result.batch_id != request.batch_id:
            raise ValueError("terminal Beam result belongs to a different session")
        return result
    return BeamBatchResult(
        protocol_version=request.protocol_version,
        batch_id=request.batch_id,
        output=None,
        failure=BeamBatchFailure(
            code="worker_result_unavailable",
            message="EngineCore did not attach gr_batch_result to the terminal output",
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
    native_request = handle.request.items[0].request
    output_id = getattr(native_request, "external_req_id", None) or session_id
    while True:
        for output in engine.step():
            if output.request_id == output_id and output.finished:
                return _terminal_result(handle.request, output)


def abort_gr_session_sync(engine: Any, session_id: str) -> None:
    # Always send the internal ID: a previous abort may already have removed
    # frontend routing while its transport acknowledgement was lost.
    engine.output_processor.abort_requests([session_id], internal=True)
    engine.engine_core.abort_requests([session_id])
    core = engine.engine_core
    if hasattr(core, "engine_core"):
        while not core.engine_core.is_gr_session_retired(session_id):
            engine.step()
    else:
        while not core.call_utility("is_gr_session_retired", session_id):
            time.sleep(0.001)


def release_gr_session_sync(engine: Any, session_id: str) -> None:
    # Terminal delivery is fenced by retirement in EngineCore.
    return None


async def submit_gr_session_async(engine: Any, request: BeamBatchRequest) -> AsyncGRSessionHandle:
    from vllm.v1.engine.output_processor import RequestOutputCollector

    native_request = request.items[0].request
    collector = RequestOutputCollector(native_request.sampling_params.output_kind, request.batch_id)
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
    engine.output_processor.abort_requests([session_id], internal=True)
    await engine.engine_core.abort_requests_async([session_id])
    while not await engine.engine_core.call_utility_async("is_gr_session_retired", session_id):
        await asyncio.sleep(0.001)


async def release_gr_session_async(engine: Any, session_id: str) -> None:
    # Terminal delivery is fenced by retirement in EngineCore.
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
