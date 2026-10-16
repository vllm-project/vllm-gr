# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online and offline frontend adapters for one-shot GR session submission."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import msgspec

from vllm_gr.config import BeamCandidateMode
from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchFailure,
    BeamBatchOutput,
    BeamBatchRequest,
    BeamBatchResult,
    BeamContractError,
    BeamOutputSequence,
    BeamRequestOutput,
)


@dataclass(slots=True)
class SyncGRSessionHandle:
    request: BeamBatchRequest


@dataclass(slots=True)
class AsyncGRSessionHandle:
    request: BeamBatchRequest
    output_collector: Any


def _normalize_inproc_result(result: BeamBatchResult) -> BeamBatchResult:
    # Strip only Struct wrappers before typed conversion. Token/parent tuples
    # stay intact, and convert still validates their elements and constraints.
    # Unexpected container shapes retain the ordinary typed wire boundary.
    try:
        if type(result) is not BeamBatchResult:
            raise TypeError
        output = result.output
        if type(output) is not BeamBatchOutput or result.failure is not None:
            raise TypeError
        if type(output.items) is not tuple:
            raise TypeError
        items = []
        astuple = msgspec.structs.astuple
        for item in output.items:
            if type(item) is not BeamRequestOutput or type(item.sequences) is not tuple:
                raise TypeError
            sequences = []
            for sequence in item.sequences:
                if type(sequence) is not BeamOutputSequence:
                    raise TypeError
                sequences.append(astuple(sequence))
            items.append((item.item_index, item.request_id, sequences))
        return msgspec.convert(
            (result.protocol_version, result.batch_id, (items,), None), type=BeamBatchResult
        )
    except TypeError:
        return msgspec.msgpack.decode(msgspec.msgpack.encode(result), type=BeamBatchResult)


def _terminal_result(request: BeamBatchRequest, output: Any) -> BeamBatchResult:
    params = getattr(output, "kv_transfer_params", None) or {}
    result = params.get("gr_batch_result")
    # kv_transfer_params is an untyped IPC field, so array-like msgspec
    # structs arrive as plain lists after a multiprocess round trip.
    # msgspec.convert trusts constructed Structs. Strip those wrappers before
    # typed conversion so inproc values receive the same nested validation.
    try:
        if isinstance(result, BeamBatchResult):
            result = _normalize_inproc_result(result)
        elif isinstance(result, (list, tuple)):
            result = msgspec.convert(result, type=BeamBatchResult)
    except msgspec.ValidationError as exc:
        # Keep one public contract-error family across inproc and wire paths;
        # msgspec.ValidationError itself is not a ValueError.
        raise BeamContractError(f"malformed terminal Beam result: {exc}") from exc
    if isinstance(result, BeamBatchResult):
        # Typed conversion checks every nested field on both transports,
        # including nonnegative indices and finite cumulative scores.
        # Check the V1 envelope and request-dependent semantics once here.
        if result.protocol_version != request.protocol_version:
            raise ValueError("terminal Beam result protocol version changed")
        if (result.output is None) == (result.failure is None):
            raise ValueError("terminal Beam result requires exactly one output or failure")
        if result.batch_id != request.batch_id:
            raise ValueError("terminal Beam result belongs to a different session")
        if (
            result.failure is not None
            and result.failure.item_index is not None
            and result.failure.item_index not in {item.item_index for item in request.items}
        ):
            raise ValueError("terminal Beam failure has an invalid item identity")
        if result.failure is not None and (
            not result.failure.code.strip() or not result.failure.message.strip()
        ):
            raise ValueError("terminal Beam failure requires a code and message")
        if result.output is not None:
            if len(result.output.items) != len(request.items):
                raise ValueError("terminal Beam result item count changed")
            options = request.output_options
            budget = request.beam_params.max_tokens
            width = request.beam_params.beam_width
            include_scores = options.include_cumulative_logprob
            include_parents = options.include_parent_beam_indices
            full_vocab = request.beam_params.candidate_mode == BeamCandidateMode.FULL_VOCAB
            expected_sequence = (
                "length",
                budget,
                None,
                include_scores,
                budget if include_parents else 0,
            )
            for item, source in zip(result.output.items, request.items):
                if (
                    item.request_id != source.request.request_id
                    or item.item_index != source.item_index
                ):
                    raise ValueError("terminal Beam result item identity changed")
                if not item.sequences or len(item.sequences) > options.num_return_sequences:
                    raise ValueError("terminal Beam result has an invalid sequence count")
                for sequence in item.sequences:
                    parents = sequence.parent_beam_indices
                    if full_vocab:
                        generated_count = len(sequence.token_ids)
                        if sequence.finish_reason == "stop":
                            if request.beam_params.ignore_stop_tokens or (
                                sequence.stop_token_id not in request.beam_params.stop_token_ids
                            ):
                                raise ValueError("terminal Beam result has an invalid stop token")
                            if options.include_stop_token:
                                if (
                                    not sequence.token_ids
                                    or sequence.token_ids[-1] != sequence.stop_token_id
                                ):
                                    raise ValueError(
                                        "terminal Beam result must end with its included stop token"
                                    )
                            else:
                                # Parent ancestry records generation steps,
                                # including an EOS omitted from returned tokens.
                                generated_count += 1
                            if not 1 <= generated_count <= budget:
                                raise ValueError(
                                    "terminal Beam result violates the effective budget"
                                )
                        elif generated_count != budget or sequence.stop_token_id is not None:
                            raise ValueError("terminal Beam result violates the effective budget")
                        if include_scores != (sequence.cumulative_logprob is not None):
                            raise ValueError(
                                "terminal Beam score presence differs from requested options"
                            )
                        if not include_parents and parents:
                            raise ValueError(
                                "terminal Beam result contains unrequested parent indices"
                            )
                        if include_parents and (
                            len(parents) != generated_count or max(parents) >= width
                        ):
                            raise ValueError("terminal Beam result has invalid parent indices")
                        continue
                    # Compare the request-dependent shape together. Typed
                    # conversion above has already checked every field's type.
                    if (
                        sequence.finish_reason,
                        len(sequence.token_ids),
                        sequence.stop_token_id,
                        sequence.cumulative_logprob is not None,
                        len(parents),
                    ) != expected_sequence:
                        # Keep precise protocol failures on malformed results.
                        if (
                            sequence.finish_reason != "length"
                            or len(sequence.token_ids) != budget
                            or sequence.stop_token_id is not None
                        ):
                            raise ValueError("terminal Beam result violates the effective budget")
                        if include_scores != (sequence.cumulative_logprob is not None):
                            raise ValueError(
                                "terminal Beam score presence differs from requested options"
                            )
                        if not include_parents and parents:
                            raise ValueError(
                                "terminal Beam result contains unrequested parent indices"
                            )
                        raise ValueError("terminal Beam result has invalid parent indices")
                    if include_parents and max(parents) >= width:
                        raise ValueError("terminal Beam result has invalid parent indices")
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
    native_request = handle.request.items[0].request
    output_id = getattr(native_request, "external_req_id", None) or session_id
    while True:
        for output in engine.step():
            if output.request_id == output_id and output.finished:
                return _terminal_result(handle.request, output)


def abort_gr_session_sync(engine: Any, session_id: str) -> None:
    # A retry must reach Core even after frontend routing has been removed.
    engine.output_processor.abort_requests([session_id], internal=True)
    core = engine.engine_core
    core.abort_requests([session_id])
    if hasattr(core, "engine_core"):
        while not core.engine_core.is_gr_session_retired(session_id):
            engine.step()
    else:
        while not core.call_utility("is_gr_session_retired", session_id):
            time.sleep(0.001)


def release_gr_session_sync(engine: Any, session_id: str) -> None:
    # EngineCore releases the scheduler state after all queued stages drain.
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
    core = engine.engine_core
    await core.abort_requests_async([session_id])
    while not await core.call_utility_async("is_gr_session_retired", session_id):
        await asyncio.sleep(0.001)


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
