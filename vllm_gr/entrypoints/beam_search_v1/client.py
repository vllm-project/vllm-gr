# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Submit, wait for, and clean up EngineCore-owned beam-search sessions."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_gr.entrypoints.beam_search_v1.exceptions import failure_exception
from vllm_gr.entrypoints.beam_search_v1.input_processor import BeamSearchInputProcessor
from vllm_gr.entrypoints.beam_search_v1.output_processor import (
    BeamSearchOutputProcessor,
)
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    BeamSearchSession,
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.logger import init_logger
from vllm_gr.v1.beam.batch_contracts import (
    ADMISSION_REJECTED_CODE,
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchRequest,
    BeamBatchResult,
)

logger = init_logger(__name__)

# Clients are created per request. Keep cleanup alive even after the request
# and its client have been dropped; asyncio.shield alone only holds weak refs.
_ASYNC_CLEANUP_TASKS: set[asyncio.Task[None]] = set()


def _run_sync_cleanup(cleanup: Callable[[], None], session_id: str) -> None:
    # The synchronous transport belongs to the calling thread. Keep ownership
    # here until acknowledgement instead of moving its RPCs to another thread.
    delay = 0.1
    while True:
        try:
            cleanup()
            return
        except EngineDeadError:
            raise
        except Exception:
            logger.exception("GR session %s cleanup failed; retrying", session_id)
            time.sleep(delay)
            delay = min(delay * 2, 5.0)


def _start_async_cleanup(
    cleanup: Callable[[], Awaitable[None]], session_id: str
) -> tuple[asyncio.Task[None], asyncio.Future[None]]:
    first_attempt: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def retry_cleanup() -> None:
        delay = 0.1
        while True:
            try:
                await cleanup()
            except EngineDeadError as exc:
                if not first_attempt.done():
                    first_attempt.set_exception(exc)
                raise
            except Exception as exc:
                if not first_attempt.done():
                    first_attempt.set_exception(exc)
                logger.exception("GR session %s cleanup failed; retrying", session_id)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)
            else:
                if not first_attempt.done():
                    first_attempt.set_result(None)
                return

    # A cancelled caller may never retrieve the first attempt's error.
    first_attempt.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    task = asyncio.create_task(retry_cleanup())
    _ASYNC_CLEANUP_TASKS.add(task)

    def cleanup_done(done: asyncio.Task[None]) -> None:
        _ASYNC_CLEANUP_TASKS.discard(done)
        if not first_attempt.done():
            first_attempt.cancel()
        if not done.cancelled() and (exc := done.exception()) is not None:
            logger.error("GR session %s cleanup failed", session_id, exc_info=exc)

    task.add_done_callback(cleanup_done)
    return task, first_attempt


def _get_engine_method(owner: Any, name: str) -> Any:
    method = getattr(owner, name, None)
    if method is None:
        raise NotImplementedError(f"A3 must install {type(owner).__name__}.{name}")
    return method


def _create_failure_result(
    batch_id: str,
    code: str,
    exc: Exception,
    *,
    protocol_version: int = BEAM_BATCH_PROTOCOL_VERSION,
) -> BeamBatchResult:
    return BeamBatchResult(
        protocol_version=protocol_version,
        batch_id=batch_id,
        output=None,
        failure=BeamBatchFailure(code=code, message=str(exc), item_index=None),
    )


def _submission_failure(exc: Exception) -> BeamBatchFailure:
    """Classify a submission that never became a session.

    Async scheduling refuses a request it cannot run before registering
    anything, which makes the exception a description of the request rather
    than of the engine. Anything else here is the transport failing.
    """
    from vllm_gr.v1.engine.gr_async_scheduler import GRAdmissionError

    code = ADMISSION_REJECTED_CODE if isinstance(exc, GRAdmissionError) else "submit_failed"
    return BeamBatchFailure(code=code, message=str(exc), item_index=None)


class BeamSearchClient:
    """Offline A2 adapter. A3 supplies the EngineCore transport methods."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.engine = llm.llm_engine
        self._boundary_tokens: tuple[str, tuple[int | None, int | None]] | None = None

    def prepare_request(self, call: OfflineBeamSearchRequest) -> BeamBatchRequest:
        BeamSearchOutputProcessor.ensure_available()
        processor = BeamSearchInputProcessor(self.engine)
        request = processor.process_offline_inputs(self.llm, call)
        self._boundary_tokens = (request.batch_id, processor.boundary_token_ids)
        return request

    def submit_once(self, request: BeamBatchRequest) -> BeamSearchSession:
        submit = _get_engine_method(self.engine, "submit_gr_session")
        abort = _get_engine_method(self.engine, "abort_gr_session")
        _get_engine_method(self.engine, "release_gr_session")
        try:
            handle = submit(request)
        except EngineDeadError:
            raise
        except Exception as exc:
            _run_sync_cleanup(lambda: abort(request.batch_id), request.batch_id)
            raise failure_exception(_submission_failure(exc)) from exc
        return BeamSearchSession(
            batch_id=request.batch_id,
            session_id=request.batch_id,
            handle=handle,
            data_parallel_rank=request.items[0].request.data_parallel_rank,
            protocol_version=request.protocol_version,
        )

    def wait_final(self, session: BeamSearchSession) -> BeamBatchResult:
        try:
            result = _get_engine_method(self.engine, "wait_gr_session")(
                session.session_id, session.handle
            )
            if not isinstance(result, BeamBatchResult):
                raise TypeError("EngineCore returned a non-BeamBatchResult terminal value")
            session.terminal = True
            session.failed = result.failure is not None
            return result
        except EngineDeadError:
            raise
        except Exception as exc:  # noqa: BLE001 - EngineCore failure boundary
            session.failed = True
            return _create_failure_result(
                session.batch_id, "execution_failed", exc, protocol_version=session.protocol_version
            )

    def build_output(
        self,
        call: OfflineBeamSearchRequest,
        request: BeamBatchRequest,
        result: BeamBatchResult,
    ) -> list[Any]:
        if result.failure is not None:
            raise failure_exception(result.failure)
        prepared = self._boundary_tokens
        boundaries = (
            prepared[1] if prepared is not None and prepared[0] == request.batch_id else None
        )
        return BeamSearchOutputProcessor(
            self.llm.get_tokenizer(), boundary_token_ids=boundaries
        ).process_outputs(call, request, result)

    def abort(self, session: BeamSearchSession) -> None:
        if session.terminal or session.closed:
            return
        self.close(session)

    def close(self, session: BeamSearchSession) -> None:
        if session.closed:
            return
        _run_sync_cleanup(lambda: self._close_session(session), session.session_id)

    def _close_session(self, session: BeamSearchSession) -> None:
        method = "release_gr_session" if session.terminal else "abort_gr_session"
        _get_engine_method(self.engine, method)(session.session_id)
        if not session.terminal:
            session.failed = True
        session.closed = True


class AsyncBeamSearchClient:
    """Online A2 adapter using the same request and EngineCore session ID."""

    def __init__(self, serving: Any) -> None:
        self.serving = serving
        self.engine = serving.engine_client
        self._boundary_tokens: tuple[str, tuple[int | None, int | None]] | None = None
        self._close_tasks: dict[str, asyncio.Task[None]] = {}

    async def prepare_request(self, call: OnlineBeamSearchRequest) -> BeamBatchRequest:
        BeamSearchOutputProcessor.ensure_available()
        processor = BeamSearchInputProcessor(self.engine)
        request = await processor.process_online_inputs(self.serving, call)
        self._boundary_tokens = (request.batch_id, processor.boundary_token_ids)
        return request

    async def submit_once(self, request: BeamBatchRequest) -> BeamSearchSession:
        submit = _get_engine_method(self.engine, "submit_gr_session")
        # Reject unsupported transports before they can accept a session that
        # this client has no way to roll back.
        _get_engine_method(self.engine, "abort_gr_session")
        _get_engine_method(self.engine, "release_gr_session")
        submission: asyncio.Future[object] | None = None
        try:
            handle = submit(request)
            if inspect.isawaitable(handle):
                submission = asyncio.ensure_future(handle)
                handle = await asyncio.shield(submission)
        except asyncio.CancelledError:
            await self._rollback_submission(request.batch_id, submission)
            raise
        except EngineDeadError:
            raise
        except Exception as exc:
            await self._rollback_submission(request.batch_id, submission)
            raise failure_exception(_submission_failure(exc)) from exc
        return BeamSearchSession(
            batch_id=request.batch_id,
            session_id=request.batch_id,
            handle=handle,
            data_parallel_rank=request.items[0].request.data_parallel_rank,
            protocol_version=request.protocol_version,
        )

    async def _rollback_submission(
        self, session_id: str, submission: asyncio.Future[object] | None
    ) -> None:
        async def cleanup() -> None:
            # Fence acceptance before aborting. Request cancellation must not
            # cancel the transport while it may still be accepting the session.
            if submission is not None:
                try:
                    await submission
                except EngineDeadError:
                    raise
                except (Exception, asyncio.CancelledError):
                    pass
            await self._abort_session(session_id)

        _, first_attempt = _start_async_cleanup(cleanup, session_id)
        try:
            await asyncio.shield(first_attempt)
        except EngineDeadError:
            raise
        except Exception:
            # The retained task owns retries before the original submission
            # exception returns to a caller that has no session handle.
            pass

    async def wait_final(self, session: BeamSearchSession) -> BeamBatchResult:
        try:
            result = _get_engine_method(self.engine, "wait_gr_session")(
                session.session_id, session.handle
            )
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, BeamBatchResult):
                raise TypeError("EngineCore returned a non-BeamBatchResult terminal value")
            session.terminal = True
            session.failed = result.failure is not None
            return result
        except EngineDeadError:
            raise
        except Exception as exc:  # noqa: BLE001 - EngineCore failure boundary
            session.failed = True
            return _create_failure_result(
                session.batch_id, "execution_failed", exc, protocol_version=session.protocol_version
            )

    def build_output(
        self,
        call: OnlineBeamSearchRequest,
        request: BeamBatchRequest,
        result: BeamBatchResult,
    ) -> list[Any]:
        if result.failure is not None:
            raise failure_exception(result.failure)
        tokenizer = self.serving.input_processor.tokenizer
        if tokenizer is None:
            raise VLLMValidationError(
                "beam_search_v1 requires tokenizer initialization",
                parameter="skip_tokenizer_init",
                value=True,
            )
        prepared = self._boundary_tokens
        boundaries = (
            prepared[1] if prepared is not None and prepared[0] == request.batch_id else None
        )
        return BeamSearchOutputProcessor(
            tokenizer, boundary_token_ids=boundaries
        ).process_online_outputs(call, request, result)

    async def abort(self, session: BeamSearchSession) -> None:
        if session.terminal or session.closed:
            return
        await self.close(session)

    async def _abort_session(self, session_id: str) -> None:
        result = _get_engine_method(self.engine, "abort_gr_session")(session_id)
        if inspect.isawaitable(result):
            await result

    async def close(self, session: BeamSearchSession) -> None:
        if session.closed:
            return
        task = self._close_tasks.get(session.session_id)
        if task is None:
            task, first_attempt = _start_async_cleanup(
                lambda: self._close_session(session), session.session_id
            )
            self._close_tasks[session.session_id] = task
            task.add_done_callback(lambda _: self._close_tasks.pop(session.session_id, None))
            await asyncio.shield(first_attempt)
            return
        await asyncio.shield(task)

    async def _close_session(self, session: BeamSearchSession) -> None:
        if session.terminal:
            result = _get_engine_method(self.engine, "release_gr_session")(session.session_id)
            if inspect.isawaitable(result):
                await result
        else:
            await self._abort_session(session.session_id)
            session.failed = True
        session.closed = True
