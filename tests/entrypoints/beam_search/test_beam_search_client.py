# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import asyncio
import gc
import threading
import weakref
from types import SimpleNamespace
from typing import Any, NoReturn, cast
from unittest.mock import AsyncMock, Mock

import pytest
from vllm.exceptions import VLLMValidationError

from vllm_gr.entrypoints.beam_search import (
    run_beam_search,
    run_beam_search_async,
)
from vllm_gr.entrypoints.beam_search.client import (
    AsyncBeamSearchClient,
    BeamSearchClient,
)
from vllm_gr.entrypoints.beam_search.exceptions import BeamSearchEngineError
from vllm_gr.entrypoints.beam_search.protocol import (
    BeamSearchSession,
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.v1.beam.batch_contracts import (
    ADMISSION_REJECTED_CODE,
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchRequest,
    BeamBatchResult,
    GRAdmissionError,
)


class _Backend:
    def __init__(self) -> None:
        self.active: set[str] = set()
        self.submit_started = asyncio.Event()
        self.allow_submit = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.allow_cleanup = asyncio.Event()
        self.cleanup_finished = asyncio.Event()
        self.cleanup_calls: list[tuple[str, str]] = []
        self.cleanup_cancelled = False
        self.submit_error = False
        self.cleanup_error = False

    async def submit_gr_session(self, request: BeamBatchRequest) -> object:
        self.active.add(request.batch_id)
        self.submit_started.set()
        await self.allow_submit.wait()
        if self.submit_error:
            raise RuntimeError("submission failed after acceptance")
        return object()

    async def _cleanup(self, method: str, session_id: str) -> None:
        self.cleanup_calls.append((method, session_id))
        self.cleanup_started.set()
        try:
            await self.allow_cleanup.wait()
        except asyncio.CancelledError:
            self.cleanup_cancelled = True
            raise
        if self.cleanup_error:
            raise RuntimeError("cleanup failed")
        self.active.remove(session_id)
        self.cleanup_finished.set()

    async def abort_gr_session(self, session_id: str) -> None:
        await self._cleanup("abort", session_id)

    async def release_gr_session(self, session_id: str) -> None:
        await self._cleanup("release", session_id)


async def _cancel(task: asyncio.Task[object]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("terminal", [False, True])  # type: ignore[misc]
def test_close_survives_cancellation_and_joins_existing_cleanup(terminal: bool) -> None:
    async def run() -> None:
        backend = _Backend()
        backend.active.add("session-1")
        client = AsyncBeamSearchClient(SimpleNamespace(engine_client=backend))
        session = BeamSearchSession(
            batch_id="session-1", session_id="session-1", handle=object(), terminal=terminal
        )

        closing = asyncio.create_task(client.close(session))
        await asyncio.wait_for(backend.cleanup_started.wait(), timeout=2)
        await _cancel(closing)

        assert not backend.cleanup_cancelled
        assert not session.closed
        assert backend.active == {"session-1"}

        retry_started = asyncio.Event()

        async def retry_close() -> None:
            retry_started.set()
            await client.close(session)

        retry = asyncio.create_task(retry_close())
        await retry_started.wait()
        await _cancel(retry)
        assert not backend.cleanup_cancelled
        assert not session.closed

        backend.allow_cleanup.set()
        await asyncio.wait_for(client.close(session), timeout=2)
        await client.close(session)
        assert session.closed
        assert not backend.active
        assert backend.cleanup_calls == [("release" if terminal else "abort", "session-1")]

    asyncio.run(run())


@pytest.mark.parametrize("submit_error", [False, True])  # type: ignore[misc]
def test_submission_rollback_survives_repeated_cancellation(submit_error: bool) -> None:
    async def run() -> None:
        backend = _Backend()
        backend.submit_error = submit_error
        client = AsyncBeamSearchClient(SimpleNamespace(engine_client=backend))
        # Only the transport boundary is exercised; request preparation is independent.
        request = cast(BeamBatchRequest, SimpleNamespace(batch_id="session-1"))
        submitting = asyncio.create_task(client.submit_once(request))
        await asyncio.wait_for(backend.submit_started.wait(), timeout=2)
        if not submit_error:
            submitting.cancel()
        backend.allow_submit.set()
        await asyncio.wait_for(backend.cleanup_started.wait(), timeout=2)
        await _cancel(submitting)

        assert not backend.cleanup_cancelled
        assert backend.active == {"session-1"}
        backend.allow_cleanup.set()
        await asyncio.wait_for(backend.cleanup_finished.wait(), timeout=2)
        assert not backend.active
        assert backend.cleanup_calls == [("abort", "session-1")]

    asyncio.run(run())


@pytest.mark.parametrize("cancel_request", [False, True])  # type: ignore[misc]
def test_failed_submission_rollback_retries_after_request_disappears(
    cancel_request: bool, caplog: pytest.LogCaptureFixture
) -> None:
    class Backend(_Backend):
        cleanup_waiter: weakref.ReferenceType[asyncio.Future[None]]

        def __init__(self) -> None:
            super().__init__()
            self.retry_started = asyncio.Event()

        async def abort_gr_session(self, session_id: str) -> None:
            self.cleanup_calls.append(("abort", session_id))
            if len(self.cleanup_calls) == 1:
                self.cleanup_started.set()
                await self.allow_cleanup.wait()
            if len(self.cleanup_calls) <= 2:
                raise RuntimeError("transient abort RPC failure")
            # Neither the request nor a transport object keeps this waiter alive.
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self.cleanup_waiter = weakref.ref(waiter)
            self.retry_started.set()
            await waiter
            self.active.remove(session_id)
            self.cleanup_finished.set()

    async def run() -> None:
        backend = Backend()
        backend.submit_error = True
        backend.allow_submit.set()
        client = AsyncBeamSearchClient(SimpleNamespace(engine_client=backend))
        request = cast(BeamBatchRequest, SimpleNamespace(batch_id="session-1"))
        submitting = asyncio.create_task(client.submit_once(request))
        await asyncio.wait_for(backend.cleanup_started.wait(), timeout=2)
        if cancel_request:
            await _cancel(submitting)
        backend.allow_cleanup.set()
        if not cancel_request:
            with pytest.raises(BeamSearchEngineError, match="submission failed after acceptance"):
                await asyncio.wait_for(submitting, timeout=2)
        del client, submitting
        # Captured exception tracebacks must not act as accidental cleanup owners.
        caplog.clear()
        gc.collect()

        await asyncio.wait_for(backend.retry_started.wait(), timeout=2)
        caplog.clear()
        gc.collect()
        waiter = backend.cleanup_waiter()
        assert waiter is not None and not waiter.done()
        assert backend.active == {"session-1"}
        assert backend.cleanup_calls == [("abort", "session-1")] * 3
        waiter.set_result(None)
        await asyncio.wait_for(backend.cleanup_finished.wait(), timeout=2)
        assert not backend.active

    asyncio.run(run())


def test_submission_requires_abort_support_before_sending() -> None:
    submit = AsyncMock(side_effect=AssertionError("must not submit without abort support"))
    client = AsyncBeamSearchClient(
        SimpleNamespace(engine_client=SimpleNamespace(submit_gr_session=submit))
    )
    request = cast(BeamBatchRequest, SimpleNamespace(batch_id="session-1"))
    with pytest.raises(NotImplementedError, match="abort_gr_session"):
        asyncio.run(client.submit_once(request))
    submit.assert_not_called()


@pytest.mark.parametrize("terminal", [False, True])  # type: ignore[misc]
def test_close_retries_after_entrypoint_and_client_exit(
    terminal: bool, caplog: pytest.LogCaptureFixture
) -> None:
    class Backend(_Backend):
        cleanup_waiter: weakref.ReferenceType[asyncio.Future[None]]

        async def wait_gr_session(self, session_id: str, handle: object) -> BeamBatchResult:
            if not terminal:
                raise RuntimeError("execution failed")
            return BeamBatchResult(
                BEAM_BATCH_PROTOCOL_VERSION,
                session_id,
                None,
                BeamBatchFailure("execution_failed", "execution failed", None),
            )

        async def _cleanup(self, method: str, session_id: str) -> None:
            self.cleanup_calls.append((method, session_id))
            if len(self.cleanup_calls) == 1:
                raise RuntimeError("transient cleanup failure")
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self.cleanup_waiter = weakref.ref(waiter)
            self.cleanup_started.set()
            await waiter
            self.active.remove(session_id)
            self.cleanup_finished.set()

    class Client(AsyncBeamSearchClient):  # type: ignore[misc]
        async def prepare_request(self, call: OnlineBeamSearchRequest) -> BeamBatchRequest:
            return _request()

    async def run() -> None:
        backend = Backend()
        backend.allow_submit.set()
        client = Client(SimpleNamespace(engine_client=backend))
        with pytest.raises(RuntimeError):
            await run_beam_search_async(
                client, OnlineBeamSearchRequest(([1, 2],), "request-1", object())
            )
        del client
        caplog.clear()
        gc.collect()
        await asyncio.wait_for(backend.cleanup_started.wait(), timeout=2)
        caplog.clear()
        gc.collect()
        waiter = backend.cleanup_waiter()
        assert waiter is not None and not waiter.done()
        assert backend.active == {"session-1"}
        assert backend.cleanup_calls == [("release" if terminal else "abort", "session-1")] * 2
        waiter.set_result(None)
        await asyncio.wait_for(backend.cleanup_finished.wait(), timeout=2)
        assert not backend.active

    asyncio.run(run())


def _request() -> BeamBatchRequest:
    return cast(
        BeamBatchRequest,
        SimpleNamespace(
            batch_id="session-1",
            items=[SimpleNamespace(request=SimpleNamespace(data_parallel_rank=None))],
        ),
    )


@pytest.mark.parametrize("operation", ["rollback", "abort", "release"])  # type: ignore[misc]
def test_sync_entrypoint_retries_cleanup_before_exiting(operation: str) -> None:
    active: set[str] = set()
    cleanup_calls = []
    thread_id = threading.get_ident()

    def submit(request: BeamBatchRequest) -> object:
        active.add(request.batch_id)
        if operation == "rollback":
            raise RuntimeError("submission failed after acceptance")
        return object()

    def wait(session_id: str, handle: object) -> BeamBatchResult:
        if operation == "abort":
            raise RuntimeError("execution failed")
        return BeamBatchResult(
            BEAM_BATCH_PROTOCOL_VERSION,
            session_id,
            None,
            BeamBatchFailure("execution_failed", "execution failed", None),
        )

    def cleanup(method: str, session_id: str) -> None:
        assert threading.get_ident() == thread_id
        cleanup_calls.append((method, session_id))
        if len(cleanup_calls) == 1:
            raise RuntimeError("transient cleanup failure")
        active.remove(session_id)

    class Client(BeamSearchClient):  # type: ignore[misc]
        def prepare_request(self, call: OfflineBeamSearchRequest) -> BeamBatchRequest:
            return _request()

        def close(self, session: BeamSearchSession) -> None:
            super().close(session)
            assert session.closed
            super().close(session)

    engine = SimpleNamespace(
        submit_gr_session=submit,
        wait_gr_session=wait,
        abort_gr_session=lambda session_id: cleanup("abort", session_id),
        release_gr_session=lambda session_id: cleanup("release", session_id),
    )
    with pytest.raises(RuntimeError):
        run_beam_search(
            Client(SimpleNamespace(llm_engine=engine)), OfflineBeamSearchRequest([[1, 2]], object())
        )
    assert not active
    assert cleanup_calls == [("release" if operation == "release" else "abort", "session-1")] * 2


@pytest.mark.parametrize("operation", ["release", "rollback"])  # type: ignore[misc]
def test_cleanup_outlives_request_and_client(operation: str) -> None:
    class Backend(_Backend):
        cleanup_waiter: weakref.ReferenceType[asyncio.Future[None]]

        async def _cleanup(self, method: str, session_id: str) -> None:
            # No event or transport object roots this future/task. The client
            # must retain its own cleanup owner after the request disappears.
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self.cleanup_waiter = weakref.ref(waiter)
            self.cleanup_started.set()
            await waiter
            self.active.remove(session_id)
            self.cleanup_finished.set()

    async def run() -> None:
        backend = Backend()
        backend.submit_error = True
        backend.allow_submit.set()

        async def request_lifetime() -> None:
            client = AsyncBeamSearchClient(SimpleNamespace(engine_client=backend))
            if operation == "rollback":
                request = cast(BeamBatchRequest, SimpleNamespace(batch_id="session-1"))
                await client.submit_once(request)
            else:
                backend.active.add("session-1")
                session = BeamSearchSession(
                    batch_id="session-1",
                    session_id="session-1",
                    handle=object(),
                    terminal=True,
                )
                await client.close(session)

        request_task = asyncio.create_task(request_lifetime())
        await asyncio.wait_for(backend.cleanup_started.wait(), timeout=2)
        await _cancel(request_task)
        del request_task
        gc.collect()

        waiter = backend.cleanup_waiter()
        assert waiter is not None and not waiter.done()
        waiter.set_result(None)
        await asyncio.wait_for(backend.cleanup_finished.wait(), timeout=2)
        assert not backend.active

    asyncio.run(run())


def test_submission_rollback_fences_delayed_acceptance() -> None:
    class Backend(_Backend):
        acceptance: asyncio.Task[object]

        async def submit_gr_session(self, request: BeamBatchRequest) -> object:
            async def accept() -> object:
                await self.allow_submit.wait()
                self.active.add(request.batch_id)
                return object()

            # Model a transport where local cancellation cannot retract a send.
            self.acceptance = asyncio.create_task(accept())
            self.submit_started.set()
            return await asyncio.shield(self.acceptance)

    async def run() -> None:
        backend = Backend()
        client = AsyncBeamSearchClient(SimpleNamespace(engine_client=backend))
        request = cast(BeamBatchRequest, SimpleNamespace(batch_id="session-1"))
        submitting = asyncio.create_task(client.submit_once(request))
        await asyncio.wait_for(backend.submit_started.wait(), timeout=2)
        submitting.cancel()
        # Let submit_once enter rollback before cancelling it a second time.
        await asyncio.sleep(0)
        await _cancel(submitting)
        assert not backend.cleanup_started.is_set()

        backend.allow_submit.set()
        await asyncio.wait_for(backend.cleanup_started.wait(), timeout=2)
        assert backend.active == {"session-1"}
        backend.allow_cleanup.set()
        await asyncio.wait_for(backend.cleanup_finished.wait(), timeout=2)
        assert not backend.active
        assert backend.cleanup_calls == [("abort", "session-1")]

    asyncio.run(run())


@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
def test_terminal_failure_precedes_tokenizer_access(online: bool) -> None:
    failure = BeamBatchFailure("beam_device_error", "original device failure", 0)
    result = BeamBatchResult(BEAM_BATCH_PROTOCOL_VERSION, "session-1", None, failure)
    with pytest.raises(BeamSearchEngineError) as caught:
        if online:
            client = AsyncBeamSearchClient(SimpleNamespace(engine_client=object()))
            call = OnlineBeamSearchRequest(([1, 2],), "external", object())
        else:
            client = BeamSearchClient(SimpleNamespace(llm_engine=object()))
            call = OfflineBeamSearchRequest([[1, 2]], object())
        client.build_output(call, _request(), result)
    assert caught.value.failure is failure


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("admission", [False, True])  # type: ignore[misc]
def test_submission_failure_classification(online: bool, admission: bool) -> None:
    from tests.helpers.beam_requests import make_batch_request

    error = GRAdmissionError("capacity exceeded") if admission else RuntimeError("transport failed")
    engine = SimpleNamespace(
        submit_gr_session=AsyncMock(side_effect=error) if online else Mock(side_effect=error),
        abort_gr_session=AsyncMock() if online else Mock(),
        release_gr_session=AsyncMock() if online else Mock(),
    )
    request = make_batch_request()
    with pytest.raises(VLLMValidationError if admission else BeamSearchEngineError) as caught:
        if online:
            asyncio.run(
                AsyncBeamSearchClient(SimpleNamespace(engine_client=engine)).submit_once(request)
            )
        else:
            BeamSearchClient(SimpleNamespace(llm_engine=engine)).submit_once(request)

    assert caught.value.__cause__ is error
    assert str(caught.value) == str(error)
    if isinstance(caught.value, BeamSearchEngineError):
        assert caught.value.failure.code == "submit_failed"
    engine.abort_gr_session.assert_called_once_with(request.batch_id)


@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
def test_admission_rejection_is_a_client_error_after_a_terminal_result(online: bool) -> None:
    """A rejection the engine sends back is the request being answered.

    The terminal payload has one shape, so the code is the whole difference
    between telling the caller what to change and reporting a server fault for
    a session that never ran.
    """
    failure = BeamBatchFailure(ADMISSION_REJECTED_CODE, "beam_width must equal 128", None)
    result = BeamBatchResult(BEAM_BATCH_PROTOCOL_VERSION, "session-1", None, failure)
    if online:
        client = AsyncBeamSearchClient(SimpleNamespace(engine_client=object()))
        call = OnlineBeamSearchRequest(([1, 2],), "external", object())
    else:
        client = BeamSearchClient(SimpleNamespace(llm_engine=object()))
        call = OfflineBeamSearchRequest([[1, 2]], object())

    with pytest.raises(VLLMValidationError) as caught:
        client.build_output(call, _request(), result)

    assert str(caught.value) == "beam_width must equal 128"


@pytest.mark.parametrize("operation", ["submit", "wait", "close"])  # type: ignore[misc]
def test_engine_dead_propagates_without_sync_cleanup_retries(operation: str) -> None:
    from vllm.v1.engine.exceptions import EngineDeadError

    from tests.helpers.beam_requests import make_batch_request

    calls = []

    def dead(*args: Any) -> NoReturn:
        calls.append(args)
        raise EngineDeadError()

    engine = SimpleNamespace(
        submit_gr_session=dead,
        wait_gr_session=dead,
        abort_gr_session=dead,
        release_gr_session=dead,
    )
    client = BeamSearchClient(SimpleNamespace(llm_engine=engine))
    session = BeamSearchSession(batch_id="A", session_id="A", handle=object())
    with pytest.raises(EngineDeadError):
        if operation == "submit":
            client.submit_once(make_batch_request())
        elif operation == "wait":
            client.wait_final(session)
        else:
            client.close(session)
    assert len(calls) == 1


def test_inproc_step_failure_reaches_the_caller_with_its_cause() -> None:
    """A failed in-process step ends the call; cleanup must neither hang nor mask it.

    This is the path that hung: cleanup aborted the session and stepped the
    engine until it retired, which never happens after a failed dispatch. It
    also checks that the ``finally`` cleanup does not replace the
    EngineDeadError carrying the step failure with a new one.
    """
    from vllm.v1.engine.core_client import InprocClient
    from vllm.v1.engine.exceptions import EngineDeadError

    from tests.helpers.beam_requests import make_batch_request
    from vllm_gr.v1.engine.beam_session_client import (
        abort_gr_session_sync,
        release_gr_session_sync,
        submit_gr_session_sync,
        wait_gr_session_sync,
    )
    from vllm_gr.v1.engine.core_client_patch import apply_engine_client_patches

    oom = RuntimeError("CUDA out of memory (test)")
    calls = {"steps": 0, "retired_checks": 0}

    def step_fn() -> NoReturn:
        calls["steps"] += 1
        raise oom

    def is_gr_session_retired(session_id: str) -> bool:
        del session_id
        calls["retired_checks"] += 1
        if calls["retired_checks"] > 10:
            raise AssertionError("cleanup is waiting for a session that cannot retire")
        return False

    apply_engine_client_patches()
    core_client = object.__new__(InprocClient)
    core_client.engine_core = SimpleNamespace(
        _handle_submit_gr_session=lambda batch: None,
        step_fn=step_fn,
        post_step=lambda model_executed: None,
        abort_requests=lambda ids: None,
        is_gr_session_retired=is_gr_session_retired,
    )
    engine = SimpleNamespace(
        engine_core=core_client,
        output_processor=SimpleNamespace(
            add_request=lambda request, prompt: None,
            abort_requests=lambda ids, internal: [],
        ),
    )
    engine.step = lambda: core_client.get_output().outputs
    engine.submit_gr_session = lambda request: submit_gr_session_sync(engine, request)
    engine.wait_gr_session = lambda sid, handle: wait_gr_session_sync(engine, sid, handle)
    engine.abort_gr_session = lambda sid: abort_gr_session_sync(engine, sid)
    engine.release_gr_session = lambda sid: release_gr_session_sync(engine, sid)

    frontend = BeamSearchClient(SimpleNamespace(llm_engine=engine))
    frontend.prepare_request = lambda call: make_batch_request()
    with pytest.raises(EngineDeadError) as raised:
        run_beam_search(frontend, OfflineBeamSearchRequest([[1, 2]], object()))

    assert raised.value.__cause__ is oom
    assert calls == {"steps": 1, "retired_checks": 0}


@pytest.mark.parametrize("operation", ["submit", "wait", "close"])  # type: ignore[misc]
def test_engine_dead_propagates_and_ends_async_cleanup(operation: str) -> None:
    from vllm.v1.engine.exceptions import EngineDeadError

    from tests.helpers.beam_requests import make_batch_request
    from vllm_gr.entrypoints.beam_search.client import _ASYNC_CLEANUP_TASKS

    async def run() -> None:
        calls = []

        async def dead(*args: Any) -> NoReturn:
            calls.append(args)
            raise EngineDeadError()

        engine = SimpleNamespace(
            submit_gr_session=dead,
            wait_gr_session=dead,
            abort_gr_session=dead,
            release_gr_session=dead,
        )
        client = AsyncBeamSearchClient(SimpleNamespace(engine_client=engine))
        session = BeamSearchSession(batch_id="A", session_id="A", handle=object())
        before = set(_ASYNC_CLEANUP_TASKS)
        with pytest.raises(EngineDeadError):
            if operation == "submit":
                await client.submit_once(make_batch_request())
            elif operation == "wait":
                await client.wait_final(session)
            else:
                await client.close(session)
        await asyncio.sleep(0)
        assert len(calls) == 1
        assert _ASYNC_CLEANUP_TASKS == before

    asyncio.run(run())


def test_engine_only_client_reuses_session_methods() -> None:
    from vllm_gr.entrypoints.beam_search.client import AsyncBeamSearchClient
    from vllm_gr.v1.beam.batch_contracts import BeamBatchResult

    async def run() -> None:
        result = BeamBatchResult(
            protocol_version=BEAM_BATCH_PROTOCOL_VERSION, batch_id="id", output=None, failure=None
        )
        engine = SimpleNamespace(
            submit_gr_session=AsyncMock(return_value="handle"),
            wait_gr_session=AsyncMock(return_value=result),
            release_gr_session=AsyncMock(),
            abort_gr_session=AsyncMock(),
        )
        client = AsyncBeamSearchClient(engine_client=engine)
        request = SimpleNamespace(
            batch_id="id",
            items=[
                SimpleNamespace(
                    request=SimpleNamespace(data_parallel_rank=None),
                )
            ],
        )
        session = await client.submit_once(request)
        assert await client.wait_final(session) is result
        await client.close(session)
        engine.release_gr_session.assert_awaited_once_with("id")
        engine.abort_gr_session.assert_not_called()
        with pytest.raises(ValueError, match="prepared"):
            await client.prepare_request(None)

    asyncio.run(run())


pytestmark = pytest.mark.cpu_test
