# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import traceback
from collections.abc import Callable
from typing import Any

from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


async def submit_gr_session_async(self, batch_request: BeamBatchRequest) -> None:
    """Submit the whole GR session once; Decode stages stay inside EngineCore."""
    requests = tuple(item.request for item in batch_request.items)
    if not requests:
        raise ValueError("GR batch session must not be empty")
    for request in requests:
        request.client_index = self.client_index
        if hasattr(self, "current_wave"):
            request.current_wave = self.current_wave
    ensure_stats_task = getattr(self, "_ensure_stats_update_task", None)
    if ensure_stats_task is not None:
        ensure_stats_task()
    engine = None
    if hasattr(self, "get_core_engine_for_request"):
        engine = self.get_core_engine_for_request(requests[0])
    to_await = self._send_input(EngineCoreRequestType.SUBMIT_GR_SESSION, batch_request, engine)
    if hasattr(self, "first_req_send_socket") and not self.engines_running:
        import msgspec

        await self.first_req_send_socket.send(msgspec.msgpack.encode(("FIRST_REQ", engine)))
    await to_await
    self._ensure_output_queue_task()


# Native InprocClient methods wrapped by the engine-dead guard, captured once.
_INPROC_NATIVE: dict[str, Callable[..., Any]] = {}


def sync_mp_submit_gr_session(self: Any, batch_request: BeamBatchRequest) -> None:
    requests = tuple(item.request for item in batch_request.items)
    if not requests:
        raise ValueError("GR batch session must not be empty")
    for request in requests:
        request.client_index = getattr(self, "client_index", 0)
        if hasattr(self, "current_wave"):
            request.current_wave = self.current_wave
    if getattr(self, "is_dp", False):
        self.engines_running = True
    self._send_input(EngineCoreRequestType.SUBMIT_GR_SESSION, batch_request)


def inproc_engine_failure(client: Any) -> BaseException | None:
    """Return the exception that ended an in-process EngineCore, if any."""
    return getattr(client, "_gr_engine_failure", None)


def _inproc_ensure_alive(self: Any) -> None:
    # The multiprocess clients raise EngineDeadError once EngineCore has died.
    # In-process, a failed step leaves EngineCore in place with work that will
    # never drain, so later calls must fail the same way instead of waiting.
    failure = inproc_engine_failure(self)
    if failure is not None:
        from vllm.v1.engine.exceptions import EngineDeadError

        raise EngineDeadError() from failure


def inproc_get_output(self: Any) -> Any:
    from vllm.v1.engine.exceptions import EngineDeadError

    _inproc_ensure_alive(self)
    try:
        return _INPROC_NATIVE["get_output"](self)
    except EngineDeadError as exc:
        self._gr_engine_failure = exc
        raise
    except Exception as exc:
        # A multiprocess EngineCore busy loop treats any step exception as fatal.
        self._gr_engine_failure = exc
        raise EngineDeadError() from exc


def inproc_add_request(self: Any, *args: Any, **kwargs: Any) -> Any:
    _inproc_ensure_alive(self)
    return _INPROC_NATIVE["add_request"](self, *args, **kwargs)


def inproc_abort_requests(self: Any, request_ids: Any) -> Any:
    # SyncMPClient aborts nothing, without raising, once EngineCore has died;
    # an empty list is a no-op in both native clients.
    if not request_ids or inproc_engine_failure(self) is not None:
        return None
    return _INPROC_NATIVE["abort_requests"](self, request_ids)


def _release_failure_frames(failure: BaseException) -> None:
    # The recorded failure keeps its traceback, and every frame on it keeps its
    # locals, such as the logits of the step that ran out of memory. Clearing
    # the frames drops those locals; messages, causes and the formatted
    # traceback (file and line) stay intact.
    seen: set[int] = set()
    pending: list[BaseException | None] = [failure]
    while pending:
        exc = pending.pop()
        if exc is None or id(exc) in seen:
            continue
        seen.add(id(exc))
        traceback.clear_frames(exc.__traceback__)
        pending.extend((exc.__cause__, exc.__context__))


def inproc_shutdown(self: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return _INPROC_NATIVE["shutdown"](self, *args, **kwargs)
    finally:
        failure = inproc_engine_failure(self)
        if failure is not None:
            _release_failure_frames(failure)


def inproc_submit_gr_session(self, batch_request):
    _inproc_ensure_alive(self)
    self.engine_core._handle_submit_gr_session(batch_request)
