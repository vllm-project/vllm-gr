# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Install atomic GR session transport and frontend lifecycle adapters."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest
from vllm_gr.v1.engine.async_llm import make_patched_process_outputs
from vllm_gr.v1.engine.core_client import submit_gr_session_async

# Native InprocClient methods wrapped by the engine-dead guard, captured once.
_INPROC_NATIVE: dict[str, Callable[..., Any]] = {}


def sync_mp_submit_gr_session(self: Any, batch_request: BeamBatchRequest) -> None:
    from vllm.v1.engine import EngineCoreRequestType

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


def inproc_submit_gr_session(self: Any, batch_request: BeamBatchRequest) -> None:
    _inproc_ensure_alive(self)
    self.engine_core._handle_submit_gr_session(batch_request)


def apply_request_type_patches() -> None:
    from vllm_gr.v1.engine.wire import register_engine_core_request_types

    register_engine_core_request_types()


def apply_engine_client_patches() -> None:
    """Install V1 submission on every supported EngineCore transport."""
    from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient

    AsyncMPClient.submit_gr_session_async = submit_gr_session_async
    SyncMPClient.submit_gr_session = sync_mp_submit_gr_session
    InprocClient.submit_gr_session = inproc_submit_gr_session
    for name, patched in (
        ("get_output", inproc_get_output),
        ("add_request", inproc_add_request),
        ("abort_requests", inproc_abort_requests),
        ("shutdown", inproc_shutdown),
    ):
        native = getattr(InprocClient, name)
        if native is not patched:
            _INPROC_NATIVE[name] = native
            setattr(InprocClient, name, patched)


# Native frontend submission methods wrapped by the dead-engine rollback.
_FRONTEND_NATIVE: dict[str, Callable[..., Any]] = {}


def llm_engine_add_request(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Roll back frontend registration when EngineCore is dead.

    Native ``LLMEngine.add_request`` registers each request (and, with n > 1,
    each child and its parent) with the output processor before handing it to
    EngineCore. A dead engine rejects the hand-off with EngineDeadError, which
    would leave the registration behind on every retry. Only the entries added
    by this call are removed; other errors keep native behavior.
    """
    from vllm.v1.engine.exceptions import EngineDeadError

    processor = self.output_processor
    states, parents = len(processor.request_states), len(processor.parent_requests)
    try:
        return _FRONTEND_NATIVE["llm_engine_add_request"](self, *args, **kwargs)
    except EngineDeadError:
        # Dicts keep insertion order, so this call's entries are the tail.
        added = list(processor.request_states)[states:]
        added += list(processor.parent_requests)[parents:]
        if added:
            processor.abort_requests(added, internal=True)
        raise


async def async_llm_add_request(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
    """AsyncLLM counterpart of ``llm_engine_add_request`` for one request.

    ``AsyncLLM.add_request`` checks ``errored`` before it registers anything,
    but EngineCore can die between that check and the hand-off.
    """
    from vllm.v1.engine.exceptions import EngineDeadError

    try:
        return await _FRONTEND_NATIVE["async_llm_add_request"](self, request, *args, **kwargs)
    except EngineDeadError:
        self.output_processor.abort_requests([request.request_id], internal=True)
        raise


def apply_async_llm_patches() -> None:
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.engine.output_processor import OutputProcessor

    from vllm_gr.v1.engine.gr_session_client import install_gr_frontend_methods

    OutputProcessor.process_outputs = make_patched_process_outputs(OutputProcessor.process_outputs)
    install_gr_frontend_methods(LLMEngine, AsyncLLM)
    for owner, name, key, patched in (
        (LLMEngine, "add_request", "llm_engine_add_request", llm_engine_add_request),
        (AsyncLLM, "_add_request", "async_llm_add_request", async_llm_add_request),
    ):
        native = getattr(owner, name)
        if native is not patched:
            _FRONTEND_NATIVE[key] = native
            setattr(owner, name, patched)


def apply_gr_session_patches() -> None:
    """Install GR transport in the frontend, including in-process engines."""
    apply_request_type_patches()
    apply_engine_client_patches()
    apply_async_llm_patches()

    from vllm_gr.v1.engine.engine_core_patch import apply_engine_core_child_patches

    apply_engine_core_child_patches()
