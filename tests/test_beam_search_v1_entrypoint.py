# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from vllm_gr.entrypoints.beam_search_v1 import (
    run_beam_search_v1,
    run_beam_search_v1_async,
)
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    AsyncBeamSearchFrontend,
    BeamSearchSession,
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest, BeamBatchResult


class _Frontend:
    def __init__(self, *, fail_wait: bool = False) -> None:
        self.events: list[str] = []
        self.fail_wait = fail_wait
        self.request: BeamBatchRequest | None = None

    def prepare_request(self, call: OfflineBeamSearchRequest) -> BeamBatchRequest:
        self.events.append("prepare")
        self.request = cast(BeamBatchRequest, {"call": call})
        return self.request

    def submit_once(self, request: BeamBatchRequest) -> BeamSearchSession:
        self.events.append("submit")
        return BeamSearchSession(
            batch_id="batch-0",
            session_id="batch-0",
            handle=request,
        )

    def wait_final(self, session: BeamSearchSession) -> BeamBatchResult:
        self.events.append("wait")
        if self.fail_wait:
            raise RuntimeError("worker failed")
        return cast(BeamBatchResult, {"session": session})

    def build_output(
        self,
        call: OfflineBeamSearchRequest,
        request: BeamBatchRequest,
        result: BeamBatchResult,
    ) -> list[Any]:
        self.events.append("build_output")
        return [call, request, result]

    def close(self, session: BeamSearchSession) -> None:
        self.events.append("close")


def test_v1_flow_submits_once_waits_once_and_closes() -> None:
    frontend = _Frontend()
    call = OfflineBeamSearchRequest(prompts=[[1, 2]], params=object())

    result = run_beam_search_v1(frontend, call)

    assert result[0] is call
    assert result[1] is frontend.request
    assert frontend.events == ["prepare", "submit", "wait", "build_output", "close"]


def test_v1_flow_closes_after_execution_failure() -> None:
    frontend = _Frontend(fail_wait=True)
    call = OfflineBeamSearchRequest(prompts=[[1, 2]], params=object())

    with pytest.raises(RuntimeError, match="worker failed"):
        run_beam_search_v1(frontend, call)

    assert frontend.events == ["prepare", "submit", "wait", "close"]


def test_grllm_v1_entry_uses_new_frontend(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.entrypoints.gr import GRLLM

    frontend = _Frontend()
    fake_llm = object.__new__(GRLLM)
    fake_llm.llm_engine = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config={"vllm_gr_beam_execution_mode": "v1"})
    )
    monkeypatch.setattr(
        GRLLM,
        "_create_beam_search_v1_client",
        lambda self: frontend,
    )

    result = GRLLM.beam_search_v1(
        fake_llm,
        [[1, 2]],
        object(),
        use_tqdm=True,
        concurrency_limit=4,
    )

    assert isinstance(result[0], OfflineBeamSearchRequest)
    assert result[0].use_tqdm is True
    assert result[0].concurrency_limit == 4


@pytest.mark.parametrize("fail_shutdown", [False, True])  # type: ignore[misc]
def test_v1_heap_is_prepared_before_serving_and_released_on_close(
    monkeypatch: pytest.MonkeyPatch, fail_shutdown: bool
) -> None:
    from vllm.entrypoints.llm import LLM

    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.utils import engine_lifecycle

    events: list[str] = []

    def shutdown() -> None:
        events.append("shutdown")
        if fail_shutdown:
            raise RuntimeError("engine shutdown failed")

    def initialize(llm: Any, **kwargs: Any) -> None:
        llm.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(additional_config={"vllm_gr_beam_execution_mode": "v1"}),
            shutdown=shutdown,
        )
        events.append("engine_ready")

    monkeypatch.setattr(LLM, "__init__", initialize)
    monkeypatch.setattr("vllm_gr.plugin.initialize_runtime", lambda: {})
    monkeypatch.setattr(
        "vllm_gr.entrypoints.openai.beam_search_patch.patch_engine_core_process_entrypoint",
        lambda: None,
    )
    monkeypatch.setattr(
        engine_lifecycle, "acquire_frontend_startup_heap", lambda: events.append("freeze")
    )
    monkeypatch.setattr(
        engine_lifecycle, "release_frontend_startup_heap", lambda: events.append("unfreeze")
    )
    monkeypatch.setattr("vllm_gr.entrypoints.gr.release_accelerator_memory", lambda: None)
    monkeypatch.setattr("vllm_gr.entrypoints.gr.gc.collect", lambda: 0)
    frontend = _Frontend()
    monkeypatch.setattr(GRLLM, "_create_beam_search_v1_client", lambda self: frontend)

    llm = GRLLM(model="/fake/v1")
    assert events == ["engine_ready", "freeze"]
    llm.beam_search_v1([[1, 2]], object())
    llm.beam_search_v1([[1, 2]], object())
    assert events == ["engine_ready", "freeze"]
    if fail_shutdown:
        with pytest.raises(RuntimeError, match="engine shutdown failed"):
            llm.shutdown()
    else:
        llm.shutdown()
    llm.shutdown()
    assert events == ["engine_ready", "freeze", "shutdown", "unfreeze"]


def test_v1_startup_heap_keeps_dynamic_cycles_collectible_and_releases_owners() -> None:
    # GC is process-wide. Exercise real collection in an isolated interpreter
    # so the test runner's objects and existing freeze ownership stay intact.
    script = textwrap.dedent(
        """
        import gc
        import weakref
        from vllm_gr.utils.engine_lifecycle import (
            acquire_frontend_startup_heap, release_frontend_startup_heap,
        )

        class Node:
            def __init__(self):
                self.peer = self

        threshold, enabled = gc.get_threshold(), gc.isenabled()
        for _ in range(2):
            startup = Node()
            startup_ref = weakref.ref(startup)
            acquire_frontend_startup_heap()
            acquire_frontend_startup_heap()
            assert gc.get_freeze_count() > 0
            del startup
            transient = Node()
            transient_ref = weakref.ref(transient)
            del transient
            gc.collect()
            assert transient_ref() is None
            assert startup_ref() is not None
            release_frontend_startup_heap()
            gc.collect()
            assert startup_ref() is not None
            release_frontend_startup_heap()
            assert gc.get_freeze_count() == 0
            gc.collect()
            assert startup_ref() is None

        assert gc.get_threshold() == threshold and gc.isenabled() == enabled
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_v1_async_flow_awaits_preparation_before_submitting_and_closes() -> None:
    frontend = AsyncMock(spec=AsyncBeamSearchFrontend)
    call = OnlineBeamSearchRequest(prompt=[1, 2], request_id="request-0", params=object())
    request = cast(BeamBatchRequest, object())
    session = BeamSearchSession(batch_id="batch-0", session_id="batch-0", handle=request)
    result = cast(BeamBatchResult, object())
    output = [object()]
    frontend.prepare_request.return_value = request
    frontend.submit_once.return_value = session
    frontend.wait_final.return_value = result
    frontend.build_output.return_value = output

    assert asyncio.run(run_beam_search_v1_async(frontend, call)) is output

    frontend.prepare_request.assert_awaited_once_with(call)
    frontend.submit_once.assert_awaited_once_with(request)
    frontend.wait_final.assert_awaited_once_with(session)
    frontend.build_output.assert_called_once_with(call, request, result)
    frontend.close.assert_awaited_once_with(session)


@pytest.mark.parametrize(  # type: ignore[misc]
    "failure_stage", ["submit_once", "wait_final"]
)
def test_v1_async_flow_closes_only_after_session_is_returned(failure_stage: str) -> None:
    frontend = AsyncMock(spec=AsyncBeamSearchFrontend)
    call = OnlineBeamSearchRequest(prompt=[1, 2], request_id="request-0", params=object())
    session = BeamSearchSession(batch_id="batch-0", session_id="batch-0", handle=object())
    frontend.submit_once.return_value = session
    getattr(frontend, failure_stage).side_effect = RuntimeError("frontend failed")

    with pytest.raises(RuntimeError, match="frontend failed"):
        asyncio.run(run_beam_search_v1_async(frontend, call))

    frontend.prepare_request.assert_awaited_once_with(call)
    if failure_stage == "submit_once":
        frontend.wait_final.assert_not_awaited()
        frontend.close.assert_not_awaited()
    else:
        frontend.close.assert_awaited_once_with(session)
