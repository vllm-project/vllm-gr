# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
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

    def prepare_request(self, call: OfflineBeamSearchRequest) -> BeamBatchRequest:
        self.events.append("prepare")
        return cast(BeamBatchRequest, {"call": call})

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
        result: BeamBatchResult,
    ) -> list[Any]:
        self.events.append("build_output")
        return [call, result]

    def close(self, session: BeamSearchSession) -> None:
        self.events.append("close")


def test_v1_flow_submits_once_waits_once_and_closes() -> None:
    frontend = _Frontend()
    call = OfflineBeamSearchRequest(prompts=[[1, 2]], params=object())

    result = run_beam_search_v1(frontend, call)

    assert result[0] is call
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
