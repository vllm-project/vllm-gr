# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any, cast

import pytest

from vllm_gr.entrypoints.beam_search_v1 import (
    BeamSearchV1Session,
    OfflineBeamSearchV1Call,
    run_beam_search_v1,
)
from vllm_gr.entrypoints.gr import GRLLM
from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest, BeamBatchResult


class _Server:
    def __init__(self, *, fail_wait: bool = False) -> None:
        self.events: list[str] = []
        self.fail_wait = fail_wait

    def prepare_request(self, call: OfflineBeamSearchV1Call) -> BeamBatchRequest:
        self.events.append("prepare")
        return cast(BeamBatchRequest, {"call": call})

    def submit_once(self, request: BeamBatchRequest) -> BeamSearchV1Session:
        self.events.append("submit")
        return BeamSearchV1Session(batch_id="batch-0", handle=request)

    def wait_final(self, session: BeamSearchV1Session) -> BeamBatchResult:
        self.events.append("wait")
        if self.fail_wait:
            raise RuntimeError("worker failed")
        return cast(BeamBatchResult, {"session": session})

    def build_output(
        self,
        call: OfflineBeamSearchV1Call,
        result: BeamBatchResult,
    ) -> list[Any]:
        self.events.append("build_output")
        return [call, result]

    def close(self, session: BeamSearchV1Session) -> None:
        self.events.append("close")


def test_v1_flow_submits_once_waits_once_and_closes() -> None:
    server = _Server()
    call = OfflineBeamSearchV1Call(prompts=[[1, 2]], params=object())

    result = run_beam_search_v1(server, call)

    assert result[0] is call
    assert server.events == ["prepare", "submit", "wait", "build_output", "close"]


def test_v1_flow_closes_after_execution_failure() -> None:
    server = _Server(fail_wait=True)
    call = OfflineBeamSearchV1Call(prompts=[[1, 2]], params=object())

    with pytest.raises(RuntimeError, match="worker failed"):
        run_beam_search_v1(server, call)

    assert server.events == ["prepare", "submit", "wait", "close"]


def test_grllm_v1_entry_uses_new_server(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _Server()
    fake_llm = object.__new__(GRLLM)
    monkeypatch.setattr(
        GRLLM,
        "_create_beam_search_v1_server",
        lambda self: server,
    )

    result = GRLLM.beam_search_v1(
        fake_llm,
        [[1, 2]],
        object(),
        use_tqdm=True,
        concurrency_limit=4,
    )

    assert isinstance(result[0], OfflineBeamSearchV1Call)
    assert result[0].use_tqdm is True
    assert result[0].concurrency_limit == 4


def test_grllm_v1_entry_is_unimplemented_without_server() -> None:
    fake_llm = object.__new__(GRLLM)

    with pytest.raises(NotImplementedError, match="A2-A5"):
        GRLLM.beam_search_v1(
            fake_llm,
            [[1, 2]],
            object(),
        )
