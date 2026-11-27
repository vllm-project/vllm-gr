# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from benchmarks.open_one_rec.online_overload import (
    Phase,
    RequestResult,
    run,
    run_request,
    summarize_recovery,
)

pytestmark = [pytest.mark.l1, pytest.mark.cpu_test]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("status", "body", "expected"),
    [
        (200, {"choices": [{"text": "sid-a"}, {"text": "sid-b"}]}, "ok"),
        (200, {"choices": [{"text": "sid-a"}]}, "invalid_response"),
        (200, {"choices": [{"text": "sid-a"}, {"text": ""}]}, "invalid_response"),
        (429, {}, "http_429"),
        (503, {}, "http_503"),
    ],
)
def test_response_outcomes(status: int, body: dict[str, object], expected: str) -> None:
    session = MagicMock(spec=aiohttp.ClientSession)
    response = session.post.return_value.__aenter__.return_value
    response.status = status
    response.json = AsyncMock(return_value=body)

    result = asyncio.run(run_request(session, "http://localhost/v1/completions", {}, 2))

    assert result.outcome == expected


def test_timeout_is_counted_separately() -> None:
    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.return_value.__aenter__.side_effect = TimeoutError

    result = asyncio.run(run_request(session, "http://localhost/v1/completions", {}, 2))

    assert result.outcome == "timeout"


def test_recovery_starts_with_overload_requests_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    async def exercise() -> None:
        release_overload = asyncio.Event()
        started_overload = asyncio.Event()
        order: list[int] = []

        async def request(*_: object) -> RequestResult:
            clock = asyncio.get_running_loop().time
            started = clock()
            index = len(order)
            order.append(index)
            if index == 1:
                started_overload.set()
                await release_overload.wait()
            if index == 3:
                assert started_overload.is_set()
                release_overload.set()
            outcome = "http_429" if index == 2 else "ok"
            return RequestResult(started, clock(), outcome)

        monkeypatch.setattr("benchmarks.open_one_rec.online_overload.run_request", request)
        phases = [Phase("normal", 1, 1), Phase("overload", 2, 1), Phase("recovery", 1, 1)]
        async with aiohttp.ClientSession() as session:
            report = await asyncio.wait_for(run(session, "", {}, 2, phases), timeout=6)
        rows = report["phases"]
        assert isinstance(rows, list)
        normal, overload, recovery = rows
        assert normal["outcomes"] == {"ok": 1}
        assert overload["outcomes"] == {"ok": 1, "http_429": 1}
        assert overload["successful_completions_rps"] == 0
        assert recovery["outcomes"] == {"ok": 1}
        assert recovery["successful_completions_rps"] == 2

    asyncio.run(exercise())


def test_recovery_windows_keep_failed_requests_in_the_denominator() -> None:
    requests = [RequestResult(0, 1, "ok")] * 5 + [RequestResult(0, 1, "timeout")]

    assert summarize_recovery(Phase("recovery", 1, 6), requests) == [
        {
            "start_s": 0,
            "end_s": 5,
            "offered_requests": 5,
            "successful_requests": 5,
            "successful_latency_p99_ms": 1000,
        },
        {
            "start_s": 5,
            "end_s": 6,
            "offered_requests": 1,
            "successful_requests": 0,
            "successful_latency_p99_ms": None,
        },
    ]
