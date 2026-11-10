# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from http import HTTPStatus

import pytest

from vllm_gr.entrypoints.beam_admission import (
    ADMISSION_TIMEOUT_ENV,
    MAX_INFLIGHT_ENV,
    MAX_WAITING_ENV,
    OUTPUT_TOP_K_ENV,
    BeamAdmissionLimiter,
    BeamAdmissionRejected,
    admitted_stream,
    beam_admission_limiter,
    beam_output_top_k,
    reset_beam_admission_limiter,
)


def test_disabled_limiter_admits_without_tracking() -> None:
    async def run() -> None:
        limiter = BeamAdmissionLimiter()
        for _ in range(1000):
            await limiter.acquire()
        assert limiter.inflight == 0
        assert limiter.waiting == 0
        assert limiter.stats()["enabled"] is False
        limiter.release()

    asyncio.run(run())


def test_full_limiter_rejects_with_retryable_status() -> None:
    async def run() -> None:
        limiter = BeamAdmissionLimiter(max_inflight=1)
        await limiter.acquire()
        with pytest.raises(BeamAdmissionRejected) as rejected:
            await limiter.acquire()
        assert rejected.value.http_status is HTTPStatus.TOO_MANY_REQUESTS
        assert limiter.stats()["rejected"] == 1
        assert limiter.inflight == 1

    asyncio.run(run())


def test_waiter_is_admitted_when_slot_is_released() -> None:
    async def run() -> None:
        limiter = BeamAdmissionLimiter(max_inflight=1, max_waiting=1, wait_timeout_s=5.0)
        await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        assert limiter.waiting == 1
        limiter.release()
        await asyncio.wait_for(waiter, timeout=2)
        assert limiter.inflight == 1
        assert limiter.waiting == 0
        limiter.release()
        assert limiter.inflight == 0

    asyncio.run(run())


def test_wait_timeout_rejects_and_keeps_counters_consistent() -> None:
    async def run() -> None:
        limiter = BeamAdmissionLimiter(max_inflight=1, max_waiting=1, wait_timeout_s=0.05)
        await limiter.acquire()
        with pytest.raises(BeamAdmissionRejected):
            await limiter.acquire()
        assert limiter.waiting == 0
        assert limiter.inflight == 1
        assert limiter.stats()["timed_out"] == 1
        limiter.release()
        assert limiter.inflight == 0

    asyncio.run(run())


def test_cancelled_waiter_does_not_leak_a_slot() -> None:
    async def run() -> None:
        limiter = BeamAdmissionLimiter(max_inflight=1, max_waiting=2, wait_timeout_s=5.0)
        await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert limiter.waiting == 0
        limiter.release()
        assert limiter.inflight == 0

    asyncio.run(run())


def test_cancellation_after_grant_returns_the_slot() -> None:
    async def run() -> None:
        limiter = BeamAdmissionLimiter(max_inflight=1, max_waiting=1, wait_timeout_s=5.0)
        await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        limiter.release()
        assert limiter.inflight == 1
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert limiter.inflight == 0
        assert limiter.waiting == 0

    asyncio.run(run())


async def _stream(events: list[str]) -> AsyncIterator[int]:
    try:
        for index in range(3):
            events.append(f"yield-{index}")
            yield index
    finally:
        events.append("closed")


def _use_limiter(monkeypatch: pytest.MonkeyPatch, limiter: BeamAdmissionLimiter) -> None:
    monkeypatch.setattr(
        "vllm_gr.entrypoints.beam_admission.beam_admission_limiter", lambda: limiter
    )


def test_admitted_stream_releases_slot_when_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = BeamAdmissionLimiter(max_inflight=1)
    _use_limiter(monkeypatch, limiter)

    async def run() -> None:
        events: list[str] = []
        values = [value async for value in admitted_stream(_stream(events))]
        assert values == [0, 1, 2]
        assert limiter.inflight == 0
        assert events[-1] == "closed"

    asyncio.run(run())


def test_admitted_stream_closes_inner_and_releases_on_early_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = BeamAdmissionLimiter(max_inflight=1)
    _use_limiter(monkeypatch, limiter)

    async def run() -> None:
        events: list[str] = []
        stream = admitted_stream(_stream(events))
        assert await stream.__anext__() == 0
        assert limiter.inflight == 1
        await stream.aclose()
        await asyncio.sleep(0.05)
        assert events[-1] == "closed"
        assert limiter.inflight == 0

    asyncio.run(run())


def test_admitted_stream_rejects_when_limiter_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = BeamAdmissionLimiter(max_inflight=1)
    _use_limiter(monkeypatch, limiter)

    async def run() -> None:
        hold = admitted_stream(_stream([]))
        assert await hold.__anext__() == 0
        with pytest.raises(BeamAdmissionRejected):
            await admitted_stream(_stream([])).__anext__()
        await hold.aclose()
        await asyncio.sleep(0.05)
        assert limiter.inflight == 0

    asyncio.run(run())


def test_env_configures_limiter_and_output_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MAX_INFLIGHT_ENV, "2")
    monkeypatch.setenv(MAX_WAITING_ENV, "4")
    monkeypatch.setenv(ADMISSION_TIMEOUT_ENV, "0.25")
    monkeypatch.setenv(OUTPUT_TOP_K_ENV, "8")
    reset_beam_admission_limiter()
    try:
        limiter = beam_admission_limiter()
        assert limiter.max_inflight == 2
        assert limiter.max_waiting == 4
        assert limiter.wait_timeout_s == pytest.approx(0.25)
        assert beam_output_top_k() == 8
        assert beam_admission_limiter() is limiter
    finally:
        reset_beam_admission_limiter()

    monkeypatch.delenv(MAX_INFLIGHT_ENV)
    monkeypatch.delenv(OUTPUT_TOP_K_ENV)
    reset_beam_admission_limiter()
    assert beam_admission_limiter().enabled is False
    assert beam_output_top_k() is None
    reset_beam_admission_limiter()


def test_limited_sequences_keeps_the_best_prefix() -> None:
    from vllm_gr.entrypoints.beam_search_v1.output_processor import _limited_sequences

    sequences = ("a", "b", "c")
    assert _limited_sequences(sequences, None) == sequences
    assert _limited_sequences(sequences, 5) == sequences
    assert _limited_sequences(sequences, 2) == ("a", "b")
