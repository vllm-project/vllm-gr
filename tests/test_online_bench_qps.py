# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import statistics

import pytest
from aiohttp import web

from tools.online_bench.qps_bench import (
    BenchConfig,
    Sample,
    SidGenerateTarget,
    _normalize_length,
    _run_shots,
    parse_rate_list,
    percentile,
    summarize,
)


def _cfg(**overrides: object) -> BenchConfig:
    values: dict[str, object] = {
        "prompt_text": "hello",
        "prompt_ids": (1, 2, 3),
        "model": "test-model",
        "beam_width": 8,
        "max_tokens": 3,
        "batch_size": 1,
        "request_rate": 50.0,
        "duration_s": 0.4,
        "warmup_shots": 0,
        "max_inflight": 32,
        "timeout_s": 5.0,
        "begin_token": "<b>",
        "end_token": "<e>",
    }
    values.update(overrides)
    return BenchConfig(**values)  # type: ignore[arg-type]


async def _serve(handler) -> tuple[web.AppRunner, int]:
    app = web.Application()
    app.router.add_post("/generate", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    # aiohttp keeps the bound sockets on the private site server object.
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, port


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([7.0], 0.99) == 7.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.50) == pytest.approx(25.0)
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.90) == pytest.approx(37.0)


def test_summarize_separates_success_from_failure() -> None:
    samples = [
        Sample("t", 50.0, 0, "r0", 0.0, 0.0, 0.010, True, "ok"),
        Sample("t", 50.0, 1, "r1", 0.0, 0.0, 0.030, True, "ok"),
        Sample("t", 50.0, 2, "r2", 0.0, 0.0, 0.500, False, "timeout", "budget"),
    ]
    summary = summarize("t", "http://endpoint", _cfg(), samples, 1.0, 2)
    assert summary.succeeded == 2
    assert summary.failed == 1
    assert summary.timed_out == 1
    assert summary.achieved_qps == pytest.approx(2.0)
    assert summary.success_rate == pytest.approx(2 / 3)
    assert summary.mean_ms == pytest.approx(20.0)
    assert summary.max_ms == pytest.approx(30.0)
    assert summary.client_backpressure == 2
    assert summary.failures == {"timeout": 1}


def test_normalize_length_trims_and_repeats() -> None:
    assert _normalize_length([1, 2, 3], 0) == [1, 2, 3]
    assert _normalize_length([1, 2, 3, 4], 2) == [1, 2]
    assert _normalize_length([1, 2], 5) == [1, 2, 1, 2, 1]


def test_parse_rate_list_rejects_non_positive() -> None:
    assert parse_rate_list("25, 100,150") == (25.0, 100.0, 150.0)
    with pytest.raises(ValueError):
        parse_rate_list("0,100")


def test_open_loop_run_records_latency_and_success() -> None:
    async def run() -> None:
        async def handler(request: web.Request) -> web.Response:
            await asyncio.sleep(0.02)
            return web.json_response({"ok": True})

        runner, port = await _serve(handler)
        cfg = _cfg()
        target = SidGenerateTarget(f"http://127.0.0.1:{port}")
        await target.start(cfg)
        try:
            samples, elapsed, backpressure = await _run_shots(
                target, cfg, rate=50.0, shots=10, base_index=0, record=True
            )
        finally:
            await target.close()
            await runner.cleanup()

        assert len(samples) == 10
        assert all(sample.ok for sample in samples)
        assert backpressure == 0
        assert 10.0 <= statistics.fmean([sample.latency_ms for sample in samples]) < 1000.0
        assert elapsed > 0.0

    asyncio.run(run())


def test_http_error_status_is_reported_per_request() -> None:
    async def run() -> None:
        async def handler(request: web.Request) -> web.Response:
            return web.json_response({"error": "overloaded"}, status=503)

        runner, port = await _serve(handler)
        cfg = _cfg()
        target = SidGenerateTarget(f"http://127.0.0.1:{port}")
        await target.start(cfg)
        try:
            samples, _, _ = await _run_shots(
                target, cfg, rate=50.0, shots=3, base_index=0, record=True
            )
        finally:
            await target.close()
            await runner.cleanup()

        assert len(samples) == 3
        assert not any(sample.ok for sample in samples)
        assert {sample.status for sample in samples} == {"http_503"}

    asyncio.run(run())
