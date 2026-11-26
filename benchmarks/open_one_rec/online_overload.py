# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Measure a OneRec service before, during, and after overload."""

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from benchmarks.metrics import linear_percentile_sorted


@dataclass(frozen=True)
class Phase:
    name: str
    rate: int
    seconds: int


@dataclass(frozen=True)
class RequestResult:
    started_at: float
    finished_at: float
    outcome: str


async def run_request(
    session: aiohttp.ClientSession, url: str, payload: dict[str, object], width: int
) -> RequestResult:
    clock = asyncio.get_running_loop().time
    started = clock()
    try:
        async with session.post(url, json=payload) as response:
            outcome = f"http_{response.status}"
            if response.status == 200:
                try:
                    body = await response.json()
                except (aiohttp.ContentTypeError, ValueError):
                    body = None
                choices = body.get("choices") if isinstance(body, dict) else None
                complete = (
                    isinstance(choices, list)
                    and len(choices) == width
                    and all(
                        isinstance(choice, dict)
                        and isinstance(choice.get("text"), str)
                        and choice["text"].strip()
                        for choice in choices
                    )
                )
                outcome = "ok" if complete else "invalid_response"
    except TimeoutError:
        outcome = "timeout"
    except aiohttp.ClientError:
        outcome = "transport_error"
    return RequestResult(started, clock(), outcome)


def p99_ms(samples: list[float]) -> float | None:
    return 1000 * linear_percentile_sorted(sorted(samples), 0.99) if samples else None


def summarize_recovery(phase: Phase, requests: list[RequestResult]) -> list[dict[str, object]]:
    windows = []
    for start in range(0, phase.seconds, 5):
        end = min(start + 5, phase.seconds)
        scheduled = requests[start * phase.rate : end * phase.rate]
        successes = [r.finished_at - r.started_at for r in scheduled if r.outcome == "ok"]
        windows.append(
            {
                "start_s": start,
                "end_s": end,
                "offered_requests": len(scheduled),
                "successful_requests": len(successes),
                "successful_latency_p99_ms": p99_ms(successes),
            }
        )
    return windows


async def run(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, object],
    width: int,
    phases: list[Phase],
) -> dict[str, object]:
    clock = asyncio.get_running_loop().time
    start = clock()
    boundaries = [start]
    cohorts: list[list[tuple[float, asyncio.Task[RequestResult]]]] = []
    for phase in phases:
        phase_start = boundaries[-1]
        boundaries.append(phase_start + phase.seconds)
        cohort = []
        for index in range(phase.rate * phase.seconds):
            due = phase_start + index / phase.rate
            await asyncio.sleep(max(0, due - clock()))
            cohort.append((due, asyncio.create_task(run_request(session, url, payload, width))))
        cohorts.append(cohort)

    await asyncio.sleep(max(0, boundaries[-1] - clock()))
    results = await asyncio.gather(*(task for cohort in cohorts for _, task in cohort))

    reports: list[dict[str, object]] = []
    offset = 0
    for phase, cohort, begin, end in zip(phases, cohorts, boundaries, boundaries[1:]):
        requests = results[offset : offset + len(cohort)]
        offset += len(cohort)
        completed = sum(r.outcome == "ok" and begin <= r.finished_at < end for r in results)
        reports.append(
            {
                "phase": phase.name,
                "duration_s": phase.seconds,
                "target_rps": phase.rate,
                "offered_requests": len(cohort),
                "started_rps": sum(begin <= r.started_at < end for r in results) / phase.seconds,
                "successful_completions_rps": completed / phase.seconds,
                "outcomes": dict(Counter(r.outcome for r in requests)),
                "start_delay_p99_ms": p99_ms(
                    [r.started_at - due for r, (due, _) in zip(requests, cohort)]
                ),
                "successful_latency_p99_ms": p99_ms(
                    [r.finished_at - r.started_at for r in requests if r.outcome == "ok"]
                ),
            }
        )
    return {
        "phases": reports,
        "recovery_windows": summarize_recovery(phases[-1], results[-len(cohorts[-1]) :]),
        "started_after_last_phase": sum(r.started_at >= boundaries[-1] for r in results),
        "successful_completions_after_last_phase": sum(
            r.outcome == "ok" and r.finished_at >= boundaries[-1] for r in results
        ),
    }


async def benchmark(
    args: argparse.Namespace, phases: list[Phase], prompt: str
) -> dict[str, object]:
    url = args.base_url.rstrip("/") + "/v1/completions"
    payload = {
        "model": args.model,
        "prompt": prompt,
        "use_beam_search": True,
        "n": args.width,
        "temperature": 0.0,
        "max_tokens": 5,
        "ignore_eos": True,
        "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
    }
    api_key = os.environ.get("VLLM_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    # A connector limit would queue requests client-side and hide server overload.
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=args.timeout),
        headers=headers,
    ) as session:
        for _ in range(args.warmup):
            result = await run_request(session, url, payload, args.width)
            if result.outcome != "ok":
                raise RuntimeError(f"warmup failed: {result.outcome}")
        report = await run(session, url, payload, args.width, phases)
        return {
            "model": args.model,
            "width": args.width,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "timeout_s": args.timeout,
            **report,
        }


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--prompt", type=Path, required=True, help="UTF-8 prompt file")
    parser.add_argument("--width", type=positive_int, required=True, help="Returned Beam count")
    parser.add_argument("--normal-rps", type=positive_int, required=True)
    parser.add_argument("--overload-rps", type=positive_int, required=True)
    parser.add_argument("--phase-seconds", type=positive_int, default=30)
    parser.add_argument(
        "--recovery-seconds", type=positive_int, help="Override --phase-seconds for recovery"
    )
    parser.add_argument(
        "--timeout", type=positive_int, default=30, help="Request timeout in seconds"
    )
    parser.add_argument("--warmup", type=positive_int, default=3)
    args = parser.parse_args()
    if args.overload_rps <= args.normal_rps:
        parser.error("--overload-rps must exceed --normal-rps")

    phases = [
        Phase("normal", args.normal_rps, args.phase_seconds),
        Phase("overload", args.overload_rps, args.phase_seconds),
        Phase("recovery", args.normal_rps, args.recovery_seconds or args.phase_seconds),
    ]
    try:
        prompt = args.prompt.read_text(encoding="utf-8")
        if not prompt.strip():
            parser.error("--prompt must contain a non-empty prompt")
        report = asyncio.run(benchmark(args, phases, prompt))
    except (OSError, UnicodeError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
