# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Open-loop online QPS benchmark for GR and competitor serving endpoints.

The benchmark answers one question: at a fixed offered request rate, what
latency and success rate does an online beam-search endpoint deliver?  Requests
are released on a schedule from a single client process, so the offered rate is
independent of how fast the server answers.  A closed-loop concurrency sweep
hides exactly the behaviour we care about, because it slows down when the server
slows down.

Three targets are supported and share the same workload description:

``gr-openai``   POST /v1/chat/completions with ``use_beam_search`` (the online
                OpenAI chain, server-side tokenization included).
``gr-grpc``     ``VllmEngine.GenerateBatch`` with pre-tokenized ``input_ids``;
                ``--batch-size N`` submits N requests in one RPC.
``sid-generate``POST /generate on a SID-GR style server (SGLang-compatible
                ``input_ids`` + ``sampling_params.n``).

Every target reports the same numbers, so rows can be compared directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

DEFAULT_BEGIN_TOKEN = "<|sid_begin|>"
DEFAULT_END_TOKEN = "<|sid_end|>"


@dataclass(frozen=True)
class BenchConfig:
    """Workload and client settings shared by every target."""

    prompt_text: str
    prompt_ids: tuple[int, ...]
    model: str
    beam_width: int
    max_tokens: int
    batch_size: int
    request_rate: float
    duration_s: float
    warmup_shots: int
    max_inflight: int
    timeout_s: float
    begin_token: str
    end_token: str
    stream_http: bool = False

    @property
    def requests_per_shot(self) -> int:
        return self.batch_size

    @property
    def shot_interval_s(self) -> float:
        return self.batch_size / self.request_rate


@dataclass
class Sample:
    """One logical request as the caller experienced it."""

    target: str
    rate: float
    index: int
    request_id: str
    scheduled_at: float
    started_at: float
    finished_at: float
    ok: bool
    status: str
    detail: str = ""

    @property
    def latency_ms(self) -> float:
        return (self.finished_at - self.started_at) * 1000.0

    @property
    def start_offset_ms(self) -> float:
        return (self.started_at - self.scheduled_at) * 1000.0


@dataclass
class RunSummary:
    """Aggregated view of one (target, rate) run."""

    target: str
    target_url: str
    rate: float
    batch_size: int
    beam_width: int
    max_tokens: int
    duration_s: float
    offered_requests: int
    succeeded: int
    failed: int
    timed_out: int
    client_backpressure: int
    achieved_qps: float
    success_rate: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float
    failures: dict[str, int] = field(default_factory=dict)

    def table_row(self) -> str:
        return (
            f"| {self.target} | {self.rate:.0f} | {self.achieved_qps:.2f} | "
            f"{self.success_rate * 100:.3f}% | {self.mean_ms:.2f} | {self.p50_ms:.2f} | "
            f"{self.p90_ms:.2f} | {self.p99_ms:.2f} | {self.max_ms:.2f} | "
            f"{self.timed_out} | {self.client_backpressure} |"
        )


TABLE_HEADER = (
    "| service | offered qps | achieved qps | success | mean ms | p50 ms | p90 ms | "
    "p99 ms | max ms | timeouts | backpressure |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, matching the repository convention."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(
    target: str,
    target_url: str,
    cfg: BenchConfig,
    samples: Sequence[Sample],
    elapsed_s: float,
    client_backpressure: int,
) -> RunSummary:
    succeeded = [sample for sample in samples if sample.ok]
    failures: dict[str, int] = {}
    for sample in samples:
        if not sample.ok:
            failures[sample.status] = failures.get(sample.status, 0) + 1
    latencies = [sample.latency_ms for sample in succeeded]
    return RunSummary(
        target=target,
        target_url=target_url,
        rate=cfg.request_rate,
        batch_size=cfg.batch_size,
        beam_width=cfg.beam_width,
        max_tokens=cfg.max_tokens,
        duration_s=elapsed_s,
        offered_requests=len(samples),
        succeeded=len(succeeded),
        failed=len(samples) - len(succeeded),
        timed_out=failures.get("timeout", 0),
        client_backpressure=client_backpressure,
        achieved_qps=len(succeeded) / elapsed_s if elapsed_s > 0 else 0.0,
        success_rate=len(succeeded) / len(samples) if samples else 0.0,
        mean_ms=statistics.fmean(latencies) if latencies else 0.0,
        p50_ms=percentile(latencies, 0.50),
        p90_ms=percentile(latencies, 0.90),
        p99_ms=percentile(latencies, 0.99),
        max_ms=max(latencies) if latencies else 0.0,
        failures=failures,
    )


class Target(Protocol):
    """One serving endpoint the benchmark can drive."""

    name: str
    url: str

    async def start(self, cfg: BenchConfig) -> None: ...

    async def send(
        self, request_ids: Sequence[str], cfg: BenchConfig
    ) -> list[tuple[bool, str, str]]: ...

    async def close(self) -> None: ...


class _HttpTarget:
    """Shared aiohttp plumbing; subclasses only build the payload and path."""

    name = "http"
    path = "/"

    def __init__(self, base_url: str) -> None:
        self.url = base_url.rstrip("/")
        self._session: Any = None

    async def start(self, cfg: BenchConfig) -> None:
        import aiohttp

        connector = aiohttp.TCPConnector(limit=max(cfg.max_inflight, cfg.batch_size))
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=cfg.timeout_s),
        )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def build_payload(self, request_id: str, cfg: BenchConfig) -> dict[str, Any]:
        raise NotImplementedError

    async def send(
        self, request_ids: Sequence[str], cfg: BenchConfig
    ) -> list[tuple[bool, str, str]]:
        async def one(request_id: str) -> tuple[bool, str, str]:
            payload = self.build_payload(request_id, cfg)
            try:
                async with self._session.post(self.url + self.path, json=payload) as response:
                    body = await response.read()
                    if response.status != 200:
                        return (
                            False,
                            f"http_{response.status}",
                            body[:200].decode("utf-8", "replace"),
                        )
                return True, "ok", ""
            except TimeoutError:
                return False, "timeout", f"exceeded {cfg.timeout_s:.1f}s"
            except Exception as exc:  # noqa: BLE001 - one request must not abort the shot
                return False, f"error:{type(exc).__name__}", str(exc)[:200]

        return list(await asyncio.gather(*(one(request_id) for request_id in request_ids)))


class OpenAIChatTarget(_HttpTarget):
    """The online OpenAI chain: text prompt, server-side tokenization."""

    name = "gr-openai"
    path = "/v1/chat/completions"

    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)

    def build_payload(self, request_id: str, cfg: BenchConfig) -> dict[str, Any]:
        return {
            "model": cfg.model,
            "messages": [{"role": "user", "content": cfg.prompt_text}],
            "use_beam_search": True,
            "n": cfg.beam_width,
            "temperature": 0.0,
            "max_tokens": cfg.max_tokens,
            "ignore_eos": True,
            "vllm_xargs": {
                "begin_token": cfg.begin_token,
                "end_token": cfg.end_token,
            },
            "request_id": request_id,
        }


class SidGenerateTarget(_HttpTarget):
    """A SID-GR style /generate endpoint that accepts pre-tokenized ids."""

    name = "sid-generate"
    path = "/generate"

    def build_payload(self, request_id: str, cfg: BenchConfig) -> dict[str, Any]:
        return {
            "input_ids": list(cfg.prompt_ids),
            "sampling_params": {
                "n": cfg.beam_width,
                "max_new_tokens": cfg.max_tokens,
                "ignore_eos": True,
            },
            "stream": False,
            "request_id": request_id,
        }


class GrGrpcTarget:
    """GR gRPC ``GenerateBatch``; one RPC carries ``--batch-size`` requests."""

    name = "gr-grpc"

    def __init__(self, address: str) -> None:
        self.url = address.split("://", 1)[-1]
        self._channel: Any = None
        self._stub: Any = None
        self._pb: Any = None
        self._grpc: Any = None

    async def start(self, cfg: BenchConfig) -> None:
        import grpc

        from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2 as pb
        from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2_grpc as rpc

        self._pb = pb
        self._grpc = grpc
        self._channel = grpc.aio.insecure_channel(self.url)
        self._stub = rpc.VllmEngineStub(self._channel)

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None

    def build_request(self, request_ids: Sequence[str], cfg: BenchConfig) -> Any:
        pb = self._pb
        return pb.BatchGenerateRequest(
            batch_id=f"bench-{request_ids[0]}",
            requests=[
                pb.GenerateRequest(
                    request_id=request_id,
                    tokenized=pb.TokenizedInput(input_ids=list(cfg.prompt_ids)),
                    sampling_params=pb.SamplingParams(
                        max_tokens=cfg.max_tokens,
                        ignore_eos=True,
                    ),
                    beam_search=pb.BeamSearchOptions(
                        beam_width=cfg.beam_width,
                        result_width=cfg.beam_width,
                    ),
                )
                for request_id in request_ids
            ],
        )

    async def send(
        self, request_ids: Sequence[str], cfg: BenchConfig
    ) -> list[tuple[bool, str, str]]:
        try:
            response = await self._stub.GenerateBatch(
                self.build_request(request_ids, cfg), timeout=cfg.timeout_s
            )
        except self._grpc.aio.AioRpcError as exc:
            status = (
                "timeout"
                if exc.code() is self._grpc.StatusCode.DEADLINE_EXCEEDED
                else f"grpc:{exc.code().name}"
            )
            return [(False, status, exc.details() or "") for _ in request_ids]
        except TimeoutError:
            return [(False, "timeout", f"exceeded {cfg.timeout_s:.1f}s") for _ in request_ids]
        if len(response.items) != len(request_ids):
            detail = f"batch response has {len(response.items)} items, expected {len(request_ids)}"
            return [(False, "bad_batch_response", detail) for _ in request_ids]
        return [(True, "ok", "") for _ in request_ids]


def make_target(name: str, *, openai_url: str, grpc_address: str, sid_url: str) -> Target:
    if name == OpenAIChatTarget.name:
        return OpenAIChatTarget(openai_url)
    if name == SidGenerateTarget.name:
        return SidGenerateTarget(sid_url)
    if name == GrGrpcTarget.name:
        return GrGrpcTarget(grpc_address)
    raise ValueError(f"unknown target {name!r}")


async def _run_shots(
    target: Target,
    cfg: BenchConfig,
    *,
    rate: float,
    shots: int,
    base_index: int,
    record: bool,
) -> tuple[list[Sample], float, int]:
    """Release ``shots`` batches on a fixed schedule and collect their results."""
    interval = cfg.batch_size / rate
    semaphore = asyncio.Semaphore(max(cfg.max_inflight, cfg.batch_size))
    samples: list[Sample] = []
    backpressure = 0
    origin = time.perf_counter() + interval
    tasks: list[asyncio.Task[None]] = []

    async def one_shot(shot: int, scheduled_at: float, request_ids: list[str]) -> None:
        started = time.perf_counter()
        try:
            results = list(await target.send(request_ids, cfg))
        except Exception as exc:  # noqa: BLE001 - transport failure fails the whole shot
            results = [(False, f"error:{type(exc).__name__}", str(exc)[:200])] * len(request_ids)
        finished = time.perf_counter()
        if len(results) != len(request_ids):
            detail = f"target returned {len(results)} results for {len(request_ids)} requests"
            results = [(False, "bad_result_count", detail)] * len(request_ids)
        if record:
            for request_id, (ok, status, detail) in zip(request_ids, results, strict=True):
                samples.append(
                    Sample(
                        target=target.name,
                        rate=rate,
                        index=base_index + shot,
                        request_id=request_id,
                        scheduled_at=scheduled_at,
                        started_at=started,
                        finished_at=finished,
                        ok=ok,
                        status=status,
                        detail=detail,
                    )
                )
        semaphore.release()

    for shot in range(shots):
        due = origin + shot * interval
        delay = due - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        if semaphore.locked():
            # Every client slot is busy: the offered rate is not sustainable even
            # before the server is asked. Reported, never silently absorbed.
            backpressure += 1
        await semaphore.acquire()
        request_ids = [
            f"bench-{target.name}-{base_index + shot}-{index}" for index in range(cfg.batch_size)
        ]
        tasks.append(asyncio.create_task(one_shot(shot, due, request_ids)))
    if tasks:
        await asyncio.gather(*tasks)

    if samples:
        elapsed = max(sample.finished_at for sample in samples) - origin
    else:
        elapsed = time.perf_counter() - origin
    return samples, max(elapsed, 1e-9), backpressure


def _planned_shots(rate: float, cfg: BenchConfig) -> int:
    return max(1, round(rate * cfg.duration_s / cfg.batch_size))


async def run_benchmark(args: argparse.Namespace, cfg: BenchConfig) -> list[RunSummary]:
    """Run every (target, rate) pair, returning one summary per run."""
    summaries: list[RunSummary] = []
    all_samples: list[Sample] = []
    for name in args.targets:
        target = make_target(
            name,
            openai_url=args.openai_url,
            grpc_address=args.grpc_address,
            sid_url=args.sid_url,
        )
        await target.start(cfg)
        try:
            if args.warmup_shots > 0:
                warm_rate = args.rates[0]
                await _run_shots(
                    target,
                    replace(cfg, request_rate=warm_rate),
                    rate=warm_rate,
                    shots=args.warmup_shots,
                    base_index=-args.warmup_shots,
                    record=False,
                )
            for rate in args.rates:
                run_cfg = replace(cfg, request_rate=rate)
                samples, elapsed, backpressure = await _run_shots(
                    target,
                    run_cfg,
                    rate=rate,
                    shots=_planned_shots(rate, run_cfg),
                    base_index=0,
                    record=True,
                )
                summary = summarize(
                    target.name, target.url, run_cfg, samples, elapsed, backpressure
                )
                print(summary.table_row(), flush=True)
                summaries.append(summary)
                all_samples.extend(samples)
        finally:
            await target.close()
    if args.output_json:
        payload = {
            "config": {
                "model": cfg.model,
                "beam_width": cfg.beam_width,
                "max_tokens": cfg.max_tokens,
                "batch_size": cfg.batch_size,
                "duration_s": cfg.duration_s,
                "max_inflight": cfg.max_inflight,
                "timeout_s": cfg.timeout_s,
                "prompt_tokens": len(cfg.prompt_ids),
                "prompt_head": cfg.prompt_text[:120],
                "targets": list(args.targets),
                "rates": list(args.rates),
            },
            "summaries": [asdict(summary) for summary in summaries],
            "samples": [asdict(sample) for sample in all_samples],
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output_json} ({len(all_samples)} samples)", flush=True)
    return summaries


def _normalize_length(ids: Sequence[int], target_len: int) -> list[int]:
    """Trim or repeat the prompt so every request has the same input length."""
    if target_len <= 0 or len(ids) == target_len:
        return list(ids)
    if len(ids) > target_len:
        return list(ids[:target_len])
    if not ids:
        return [0] * target_len
    repeated = list(ids)
    while len(repeated) < target_len:
        repeated.extend(ids)
    return repeated[:target_len]


def load_prompt_text(args: argparse.Namespace) -> str:
    if args.prompt_text is not None:
        return args.prompt_text
    return Path(args.prompt_file).read_text(encoding="utf-8").strip()


def load_prompt_ids(args: argparse.Namespace, prompt_text: str) -> tuple[int, ...]:
    if args.prompt_ids_file:
        ids = [int(value) for value in json.loads(Path(args.prompt_ids_file).read_text())]
    else:
        source = args.tokenizer or args.model
        if not source:
            raise SystemExit("--tokenizer or --model is required to build input_ids")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=args.trust_remote_code)
        ids = list(tokenizer.encode(prompt_text, add_special_tokens=args.add_special_tokens))
    return tuple(_normalize_length(ids, args.input_len))


def parse_name_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_rate_list(value: str) -> tuple[float, ...]:
    rates = tuple(float(item) for item in value.split(",") if item.strip())
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("--request-rate needs positive numbers, e.g. 25,50,75,100,150")
    return rates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--targets",
        default="gr-openai",
        help="comma separated subset of gr-openai,sid-generate,gr-grpc",
    )
    parser.add_argument("--openai-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sid-url", default="http://127.0.0.1:8001")
    parser.add_argument("--grpc-address", default="127.0.0.1:50051")
    parser.add_argument("--model", default="OpenOneRec/OneRec-1.7B")
    parser.add_argument("--prompt-file", help="UTF-8 text prompt used by text targets")
    parser.add_argument("--prompt-text", help="inline prompt instead of --prompt-file")
    parser.add_argument("--prompt-ids-file", help="JSON list of prompt token ids")
    parser.add_argument("--tokenizer", help="tokenizer path; defaults to --model")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument(
        "--input-len",
        type=int,
        default=0,
        help="trim or repeat the prompt to this token length (0 keeps it as is)",
    )
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="requests per shot: one GenerateBatch RPC, or concurrent HTTP calls",
    )
    parser.add_argument(
        "--request-rate",
        default="25,50,75,100,150",
        help="offered requests per second, comma separated",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup-shots", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--begin-token", default=DEFAULT_BEGIN_TOKEN)
    parser.add_argument("--end-token", default=DEFAULT_END_TOKEN)
    parser.add_argument("--output-json", help="write summaries and raw samples here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.targets = parse_name_list(args.targets)
    try:
        args.rates = parse_rate_list(args.request_rate)
    except ValueError as exc:
        parser.error(str(exc))
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.prompt_file and args.prompt_text is None:
        parser.error("one of --prompt-file or --prompt-text is required")

    prompt_text = load_prompt_text(args)
    prompt_ids = load_prompt_ids(args, prompt_text)
    cfg = BenchConfig(
        prompt_text=prompt_text,
        prompt_ids=prompt_ids,
        model=args.model,
        beam_width=args.beam_width,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        request_rate=args.rates[0],
        duration_s=args.duration,
        warmup_shots=args.warmup_shots,
        max_inflight=args.max_inflight,
        timeout_s=args.timeout,
        begin_token=args.begin_token,
        end_token=args.end_token,
    )

    print(
        f"prompt_tokens={len(prompt_ids)} beam_width={cfg.beam_width} "
        f"max_tokens={cfg.max_tokens} batch={cfg.batch_size}",
        flush=True,
    )
    print(TABLE_HEADER, flush=True)
    summaries = asyncio.run(run_benchmark(args, cfg))
    failing = [summary for summary in summaries if summary.failures]
    if failing:
        print("\nfailures:", flush=True)
        for summary in failing:
            print(f"- {summary.target} @ {summary.rate:g} qps: {summary.failures}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
