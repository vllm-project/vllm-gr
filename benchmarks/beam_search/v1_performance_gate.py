# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Collect comparable offline batch and online concurrency V1 timings."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from benchmarks.beam_search.v1_sid_validation import (
    validate_sid_text,
    validate_sid_tokens,
)


def distribution(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("latency samples must be positive finite values")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "min_ms": ordered[0],
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(0.50),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1],
        "std_ms": statistics.pstdev(ordered),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> None:
    import yaml

    from tools.build_constraint_table import build_constraint_table

    artifact = args.output_dir / "constraint_table.safetensors"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_digest, tokenizer_digest, _ = build_constraint_table(
        triples_path=str(args.triples),
        model=args.model,
        revision=args.revision,
        output_path=str(artifact),
        trust_remote_code=True,
    )
    config = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "graph_enabled": True,
            "enable_kv_pool_ping_pong": getattr(args, "kv_pool_ping_pong", True),
            "worker_decision": True,
            "max_width": args.beam_width,
            "max_decode_steps": args.generated_steps,
        },
        "constraint_table": {
            "enabled": True,
            "backend": "cuda",
            "max_top_k": args.beam_width,
            "path": str(artifact.resolve()),
            "format": "constraint_table_v1",
            "artifact_digest": artifact_digest,
            "tokenizer_digest": tokenizer_digest,
        },
    }
    (args.output_dir / "vllm_gr_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    write_json(
        args.output_dir / "setup.json",
        {
            "benchmark_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model": args.model,
            "revision": args.revision,
            "triples": str(args.triples.resolve()),
            "prompt": str(args.prompt.resolve()),
            "input_tokens_including_sid_begin": args.input_len,
            "beam_width": args.beam_width,
            "generated_steps": args.generated_steps,
            "batch_sizes": args.batch_sizes,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": max(args.batch_sizes),
            "max_num_batched_tokens": args.max_model_len,
            "temperature": 0.0,
            "ignore_eos": True,
            "request_timeout_seconds": args.request_timeout,
            "max_tokens_including_sid_boundaries": args.generated_steps + 2,
            "artifact_digest": artifact_digest,
            "tokenizer_digest": tokenizer_digest,
            "execution_mode": "v1",
            "graph_enabled": True,
            "enable_kv_pool_ping_pong": getattr(args, "kv_pool_ping_pong", True),
            "prefix_caching": False,
            "chunked_prefill": False,
            "timing": "end-to-end caller time; warmup excluded; generated beam IDs/scores/text materialized offline (full/compact normalized)",
            "online_streaming": False,
        },
    )


def case_result(
    mode: str,
    count: int,
    group_samples: list[float],
    request_samples: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    total_seconds = sum(group_samples) / 1000.0
    requests = count * len(group_samples)
    return {
        "mode": mode,
        "batch_size" if mode == "offline" else "concurrency": count,
        "logical_requests": requests,
        "measured_wall_seconds": total_seconds,
        "group_latency_ms": distribution(group_samples),
        "request_latency_ms": distribution(request_samples),
        "requests_per_second": requests / total_seconds,
        "logical_generated_tokens_per_second": requests * args.generated_steps / total_seconds,
        "beam_generated_tokens_per_second": (
            requests * args.generated_steps * args.beam_width / total_seconds
        ),
        "group_samples_ms": group_samples,
        "request_samples_ms": request_samples,
    }


def validate_online_response(
    body: dict[str, Any], *, index: int, beam_width: int, prompt_tokens: int
) -> None:
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != beam_width:
        raise RuntimeError(f"online request={index}: incomplete V1 output")
    for beam_index, choice in enumerate(choices):
        validate_sid_text(choice.get("text"), f"online request={index} beam={beam_index}")
    usage = body.get("usage")
    if usage is not None:
        expected_completion = beam_width * 5
        if (
            not isinstance(usage, dict)
            or usage.get("prompt_tokens") != prompt_tokens
            or usage.get("completion_tokens") != expected_completion
            or usage.get("total_tokens") != prompt_tokens + expected_completion
        ):
            raise RuntimeError(f"online request={index}: inconsistent token usage: {usage!r}")


def error_details(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(error_details(child) for child in error.exceptions)
    return f"{type(error).__name__}: {error}"


def _measure_offline_batch(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    params: Any,
    tokenizer: Any,
    prompt_token_ids: list[list[int]],
    count: int,
    beam_width: int,
    output_snapshots: list[Any] | None = None,
) -> float:
    """Consume and validate one batch, releasing its outputs before the next call."""
    import torch

    start = time.perf_counter_ns()
    outputs = llm.beam_search_v1(prompts, params, use_tqdm=False)
    # Consume identical generated IDs, scores and decoded text on both revisions.
    # The base runtime returns full outputs; the candidate defaults to compact.
    # Do not import candidate-only output classes in this shared A/B driver.
    consumed = []
    for request_index, output in enumerate(outputs):
        prompt_ids = prompt_token_ids[request_index] if request_index < count else []
        candidates = []
        for sequence in output.sequences:
            if hasattr(sequence, "token_ids"):
                tokens = list(sequence.token_ids)
                score = float(sequence.cumulative_logprob)
            else:
                # Preserve legacy lazy full-text materialization, then normalize
                # to the same generated-only result that compact consumers use.
                _ = sequence.text
                tokens = sequence.tokens[len(prompt_ids) :]
                score = float(sequence.cum_logprob)
            text = tokenizer.decode(
                tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            candidates.append((tokens, score, text))
        consumed.append(candidates)
    torch.accelerator.synchronize()
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if len(consumed) != count or any(len(row) != beam_width for row in consumed):
        raise RuntimeError("offline: incomplete V1 output")
    for request_index, (output, candidates) in enumerate(zip(outputs, consumed, strict=True)):
        prompt_ids = prompt_token_ids[request_index]
        compact_prompt = getattr(output, "prompt_token_ids", None)
        compact_prompt_matches = compact_prompt is not None and tuple(compact_prompt) == tuple(
            prompt_ids
        )
        for beam_index, (sequence, (tokens, score, text)) in enumerate(
            zip(output.sequences, candidates, strict=True)
        ):
            prompt_matches = (
                compact_prompt_matches
                if hasattr(sequence, "token_ids")
                else list(sequence.tokens[: len(prompt_ids)]) == prompt_ids
            )
            context = f"offline B={count} request={request_index} beam={beam_index}"
            if not prompt_matches:
                raise RuntimeError(f"{context}: output prompt/order mismatch")
            if not math.isfinite(score) or not isinstance(text, str):
                raise RuntimeError(f"{context}: invalid score/text")
            validate_sid_tokens(tokenizer, tokens, context)
    if output_snapshots is not None:
        # Store only ordinary Python values after timing and validation. Keep
        # the output-release contract and the measured consumption work intact.
        output_snapshots.append(
            [
                [
                    {"token_ids": [int(token) for token in tokens], "score": score}
                    for tokens, score, _ in row
                ]
                for row in consumed
            ]
        )
    return elapsed


def run_offline(args: argparse.Namespace, prompt: str, result: dict[str, Any]) -> None:
    import yaml

    from vllm_gr.entrypoints import gr
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    runtime_source = Path(gr.__file__).resolve()
    if root := os.environ.get("V1_PERF_REPO_ROOT"):
        if not runtime_source.is_relative_to(Path(root).resolve() / "vllm_gr"):
            raise RuntimeError(f"benchmark imported the wrong runtime: {runtime_source}")
    result["runtime_source"] = str(runtime_source)
    config = yaml.safe_load((args.output_dir / "vllm_gr_config.yaml").read_text())
    result["gr_config"] = config
    engine_args = {
        "model": args.model,
        "trust_remote_code": True,
        "vllm_gr_config": config,
        "async_scheduling": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_model_len,
        "max_num_seqs": max(args.batch_sizes),
        "max_logprobs": args.beam_width,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": False,
        "seed": 0,
    }
    if args.revision:
        engine_args["revision"] = args.revision
    params = BeamSearchParams(
        beam_width=args.beam_width,
        max_tokens=args.generated_steps + 2,
        temperature=0.0,
        ignore_eos=True,
        include_stop_str_in_output=True,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
    )
    with GRLLM(**engine_args) as llm:
        tokenizer = llm.get_tokenizer()
        newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_tokens) != 1:
            raise ValueError("benchmark padding requires one token for a newline")
        prompt_token_ids = []
        for index in range(max(args.batch_sizes)):
            tokens = tokenizer.encode(f"{prompt}\nRequest {index}", add_special_tokens=False)
            tokens = [int(value) for value in tokens[-(args.input_len - 1) :]]
            tokens = [int(newline_tokens[0])] * (args.input_len - 1 - len(tokens)) + tokens
            prompt_token_ids.append(tokens)
        if args.input_len + args.generated_steps + 1 > args.max_model_len:
            raise ValueError("prompt exceeds max_model_len")
        write_json(
            args.output_dir / "prompt_token_ids.json",
            {
                "input_tokens_including_sid_begin": args.input_len,
                "prompts": prompt_token_ids,
            },
        )
        for count in args.batch_sizes:
            prompts = [{"prompt_token_ids": tokens} for tokens in prompt_token_ids[:count]]
            group_samples = []
            snapshots: list[Any] | None = [] if getattr(args, "record_outputs", False) else None
            for iteration in range(args.warmup_runs + args.measured_runs):
                elapsed = _measure_offline_batch(
                    llm,
                    prompts,
                    params,
                    tokenizer,
                    prompt_token_ids,
                    count,
                    args.beam_width,
                    **(
                        {"output_snapshots": snapshots}
                        if snapshots is not None and iteration >= args.warmup_runs
                        else {}
                    ),
                )
                if iteration >= args.warmup_runs:
                    group_samples.append(elapsed)
            case = case_result(
                "offline",
                count,
                group_samples,
                [value for value in group_samples for _ in range(count)],
                args,
            )
            case["prompt_tokens_including_sid_begin"] = [args.input_len] * count
            if snapshots is not None:
                case["outputs"] = snapshots
            result["cases"].append(case)
            write_json(args.output_dir / "offline.json", result)
            print(
                json.dumps(
                    {key: value for key, value in case.items() if key != "outputs"}, sort_keys=True
                ),
                flush=True,
            )


async def run_online(args: argparse.Namespace, result: dict[str, Any]) -> None:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    url = f"http://127.0.0.1:{args.port}/v1/completions"
    offline_result = json.loads((args.output_dir / "offline.json").read_text(encoding="utf-8"))
    if not offline_result.get("complete"):
        raise RuntimeError("offline V1 measurements are incomplete")
    offline_cases = {case["batch_size"]: case for case in offline_result["cases"]}
    prompt_data = json.loads(
        (args.output_dir / "prompt_token_ids.json").read_text(encoding="utf-8")
    )
    prompt_token_ids = prompt_data["prompts"]
    if (
        prompt_data["input_tokens_including_sid_begin"] != args.input_len
        or len(prompt_token_ids) != max(args.batch_sizes)
        or any(len(tokens) + 1 != args.input_len for tokens in prompt_token_ids)
    ):
        raise ValueError("offline and online prompt token lengths differ")

    async def one_request(session: aiohttp.ClientSession, index: int) -> float:
        payload = {
            "model": args.served_model_name or args.model,
            "prompt": prompt_token_ids[index],
            "use_beam_search": True,
            "n": args.beam_width,
            "max_tokens": args.generated_steps + 2,
            "temperature": 0.0,
            "ignore_eos": True,
            "include_stop_str_in_output": True,
            "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
        }
        start = time.perf_counter_ns()
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                body = (await response.text())[:500]
                raise RuntimeError(f"online request failed ({response.status}): {body}")
            body = await response.json()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        validate_online_response(
            body,
            index=index,
            beam_width=args.beam_width,
            prompt_tokens=len(prompt_token_ids[index]),
        )
        return elapsed

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for count in args.batch_sizes:
            group_samples = []
            request_samples = []
            for iteration in range(args.warmup_runs + args.measured_runs):
                start = time.perf_counter_ns()
                async with asyncio.TaskGroup() as group:
                    tasks = [
                        group.create_task(one_request(session, index)) for index in range(count)
                    ]
                latencies = [task.result() for task in tasks]
                group_elapsed = (time.perf_counter_ns() - start) / 1e6
                if iteration >= args.warmup_runs:
                    group_samples.append(group_elapsed)
                    request_samples.extend(latencies)
            case = case_result("online", count, group_samples, request_samples, args)
            case["prompt_tokens_including_sid_begin"] = offline_cases[count][
                "prompt_tokens_including_sid_begin"
            ]
            result["cases"].append(case)
            write_json(args.output_dir / "online.json", result)
            print(json.dumps(case, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "offline", "online"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--revision")
    parser.add_argument("--triples", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--generated-steps", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--port", type=int, default=7001)
    parser.add_argument(
        "--kv-pool-ping-pong",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gather into the next KV bank; disable for the optimization baseline.",
    )
    parser.add_argument(
        "--record-outputs",
        action="store_true",
        help="Save measured beam token IDs and scores after the timed section.",
    )
    args = parser.parse_args()
    if (
        args.beam_width <= 0
        or args.generated_steps != 3
        or args.warmup_runs < 0
        or args.measured_runs <= 0
        or not args.batch_sizes
        or any(size <= 0 for size in args.batch_sizes)
        or len(set(args.batch_sizes)) != len(args.batch_sizes)
        or args.max_model_len <= 0
        or args.input_len <= 1
        or not 0 < args.gpu_memory_utilization <= 1
        or args.request_timeout <= 0
    ):
        parser.error("invalid V1 performance geometry or timing parameters")
    if not args.triples.is_file() or not args.prompt.is_file():
        parser.error("--triples and --prompt must be existing files")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
        return
    if not (args.output_dir / "vllm_gr_config.yaml").is_file():
        raise FileNotFoundError("V1 config missing; run prepare first")
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("empty benchmark prompt")
    result: dict[str, Any] = {
        "mode": args.mode,
        "api": (
            "GRLLM.beam_search_v1" if args.mode == "offline" else "OpenAI HTTP V1 beam dispatch"
        ),
        "cases": [],
        "complete": False,
    }
    try:
        if args.mode == "offline":
            run_offline(args, prompt, result)
        else:
            asyncio.run(run_online(args, result))
    except BaseException as exc:
        result["error"] = error_details(exc)
        write_json(args.output_dir / f"{args.mode}.json", result)
        raise
    result["complete"] = True
    write_json(args.output_dir / f"{args.mode}.json", result)


if __name__ == "__main__":
    main()
