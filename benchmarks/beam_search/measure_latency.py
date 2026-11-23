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
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_sid_validator() -> ModuleType:
    root = os.environ.get("V1_PERF_REPO_ROOT")
    if root is None:
        return import_module("benchmarks.beam_search.sid_validation")

    # A/B runs this candidate driver on both checkouts. Load the selected
    # checkout's validator directly so an inherited PYTHONPATH cannot supply
    # the candidate's module when only the old filename exists in the base.
    path = Path(root) / "benchmarks/beam_search/sid_validation.py"
    if not path.exists():
        path = path.with_name("v1_sid_validation.py")
    spec = spec_from_file_location("_beam_sid_validation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SID validator from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sid_validator = _load_sid_validator()
validate_sid_text = _sid_validator.validate_sid_text
validate_sid_tokens = _sid_validator.validate_sid_tokens

# A/B executes this candidate driver against both runtimes. Resolve the
# shared statistics beside the driver because an older base can have its own
# benchmarks package without metrics.py. Runtime imports still use PYTHONPATH.
_metrics_path = Path(__file__).resolve().parents[1] / "metrics.py"
_metrics_spec = spec_from_file_location("_gr_benchmark_metrics", _metrics_path)
if _metrics_spec is None or _metrics_spec.loader is None:
    raise ImportError(f"Cannot load benchmark statistics from {_metrics_path}")
_metrics = module_from_spec(_metrics_spec)
_metrics_spec.loader.exec_module(_metrics)
linear_percentile_sorted = _metrics.linear_percentile_sorted


def distribution(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("latency samples must be positive finite values")
    ordered = sorted(values)

    return {
        "min_ms": ordered[0],
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": linear_percentile_sorted(ordered, 0.50),
        "p90_ms": linear_percentile_sorted(ordered, 0.90),
        "p95_ms": linear_percentile_sorted(ordered, 0.95),
        "p99_ms": linear_percentile_sorted(ordered, 0.99),
        "max_ms": ordered[-1],
        "std_ms": statistics.pstdev(ordered),
    }


def resolve_beam_search(llm: Any) -> Any:
    """Select the public Beam method exposed by the loaded runtime.

    This driver runs against both the pre-rename and the renamed checkout.
    Pre-rename runtimes expose ``beam_search_v1`` while ``beam_search`` is a
    retired stub; the renamed runtime exposes only ``beam_search``.
    """
    method = getattr(llm, "beam_search_v1", None)
    return method if method is not None else llm.beam_search


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> None:
    import yaml

    module_name = "tools.constraints.build_constraint_table"
    try:
        builder = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name not in {"tools.constraints", module_name}:
            raise
        # Keep this fallback until all supported CI bases have the new path and
        # replaying checkouts from before the move is no longer supported.
        builder = import_module("tools.build_constraint_table")

    artifact = args.output_dir / "constraint_table.safetensors"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_digest, tokenizer_digest, _ = builder.build_constraint_table(
        triples_path=str(args.triples),
        model=args.model,
        revision=args.revision,
        output_path=str(artifact),
        trust_remote_code=True,
    )
    config = {
        "beam_graph_enabled": True,
        "worker_decision": True,
        "beam_width": [args.beam_width] * args.generated_steps,
        "max_decode_steps": args.generated_steps,
        "constraint_mode": "table",
        "constraint_table": {
            "backend": "cuda",
            "max_top_k": args.beam_width,
            "path": str(artifact.resolve()),
            "format": "constraint_table_v1",
            "digests": {"artifact": artifact_digest, "tokenizer": tokenizer_digest},
        },
        "candidate_width": [args.beam_width] * args.generated_steps,
    }
    if args.ab_prefill_grid:
        config.update(
            prefill_graph_enabled=True,
            prefill_token_buckets=[1024, 2048, 4096],
            beam_batch_sizes=[1, 2, 4],
        )
    (args.output_dir / "gr-config.yaml").write_text(
        yaml.safe_dump({"attention-config": {"backend": "CUSTOM"}, "gr": config}, sort_keys=False),
        encoding="utf-8",
    )
    write_json(
        args.output_dir / "setup.json",
        {
            "gr": config,
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
            "graph_enabled": True,
            "prefix_caching": False,
            "chunked_prefill": False,
            "timing": "end-to-end caller time; warmup excluded; generated beam IDs/scores/text materialized offline (full/compact normalized)",
            "online_streaming": False,
            "candidate_width": [args.beam_width] * 3,
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
) -> float:
    """Consume and validate one batch, releasing its outputs before the next call."""
    import torch

    start = time.perf_counter_ns()
    # Pin compact on both revisions: the base defaults to compact and the head
    # defaults to full, whose lazy text materialization would add PR-side work.
    outputs = resolve_beam_search(llm)(prompts, params, use_tqdm=False, output_mode="compact")
    # Consume identical generated IDs, scores and decoded text on both revisions.
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
    return elapsed


def _legacy_ab_config(gr: dict[str, Any]) -> dict[str, Any]:
    """Translate the fixed A/B workload to the base revision's schema."""
    widths = gr["beam_width"]
    steps = gr["max_decode_steps"]
    if not widths or widths != [widths[0]] * steps or gr["candidate_width"] != widths:
        raise ValueError("Legacy A/B requires one shared candidate and beam width")
    table = gr["constraint_table"]
    return {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "graph_enabled": gr["beam_graph_enabled"],
            "worker_decision": gr["worker_decision"],
            "max_width": widths[0],
            "max_decode_steps": steps,
            "online_batch_sizes": gr["beam_batch_sizes"],
        },
        "constraint_table": {
            "enabled": True,
            "backend": table["backend"],
            "max_top_k": table["max_top_k"],
            "path": table["path"],
            "format": table["format"],
            "artifact_digest": table["digests"]["artifact"],
            "tokenizer_digest": table["digests"]["tokenizer"],
            "candidate_width": widths[0],
        },
    }


def run_offline(args: argparse.Namespace, prompt: str, result: dict[str, Any]) -> None:
    from vllm_gr.entrypoints import gr
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    runtime_source = Path(gr.__file__).resolve()
    if root := os.environ.get("V1_PERF_REPO_ROOT"):
        if not runtime_source.is_relative_to(Path(root).resolve() / "vllm_gr"):
            raise RuntimeError(f"benchmark imported the wrong runtime: {runtime_source}")
    result["runtime_source"] = str(runtime_source)
    gr = json.loads((args.output_dir / "setup.json").read_text(encoding="utf-8"))["gr"]
    engine_args = {
        "model": args.model,
        "trust_remote_code": True,
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
    if args.ab_legacy_config:
        engine_args["vllm_gr_config"] = _legacy_ab_config(gr)
    else:
        engine_args.update(gr=gr, attention_config={"backend": "CUSTOM"})
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
            for iteration in range(args.warmup_runs + args.measured_runs):
                elapsed = _measure_offline_batch(
                    llm, prompts, params, tokenizer, prompt_token_ids, count, args.beam_width
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
            result["cases"].append(case)
            write_json(args.output_dir / "offline.json", result)
            print(json.dumps(case, sort_keys=True), flush=True)


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
        "--ab-prefill-grid", action="store_true", help="Capture the B=1,2,4 performance workload"
    )
    parser.add_argument(
        "--ab-legacy-config",
        action="store_true",
        help="Use schema-v1 startup for an offline A/B pass",
    )
    args = parser.parse_args()
    if args.ab_legacy_config and (not args.ab_prefill_grid or args.mode == "online"):
        parser.error("--ab-legacy-config requires an offline A/B pass")
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
    if not (args.output_dir / "setup.json").is_file():
        raise FileNotFoundError("Benchmark setup missing; run prepare first")
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("empty benchmark prompt")
    result: dict[str, Any] = {
        "mode": args.mode,
        "api": ("GRLLM.beam_search" if args.mode == "offline" else "OpenAI HTTP V1 beam dispatch"),
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
