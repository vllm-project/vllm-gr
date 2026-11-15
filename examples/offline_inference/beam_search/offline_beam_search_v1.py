#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run offline SID Beam Search, optionally after thinking on the same input."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, cast

import regex as re

os.environ.setdefault("USE_TF", "0")


BEGIN_TOKEN = "<|sid_begin|>"
END_TOKEN = "<|sid_end|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Constraint triples JSON; required for Beam, unused with --enable-thinking",
    )
    parser.add_argument(
        "--constraint-backend",
        choices=("cuda", "npu"),
        default="cuda",
        help="Constraint Table device backend (CUDA or NPU)",
    )
    parser.add_argument("--prompt", type=Path, help="Complete direct-mode prompt document")
    parser.add_argument(
        "--cache-mode",
        choices=("off", "cold", "shared-prefix"),
        default="cold",
        help=(
            "Prefix-cache benchmark mode: off disables APC, cold keeps isolated salts, "
            "and shared-prefix uses one salt for a JSON Lines workload"
        ),
    )
    parser.add_argument(
        "--workload-jsonl",
        type=Path,
        help=(
            "JSON Lines workload; each row has either text or prompt_token_ids. "
            "The workload bypasses fixed-length prompt truncation and padding"
        ),
    )
    parser.add_argument(
        "--shared-cache-salt",
        default="offline-beam-v1-shared-prefix",
        help="Cache salt shared by every request in shared-prefix and off workload modes",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Run ordinary LLM thinking instead of constrained Beam SID generation",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use the same complete --prompt for thinking, then constrained SID Beam Search",
    )
    parser.add_argument(
        "--cot-max-tokens",
        type=int,
        default=1024,
        help="Thinking budget before Beam; default policy aborts if </think> is not reached",
    )
    parser.add_argument(
        "--cot-on-limit",
        choices=("error", "truncate"),
        default="error",
        help="On an exhausted CoT budget: fail, or explicitly close the span and continue",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="CoT plus final answer output budget"
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Thinking mode sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Thinking mode sampling")
    parser.add_argument("--top-k", type=int, default=20, help="Thinking mode sampling")
    parser.add_argument("--seed", type=int, default=0, help="Thinking mode random seed")
    parser.add_argument(
        "--execution",
        choices=("eager", "graph"),
        default="eager",
        help="NPU Decode execution mode; CUDA keeps its existing graph behavior",
    )
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument(
        "--generated-steps",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Effective SID length; constrained Beam requires 3 (unused with --enable-thinking)",
    )
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument(
        "--dynamic-beam-widths",
        help="NPU per-stage retained widths, e.g. 128,64,32 (beam-width remains capacity)",
    )
    parser.add_argument(
        "--result-width",
        type=int,
        help="NPU final sequence count; defaults to the last stage width",
    )
    parser.add_argument(
        "--candidate-width",
        type=int,
        help="Candidates per beam row (1..256); defaults to --beam-width",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="KV cache block size override for reproducible cache-reuse comparisons "
        "(for example Ascend block_size=32)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--print-beams", type=int, default=5)
    parser.add_argument("--repeat", type=int, help="Benchmark this many batches without a profiler")
    parser.add_argument(
        "--warmup-runs", type=int, default=3, help="Untimed warmup batches (requires --repeat)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Save Beam benchmark (--repeat required), or thinking output and timings",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Save Worker profiler traces; exclude engine startup and output printing",
    )
    parser.add_argument(
        "--profile-warmup-runs",
        type=int,
        default=1,
        help="Unprofiled warmup batches before collection (requires --profile-dir)",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=1,
        help="Complete batches to profile; print only the last result (requires --profile-dir)",
    )
    args = parser.parse_args()
    if args.cot and args.enable_thinking:
        parser.error("--cot and --enable-thinking are separate modes; choose one")
    if args.cot and args.cot_max_tokens <= 0:
        parser.error("--cot-max-tokens must be positive")
    if args.workload_jsonl is not None:
        if args.enable_thinking or args.cot:
            parser.error("--workload-jsonl only supports direct constrained SID Beam mode")
        if args.profile_dir is not None:
            parser.error("--workload-jsonl currently requires --repeat, not --profile-dir")
        if args.repeat is None:
            parser.error("--workload-jsonl requires --repeat")
        if args.prompt is not None:
            parser.error("--prompt and --workload-jsonl are mutually exclusive")
    elif args.prompt is None:
        parser.error("--prompt is required unless --workload-jsonl is provided")
    if args.cache_mode == "shared-prefix" and args.workload_jsonl is None:
        parser.error("--cache-mode shared-prefix requires --workload-jsonl")
    if not args.shared_cache_salt:
        parser.error("--shared-cache-salt must be nonempty")
    if not args.enable_thinking and args.catalog is None:
        parser.error("--catalog is required unless --enable-thinking is set")
    if not args.enable_thinking and args.generated_steps != 3:
        parser.error(
            "constrained Beam requires --generated-steps 3 to match constraint_table_v1 depth"
        )
    if args.enable_thinking or args.cot:
        if args.enable_thinking and (
            args.dynamic_beam_widths is not None or args.result_width is not None
        ):
            parser.error("--enable-thinking cannot be combined with dynamic Beam options")
        if args.max_tokens <= 0 or args.max_model_len <= 0:
            parser.error("--max-tokens and --max-model-len must be positive")
        if not 0.0 <= args.temperature <= 2.0 or not 0.0 < args.top_p <= 1.0:
            parser.error("--temperature must be in [0, 2] and --top-p in (0, 1]")
        if args.top_k < -1:
            parser.error("--top-k must be -1, 0, or positive")
    if args.dynamic_beam_widths is not None:
        try:
            widths = [int(value.strip()) for value in args.dynamic_beam_widths.split(",")]
        except ValueError:
            parser.error("--dynamic-beam-widths must contain comma-separated integers")
        if len(widths) != args.generated_steps or any(
            not 1 <= width <= args.beam_width for width in widths
        ):
            parser.error(
                "--dynamic-beam-widths needs one width per generated step, each in [1, beam-width]"
            )
        args.dynamic_beam_widths = widths
    if args.dynamic_beam_widths is not None or args.result_width is not None:
        if args.constraint_backend != "npu":
            parser.error("dynamic-beam-widths and result-width currently require the NPU backend")
        final_width = (
            args.dynamic_beam_widths[-1]
            if args.dynamic_beam_widths is not None
            else args.beam_width
        )
        if args.result_width is not None and not 1 <= args.result_width <= final_width:
            parser.error("--result-width must be in [1, last-stage-width]")
    if args.profile_warmup_runs < 0 or args.profile_runs <= 0:
        parser.error("--profile-warmup-runs must be nonnegative and --profile-runs positive")
    if args.warmup_runs < 0 or (args.repeat is not None and args.repeat <= 0):
        parser.error("--warmup-runs must be nonnegative and --repeat positive")
    if args.repeat is not None and args.profile_dir is not None:
        parser.error(
            "--repeat cannot be combined with --profile-dir; profile overhead affects latency"
        )
    if (
        args.output_json is not None
        and args.repeat is None
        and not (args.enable_thinking or args.cot)
    ):
        parser.error("--output-json requires --repeat")
    return args


def profiler_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile_dir is None:
        return {}
    profile_dir = args.profile_dir.resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    return {
        "profiler_config": {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_use_gzip": False,
            "torch_profiler_dump_cuda_time_total": False,
            "ignore_frontend": True,
        }
    }


def isolate_prompts(prompts: list[Any], suffix: str) -> list[Any]:
    # Keep prefix caching enabled, but force full Prefill for every row/run.
    return [
        dict(
            prompt,
            prompt_token_ids=list(prompt["prompt_token_ids"]),
            cache_salt=f"{prompt['cache_salt']}-{suffix}",
        )
        for prompt in prompts
    ]


@dataclass(frozen=True)
class PrefixWorkload:
    """Final token IDs for a shared-prefix benchmark workload."""

    prompt_token_ids: tuple[tuple[int, ...], ...]
    common_prefix_tokens: int

    @property
    def max_prompt_tokens(self) -> int:
        return max(len(ids) for ids in self.prompt_token_ids)

    @property
    def prompt_lengths(self) -> tuple[int, ...]:
        return tuple(len(ids) for ids in self.prompt_token_ids)


def load_prefix_workload(path: Path, tokenizer: Any) -> PrefixWorkload:
    """Load complete final prompts without truncation, padding, or chat templating."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: workload row is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: workload row must be a JSON object")
        keys = set(row)
        if keys == {"text"}:
            if not isinstance(row["text"], str):
                raise ValueError(f"{path}:{line_number}: text must be a string")
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            if not isinstance(ids, list) or not all(
                isinstance(token, int) and not isinstance(token, bool) for token in ids
            ):
                raise ValueError(f"{path}:{line_number}: text did not tokenize to an int array")
        elif keys == {"prompt_token_ids"}:
            raw_ids = row["prompt_token_ids"]
            if not isinstance(raw_ids, list) or any(
                not isinstance(token, int) or isinstance(token, bool) for token in raw_ids
            ):
                raise ValueError(
                    f"{path}:{line_number}: prompt_token_ids must be a nonempty int array"
                )
            ids = raw_ids
        else:
            raise ValueError(
                f"{path}:{line_number}: each row must contain exactly one of "
                "text or prompt_token_ids"
            )
        if not ids:
            raise ValueError(f"{path}:{line_number}: prompt is empty")
        rows.append({"prompt_token_ids": [int(token) for token in ids]})

    if not rows:
        raise ValueError(f"{path}: workload is empty")

    prompt_ids = tuple(tuple(row["prompt_token_ids"]) for row in rows)
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError(f"{path}: workload contains duplicate complete prompts")

    return PrefixWorkload(prompt_ids, common_prefix_length(prompt_ids))


def common_prefix_length(prompt_ids: tuple[tuple[int, ...], ...]) -> int:
    common_prefix = list(prompt_ids[0])
    for ids in prompt_ids[1:]:
        prefix_len = 0
        for expected, actual in zip(common_prefix, ids):
            if expected != actual:
                break
            prefix_len += 1
        del common_prefix[prefix_len:]
    return len(common_prefix)


def restrict_workload(workload: PrefixWorkload, required: int) -> PrefixWorkload:
    """Restrict statistics and requests to the rows this run consumes."""
    if required >= len(workload.prompt_token_ids):
        return workload
    selected = workload.prompt_token_ids[:required]
    return PrefixWorkload(selected, common_prefix_length(selected))


def prompt_fits_context(effective_input_len: int, generated_steps: int, max_model_len: int) -> bool:
    """Accept an exact fit; ``effective_input_len`` already includes SID begin."""
    return effective_input_len + generated_steps <= max_model_len


def make_workload_requests(
    workload: PrefixWorkload,
    args: argparse.Namespace,
    tokens_input: Callable[..., Any],
) -> list[Any]:
    requests = []
    for index, ids in enumerate(workload.prompt_token_ids):
        if args.cache_mode == "cold":
            salt = f"offline-beam-v1-workload-{index}"
        else:
            salt = args.shared_cache_salt
        requests.append(tokens_input(list(ids), cache_salt=salt))
    return requests


def workload_report(workload: PrefixWorkload, path: Path, required: int) -> dict[str, Any]:
    lengths = workload.prompt_lengths
    return {
        "path": str(path.resolve()),
        "requests": len(workload.prompt_token_ids),
        "required_requests": required,
        "common_prefix_tokens": workload.common_prefix_tokens,
        "min_prompt_tokens": min(lengths),
        "max_prompt_tokens": max(lengths),
        "seed_and_measurement_strategy": (
            "one non-timed seed batch, then every warmup and timed batch"
        ),
    }


def observed_prefix_cache_stats(llm: Any) -> dict[str, Any] | None:
    """Best-effort engine prefix-cache counters for benchmark evidence."""
    try:
        observed = llm.collective_rpc("get_prefix_cache_hit_rate")[0]
    except Exception:
        return None
    if isinstance(observed, dict):
        return dict(observed)
    if isinstance(observed, (int, float)):
        return {"hit_rate": float(observed)}
    return None


def observed_block_size(llm: Any) -> int | None:
    """Resolve the effective KV cache block size when the engine exposes it."""
    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    for config in (vllm_config, engine, getattr(llm, "llm_config", None)):
        cache_config = getattr(config, "cache_config", None) or config
        block_size = getattr(cache_config, "block_size", None)
        if isinstance(block_size, int):
            return block_size
    return None


def beam_row_evidence(run: int, row: int, request: Any, output: Any) -> dict[str, Any]:
    """Per-row SID order and scores collected outside the timing window."""
    return {
        "run": run,
        "row": row,
        "cache_salt": request["cache_salt"],
        "sid_order": [
            [int(token) for token in sequence.token_ids] for sequence in output.sequences
        ],
        "scores": [float(sequence.cumulative_logprob) for sequence in output.sequences],
    }


def next_benchmark_requests(
    args: argparse.Namespace,
    requests: list[Any],
    workload_requests: list[Any] | None,
    cursor: int,
    suffix: str,
) -> tuple[list[Any], int]:
    if workload_requests is None:
        selected = requests
    else:
        end = cursor + args.batch_size
        selected = workload_requests[cursor:end]
        if len(selected) != args.batch_size:
            raise RuntimeError("prefix workload was exhausted before benchmark completion")
        cursor = end
    if args.cache_mode == "cold":
        return isolate_prompts(selected, suffix), cursor
    copied = [
        dict(request, prompt_token_ids=list(request["prompt_token_ids"]))
        for request in selected
    ]
    return copied, cursor


def execute_beam_batch(
    llm: Any, args: argparse.Namespace, prompts: list[Any], params: Any
) -> list[Any]:
    if not getattr(args, "cot", False):
        return cast(list[Any], llm.beam_search_v1(prompts, params, output_mode="full"))

    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    close_id = single_thinking_token(tokenizer, "</think>")
    started = perf_counter()
    completions = llm.generate(
        prompts,
        SamplingParams(
            max_tokens=args.cot_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            stop_token_ids=[close_id],
            detokenize=False,
        ),
        use_tqdm=False,
    )
    thinking_ms = (perf_counter() - started) * 1000.0
    if len(completions) != len(prompts):
        raise RuntimeError("CoT returned a different request count; Beam was not started")
    separator = list(tokenizer.encode("\n\n", add_special_tokens=False))
    beam_prompts = []
    records = []
    for index, (prompt, result) in enumerate(zip(prompts, completions, strict=True)):
        # LLM.generate returns results in input order. Verify their source IDs
        # before associating each continuation with its SID request.
        if result.prompt_token_ids != prompt["prompt_token_ids"]:
            raise RuntimeError(f"request {index}: CoT prompt/order mismatch; Beam was not started")
        if len(result.outputs) != 1:
            raise RuntimeError("CoT requires one continuation per request")
        completion = result.outputs[0]
        ids = list(completion.token_ids)
        forced_close = False
        if (
            ids
            and ids[-1] != close_id
            and completion.finish_reason == "length"
            and len(ids) == args.cot_max_tokens
            and getattr(args, "cot_on_limit", "error") == "truncate"
        ):
            ids.append(close_id)
            forced_close = True
        if not ids or ids[-1] != close_id:
            raise RuntimeError(
                f"request {index}: CoT did not reach </think> "
                f"(finish_reason={completion.finish_reason}); increase --cot-max-tokens "
                "or check the model/prompt. Beam was not started."
            )
        combined = [*prompt["prompt_token_ids"], *ids, *separator]
        if len(combined) + args.generated_steps + 2 > args.max_model_len:
            raise ValueError("CoT plus SID generation exceeds --max-model-len")
        # Keep the initial token-handoff feature independent of partial-prefix
        # reuse between ordinary Decode and the GR Beam runtime.
        beam_prompts.append(
            dict(
                prompt,
                prompt_token_ids=combined,
                cache_salt=f"{prompt.get('cache_salt', 'offline-cot')}-sid-row-{index}",
            )
        )
        records.append(
            {
                "request": index,
                "thinking_token_ids": ids,
                "generated_thinking_token_ids": list(completion.token_ids),
                "thinking_text": tokenizer.decode(ids, skip_special_tokens=False),
                "thinking_tokens": len(ids),
                "generated_thinking_tokens": len(completion.token_ids),
                "thinking_truncated": forced_close,
                "forced_close": forced_close,
                "beam_prompt_tokens": len(combined),
            }
        )
    started = perf_counter()
    outputs: list[Any] = llm.beam_search_v1(beam_prompts, params, output_mode="full")
    if len(outputs) != len(prompts):
        raise RuntimeError("Beam returned a different request count after CoT")
    args.cot_last_report = {
        "thinking_ms": thinking_ms,
        "beam_ms": (perf_counter() - started) * 1000.0,
        "requests": records,
        "kv_handoff": "token IDs only; separate cache salts for thinking and SID Prefill",
    }
    return outputs


def benchmark_beam_search(
    llm: Any,
    args: argparse.Namespace,
    prompts: list[Any],
    params: Any,
    workload_requests: list[Any] | None = None,
    workload: PrefixWorkload | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    cursor = 0
    if workload_requests is not None:
        seed, cursor = next_benchmark_requests(args, prompts, workload_requests, cursor, "seed")
        execute_beam_batch(llm, args, seed, params)

    for iteration in range(args.warmup_runs):
        warmup, cursor = next_benchmark_requests(
            args, prompts, workload_requests, cursor, f"warmup-{iteration}"
        )
        execute_beam_batch(llm, args, warmup, params)

    latencies_ms: list[float] = []
    returned_beams: list[int] = []
    outputs: list[Any] = []
    measured_rows: list[dict[str, Any]] = []
    prefix_cache_before = (
        observed_prefix_cache_stats(llm) if workload_requests is not None else None
    )
    for iteration in range(args.repeat):
        row_start = cursor
        measured, cursor = next_benchmark_requests(
            args, prompts, workload_requests, cursor, f"benchmark-{iteration}"
        )
        started = perf_counter()
        outputs = execute_beam_batch(llm, args, measured, params)
        latencies_ms.append((perf_counter() - started) * 1000.0)
        if len(outputs) != len(prompts):
            raise RuntimeError(f"expected {len(prompts)} request outputs, got {len(outputs)}")
        # Candidate counting and cache/output evidence stay outside timing.
        returned_beams.append(sum(len(output.sequences) for output in outputs))
        if workload_requests is not None:
            measured_rows.extend(
                beam_row_evidence(iteration, row_start + index, request, output)
                for index, (request, output) in enumerate(zip(measured, outputs, strict=True))
            )
    prefix_cache_after = (
        observed_prefix_cache_stats(llm) if workload_requests is not None else None
    )

    ordered = sorted(latencies_ms)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    total_seconds = sum(latencies_ms) / 1000.0
    if workload_requests is None:
        timing_scope = (
            "beam_search_v1 API call until output objects return; excludes engine startup, "
            "warmup, pre-call prompt preparation and post-call result inspection/printing"
        )
    else:
        timing_scope = (
            "beam_search_v1 API call until output objects return; excludes engine startup, "
            "seed, warmup, pre-call prompt preparation and post-call result inspection/printing"
        )
    report: dict[str, Any] = {
        "model": str(args.model.resolve()),
        "catalog": str(args.catalog.resolve()),
        "prompt": str(args.prompt.resolve()) if args.prompt is not None else None,
        "constraint_backend": args.constraint_backend,
        "execution": args.execution,
        "batch_size": args.batch_size,
        "beam_width": args.beam_width,
        "input_len": None if workload_requests is not None else args.input_len,
        "generated_steps": args.generated_steps,
        "length_penalty": args.length_penalty,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "warmup_runs": args.warmup_runs,
        "repeat": args.repeat,
        "prefix_caching": (
            f"{args.cache_mode}; isolated salt per row and run (cold Prefill)"
            if workload_requests is None
            else (
                f"{args.cache_mode}; one non-timed seed batch, then unique workload rows; "
                "cold mode still isolates salts per row and run"
            )
        ),
        "timing_scope": timing_scope,
        "batch_latency_ms": {
            "mean": mean(latencies_ms),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "min": ordered[0],
            "max": ordered[-1],
        },
        "total_timed_seconds": total_seconds,
        "requests_per_second": len(prompts) * len(latencies_ms) / total_seconds,
        "returned_beams_per_second": sum(returned_beams) / total_seconds,
        "runs": [
            {"run": index + 1, "batch_latency_ms": latency, "returned_beams": beams}
            for index, (latency, beams) in enumerate(zip(latencies_ms, returned_beams, strict=True))
        ],
    }
    if getattr(args, "dynamic_beam_widths", None) is not None:
        report["dynamic_beam_widths"] = list(args.dynamic_beam_widths)
    if getattr(args, "result_width", None) is not None:
        report["result_width"] = args.result_width
    if workload is not None:
        assert args.workload_jsonl is not None
        required = args.batch_size * (1 + args.warmup_runs + args.repeat)
        report["workload"] = workload_report(workload, args.workload_jsonl, required)
        report["workload"]["block_size_requested"] = getattr(args, "block_size", None)
        report["workload"]["block_size_effective"] = observed_block_size(llm)
        if workload_requests is not None:
            report["workload"]["prefix_cache_stats"] = {
                "before_seed": prefix_cache_before,
                "after_measured": prefix_cache_after,
                "source": "collective_rpc:get_prefix_cache_hit_rate; null means unavailable",
            }
            report["workload"]["measured_rows"] = measured_rows
    if getattr(args, "cot", False):
        report.update(
            mode="cot_beam",
            input_len=None,
            cot_input_tokens=len(prompts[0]["prompt_token_ids"]),
            cot_max_tokens=args.cot_max_tokens,
            cot_on_limit=args.cot_on_limit,
            timing_scope="CoT generation, token handoff and beam_search_v1 until outputs return; "
            "excludes startup, warmup and printing",
        )
    return outputs, report


def run_beam_search(
    llm: Any, args: argparse.Namespace, prompts: list[Any], params: Any
) -> list[Any]:
    if args.profile_dir is None:
        outputs: list[Any] = execute_beam_batch(llm, args, prompts, params)
        return outputs

    def run_once(iteration: int) -> list[Any]:
        isolated = isolate_prompts(prompts, f"profile-{iteration}")
        result: list[Any] = execute_beam_batch(llm, args, isolated, params)
        return result

    for iteration in range(args.profile_warmup_runs):
        run_once(iteration)
    llm.start_profile(f"offline_beam_v1_{args.execution}_b{args.batch_size}_w{args.beam_width}")
    try:
        for iteration in range(
            args.profile_warmup_runs, args.profile_warmup_runs + args.profile_runs
        ):
            outputs = run_once(iteration)
    finally:
        llm.stop_profile()
    return outputs


def load_triples(path: Path) -> list[list[str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("triples") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} is not a nonempty Constraint Table triples document")
    triples = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not all(key in row for key in ("a", "b", "c")):
            raise ValueError(f"invalid triples row {index}: {row!r}")
        triples.append([str(row["a"]), str(row["b"]), str(row["c"])])
    return sorted(triples)


def build_artifact(
    model: Path,
    catalog: Path,
    artifact_path: Path,
) -> tuple[str, str]:
    from transformers import AutoConfig, AutoTokenizer

    from vllm_gr.constraints.artifact import save_constraint_table_artifact
    from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
    from vllm_gr.constraints.json_compiler import compile_relations_json
    from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    tokenizer_digest = compute_tokenizer_digest(tokenizer)
    table = ConstraintTableArtifactCompiler(
        tokenizer,
        int(model_config.vocab_size),
        tokenizer_digest=tokenizer_digest,
    ).compile(compile_relations_json(load_triples(catalog)))
    artifact_digest = save_constraint_table_artifact(table, artifact_path)
    return artifact_digest, tokenizer_digest


def make_prompt(tokenizer: Any, path: Path, input_len: int) -> list[int]:
    # input_len includes the begin token appended by beam_search_v1.
    submitted_len = input_len - 1
    if submitted_len <= 0:
        raise ValueError("--input-len must be at least 2")
    source = tokenizer.encode(path.read_text(encoding="utf-8"), add_special_tokens=False)
    if len(source) >= submitted_len:
        return [int(token) for token in source[-submitted_len:]]
    newline = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline) != 1:
        raise RuntimeError(f"newline must map to exactly one token, got {newline}")
    return [int(newline[0])] * (submitted_len - len(source)) + [int(token) for token in source]


def _read_thinking_input(path: Path) -> str:
    """Read plain text or unambiguously encoded ChatML, without guessing escapes."""
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith('"'):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            if re.match(r'^"\s*<\|im_start\|>', text.lstrip()):
                raise ValueError("JSON-encoded ChatML must be a complete JSON string") from exc
        else:
            if isinstance(decoded, str) and decoded.lstrip().startswith("<|im_start|>"):
                text = decoded
    if (
        text.lstrip().startswith("<|im_start|>")
        and "\\n" in text
        and "\n" not in text.rstrip("\r\n")
    ):
        raise ValueError(
            "ambiguous single-line ChatML with literal \\n; use real newlines or a "
            "JSON string containing ChatML, with body backslashes escaped by JSON"
        )
    return text


def _template_thinking_prompt(tokenizer: Any, text: str) -> list[int]:
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        return_dict=False,
        enable_thinking=True,
        add_generation_prompt=True,
    )
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise TypeError("chat template must return a flat list of token IDs")
    if not tokens:
        raise ValueError("chat template returned an empty prompt")
    return tokens


def single_thinking_token(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != text:
        raise ValueError(f"CoT requires a dedicated {text} token in the model tokenizer")
    return int(ids[0])


def _reopen_empty_thinking_span(tokenizer: Any, ids: list[int]) -> list[int]:
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    if not re.search(r"<think>\s*</think>\s*$", rendered):
        return ids
    # Keep every earlier ID: an empty template span may be removed only when
    # its opening tag is a real token boundary, not text inside another token.
    open_id = single_thinking_token(tokenizer, "<think>")
    try:
        open_index = len(ids) - 1 - ids[::-1].index(open_id)
    except ValueError as exc:
        raise ValueError("chat template did not preserve the <think> special token") from exc
    if not re.fullmatch(
        r"<think>\s*</think>\s*",
        tokenizer.decode(ids[open_index:], skip_special_tokens=False),
    ):
        raise ValueError("chat template did not preserve the terminal <think> token boundary")
    return [*ids[: open_index + 1], *tokenizer.encode("\n", add_special_tokens=False)]


def _serialized_thinking_prompt(tokenizer: Any, text: str) -> list[int]:
    single_thinking_token(tokenizer, "<think>")
    single_thinking_token(tokenizer, "</think>")
    user_end = text.rfind("<|im_end|>")
    prefix, suffix = text[:user_end], text[user_end:]
    if user_end < 0 or not re.fullmatch(
        r"<\|im_end\|>\s*<\|im_start\|>assistant\s*(?:<think>\s*(?:</think>\s*)?)?",
        suffix,
    ):
        raise ValueError("CoT expects serialized chat ending in an empty assistant/think prefix")
    prefix = re.sub(r"/no_think(?=\s*$)", "/think", prefix)
    text = prefix + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    return list(tokenizer.encode(text, add_special_tokens=False))


def make_thinking_prompt(tokenizer: Any, path: Path) -> list[int]:
    # Preserve the complete question and template; Beam's fixed-length prompt
    # padding/truncation and SID boundary tokens do not apply to normal chat.
    text = _read_thinking_input(path)
    if text.lstrip().startswith("<|im_start|>"):
        return _serialized_thinking_prompt(tokenizer, text)
    return _reopen_empty_thinking_span(tokenizer, _template_thinking_prompt(tokenizer, text))


def make_cot_prompt(tokenizer: Any, path: Path) -> list[int]:
    """Preserve the full input; reopen only the terminal empty thinking span."""
    single_thinking_token(tokenizer, "<think>")
    single_thinking_token(tokenizer, "</think>")
    ids = make_thinking_prompt(tokenizer, path)
    # Some OneRec templates accept enable_thinking but leave the tag to the model.
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    if re.search(r"</think>\s*$", rendered):
        raise ValueError("CoT cannot replace a nonempty thinking span from the chat template")
    if not re.search(r"<think>\s*$", rendered):
        ids.extend(tokenizer.encode("<think>\n", add_special_tokens=False))
    if not ids:
        raise ValueError("CoT prompt is empty")
    return ids


def run_thinking(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import tokens_input

    from vllm_gr.entrypoints.gr import GRLLM

    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), trust_remote_code=True)
    prompt_ids = make_thinking_prompt(tokenizer, args.prompt.resolve())
    if len(prompt_ids) + args.max_tokens > args.max_model_len:
        raise ValueError(
            f"thinking prompt ({len(prompt_ids)}) + --max-tokens ({args.max_tokens}) "
            f"exceeds --max-model-len ({args.max_model_len}); the prompt was not truncated"
        )
    sampling = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
    }
    params = SamplingParams(**sampling)
    prompts = [tokens_input(list(prompt_ids)) for _ in range(args.batch_size)]
    config = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "constrained_generation": False,
        "constraint_table": {"enabled": False, "backend": args.constraint_backend, "max_top_k": 1},
    }
    device_kwargs = {"block_size": 128} if args.constraint_backend == "npu" else {}
    print(
        f"mode=thinking batch_size={args.batch_size} input_tokens={len(prompt_ids)} "
        f"max_tokens={args.max_tokens}; ordinary LLM generation, constraints disabled"
    )
    with GRLLM(
        model=str(args.model.resolve()),
        trust_remote_code=True,
        seed=args.seed,
        vllm_gr_config=config,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(args.max_model_len, args.batch_size * len(prompt_ids)),
        max_num_seqs=max(4, args.batch_size),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.execution == "eager",
        **device_kwargs,
        **profiler_engine_kwargs(args),
    ) as llm:

        def generate() -> list[Any]:
            outputs: list[Any] = llm.generate(prompts, params, use_tqdm=False)
            if len(outputs) != args.batch_size:
                raise RuntimeError(f"expected {args.batch_size} outputs, got {len(outputs)}")
            return outputs

        latencies_ms: list[float] = []
        generated_tokens: list[int] = []
        if args.repeat is not None:
            for _ in range(args.warmup_runs):
                generate()
            for _ in range(args.repeat):
                started = perf_counter()
                outputs = generate()
                latencies_ms.append((perf_counter() - started) * 1000.0)
                generated_tokens.append(
                    sum(
                        len(candidate.token_ids)
                        for output in outputs
                        for candidate in output.outputs
                    )
                )
        elif args.profile_dir is not None:
            for _ in range(args.profile_warmup_runs):
                generate()
            llm.start_profile(f"offline_thinking_{args.execution}_b{args.batch_size}")
            try:
                for _ in range(args.profile_runs):
                    outputs = generate()
            finally:
                llm.stop_profile()
        else:
            outputs = generate()

        results = []
        for request_index, output in enumerate(outputs):
            for candidate in output.outputs:
                result = {
                    "request": request_index,
                    "text": candidate.text,
                    "token_ids": list(candidate.token_ids),
                    "generated_tokens": len(candidate.token_ids),
                    "finish_reason": candidate.finish_reason,
                    "stop_reason": candidate.stop_reason,
                }
                results.append(result)
                print(
                    f"request={request_index} generated_tokens={len(candidate.token_ids)} "
                    f"finish_reason={candidate.finish_reason}\n{candidate.text}"
                )

    report = {
        "mode": "thinking",
        "model": str(args.model.resolve()),
        "prompt": str(args.prompt.resolve()),
        "execution": args.execution,
        "batch_size": args.batch_size,
        "input_tokens": len(prompt_ids),
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "sampling": sampling,
        "results": results,
    }
    if latencies_ms:
        benchmark = {
            "timing_scope": "generate API until output objects return; excludes startup and warmup",
            "prefix_caching": False,
            "repeat": args.repeat,
            "warmup_runs": args.warmup_runs,
            "mean_batch_latency_ms": mean(latencies_ms),
            "runs": [
                {"batch_latency_ms": latency, "generated_tokens": count}
                for latency, count in zip(latencies_ms, generated_tokens, strict=True)
            ],
        }
        report["benchmark"] = benchmark
        print(json.dumps({"thinking_benchmark": benchmark}, ensure_ascii=False))
    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"thinking_json={output_path}")


def run(args: argparse.Namespace, artifact_path: Path) -> None:
    from vllm.inputs import tokens_input

    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    is_npu = args.constraint_backend == "npu"
    # Candidate width K is the per-row top-k; None means it follows the beam
    # width (the pre-decoupling behaviour).
    candidate_width = args.candidate_width if args.candidate_width is not None else args.beam_width
    cot_prompt_ids = None
    workload: PrefixWorkload | None = None
    effective_input_len = args.input_len
    if args.workload_jsonl is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), trust_remote_code=True)
        workload = load_prefix_workload(args.workload_jsonl.resolve(), tokenizer)
        required_requests = args.batch_size * (1 + args.warmup_runs + args.repeat)
        if len(workload.prompt_token_ids) < required_requests:
            raise ValueError(
                f"--workload-jsonl needs at least {required_requests} unique prompts for "
                f"one seed batch, {args.warmup_runs} warmup batches, and {args.repeat} "
                "timed batches"
            )
        workload = restrict_workload(workload, required_requests)
        effective_input_len = workload.max_prompt_tokens + 1
        if not prompt_fits_context(
            effective_input_len, args.generated_steps, args.max_model_len
        ):
            raise ValueError("workload prompt and generated tokens exceed --max-model-len")
    prefill_capacity = args.batch_size * (effective_input_len + 1)
    if getattr(args, "cot", False):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), trust_remote_code=True)
        cot_prompt_ids = make_cot_prompt(tokenizer, args.prompt.resolve())
        separator_len = len(tokenizer.encode("\n\n", add_special_tokens=False))
        forced_close_reserve = int(args.cot_on_limit == "truncate")
        longest_prompt = (
            len(cot_prompt_ids) + args.cot_max_tokens + forced_close_reserve + separator_len + 1
        )
        if longest_prompt + args.generated_steps + 1 > args.max_model_len:
            raise ValueError(
                "full prompt plus --cot-max-tokens and SID budget exceeds --max-model-len"
            )
        prefill_capacity = args.batch_size * longest_prompt

    artifact_digest, tokenizer_digest = build_artifact(
        args.model.resolve(),
        args.catalog.resolve(),
        artifact_path,
    )
    beam_config: dict[str, object] = {
        "execution_mode": "v1",
        "graph_enabled": not is_npu or args.execution == "graph",
        "graph_batch_size": args.batch_size,
        "worker_decision": True,
        "max_width": args.beam_width,
        "max_decode_steps": args.generated_steps,
    }
    vllm_gr_config: dict[str, Any] = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": beam_config,
        "constraint_table": {
            "enabled": True,
            "backend": args.constraint_backend,
            "max_top_k": candidate_width,
            "path": str(artifact_path),
            "format": "constraint_table_v1",
            "artifact_digest": artifact_digest,
            "tokenizer_digest": tokenizer_digest,
        },
    }
    dynamic_widths = getattr(args, "dynamic_beam_widths", None)
    result_width = getattr(args, "result_width", None)
    if dynamic_widths is not None and len(dynamic_widths) > 1:
        beam_config["graph_widths"] = sorted(set(dynamic_widths[:-1]))
    if args.candidate_width is not None:
        # Omitted means K == beam_width; state it explicitly when decoupled.
        vllm_gr_config["constraint_table"]["candidate_width"] = args.candidate_width
    params = BeamSearchParams(
        beam_width=args.beam_width,
        dynamic_beam_widths=dynamic_widths,
        result_width=result_width,
        # Public max_tokens includes the begin and end boundary positions.
        max_tokens=args.generated_steps + 2,
        temperature=0.0,
        ignore_eos=True,
        include_stop_str_in_output=True,
        begin_token=BEGIN_TOKEN,
        end_token=END_TOKEN,
        length_penalty=args.length_penalty,
    )

    with GRLLM(
        model=str(args.model.resolve()),
        trust_remote_code=True,
        vllm_gr_config=vllm_gr_config,
        async_scheduling=True,
        enable_prefix_caching=args.cache_mode != "off",
        enable_chunked_prefill=False,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        # The V1 input processor appends the begin token to every prompt. Keep one
        # additional token of capacity to match the pre-workload headroom.
        max_num_batched_tokens=max(args.max_model_len, prefill_capacity),
        max_num_seqs=max(args.beam_width, args.batch_size),
        max_logprobs=candidate_width,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=is_npu and args.execution == "eager",
        **profiler_engine_kwargs(args),
    ) as llm:
        tokenizer = llm.get_tokenizer()
        workload_requests = None
        if workload is not None:
            workload_requests = make_workload_requests(workload, args, tokens_input)
            request_prompts = workload_requests[: args.batch_size]
            print(
                f"mode=prefix_cache_{args.cache_mode} workload_requests="
                f"{len(workload.prompt_token_ids)} common_prefix_tokens="
                f"{workload.common_prefix_tokens} max_prompt_tokens="
                f"{workload.max_prompt_tokens}"
            )
        else:
            if cot_prompt_ids is not None:
                prompt_token_ids = cot_prompt_ids
                print(
                    f"mode=cot_beam input_tokens={len(prompt_token_ids)}; "
                    "--input-len is not used"
                )
            else:
                assert args.prompt is not None
                prompt_token_ids = make_prompt(tokenizer, args.prompt.resolve(), args.input_len)
            # Keep prefix caching enabled, but prevent duplicate batch rows from
            # sharing the same prompt block before their atomic Prefill completes.
            request_prompts = [
                tokens_input(
                    list(prompt_token_ids),
                    cache_salt=f"offline-beam-v1-row-{request_index}",
                )
                for request_index in range(args.batch_size)
            ]
        benchmark = None
        if args.repeat is not None:
            outputs, benchmark = benchmark_beam_search(
                llm,
                args,
                request_prompts,
                params,
                workload_requests=workload_requests,
                workload=workload,
            )
        else:
            outputs = run_beam_search(llm, args, request_prompts, params)
        if getattr(args, "cot", False):
            print(json.dumps({"cot_beam": args.cot_last_report}, ensure_ascii=False))
            if any(row["thinking_truncated"] for row in args.cot_last_report["requests"]):
                print(
                    "CoT budget exhausted: explicitly truncated (--cot-on-limit=truncate); "
                    "this is not a naturally completed thinking span"
                )
        if is_npu and args.execution == "graph":
            stats = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
            print(
                json.dumps(
                    {"npu_beam_graph": stats},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            decode_widths = dynamic_widths[:-1] if dynamic_widths is not None else [args.beam_width]
            expected_shapes = {(args.batch_size, width) for width in decode_widths}
            captured_shapes = {tuple(shape) for shape in stats.get("captured_shapes", ())}
            # N=1 finishes in Prefill and must not require a Decode graph.
            if args.generated_steps > 1 and (
                not expected_shapes.issubset(captured_shapes) or not stats["full_dispatches"]
            ):
                raise RuntimeError("NPU Beam graph capture/replay was not observed")
            if benchmark is not None:
                benchmark["npu_beam_graph"] = stats

        if benchmark is not None:
            print(json.dumps({"beam_benchmark": benchmark}, ensure_ascii=False, sort_keys=True))
            if args.output_json is not None:
                output_path = args.output_json.resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(f"benchmark_json={output_path}")

        for request_index, output in enumerate(outputs):
            sequences = output.sequences
            print(f"request={request_index} returned_beams={len(sequences)}")
            for rank, sequence in enumerate(sequences[: args.print_beams]):
                token_ids = [int(token) for token in sequence.tokens]
                sid_token_ids = token_ids[-(args.generated_steps + 2) :]
                print(
                    json.dumps(
                        {
                            "request": request_index,
                            "rank": rank,
                            "cumulative_logprob": float(sequence.cum_logprob),
                            "finish_reason": sequence.finish_reason,
                            "stop_reason": sequence.stop_reason,
                            "sid_token_ids": sid_token_ids,
                            "sid_tokens": tokenizer.convert_ids_to_tokens(sid_token_ids),
                        },
                        ensure_ascii=False,
                    )
                )
        if getattr(args, "cot", False) and args.output_json is not None:
            report = {
                "mode": "cot_beam",
                "cot": args.cot_last_report,
                "benchmark": benchmark,
                "results": [
                    [
                        {"tokens": list(seq.tokens), "cumulative_logprob": float(seq.cum_logprob)}
                        for seq in output.sequences
                    ]
                    for output in outputs
                ],
            }
            path = args.output_json.resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


def main() -> None:
    args = parse_args()
    paths: list[Path] = [args.model]
    if not args.enable_thinking:
        paths.append(args.catalog)
    if args.prompt is not None:
        paths.append(args.prompt)
    if args.workload_jsonl is not None:
        paths.append(args.workload_jsonl)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.enable_thinking:
        run_thinking(args)
        return
    if (
        args.workload_jsonl is None
        and not args.cot
        and args.input_len + args.generated_steps + 1 > args.max_model_len
    ):
        raise ValueError("prompt and generated tokens exceed --max-model-len")
    if args.print_beams <= 0:
        raise ValueError("--print-beams must be positive")

    with tempfile.TemporaryDirectory(prefix="vllm-gr-beam-v1-") as directory:
        artifact_path = Path(directory).resolve() / "constraint_table.safetensors"
        run(args, artifact_path)


if __name__ == "__main__":
    main()
