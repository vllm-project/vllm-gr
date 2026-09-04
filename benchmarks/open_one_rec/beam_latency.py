# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reproducible OneRec beam latency benchmark.

The benchmark measures one fixed request at several normalized input lengths.
Each length is run as a prefix-cache miss immediately followed by a hit.  The
``compare`` mode launches isolated eager and graph processes so CUDA graph
capture and teardown cannot contaminate the other mode.

The benchmark is intentionally a Python entry point rather than a pytest test:
model construction, CUDA graph capture, and prefix-cache preparation are setup
work, while the JSON output contains the measured request samples and the
correctness checks needed to interpret them.

Example::

    python -m benchmarks.open_one_rec.beam_latency \
        --mode compare \
        --model /path/to/OneRec-1.7B \
        --catalog /path/to/video_constraint_triples.json \
        --prompt tests/resources/single_one_rec_prompt.txt \
        --output-json results/beam_latency.json

The caller must reserve an exclusive accelerator.  On the shared vllm-gr
development host, wrap the command with ``gpu run`` as documented in
``docs/benchmarks/beam.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_INPUT_LENGTHS = (1024, 2048, 5120, 8192, 10240)
DEFAULT_WARMUPS = 2
DEFAULT_RUNS = 5
DEFAULT_BEAM_WIDTH = 128
DEFAULT_MAX_TOKENS = 5
DEFAULT_BEGIN_TOKEN = "<|sid_begin|>"
DEFAULT_END_TOKEN = "<|sid_end|>"
BENCHMARK_SCHEMA_VERSION = 1
_MODE_CHOICES = ("eager", "graph", "compare")
_TIMED_METRICS = ("prefill_miss", "prefill_hit", "decode", "e2e_miss", "e2e_hit")


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize one measured metric in milliseconds."""
    if not values:
        raise ValueError("cannot summarize an empty sample")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean_ms": statistics.fmean(numeric),
        "p50_ms": _percentile(numeric, 50.0),
        "p90_ms": _percentile(numeric, 90.0),
        "p95_ms": _percentile(numeric, 95.0),
        "min_ms": min(numeric),
        "max_ms": max(numeric),
        "std_ms": statistics.pstdev(numeric) if len(numeric) > 1 else 0.0,
    }


def _prepare_prompt_ids(
    tokenizer: Any,
    source_text: str,
    input_length: int,
    *,
    begin_token: str | None,
) -> tuple[list[int], dict[str, Any]]:
    """Normalize one prompt to the model input length used by the benchmark."""
    if input_length < 1:
        raise ValueError(f"input length must be positive, got {input_length}")
    source_ids = list(tokenizer.encode(source_text, add_special_tokens=False))
    begin_count = 1 if begin_token is not None else 0
    submitted_length = input_length - begin_count
    if submitted_length < 1:
        raise ValueError(
            f"input length {input_length} is too short for begin token {begin_token!r}"
        )

    if len(source_ids) >= submitted_length:
        prompt_ids = source_ids[-submitted_length:]
        adjustment = "left_truncate"
    else:
        newline_ids = list(tokenizer.encode("\n", add_special_tokens=False))
        if len(newline_ids) != 1:
            raise ValueError(f"newline must encode to one token, got {newline_ids}")
        prompt_ids = [int(newline_ids[0])] * (submitted_length - len(source_ids)) + source_ids
        adjustment = "left_pad_newline"

    if len(prompt_ids) != submitted_length:
        raise AssertionError(
            f"prompt normalization produced {len(prompt_ids)} tokens; expected {submitted_length}"
        )
    return [int(token) for token in prompt_ids], {
        "source_prompt_tokens": len(source_ids),
        "submitted_prompt_tokens": len(prompt_ids),
        "begin_tokens_appended_by_serving": begin_count,
        "model_input_tokens": input_length,
        "adjustment": adjustment,
    }


def _digest_sequences(sequences: Iterable[Any]) -> str:
    """Hash complete ordered beam token sequences."""
    canonical = [[int(token) for token in sequence.tokens] for sequence in sequences]
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_beam_output(
    output: Any,
    *,
    beam_width: int,
    model_input_tokens: int,
    decode_steps: int,
    end_token_id: int | None,
) -> dict[str, Any]:
    """Validate beam count, sequence geometry, and the configured end token."""
    if len(output) != 1:
        raise AssertionError(f"expected one request in output, got {len(output)}")
    sequences = output[0].sequences
    if len(sequences) != beam_width:
        raise AssertionError(f"expected {beam_width} beams, got {len(sequences)}")

    expected_length = model_input_tokens + decode_steps + (1 if end_token_id is not None else 0)
    lengths = {len(sequence.tokens) for sequence in sequences}
    if lengths != {expected_length}:
        raise AssertionError(
            f"unexpected sequence lengths {sorted(lengths)}; expected {expected_length}"
        )
    if end_token_id is not None and any(
        sequence.tokens[-1] != end_token_id for sequence in sequences
    ):
        raise AssertionError("at least one beam does not end with the configured end token")

    return {
        "beam_count": len(sequences),
        "sequence_length": expected_length,
        "output_token_digest": _digest_sequences(sequences),
    }


def _stable_digests(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    digests = sorted({str(record["output_token_digest"]) for record in records})
    return {"count": len(digests), "digests": digests, "stable": len(digests) == 1}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=_MODE_CHOICES, required=True)
    parser.add_argument("--model", required=True, help="OneRec model path or model ID")
    parser.add_argument("--catalog", required=True, help="Constraint Tables triples JSON")
    parser.add_argument("--prompt", type=Path, required=True, help="Fixed prompt text file")
    parser.add_argument(
        "--input-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_INPUT_LENGTHS),
        metavar="TOKENS",
    )
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--begin-token", default=DEFAULT_BEGIN_TOKEN)
    parser.add_argument("--end-token", default=DEFAULT_END_TOKEN)
    parser.add_argument(
        "--constraint-backend",
        default="constraint_table",
        choices=("constraint_table",),
        help="Constraint Tables backend used by this benchmark",
    )
    parser.add_argument(
        "--attention-backend",
        default="CUSTOM",
        choices=("CUSTOM",),
        help="CUSTOM attention is required for the timed beam-request protocol",
    )
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM asynchronous scheduling (default: enabled)",
    )
    parser.add_argument("--output-json", type=Path, help="Write the result JSON to this path")
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Write an issue-ready eager-versus-graph summary table (compare mode only)",
    )
    parser.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not args.input_lengths or any(length < 1 for length in args.input_lengths):
        raise ValueError("--input-lengths must contain positive integers")
    if len(set(args.input_lengths)) != len(args.input_lengths):
        raise ValueError("--input-lengths must not contain duplicates")
    if args.beam_width < 1:
        raise ValueError("--beam-width must be positive")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.beam_width > 256:
        raise ValueError("--beam-width exceeds vllm-gr's current maximum of 256")


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    metadata: dict[str, Any] = {"root": str(repo_root)}
    try:
        metadata["sha"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        metadata["dirty"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        metadata["sha"] = os.environ.get("VLLM_GR_REPOSITORY_SHA")
        metadata["dirty"] = None
    return metadata


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return versions[0] if versions else None


def _environment_metadata(torch: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "vllm": _package_version("vllm"),
        "torch": _package_version("torch") or getattr(torch, "__version__", None),
        "cuda_runtime": str(getattr(torch.version, "cuda", None)),
        "vllm_gr": _package_version("vllm-gr"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "driver_version": _driver_version(),
        "repository": _git_metadata(),
    }
    try:
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        metadata["device"] = {
            "logical_index": device_index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "total_memory_mib": int(properties.total_memory // (1024 * 1024)),
            "multi_processor_count": int(properties.multi_processor_count),
        }
    except (AttributeError, RuntimeError):
        metadata["device"] = None
    return metadata


def _synchronize(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_sampling_params(
    beam_search_params_type: Any,
    *,
    beam_width: int,
    max_tokens: int,
    begin_token: str | None,
    end_token: str | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "beam_width": beam_width,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": False,
    }
    if begin_token is not None:
        kwargs["begin_token"] = begin_token
    if end_token is not None:
        kwargs["end_token"] = end_token
    try:
        return beam_search_params_type(**kwargs)
    except TypeError:
        # Keep the benchmark usable with the older compatibility class while
        # still making the marker configuration explicit when supported.
        fallback = {
            key: value for key, value in kwargs.items() if key not in {"begin_token", "end_token"}
        }
        params = beam_search_params_type(**fallback)
        if begin_token is not None and hasattr(params, "begin_token"):
            params.begin_token = begin_token
        if end_token is not None and hasattr(params, "end_token"):
            params.end_token = end_token
        return params


def _run_single_mode(args: argparse.Namespace) -> dict[str, Any]:
    """Run one mode and return the complete machine-readable result."""
    import torch
    from vllm.inputs import TokensPrompt

    from vllm_gr.config import GRConfig
    from vllm_gr.entrypoints import gr as gr_module
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the OneRec beam latency benchmark")

    source_text = args.prompt.read_text(encoding="utf-8")
    mode_graph = args.mode == "graph"
    begin_token = args.begin_token or None
    end_token = args.end_token or None
    precomputed_steps = args.max_tokens - int(begin_token is not None) - int(end_token is not None)
    if precomputed_steps < 1:
        raise ValueError(
            "--max-tokens must leave at least one generated token after begin/end markers"
        )
    params = _make_sampling_params(
        BeamSearchParams,
        beam_width=args.beam_width,
        max_tokens=args.max_tokens,
        begin_token=begin_token,
        end_token=end_token,
    )

    current_steps: list[tuple[int, float]] | None = None
    original_step: Callable[..., Any] = gr_module._step_engine_and_collect_outputs

    def timed_step(*step_args: Any, **step_kwargs: Any) -> Any:
        nonlocal current_steps
        token = int(step_kwargs.get("token", step_args[2] if len(step_args) > 2 else -1))
        started = time.perf_counter_ns()
        result = original_step(*step_args, **step_kwargs)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if current_steps is None:
            raise RuntimeError("engine step occurred outside a measured request")
        current_steps.append((token, elapsed_ms))
        return result

    gr_module._step_engine_and_collect_outputs = timed_step
    lengths: dict[str, Any] = {}
    contract: dict[str, Any] = {
        "input_lengths": [int(length) for length in args.input_lengths],
        "batch_size": 1,
        "beam_width": args.beam_width,
        "max_logprobs": args.beam_width * 2,
        "max_tokens_including_markers": args.max_tokens,
        "generated_steps": precomputed_steps,
        "dtype": "bfloat16",
        "mode": args.mode,
        "prefix_cache_reset_before_each_miss": True,
        "hit_immediately_follows_miss": True,
        "warmups_per_length": args.warmups,
        "measured_pairs_per_length": args.runs,
        "timing_boundary": "beam_search return plus CUDA synchronize",
        "step_boundary": "one _step_engine_and_collect_outputs call",
    }

    try:
        with GRLLM(
            model=args.model,
            trust_remote_code=True,
            max_logprobs=args.beam_width * 2,
            catalog_path=args.catalog,
            constraint_backend=args.constraint_backend,
            attention_config={"backend": args.attention_backend},
            dtype="bfloat16",
            enforce_eager=not mode_graph,
            beam_graph_enabled=mode_graph,
            beam_max_width=args.beam_width,
            beam_max_decode_steps=precomputed_steps,
            async_scheduling=args.async_scheduling,
            disable_log_stats=True,
            max_num_seqs=1,
            enable_prefix_caching=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
        ) as llm:
            tokenizer = llm.get_tokenizer()
            end_token_id = (
                tokenizer.convert_tokens_to_ids(end_token) if end_token is not None else None
            )
            if end_token_id in (None, -1):
                raise ValueError(f"end token is not in the tokenizer vocabulary: {end_token!r}")

            config = GRConfig.from_vllm_config(getattr(llm.llm_engine, "vllm_config", None))
            vllm_config = getattr(llm.llm_engine, "vllm_config", None)
            model_config = getattr(vllm_config, "model_config", None)
            resolved = {
                "beam_graph_enabled": getattr(config, "beam_graph_enabled", None),
                "beam_worker_decision": getattr(config, "beam_worker_decision", None),
                "beam_self_advance": getattr(config, "beam_self_advance", None),
                "constraint_backend": str(getattr(config, "constraint_backend", None)),
                "enforce_eager": bool(getattr(model_config, "enforce_eager", False)),
                "async_scheduling": args.async_scheduling,
            }

            for input_length in args.input_lengths:
                prompt_ids, prompt_info = _prepare_prompt_ids(
                    tokenizer,
                    source_text,
                    input_length,
                    begin_token=begin_token,
                )
                prompt = TokensPrompt(prompt_token_ids=prompt_ids)
                warmup_records: list[dict[str, Any]] = []
                measured: dict[str, list[dict[str, Any]]] = {"miss": [], "hit": []}

                def run_request(
                    cache_state: str,
                    *,
                    keep_record: bool,
                    prompt: Any = prompt,
                    input_length: int = input_length,
                    measured: dict[str, list[dict[str, Any]]] = measured,
                    warmup_records: list[dict[str, Any]] = warmup_records,
                ) -> dict[str, Any]:
                    nonlocal current_steps
                    current_steps = []
                    _synchronize(torch)
                    started = time.perf_counter_ns()
                    output = llm.beam_search([prompt], params, use_tqdm=False, concurrency_limit=1)
                    _synchronize(torch)
                    e2e_ms = (time.perf_counter_ns() - started) / 1e6
                    steps = current_steps
                    current_steps = None
                    if steps is None or [token for token, _ in steps] != list(
                        range(precomputed_steps)
                    ):
                        raise RuntimeError(
                            f"unexpected engine steps for input length {input_length}: {steps}"
                        )
                    validation = _validate_beam_output(
                        output,
                        beam_width=args.beam_width,
                        model_input_tokens=input_length,
                        decode_steps=precomputed_steps,
                        end_token_id=end_token_id,
                    )
                    record = {
                        "cache_state": cache_state,
                        "input_length": input_length,
                        "prefill_ms": steps[0][1],
                        "decode_engine_ms": sum(value for token, value in steps if token > 0),
                        "e2e_ms": e2e_ms,
                        "step_ms": [
                            {"token": token, "elapsed_ms": elapsed} for token, elapsed in steps
                        ],
                        **validation,
                    }
                    if keep_record:
                        measured[cache_state].append(record)
                    else:
                        warmup_records.append(record)
                    return record

                for _ in range(args.warmups):
                    if llm.reset_prefix_cache() is False:
                        raise RuntimeError("prefix-cache reset was rejected during warmup")
                    run_request("miss", keep_record=False)
                    run_request("hit", keep_record=False)
                for _ in range(args.runs):
                    if llm.reset_prefix_cache() is False:
                        raise RuntimeError("prefix-cache reset was rejected during measurement")
                    run_request("miss", keep_record=True)
                    run_request("hit", keep_record=True)

                miss = measured["miss"]
                hit = measured["hit"]
                metric_values: dict[str, list[float]] = {
                    "prefill_miss": [float(record["prefill_ms"]) for record in miss],
                    "prefill_hit": [float(record["prefill_ms"]) for record in hit],
                    "decode": [float(record["decode_engine_ms"]) for record in miss],
                    "e2e_miss": [float(record["e2e_ms"]) for record in miss],
                    "e2e_hit": [float(record["e2e_ms"]) for record in hit],
                }
                medians = {
                    name: _percentile(values, 50.0) for name, values in metric_values.items()
                }
                medians["overhead"] = (
                    medians["e2e_miss"] - medians["prefill_miss"] - medians["decode"]
                )
                lengths[str(input_length)] = {
                    "prompt": prompt_info,
                    "median_ms": medians,
                    "summary_ms": {
                        name: _summarize(values) for name, values in metric_values.items()
                    },
                    "raw": measured,
                    "warmup_count": len(warmup_records),
                    "output_stability": {
                        "miss": _stable_digests(miss),
                        "hit": _stable_digests(hit),
                    },
                }
    finally:
        gr_module._step_engine_and_collect_outputs = original_step

    try:
        import vllm_gr

        vllm_gr_path = str(Path(vllm_gr.__file__).resolve())
        vllm_gr_version = getattr(vllm_gr, "__version__", None)
    except (ImportError, AttributeError):
        vllm_gr_path = None
        vllm_gr_version = None
    environment = _environment_metadata(torch)
    environment["vllm_gr_import"] = vllm_gr_path
    environment["vllm_gr_version"] = vllm_gr_version
    repository = environment.get("repository", {})
    environment["vllm_gr_commit"] = repository.get("sha")
    environment["vllm_gr_dirty"] = repository.get("dirty")
    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": "onerec_beam_latency",
        "mode": args.mode,
        "model": args.model,
        "catalog": str(Path(args.catalog).resolve()),
        "prompt": str(args.prompt.resolve()),
        "contract": contract,
        "resolved": resolved,
        "environment": environment,
        "lengths": lengths,
    }
    return result


def _child_args(args: argparse.Namespace, mode: str, output: Path) -> list[str]:
    values: list[str] = [
        "--mode",
        mode,
        "--model",
        args.model,
        "--catalog",
        args.catalog,
        "--prompt",
        str(args.prompt),
        "--beam-width",
        str(args.beam_width),
        "--max-tokens",
        str(args.max_tokens),
        "--warmups",
        str(args.warmups),
        "--runs",
        str(args.runs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--begin-token",
        args.begin_token,
        "--end-token",
        args.end_token,
        "--constraint-backend",
        args.constraint_backend,
        "--attention-backend",
        args.attention_backend,
        "--output-json",
        str(output),
        "--input-lengths",
        *(str(length) for length in args.input_lengths),
        "--quiet",
    ]
    if not args.async_scheduling:
        values.append("--no-async-scheduling")
    return values


def _compare_mode_results(eager: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    """Build a same-input eager/graph comparison from two mode results."""
    if eager.get("contract", {}).get("input_lengths") != graph.get("contract", {}).get(
        "input_lengths"
    ):
        raise ValueError("eager and graph input-length contracts differ")
    comparison: dict[str, Any] = {}
    for input_length in eager["contract"]["input_lengths"]:
        key = str(input_length)
        eager_row = eager["lengths"][key]
        graph_row = graph["lengths"][key]
        eager_medians = eager_row["median_ms"]
        graph_medians = graph_row["median_ms"]
        deltas = {
            metric: float(graph_medians[metric] - eager_medians[metric])
            for metric in (*_TIMED_METRICS, "overhead")
        }
        improvements = {}
        for metric in (*_TIMED_METRICS, "overhead"):
            baseline = float(eager_medians[metric])
            improvements[metric] = (
                (baseline - float(graph_medians[metric])) / baseline * 100.0
                if baseline != 0.0
                else None
            )
        eager_digests = [
            record["output_token_digest"]
            for state in ("miss", "hit")
            for record in eager_row["raw"].get(state, [])
        ]
        graph_digests = [
            record["output_token_digest"]
            for state in ("miss", "hit")
            for record in graph_row["raw"].get(state, [])
        ]
        eager_outputs_stable = bool(eager_digests) and len(set(eager_digests)) == 1
        graph_outputs_stable = bool(graph_digests) and len(set(graph_digests)) == 1
        comparison[key] = {
            "eager_median_ms": eager_medians,
            "graph_median_ms": graph_medians,
            "delta_graph_minus_eager_ms": deltas,
            "graph_improvement_percent": improvements,
            "output_digests_identical": eager_digests == graph_digests,
            "eager_outputs_stable": eager_outputs_stable,
            "graph_outputs_stable": graph_outputs_stable,
            "output_stability_pass": eager_outputs_stable and graph_outputs_stable,
        }
    return comparison


def _markdown_value(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{float(value):.2f}%"
    return f"{float(value):.3f}"


def _markdown_summary(result: dict[str, Any]) -> str:
    """Render a compact table suitable for pasting into a GitHub issue."""
    if result.get("mode") != "compare":
        raise ValueError("Markdown summaries require compare mode")
    environment = result.get("environment", {})
    device = environment.get("device") or {}
    device_name = device.get("name", "unknown GPU")
    device_memory = device.get("total_memory_mib")
    if device_memory is not None:
        device_name = f"{device_name} ({device_memory} MiB)"
    repository = environment.get("repository") or {}
    commit = environment.get("vllm_gr_commit") or repository.get("sha") or "unknown"
    dirty = environment.get("vllm_gr_dirty")
    tree_state = "dirty" if dirty else "clean" if dirty is False else "unknown tree state"
    contract = result.get("contract", {})

    lines = [
        "# OneRec beam latency benchmark",
        "",
        f"- Hardware: {device_name}",
        f"- vLLM: {environment.get('vllm') or 'unknown'}",
        (
            f"- vllm-gr: {environment.get('vllm_gr_version') or 'unknown'} "
            f"(`{commit}`, {tree_state})"
        ),
        (
            f"- Contract: beam width {contract.get('beam_width')}, "
            f"max tokens {contract.get('max_tokens_including_markers')}, "
            f"{contract.get('measured_pairs_per_length')} measured miss/hit pairs per length"
        ),
        "",
        (
            "| Input tokens | Eager prefill miss (ms) | Graph prefill miss (ms) | "
            "Eager E2E miss (ms) | Graph E2E miss (ms) | Graph E2E improvement | Output parity |"
        ),
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for input_length in contract.get("input_lengths", []):
        row = result["comparison"][str(input_length)]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(input_length),
                    _markdown_value(row["eager_median_ms"]["prefill_miss"]),
                    _markdown_value(row["graph_median_ms"]["prefill_miss"]),
                    _markdown_value(row["eager_median_ms"]["e2e_miss"]),
                    _markdown_value(row["graph_median_ms"]["e2e_miss"]),
                    _markdown_value(row["graph_improvement_percent"]["e2e_miss"], percent=True),
                    "PASS"
                    if row["output_digests_identical"] and row["output_stability_pass"]
                    else "FAIL",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _write_result(
    result: dict[str, Any], output: Path | None, *, print_result: bool = True
) -> None:
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    if print_result:
        print(serialized, end="")


def _write_markdown(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_markdown_summary(result), encoding="utf-8")


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_json
    if output is None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="vllm_gr_beam_latency_") as temporary:
            return _run_compare_with_paths(args, Path(temporary), None, print_result=not args.quiet)
    output.parent.mkdir(parents=True, exist_ok=True)
    return _run_compare_with_paths(args, output.parent, output, print_result=not args.quiet)


def _run_compare_with_paths(
    args: argparse.Namespace,
    directory: Path,
    output: Path | None,
    *,
    print_result: bool = True,
) -> dict[str, Any]:
    stem = output.stem if output is not None else "beam_latency"
    eager_path = directory / f"{stem}.eager.json"
    graph_path = directory / f"{stem}.graph.json"
    for mode, component_path in (("eager", eager_path), ("graph", graph_path)):
        command = [sys.executable, "-m", "benchmarks.open_one_rec.beam_latency"]
        command.extend(_child_args(args, mode, component_path))
        subprocess.run(command, check=True)
    eager = json.loads(eager_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": "onerec_beam_latency",
        "mode": "compare",
        "contract": eager["contract"],
        "environment": eager["environment"],
        "eager": eager,
        "graph": graph,
        "comparison": _compare_mode_results(eager, graph),
        "component_results": (
            {"eager": str(eager_path), "graph": str(graph_path)} if output is not None else None
        ),
    }
    if output is not None:
        _write_result(result, output, print_result=print_result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if args.output_markdown is not None and args.mode != "compare":
            raise ValueError("--output-markdown is only supported with --mode compare")
        for path, label in ((args.prompt, "prompt"), (Path(args.catalog), "catalog")):
            if not path.is_file():
                raise FileNotFoundError(f"{label} file not found: {path}")
        if args.mode == "compare":
            result = _run_compare(args)
            if args.output_json is None and not args.quiet:
                _write_result(result, None)
            if args.output_markdown is not None:
                _write_markdown(result, args.output_markdown)
        else:
            result = _run_single_mode(args)
            _write_result(result, args.output_json, print_result=not args.quiet)
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        AssertionError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"beam latency benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
