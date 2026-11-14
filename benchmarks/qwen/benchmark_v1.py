# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Benchmark the offline ``GRLLM.beam_search_v1`` API.

The benchmark is self-contained within the vllm-gr repository. It consumes a
JSONL workload produced by ``benchmarks/qwen/make_workload.py``, builds a
synthetic three-level constraint table when needed, compiles the canonical
safetensors artifact, runs warmup/measured requests, and optionally exports a
gzip Perfetto timeline.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_gr.config.benchmark_options import (
    add_recogem_arguments,
    beam_optimization_config,
    configure_benchmark_precision,
    optimization_settings,
)

MAX_BEAM_STEPS = 3


def configure_multiprocessing() -> str:
    """Force CUDA-safe process creation before importing torch or vLLM."""
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return "spawn"


@dataclass(frozen=True, slots=True)
class ConstraintArtifact:
    path: Path
    artifact_digest: str
    tokenizer_digest: str
    vocab_size: int


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _write_json(path: Path | str, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _numeric_median(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute the median of an empty sequence")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(quantile * (len(ordered) - 1)))
    return ordered[index]


def _metric_summaries(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in (
        "wall_ms",
        "output_normalization_ms",
        "request_latency_ms_p50",
        "request_latency_ms_p95",
        "qps",
        "generated_tokens_per_s",
        "beam_candidates_per_s",
    ):
        samples = [float(run[field]) for run in runs]
        result[f"{field}_samples"] = samples
        result[f"{field}_median"] = _numeric_median(samples)
    return result


def _load_workload(
    path: Path,
    *,
    requests: int,
    vocab_size: int,
    token_id_start: int = 0,
    token_id_end: int | None = None,
) -> list[dict[str, Any]]:
    token_id_end = vocab_size if token_id_end is None else token_id_end
    if not 0 <= token_id_start < token_id_end <= vocab_size:
        raise ValueError("invalid workload token interval")
    if not path.is_file():
        raise FileNotFoundError(f"workload not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"workload line {line_no} must be a JSON object")
            rows.append(row)
            if len(rows) == requests:
                break
    if len(rows) < requests:
        raise ValueError(f"workload has {len(rows)} rows, expected {requests}")
    for index, row in enumerate(rows):
        input_ids = row.get("input_ids")
        if not isinstance(input_ids, list) or not input_ids:
            raise ValueError(f"workload row {index} must contain non-empty input_ids")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not token_id_start <= token_id < token_id_end
            for token_id in input_ids
        ):
            raise ValueError(
                f"workload row {index} contains a token ID outside [{token_id_start}, {token_id_end})"
            )
        row.setdefault("request_id", f"beamcmp-{index}")
    return rows


def _load_model_assets(args: argparse.Namespace) -> tuple[Any, int]:
    from transformers import AutoConfig, AutoTokenizer

    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    model_config = AutoConfig.from_pretrained(args.model, **load_kwargs)
    vocab_size = getattr(model_config, "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"model {args.model!r} does not expose a positive integer vocab_size")
    return tokenizer, vocab_size


def _spread_items(items: list[tuple[int, str]], count: int) -> list[tuple[int, str]]:
    if count == 1:
        return [items[len(items) // 2]]
    # Select quantile centers so synthetic IDs span the available model vocab
    # without concentrating at token IDs near zero.
    return [items[((2 * index + 1) * len(items)) // (2 * count)] for index in range(count)]


def _synthetic_constraint_tokens(
    tokenizer: Any,
    *,
    count: int,
    vocab_size: int,
    layout: str,
    token_id_start: int,
    token_id_end: int,
) -> list[str]:
    if not 0 <= token_id_start < token_id_end <= vocab_size:
        raise ValueError("token interval exceeds model vocabulary or is empty")
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError("the model tokenizer does not expose get_vocab()")
    vocab = get_vocab()
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ())}
    candidates: list[tuple[int, str]] = []
    seen_ids: set[int] = set()
    for token, token_id in sorted(vocab.items(), key=lambda item: int(item[1])):
        token_id = int(token_id)
        if (
            token_id in seen_ids
            or token_id in special_ids
            or not token_id_start <= token_id < token_id_end
        ):
            continue
        if tokenizer.convert_tokens_to_ids(token) != token_id:
            continue
        seen_ids.add(token_id)
        candidates.append((token_id, str(token)))
    if len(candidates) < count:
        raise ValueError(
            f"interval [{token_id_start}, {token_id_end}) has {len(candidates)} usable tokens; "
            f"{count} are required for synthetic constraints"
        )
    selected = candidates[:count] if layout == "low" else _spread_items(candidates, count)
    if len({token_id for token_id, _ in selected}) != count:
        raise RuntimeError("synthetic token selection produced duplicate token IDs")
    return [token for _, token in selected]


def build_synthetic_constraint_document(
    tokenizer: Any,
    *,
    beam_width: int,
    vocab_size: int,
    layout: str,
    token_id_start: int,
    token_id_end: int,
) -> dict[str, Any]:
    """Build a three-step table with exactly ``beam_width`` valid paths."""
    from vllm_gr.constraints.json_compiler import compile_relations_json

    tokens = _synthetic_constraint_tokens(
        tokenizer,
        count=beam_width * MAX_BEAM_STEPS,
        vocab_size=vocab_size,
        layout=layout,
        token_id_start=token_id_start,
        token_id_end=token_id_end,
    )
    triples = [
        [
            tokens[index],
            tokens[beam_width + index],
            tokens[2 * beam_width + index],
        ]
        for index in range(beam_width)
    ]
    return compile_relations_json(triples)


def ensure_constraint_document(
    args: argparse.Namespace, *, tokenizer: Any, vocab_size: int
) -> Path:
    path = Path(args.constraint_triples)
    if not args.generate_constraint_triples:
        if not path.is_file():
            raise FileNotFoundError(
                f"constraint triples not found: {path}; provide the file or omit the option "
                "to generate a synthetic table"
            )
        return path
    document = build_synthetic_constraint_document(
        tokenizer,
        beam_width=args.beam_width,
        vocab_size=vocab_size,
        layout=args.synthetic_token_layout,
        token_id_start=args.token_id_start,
        token_id_end=args.token_id_end,
    )
    _write_json(path, document)
    return path


def _validate_constraint_range(table: Any, start: int, end: int) -> None:
    for name in ("first_token_ids", "prefix1_values", "prefix2_values"):
        ids = getattr(table, name)
        if ids.size and (int(ids.min()) < start or int(ids.max()) >= end):
            raise ValueError(f"constraint table {name} contains IDs outside [{start}, {end})")


def build_constraint_artifact(
    args: argparse.Namespace,
    triples_path: Path,
    *,
    tokenizer: Any,
    vocab_size: int,
) -> ConstraintArtifact:
    """Compile triples into the canonical artifact required by Beam Search V1."""
    from vllm_gr.constraints.artifact import save_constraint_table_artifact
    from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
    from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

    try:
        document = json.loads(triples_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read constraint triples {triples_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("constraint triples document must be a JSON object")
    tokenizer_digest = compute_tokenizer_digest(tokenizer)
    table = ConstraintTableArtifactCompiler(
        tokenizer,
        vocab_size,
        tokenizer_digest=tokenizer_digest,
    ).compile(document)
    _validate_constraint_range(table, args.token_id_start, args.token_id_end)
    output_path = Path(args.constraint_artifact).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_digest = save_constraint_table_artifact(table, output_path)
    return ConstraintArtifact(
        path=output_path,
        artifact_digest=artifact_digest,
        tokenizer_digest=tokenizer_digest,
        vocab_size=vocab_size,
    )


def canonical_gr_config(args: argparse.Namespace, artifact: ConstraintArtifact) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "beam": {
            "execution_mode": "v1",
            "graph_enabled": not args.disable_beam_graph,
            "enable_kv_pool_ping_pong": args.kv_pool_ping_pong,
            **beam_optimization_config(args),
            "max_width": args.beam_width,
            "max_decode_steps": args.beam_steps,
        },
        "constraint_table": {
            "enabled": True,
            "backend": "cuda",
            "path": str(artifact.path),
            "format": "constraint_table_v1",
            "artifact_digest": artifact.artifact_digest,
            "tokenizer_digest": artifact.tokenizer_digest,
            "max_top_k": args.beam_width,
        },
        "attention_backend": args.attention_backend or None,
    }


def _sequence_token_ids(sequence: Any, prompt_ids: list[int]) -> list[int]:
    tokens = list(getattr(sequence, "tokens", ()))
    if tokens[: len(prompt_ids)] == prompt_ids:
        return [int(token) for token in tokens[len(prompt_ids) :]]
    token_ids = getattr(sequence, "token_ids", None)
    if token_ids is not None:
        return [int(token) for token in token_ids]
    return [int(token) for token in tokens]


def _normalize_output(
    output_rows: Any, row: dict[str, Any], *, expected_beam_width: int
) -> dict[str, Any]:
    outputs = list(output_rows)
    if len(outputs) != 1:
        raise RuntimeError(
            f"beam_search_v1 returned {len(outputs)} request outputs; expected exactly 1"
        )
    prompt_ids = [int(token) for token in row["input_ids"]]
    sequences = list(getattr(outputs[0], "sequences", ()))
    if len(sequences) != expected_beam_width:
        raise RuntimeError(
            f"beam_search_v1 returned {len(sequences)} beams; expected {expected_beam_width}"
        )
    beams = []
    for rank, sequence in enumerate(sequences):
        score = getattr(sequence, "cum_logprob", None)
        if score is None:
            score = getattr(sequence, "cumulative_logprob", None)
        finish_reason = getattr(sequence, "finish_reason", None)
        beams.append(
            {
                "rank": rank,
                "token_ids": _sequence_token_ids(sequence, prompt_ids),
                "score": float(score) if score is not None else None,
                "finish_reason": (str(finish_reason) if finish_reason is not None else None),
            }
        )
    return {
        "request_id": row["request_id"],
        "workload_id": row["request_id"],
        "prompt_tokens": len(prompt_ids),
        "beams": beams,
    }


def _run_once(
    llm: Any,
    workload: list[dict[str, Any]],
    params: Any,
    *,
    beam_width: int,
    beam_steps: int,
    record_outputs: bool,
) -> dict[str, Any]:
    raw_outputs: list[tuple[dict[str, Any], Any]] = []
    request_latencies_ms: list[float] = []
    started = time.perf_counter()
    for row in workload:
        request_started = time.perf_counter()
        # Beam Search V1 currently accepts exactly one prompt per offline call.
        outputs = llm.beam_search_v1(
            [{"prompt_token_ids": row["input_ids"]}],
            params,
            use_tqdm=False,
            concurrency_limit=1,
            output_mode="full",
        )
        request_latencies_ms.append((time.perf_counter() - request_started) * 1000.0)
        if record_outputs:
            raw_outputs.append((row, outputs))
    wall_ms = (time.perf_counter() - started) * 1000.0
    normalize_started = time.perf_counter()
    normalized = (
        [
            _normalize_output(outputs, row, expected_beam_width=beam_width)
            for row, outputs in raw_outputs
        ]
        if record_outputs
        else []
    )
    output_normalization_ms = (time.perf_counter() - normalize_started) * 1000.0
    request_count = len(workload)
    elapsed_s = wall_ms / 1000.0
    generated_tokens = request_count * beam_steps
    beam_candidates = generated_tokens * beam_width
    return {
        "wall_ms": wall_ms,
        "qps": request_count / elapsed_s if elapsed_s > 0 else None,
        "generated_tokens_per_s": generated_tokens / elapsed_s if elapsed_s > 0 else None,
        "beam_candidates_per_s": beam_candidates / elapsed_s if elapsed_s > 0 else None,
        "request_latencies_ms": request_latencies_ms,
        "request_latency_ms_p50": _percentile(request_latencies_ms, 0.50),
        "request_latency_ms_p95": _percentile(request_latencies_ms, 0.95),
        "output_normalization_ms": output_normalization_ms,
        "outputs": normalized,
    }


def _timeline_target(args: argparse.Namespace) -> Path | None:
    if not args.timeline_output:
        return None
    path = Path(args.timeline_output).resolve()
    if not path.name.endswith(".json.gz"):
        raise ValueError("--timeline-output must end with .json.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _profiler_config(args: argparse.Namespace, trace_dir: Path) -> dict[str, Any]:
    return {
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir),
        "torch_profiler_with_stack": args.timeline_with_stack,
        "torch_profiler_record_shapes": args.timeline_record_shapes,
        "torch_profiler_with_memory": args.timeline_with_memory,
        "torch_profiler_use_gzip": True,
        "torch_profiler_dump_cuda_time_total": False,
        "ignore_frontend": True,
    }


def _run_timeline(
    llm: Any,
    workload: list[dict[str, Any]],
    params: Any,
    args: argparse.Namespace,
    *,
    trace_dir: Path,
    target: Path,
) -> dict[str, Any]:
    profile_prefix = target.name.removesuffix(".json.gz")
    before = set(trace_dir.glob("*.pt.trace.json.gz"))
    llm.start_profile(profile_prefix)
    try:
        runs = [
            _run_once(
                llm,
                workload[: args.timeline_requests],
                params,
                beam_width=args.beam_width,
                beam_steps=args.beam_steps,
                record_outputs=True,
            )
            for _ in range(args.timeline_repeat)
        ]
    finally:
        llm.stop_profile()
    traces = sorted(
        set(trace_dir.glob("*.pt.trace.json.gz")) - before,
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(traces) != 1:
        names = [path.name for path in traces]
        raise RuntimeError(
            f"expected exactly one Worker timeline trace, found {len(traces)}: {names}"
        )
    traces[0].replace(target)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"timeline trace was not written: {target}")
    return {
        "repeat": args.timeline_repeat,
        "wall_ms_samples": [run["wall_ms"] for run in runs],
    }


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment(torch: Any) -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    def package_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version,
        "torch": getattr(torch, "__version__", None),
        "vllm": package_version("vllm"),
        "vllm_gr": package_version("vllm-gr"),
        "cuda_available": cuda_available,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_device_capability": (torch.cuda.get_device_capability(0) if cuda_available else None),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    multiprocessing_method = configure_multiprocessing()
    precision = configure_benchmark_precision(args)
    import torch

    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    tokenizer, vocab_size = _load_model_assets(args)
    if not 0 <= args.token_id_start < args.token_id_end <= vocab_size:
        raise ValueError(f"require 0 <= token-id-start < token-id-end <= {vocab_size}")
    workload = _load_workload(
        Path(args.workload_jsonl),
        requests=args.requests,
        vocab_size=vocab_size,
        token_id_start=args.token_id_start,
        token_id_end=args.token_id_end,
    )
    constraint_path = ensure_constraint_document(
        args, tokenizer=tokenizer, vocab_size=vocab_size
    ).resolve()
    constraint_artifact = build_constraint_artifact(
        args,
        constraint_path,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
    )
    gr_config = canonical_gr_config(args, constraint_artifact)
    params = BeamSearchParams(
        beam_width=args.beam_width,
        max_tokens=args.beam_steps,
        temperature=args.temperature,
        ignore_eos=args.ignore_eos,
        length_penalty=args.length_penalty,
    )
    max_prompt_len = max(len(row["input_ids"]) for row in workload)
    max_model_len = args.max_model_len or max_prompt_len + args.beam_steps
    max_num_batched_tokens = args.max_num_batched_tokens or max(
        max_prompt_len, args.beam_width, max_model_len
    )
    timeline_target = _timeline_target(args)
    timeline_temp = (
        tempfile.TemporaryDirectory(
            prefix=f".{timeline_target.name.removesuffix('.json.gz')}.",
            dir=timeline_target.parent,
        )
        if timeline_target is not None
        else None
    )
    timeline_trace_dir = Path(timeline_temp.name) if timeline_temp is not None else None
    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": args.trust_remote_code,
        "max_logprobs": args.beam_width,
        "vllm_gr_config": gr_config,
        "async_scheduling": True,
        "enable_prefix_caching": args.enable_prefix_caching,
        "max_num_seqs": 1,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "dtype": args.dtype,
        "seed": args.seed,
    }
    if timeline_trace_dir is not None:
        engine_kwargs["profiler_config"] = _profiler_config(args, timeline_trace_dir)

    timeline_run = None
    memory = None
    try:
        with GRLLM(**engine_kwargs) as llm:
            for _ in range(args.warmup_runs):
                _run_once(
                    llm,
                    workload,
                    params,
                    beam_width=args.beam_width,
                    beam_steps=args.beam_steps,
                    record_outputs=False,
                )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            measured = [
                _run_once(
                    llm,
                    workload,
                    params,
                    beam_width=args.beam_width,
                    beam_steps=args.beam_steps,
                    record_outputs=True,
                )
                for _ in range(args.repeat)
            ]
            if torch.cuda.is_available():
                memory = {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "scope": "post-warmup inference, including model, KV and captured graphs; excludes profiler",
                }
            if timeline_target is not None and timeline_trace_dir is not None:
                timeline_run = _run_timeline(
                    llm,
                    workload,
                    params,
                    args,
                    trace_dir=timeline_trace_dir,
                    target=timeline_target,
                )
    finally:
        if timeline_temp is not None:
            timeline_temp.cleanup()

    repo_root = Path(__file__).resolve().parents[2]
    return {
        "framework": "vllm-gr",
        "api": "GRLLM.beam_search_v1",
        "execution_mode": "offline_sequential_single_prompt",
        "model": args.model,
        "vllm_gr_repo": str(repo_root),
        "vllm_gr_commit": _git_commit(repo_root),
        "workload_jsonl": args.workload_jsonl,
        "constraint_triples": str(constraint_path),
        "constraint_artifact": str(constraint_artifact.path),
        "constraint_artifact_digest": constraint_artifact.artifact_digest,
        "constraint_tokenizer_digest": constraint_artifact.tokenizer_digest,
        "synthetic_constraint_triples": args.generate_constraint_triples,
        "synthetic_token_layout": (
            args.synthetic_token_layout if args.generate_constraint_triples else None
        ),
        "context_len": args.context_len,
        "token_id_start": args.token_id_start,
        "token_id_end_exclusive": args.token_id_end,
        "beam_steps": args.beam_steps,
        "decode_iterations": args.beam_steps - 1,
        "beam_width": args.beam_width,
        "requests": len(workload),
        "warmup_runs": args.warmup_runs,
        "repeat": args.repeat,
        "timeline": (
            {
                "output": str(timeline_target),
                "requests": args.timeline_requests,
                "repeat": args.timeline_repeat,
                "profile_run_excluded_from_metrics": True,
                "wall_ms_samples": timeline_run["wall_ms_samples"],
                "phase_semantics": (
                    "stage 0 is Prefill and emits the first Beam result; "
                    f"stages 1..{args.beam_steps - 1} are Decode"
                ),
            }
            if timeline_target is not None and timeline_run is not None
            else None
        ),
        "sampling_params": {
            "beam_width": args.beam_width,
            "max_tokens": args.beam_steps,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "length_penalty": args.length_penalty,
        },
        "beam_optimizations": optimization_settings(args),
        "memory": memory,
        "precision": precision,
        "engine_config": {
            "worker_multiprocessing_method": multiprocessing_method,
            "async_scheduling": True,
            "constraint_backend": "canonical_cuda_constraint_table",
            "beam_graph_enabled": not args.disable_beam_graph,
            "enable_kv_pool_ping_pong": args.kv_pool_ping_pong,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": 1,
            "enable_prefix_caching": args.enable_prefix_caching,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "attention_backend": args.attention_backend,
            "enforce_eager": args.enforce_eager,
            "dtype": args.dtype,
        },
        **_metric_summaries(measured),
        "runs": measured,
        "environment": _environment(torch),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model path or model ID.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--workload-jsonl", required=True)
    parser.add_argument(
        "--token-id-start", type=int, required=True, help="Shared input/output lower bound."
    )
    parser.add_argument(
        "--token-id-end", type=int, required=True, help="Shared exclusive upper bound."
    )
    parser.add_argument("--constraint-triples")
    parser.add_argument("--constraint-artifact")
    parser.add_argument("--generate-constraint-triples", action="store_true")
    parser.add_argument(
        "--synthetic-token-layout",
        choices=("spread", "low"),
        default="spread",
        help="Synthetic token distribution within the configured token interval.",
    )
    parser.add_argument("--context-len", type=int, default=5000)
    parser.add_argument(
        "--beam-steps",
        "--decode-steps",
        dest="beam_steps",
        type=int,
        default=3,
        help="Total generated Beam stages: one Prefill plus beam_steps-1 Decode stages.",
    )
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--attention-backend", default="CUSTOM")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-beam-graph", action="store_true")
    parser.add_argument(
        "--kv-pool-ping-pong",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gather KV directly into the next Decode bank; disable for write-back A/B.",
    )
    add_recogem_arguments(parser)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--timeline-output",
        help="Write a gzip Perfetto/Chrome trace from separate profiled runs.",
    )
    parser.add_argument("--timeline-requests", type=int, default=1)
    parser.add_argument(
        "--timeline-repeat",
        type=int,
        help="Number of profiled runs; defaults to --repeat.",
    )
    parser.add_argument("--timeline-record-shapes", action="store_true")
    parser.add_argument("--timeline-with-stack", action="store_true")
    parser.add_argument("--timeline-with-memory", action="store_true")
    return parser


def finalize_args(args: argparse.Namespace) -> None:
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json = str(output)
    args.workload_jsonl = str(Path(args.workload_jsonl).expanduser().resolve())
    if args.constraint_triples is None:
        args.constraint_triples = str(
            output.parent / f"synthetic_constraint_beam{args.beam_width}.json"
        )
        args.generate_constraint_triples = True
    else:
        args.constraint_triples = str(Path(args.constraint_triples).expanduser().resolve())
    if args.constraint_artifact is None:
        args.constraint_artifact = str(
            output.parent / f"constraint_table_beam{args.beam_width}.safetensors"
        )
    else:
        args.constraint_artifact = str(Path(args.constraint_artifact).expanduser().resolve())
    if args.timeline_output:
        args.timeline_output = str(Path(args.timeline_output).expanduser().resolve())
    if args.timeline_repeat is None:
        args.timeline_repeat = args.repeat


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.beam_steps <= MAX_BEAM_STEPS:
        raise ValueError(f"--beam-steps must be in [1, {MAX_BEAM_STEPS}]")
    if not 0 <= args.token_id_start < args.token_id_end:
        raise ValueError("require 0 <= token-id-start < token-id-end")
    if not 1 <= args.beam_width <= 256:
        raise ValueError("--beam-width must be in [1, 256]")
    if args.context_len <= 0 or args.requests <= 0 or args.repeat <= 0:
        raise ValueError("context-len, requests, and repeat must be positive")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if not 1 <= args.timeline_requests <= args.requests:
        raise ValueError("--timeline-requests must be in [1, requests]")
    if args.timeline_repeat <= 0:
        raise ValueError("--timeline-repeat must be positive")
    _timeline_target(args)


def main() -> None:
    args = build_parser().parse_args()
    finalize_args(args)
    validate_args(args)
    configure_multiprocessing()
    summary = run_benchmark(args)
    _write_json(args.output_json, summary)
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "timeline": (summary["timeline"] or {}).get("output"),
                "api": summary["api"],
                "wall_ms_median": summary["wall_ms_median"],
                "qps_median": summary["qps_median"],
                "generated_tokens_per_s_median": summary["generated_tokens_per_s_median"],
                "beam_candidates_per_s_median": summary["beam_candidates_per_s_median"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
