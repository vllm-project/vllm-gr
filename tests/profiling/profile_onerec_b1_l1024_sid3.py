# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture an attributable B=1 OneRec latency/correctness baseline.

One invocation records one unambiguous backend/execution/prefix-cache case.
Run separate processes for the matrix so engine-global graph and prefix-cache
settings cannot bleed between cases.  This is a profiling utility, not a test:
missing accelerator hardware is an environment failure rather than a skip.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

_DEPENDENCIES = (
    "vllm-gr",
    "vllm",
    "vllm-ascend",
    "xllm-ops",
    "torch",
    "torch-npu",
    "triton",
    "triton-ascend",
)

_RUNNER_CLASS_ALLOWLIST = {
    "gpu": {"vllm.v1.worker.gpu_model_runner.GPUModelRunner"},
    "npu": {"vllm_ascend.worker.model_runner_v1.NPUModelRunner"},
}

_TRACE_STAGE_MARKERS = {
    ("gpu", "eager"): ("prefill_beam_bootstrap", "decode_result_d2h"),
    ("gpu", "graph"): ("prefill_beam_bootstrap", "decode_result_d2h"),
    ("npu", "eager"): ("gr_npu_prefill_stage", "gr_npu_decode_stage"),
    ("npu", "graph"): ("gr_npu_prefill_stage", "gr_npu_decode_stage"),
    ("npu", "cascade"): ("gr_npu_prefill_stage", "gr_npu_decode_stage"),
}


class EnvironmentNotValidatedError(RuntimeError):
    """The requested accelerator runtime is absent or unusable."""


class EvidenceUnavailableError(RuntimeError):
    """A mandatory Step 0 measurement could not be collected or verified."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--backend", choices=("gpu", "npu"), required=True)
    parser.add_argument("--execution", choices=("eager", "graph", "cascade"), required=True)
    parser.add_argument("--prefix-cache", choices=("miss", "hit", "disabled"), required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--pr-324-sha", required=True)
    parser.add_argument("--model-digest")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--generation-steps", type=int, default=3)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--slo-ms", type=float)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--persistent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the persistent final-only request and runner-state path",
    )
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("this frozen baseline only supports batch_size=1")
    if args.backend == "gpu" and args.execution == "cascade":
        raise ValueError("cascade execution is NPU-only")
    if args.persistent and not args.async_scheduling:
        raise ValueError("persistent profiling requires async scheduling")
    for name in ("input_len", "beam_width", "generation_steps", "runs"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.slo_ms is not None and args.slo_ms <= 0:
        raise ValueError("slo_ms must be positive when provided")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    for name in ("base_sha", "pr_324_sha"):
        value = getattr(args, name)
        if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
            cli_name = name.replace("_", "-")
            raise ValueError(f"--{cli_name} must be a complete 40-hex commit SHA")


def _require_v1_model_runner(environ: Mapping[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    value = env.get("VLLM_USE_V2_MODEL_RUNNER")
    if value != "0":
        raise RuntimeError(
            "Step 0 profiling requires ModelRunner V1: set "
            "VLLM_USE_V2_MODEL_RUNNER=0 explicitly (found " + repr(value) + ")"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(model: str, supplied_digest: str | None, hf_commit: str | None) -> dict:
    if supplied_digest:
        return {"requested": model, "digest": supplied_digest, "digest_kind": "supplied"}
    path = Path(model).expanduser()
    if path.is_file():
        return {
            "requested": model,
            "digest": _sha256_file(path),
            "digest_kind": "file_sha256",
        }
    if path.is_dir():
        # Hashing multi-GB weight files would perturb the benchmark host cache.
        # The manifest digest is deliberately labelled as such; callers that
        # have a content digest should pass --model-digest.
        manifest = [
            (str(item.relative_to(path)), item.stat().st_size)
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ]
        payload = json.dumps(manifest, separators=(",", ":")).encode()
        return {
            "requested": model,
            "resolved_path": str(path.resolve()),
            "digest": hashlib.sha256(payload).hexdigest(),
            "digest_kind": "path_size_manifest_sha256",
            "hf_commit": hf_commit,
        }
    payload = json.dumps(
        {"model": model, "hf_commit": hf_commit}, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "requested": model,
        "digest": hashlib.sha256(payload).hexdigest(),
        "digest_kind": "model_revision_identity_sha256",
        "hf_commit": hf_commit,
    }


def _run_command(argv: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _source_metadata(repo_root: Path, args: argparse.Namespace) -> dict:
    head = _run_command(("git", "rev-parse", "HEAD"), repo_root)
    status = _run_command(("git", "status", "--porcelain"), repo_root)
    branch = _run_command(("git", "branch", "--show-current"), repo_root)
    if not head:
        raise RuntimeError(f"cannot resolve commit SHA from {repo_root}")
    if status is None:
        raise EvidenceUnavailableError(f"cannot determine git worktree status in {repo_root}")
    if branch is None:
        raise EvidenceUnavailableError(f"cannot determine git branch in {repo_root}")
    return {
        "repo_root": str(repo_root.resolve()),
        "head_sha": head,
        "branch": branch,
        "dirty": bool(status),
        "base_sha": args.base_sha,
        "pr_324_sha": args.pr_324_sha,
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in _DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _summary(
    samples_ms: Sequence[float], slo_ms: float | None = None
) -> dict[str, float | int | None]:
    if not samples_ms:
        raise ValueError("at least one latency sample is required")
    ordered = sorted(float(value) for value in samples_ms)

    def percentile(percent: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * percent / 100
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        fraction = rank - low
        return ordered[low] + (ordered[high] - ordered[low]) * fraction

    mean_ms = statistics.fmean(ordered)
    elapsed_seconds = sum(ordered) / 1000
    return {
        "count": len(ordered),
        "mean_ms": mean_ms,
        "median_ms": statistics.median(ordered),
        "p90_ms": percentile(90),
        "p99_ms": percentile(99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stddev_ms": statistics.pstdev(ordered),
        "throughput_requests_per_second": len(ordered) / elapsed_seconds,
        "slo_ms": slo_ms,
        "goodput_requests_per_second": (
            sum(value <= slo_ms for value in ordered) / elapsed_seconds
            if slo_ms is not None
            else None
        ),
    }


def _jsonable(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if depth >= 6:
        return repr(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name), depth + 1)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item, depth + 1) for item in value]
    return repr(value)


def _effective_config_snapshot(config: Any) -> dict:
    fields = getattr(config, "__dataclass_fields__", {})
    return {
        "compute_hash": config.compute_hash(),
        "fields": {
            name: _jsonable(getattr(config, name)) for name in fields if hasattr(config, name)
        },
    }


def _metric_totals(metrics: Sequence[Any]) -> dict[str, float]:
    wanted = {"vllm:prefix_cache_queries", "vllm:prefix_cache_hits"}
    totals = {name: 0.0 for name in wanted}
    found: set[str] = set()
    for metric in metrics:
        name = getattr(metric, "name", None)
        if name in wanted and hasattr(metric, "value"):
            totals[name] += float(metric.value)
            found.add(name)
    missing = wanted - found
    if missing:
        raise EvidenceUnavailableError(
            "required prefix-cache metrics are unavailable: " + ", ".join(sorted(missing))
        )
    return totals


def _metric_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    return {
        key: after.get(key, 0.0) - before.get(key, 0.0) for key in sorted(set(before) | set(after))
    }


def _worker_diagnostics(llm: Any, backend: str) -> list[dict[str, Any]]:
    try:
        snapshots = llm.llm_engine.collective_rpc("get_gr_profile_runtime_stats")
    except Exception as error:
        raise EvidenceUnavailableError(
            f"{backend} Worker named profiling diagnostic RPC is unavailable"
        ) from error
    if not snapshots:
        raise EvidenceUnavailableError("Worker profiling diagnostic returned no snapshots")
    for snapshot in snapshots:
        if snapshot.get("backend") != backend:
            raise EvidenceUnavailableError(
                f"Worker diagnostic backend mismatch: {snapshot.get('backend')!r} != {backend!r}"
            )
        runner_class = snapshot.get("runner_class")
        if not isinstance(runner_class, str):
            raise EvidenceUnavailableError("Worker diagnostic omitted runner_class")
        allowed = _RUNNER_CLASS_ALLOWLIST[backend]
        if runner_class not in allowed:
            raise RuntimeError(
                f"unsupported {backend} ModelRunner class {runner_class!r}; "
                f"required V1 class: {sorted(allowed)}"
            )
        if not isinstance(snapshot.get("memory"), dict):
            raise EvidenceUnavailableError("Worker diagnostic omitted device memory")
        if not isinstance(snapshot.get("beam_graph"), dict):
            raise EvidenceUnavailableError("Worker diagnostic omitted Beam graph state")
    return snapshots


def _validate_graph_evidence(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> None:
    if len(before) != len(after):
        raise EvidenceUnavailableError("Worker diagnostic cardinality changed during profiling")
    required_dispatches = args.runs + (1 if args.profile_dir is not None else 0)
    for old, new in zip(before, after):
        old_graph = old["beam_graph"]
        new_graph = new["beam_graph"]
        dispatch_delta = int(new_graph["full_dispatches"]) - int(old_graph["full_dispatches"])
        if args.execution == "graph":
            captured = {int(width) for width in new_graph["captured_widths"]}
            if args.beam_width not in captured:
                raise EvidenceUnavailableError(
                    f"exact Beam width {args.beam_width} was not captured: {sorted(captured)}"
                )
            if dispatch_delta < required_dispatches:
                raise EvidenceUnavailableError(
                    "Beam FULL graph dispatch evidence is incomplete: "
                    f"delta={dispatch_delta}, required>={required_dispatches}"
                )
        elif dispatch_delta != 0:
            raise EvidenceUnavailableError(
                f"non-graph execution unexpectedly dispatched FULL graph {dispatch_delta} times"
            )


def _validate_prefix_evidence(
    cache_mode: str,
    delta: Mapping[str, float],
    observed: Sequence[Mapping[str, Any]] | None = None,
    expected_requests: int | None = None,
) -> None:
    if observed is not None:
        if not observed:
            raise EvidenceUnavailableError("direct prefix observer recorded no queries")
        if expected_requests is not None and len(observed) != expected_requests:
            raise EvidenceUnavailableError(
                "direct prefix observer request count mismatch: "
                f"observed={len(observed)}, expected={expected_requests}"
            )
        hits = [int(event["computed_tokens"]) for event in observed]
        enabled = [bool(event["cache_enabled"]) for event in observed]
        if cache_mode == "disabled":
            if any(enabled) or any(hits):
                raise EvidenceUnavailableError(
                    "disabled prefix cache returned effective queries: "
                    f"enabled={enabled}, computed={hits}"
                )
            return
        if not all(enabled):
            raise EvidenceUnavailableError(
                f"cache-enabled case observed disabled lookups: {enabled}"
            )
        if cache_mode == "miss" and any(hits):
            raise EvidenceUnavailableError(f"prefix miss returned computed tokens: {hits}")
        if cache_mode == "hit" and any(hit <= 0 for hit in hits):
            raise EvidenceUnavailableError(f"prefix hit was not observed: {hits}")
        return

    queries = delta["vllm:prefix_cache_queries"]
    hits = delta["vllm:prefix_cache_hits"]
    if cache_mode == "disabled":
        if queries != 0 or hits != 0:
            raise EvidenceUnavailableError(
                f"disabled prefix cache reported queries={queries}, hits={hits}"
            )
    elif cache_mode == "miss":
        # The frozen self-advance baseline may query/rebind the prefix during
        # decode inside the same top-level request. A cold request therefore
        # needs a non-perfect hit rate, not necessarily zero total hits.
        if queries <= 0 or hits >= queries:
            raise EvidenceUnavailableError(
                f"prefix miss was not observed: queries={queries}, hits={hits}"
            )
    elif queries <= 0 or hits <= 0:
        raise EvidenceUnavailableError(
            f"prefix hit was not observed: queries={queries}, hits={hits}"
        )


def _prefix_hit_rate(delta: Mapping[str, float]) -> float | None:
    queries = delta["vllm:prefix_cache_queries"]
    return delta["vllm:prefix_cache_hits"] / queries if queries > 0 else None


def _install_inproc_prefix_observer(llm: Any) -> list[dict[str, Any]]:
    """Observe the actual UniProc KV lookup without changing its result."""
    client = getattr(llm.llm_engine, "engine_core", None)
    core = getattr(client, "engine_core", None)
    manager = getattr(getattr(core, "scheduler", None), "kv_cache_manager", None)
    original = getattr(manager, "get_computed_blocks", None)
    if not callable(original):
        raise EvidenceUnavailableError(
            "persistent prefix observer requires an in-process KVCacheManager"
        )
    events: list[dict[str, Any]] = []
    cache_enabled = bool(llm.llm_engine.vllm_config.cache_config.enable_prefix_caching)

    def observed(request):
        blocks, computed_tokens = original(request)
        query_tokens = getattr(request, "num_prompt_tokens", None)
        if query_tokens is None:
            query_tokens = request.num_tokens
        events.append(
            {
                "request_id": str(request.request_id),
                "query_tokens": int(query_tokens),
                "computed_tokens": int(computed_tokens),
                "num_computed_tokens_before": int(request.num_computed_tokens),
                "gr_issued_steps": int(getattr(request, "gr_issued_steps", 0)),
                "persistent": bool(
                    getattr(request, "gr_persistent_request", False)
                    or getattr(request, "gr_persistent_runner_state", False)
                ),
                "cache_enabled": cache_enabled,
            }
        )
        return blocks, computed_tokens

    manager.get_computed_blocks = observed
    return events


def _validate_v1_engine_config(config: Any) -> None:
    use_v2 = getattr(config, "use_v2_model_runner", None)
    if use_v2 is not False:
        raise RuntimeError(
            "Step 0 requires VllmConfig.use_v2_model_runner is False, " f"found {use_v2!r}"
        )


def _preflight_backend(backend: str) -> None:
    try:
        import torch
    except ImportError as error:
        raise EnvironmentNotValidatedError("PyTorch is not installed") from error
    if backend == "gpu":
        if not torch.cuda.is_available():
            raise EnvironmentNotValidatedError("requested CUDA accelerator is unavailable")
        return
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise EnvironmentNotValidatedError("torch-npu is not installed") from error
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise EnvironmentNotValidatedError("requested NPU accelerator is unavailable")


def _driver_accelerator_info(backend: str) -> dict:
    if backend == "gpu":
        command = (
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    else:
        command = ("npu-smi", "info")
    output = _run_command(command)
    if not output:
        raise EvidenceUnavailableError(
            f"cannot collect {backend} driver/device evidence using {command[0]}"
        )
    return {"command": list(command), "output": output}


def _validate_trace_evidence(
    trace_files: Sequence[Path], backend: str, execution: str
) -> dict[str, Any]:
    """Require non-empty trace artifacts containing GR stage boundaries."""
    if not trace_files:
        raise EvidenceUnavailableError("profiling completed without a trace artifact")
    empty = [str(path) for path in trace_files if not path.is_file() or path.stat().st_size == 0]
    if empty:
        raise EvidenceUnavailableError(f"profiling produced empty trace artifacts: {empty}")

    try:
        required = _TRACE_STAGE_MARKERS[(backend, execution)]
    except KeyError as error:
        raise EvidenceUnavailableError(
            f"no trace marker contract for backend={backend!r}, execution={execution!r}"
        ) from error
    observed: set[str] = set()
    needles = {name: name.encode() for name in required}
    for path in trace_files:
        # torch profiler is configured without gzip. Scan in bounded chunks so
        # a multi-GB Chrome trace is not materialized in host memory.
        with path.open("rb") as stream:
            carry = b""
            while chunk := stream.read(1024 * 1024):
                data = carry + chunk
                for name, needle in needles.items():
                    if needle in data:
                        observed.add(name)
                if len(observed) == len(required):
                    break
                carry = data[-128:]
        if len(observed) == len(required):
            break
    missing = sorted(set(required) - observed)
    if missing:
        raise EvidenceUnavailableError(
            "trace omitted required GR stage markers: " + ", ".join(missing)
        )
    return {"required_stage_markers": list(required), "observed_stage_markers": sorted(observed)}


def _memory_breakdown_gate() -> dict[str, Any]:
    """Describe unavailable allocator attribution without inventing values."""
    components = (
        "native_kv_bytes",
        "beam_kv_bytes",
        "workspace_bytes",
        "graph_pool_bytes",
        "gr_state_bytes",
    )
    return {
        "status": "FAIL",
        "measurements": {name: None for name in components},
        "limitation": (
            "ModelRunner V1 currently exposes aggregate device allocator counters, "
            "but no reliable per-owner attribution for Native KV, Beam KV, workspace, "
            "Graph pool, and GR state. Step 0 cannot pass this evidence gate until "
            "backend diagnostics expose those values."
        ),
    }


def _trace_coverage_gate() -> dict[str, str]:
    """State the known process boundary of the current Worker profiler."""
    return {
        "status": "FAIL",
        "limitation": (
            "The vLLM Worker torch profiler captures Worker/device stages but not "
            "EngineCore Scheduler/Request/metadata CPU time in the synchronous GRLLM "
            "client. Step 0 cannot pass until an EngineCore-side or independent host "
            "trace is captured and correlated with this device trace."
        ),
    }


def _process_rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def _engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": args.dtype,
        "max_logprobs": args.beam_width,
        "catalog_path": str(args.catalog),
        "constraint_backend": "constraint_table",
        "beam_graph_enabled": args.execution == "graph",
        "beam_max_width": args.beam_width,
        "beam_max_decode_steps": args.generation_steps,
        "attention_config": {"backend": "CUSTOM"},
        "max_num_seqs": 1,
        "enable_prefix_caching": args.prefix_cache != "disabled",
        "disable_log_stats": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "async_scheduling": args.async_scheduling,
        "enforce_eager": args.execution != "graph",
        "seed": args.seed,
    }
    additional_config: dict[str, Any] = {}
    if args.backend == "npu":
        additional_config["vllm_gr_npu_decode_execution"] = (
            "cascade" if args.execution == "cascade" else "xllm"
        )
    if args.persistent:
        additional_config.update(
            {
                "vllm_gr_persistent_request": True,
                "vllm_gr_persistent_runner_state": True,
                "vllm_gr_scheduling_mode": "async",
            }
        )
    if additional_config:
        kwargs["additional_config"] = additional_config
    return kwargs


def _digest_payload(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_logprob(logprob: Any) -> dict[str, Any]:
    rank = getattr(logprob, "rank", None)
    return {
        "logprob_hex": float(getattr(logprob, "logprob")).hex(),
        "rank": int(rank) if rank is not None else None,
        "decoded_token": getattr(logprob, "decoded_token", None),
    }


def _canonical_logprobs(logprobs: Any) -> list[Any]:
    """Normalize FlatLogprobs and list[dict[int, Logprob]] identically."""
    flat_fields = ("start_indices", "end_indices", "token_ids", "logprobs", "ranks")
    if all(hasattr(logprobs, name) for name in flat_fields):
        decoded_tokens = getattr(logprobs, "decoded_tokens", ())
        positions = []
        for start, end in zip(logprobs.start_indices, logprobs.end_indices):
            entries = []
            for index in range(start, end):
                rank = logprobs.ranks[index]
                entries.append(
                    {
                        "token_id": int(logprobs.token_ids[index]),
                        "logprob_hex": float(logprobs.logprobs[index]).hex(),
                        "rank": int(rank) if rank is not None else None,
                        "decoded_token": (
                            decoded_tokens[index] if index < len(decoded_tokens) else None
                        ),
                    }
                )
            positions.append(sorted(entries, key=lambda item: item["token_id"]))
        return positions

    positions = []
    for position in logprobs:
        if position is None:
            positions.append(None)
            continue
        entries = [
            {"token_id": int(token_id), **_canonical_logprob(logprob)}
            for token_id, logprob in position.items()
        ]
        positions.append(sorted(entries, key=lambda item: item["token_id"]))
    return positions


def _output_fingerprints(outputs: Sequence[Any]) -> dict[str, Any]:
    tokens = [
        [[int(token) for token in sequence.tokens] for sequence in output.sequences]
        for output in outputs
    ]
    scores = [
        [float(sequence.cum_logprob).hex() for sequence in output.sequences] for output in outputs
    ]
    logprobs = [
        [_canonical_logprobs(sequence.logprobs) for sequence in output.sequences]
        for output in outputs
    ]
    full_abi = [
        {
            "sequences": [
                {
                    "tokens": [int(token) for token in sequence.tokens],
                    "cum_logprob_hex": float(sequence.cum_logprob).hex(),
                    "logprobs": _canonical_logprobs(sequence.logprobs),
                    "text": sequence.text,
                    "finish_reason": sequence.finish_reason,
                    "stop_reason": _jsonable(sequence.stop_reason),
                }
                for sequence in output.sequences
            ]
        }
        for output in outputs
    ]
    return {
        "token_digest": _digest_payload(tokens),
        "score_digest": _digest_payload(scores),
        "logprob_digest": _digest_payload(logprobs),
        "full_abi_digest": _digest_payload(full_abi),
        "parent_lineage_available": False,
        "limitation": "current BeamSearchSequence output ABI does not expose parent lineage",
    }


def _prepare_prompt(llm: Any, prompt_path: Path, input_len: int) -> tuple[list[dict], int]:
    tokenizer = llm.get_tokenizer()
    source_ids = tokenizer.encode(prompt_path.read_text(encoding="utf-8"), add_special_tokens=False)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise RuntimeError("newline must encode to exactly one token")
    target = input_len - 1
    prompt_ids = (
        source_ids[-target:]
        if len(source_ids) >= target
        else [int(newline_ids[0])] * (target - len(source_ids)) + source_ids
    )
    end_token_id = int(tokenizer.convert_tokens_to_ids("<|sid_end|>"))
    return [{"prompt_token_ids": prompt_ids}], end_token_id


def _catalog_triple_token_ids(catalog_path: Path, tokenizer: Any) -> set[tuple[int, int, int]]:
    document = json.loads(catalog_path.read_text(encoding="utf-8"))
    triples = document.get("triples") if isinstance(document, dict) else None
    if not isinstance(triples, list) or not triples:
        raise EvidenceUnavailableError("constraint catalog has no triples")
    result: set[tuple[int, int, int]] = set()
    for index, triple in enumerate(triples):
        if not isinstance(triple, dict) or set(triple) != {"a", "b", "c"}:
            raise EvidenceUnavailableError(f"invalid constraint triple #{index}")
        token_ids = tuple(
            int(tokenizer.convert_tokens_to_ids(triple[field]))
            for field in ("a", "b", "c")
        )
        if any(token_id < 0 for token_id in token_ids):
            raise EvidenceUnavailableError(
                f"constraint triple #{index} contains an unknown tokenizer token"
            )
        result.add(token_ids)
    return result


def _run(args: argparse.Namespace) -> dict:
    _require_v1_model_runner()
    # This must precede every vllm_gr/vllm import so missing accelerator
    # dependencies receive the explicit environment-not-validated outcome.
    _preflight_backend(args.backend)
    repo_root = Path(__file__).resolve().parents[2]
    started_at = datetime.now(timezone.utc).isoformat()
    host_before_engine = _process_rss_bytes()
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": started_at,
        "implementation": args.implementation,
        "source": _source_metadata(repo_root, args),
        "command": [sys.executable, *sys.argv],
        "requested_config": _jsonable(vars(args)),
        "dependencies": _dependency_versions(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "accelerator_cli": _driver_accelerator_info(args.backend),
        },
        "artifacts": {
            "catalog": {
                "path": str(args.catalog.resolve()),
                "sha256": _sha256_file(args.catalog),
            },
            "prompt": {
                "path": str(args.prompt.resolve()),
                "sha256": _sha256_file(args.prompt),
            },
        },
        "engine_kwargs": _jsonable(_engine_kwargs(args)),
    }

    from vllm_gr.config import GRConfig
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.plugin import initialize_runtime
    from vllm_gr.sampling_params import BeamSearchParams

    initialize_runtime()
    kwargs = _engine_kwargs(args)
    trace_files_before: set[Path] = set()
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        trace_files_before = set(args.profile_dir.rglob("*"))
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile_dir.resolve()),
            "torch_profiler_with_stack": True,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": True,
            "torch_profiler_use_gzip": False,
            "ignore_frontend": False,
        }

    worker_startup: list[dict] = []
    worker_before: list[dict] = []
    worker_after: list[dict] = []
    samples_ms: list[float] = []
    fingerprints: list[dict[str, Any]] = []
    host_engine_ready = 0
    host_after_runs = 0
    effective_config: dict = {}
    model_identity: dict = {}
    prefix_before: dict[str, float] = {}
    prefix_after: dict[str, float] = {}
    prefix_observer_events: list[dict[str, Any]] = []
    measured_prefix_events: list[dict[str, Any]] = []

    with GRLLM(model=args.model, **kwargs) as llm:
        config = llm.llm_engine.vllm_config
        _validate_v1_engine_config(config)
        actual_platform = str(config.device_config.device_type)
        expected_platform = "cuda" if args.backend == "gpu" else "npu"
        if actual_platform != expected_platform:
            raise RuntimeError(
                f"requested backend {args.backend!r}, detected device {actual_platform!r}"
            )
        host_engine_ready = _process_rss_bytes()
        # Fail before any workload execution if the named safe-RPC hook,
        # backend identity, runner class, or device-memory evidence is absent.
        worker_startup = _worker_diagnostics(llm, args.backend)
        effective_config = _effective_config_snapshot(config)
        effective_config["use_v2_model_runner"] = config.use_v2_model_runner
        resolved_gr = GRConfig.from_vllm_config(config)
        effective_config["vllm_gr"] = _jsonable(resolved_gr)
        hf_config = getattr(config.model_config, "hf_config", None)
        model_identity = _model_identity(
            args.model,
            args.model_digest,
            getattr(hf_config, "_commit_hash", None),
        )
        if args.persistent:
            prefix_observer_events = _install_inproc_prefix_observer(llm)

        prompts, end_token_id = _prepare_prompt(llm, args.prompt, args.input_len)
        valid_triple_ids = _catalog_triple_token_ids(args.catalog, llm.get_tokenizer())
        params = BeamSearchParams(
            beam_width=args.beam_width,
            max_tokens=args.generation_steps + 2,
            temperature=0.0,
            ignore_eos=False,
            begin_token="<|sid_begin|>",
            end_token="<|sid_end|>",
        )

        def reset_for_miss() -> None:
            if args.prefix_cache == "miss" and not llm.reset_prefix_cache():
                raise RuntimeError("prefix-cache reset was rejected")

        def run_one() -> tuple[float, dict[str, Any]]:
            reset_for_miss()
            start_ns = time.perf_counter_ns()
            outputs = llm.beam_search(prompts, params)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            if len(outputs) != 1 or len(outputs[0].sequences) != args.beam_width:
                raise AssertionError("OneRec output has the wrong request/beam count")
            expected_len = args.input_len + args.generation_steps + 1
            if any(
                len(sequence.tokens) != expected_len or sequence.tokens[-1] != end_token_id
                for sequence in outputs[0].sequences
            ):
                raise AssertionError("OneRec output violated the fixed generation contract")
            invalid = [
                tuple(int(token) for token in sequence.tokens[-4:-1])
                for sequence in outputs[0].sequences
                if tuple(int(token) for token in sequence.tokens[-4:-1])
                not in valid_triple_ids
            ]
            if invalid:
                raise AssertionError(
                    f"OneRec output contains {len(invalid)} triples outside the constraint catalog"
                )
            fingerprint = _output_fingerprints(outputs)
            fingerprint["constraint_validation"] = {
                "catalog_triples": len(valid_triple_ids),
                "valid_sequences": len(outputs[0].sequences),
                "invalid_sequences": 0,
            }
            return elapsed_ms, fingerprint

        if args.prefix_cache == "hit":
            if not llm.reset_prefix_cache():
                raise RuntimeError("prefix-cache reset was rejected before hit priming")
            run_one()
        for _ in range(args.warmups):
            run_one()
        prefix_before = _metric_totals(llm.get_metrics())
        prefix_observer_start = len(prefix_observer_events)
        # Graph dispatch deltas must start immediately before measured runs;
        # startup capture, hit priming, and warmups are not measurement proof.
        worker_before = _worker_diagnostics(llm, args.backend)
        for _ in range(args.runs):
            elapsed_ms, output_fingerprints = run_one()
            samples_ms.append(elapsed_ms)
            fingerprints.append(output_fingerprints)
        prefix_after = _metric_totals(llm.get_metrics())
        measured_prefix_events = prefix_observer_events[prefix_observer_start:]

        if args.profile_dir is not None:
            llm.start_profile(
                f"step0_{args.implementation}_{args.backend}_{args.execution}_{args.prefix_cache}"
            )
            try:
                run_one()
            finally:
                llm.stop_profile()
        worker_after = _worker_diagnostics(llm, args.backend)
        host_after_runs = _process_rss_bytes()

    prefix_delta = _metric_delta(prefix_before, prefix_after)
    _validate_prefix_evidence(
        args.prefix_cache,
        prefix_delta,
        measured_prefix_events if args.persistent else None,
        args.runs if args.persistent else None,
    )
    _validate_graph_evidence(worker_before, worker_after, args)
    unique_fingerprints = {
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
        for fingerprint in fingerprints
    }
    if len(unique_fingerprints) != 1:
        raise AssertionError("output ABI fingerprint was not repeatable")
    output_fingerprints = json.loads(next(iter(unique_fingerprints)))
    trace_files: list[Path] = []
    trace_evidence: dict[str, Any] | None = None
    if args.profile_dir is not None:
        trace_files = sorted(
            path.resolve()
            for path in set(args.profile_dir.rglob("*")) - trace_files_before
            if path.is_file()
        )
        trace_evidence = _validate_trace_evidence(trace_files, args.backend, args.execution)
    memory_breakdown = _memory_breakdown_gate()
    trace_coverage = _trace_coverage_gate()
    result.update(
        {
            "status": "complete",
            "exit_gate": "FAIL",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "effective_config": effective_config,
            "artifacts": {**result["artifacts"], "model": model_identity},
            "runtime": {
                "worker_startup": worker_startup,
                "worker_before": worker_before,
                "worker_after": worker_after,
                "prefix_cache_metric_delta": prefix_delta,
                "prefix_cache_hit_rate": _prefix_hit_rate(prefix_delta),
                "prefix_cache_observer": (
                    {
                        "source": "inproc_kv_cache_manager.get_computed_blocks",
                        "all_events": prefix_observer_events,
                        "measured_events": measured_prefix_events,
                        "measured_query_tokens": sum(
                            event["query_tokens"] for event in measured_prefix_events
                        ),
                        "measured_hit_tokens": sum(
                            event["computed_tokens"] for event in measured_prefix_events
                        ),
                    }
                    if args.persistent
                    else None
                ),
            },
            "memory": {
                "host_rss_before_engine_bytes": host_before_engine,
                "host_rss_engine_ready_bytes": host_engine_ready,
                "host_rss_after_runs_bytes": host_after_runs,
                "host_rss_after_shutdown_bytes": _process_rss_bytes(),
                "component_breakdown": memory_breakdown,
            },
            "evidence_gates": {
                "memory_component_breakdown": memory_breakdown,
                "engine_core_trace_coverage": trace_coverage,
            },
            "e2e": {
                "samples_ms": samples_ms,
                "summary": _summary(samples_ms, args.slo_ms),
                "output_fingerprints": output_fingerprints,
            },
            "profile": {
                "enabled": args.profile_dir is not None,
                "directory": str(args.profile_dir.resolve()) if args.profile_dir else None,
                "trace_files": [str(path) for path in trace_files],
                "stage_marker_evidence": trace_evidence,
                "coverage_gate": trace_coverage,
            },
        }
    )
    return result


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _failure_result(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    environment_failure = isinstance(error, EnvironmentNotValidatedError)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        source: Any = _source_metadata(repo_root, args)
    except Exception as source_error:  # Keep the primary failure authoritative.
        source = {
            "repo_root": str(repo_root.resolve()),
            "collection_error": {
                "type": type(source_error).__name__,
                "message": str(source_error),
            },
            "base_sha": args.base_sha,
            "pr_324_sha": args.pr_324_sha,
        }
    return {
        "schema_version": 1,
        "status": "failed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": args.implementation,
        "source": source,
        "command": [sys.executable, *sys.argv],
        "requested_config": _jsonable(vars(args)),
        "dependencies": _dependency_versions(),
        "validation_status": ("ENVIRONMENT_NOT_VALIDATED" if environment_failure else "FAIL"),
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        result = _run(args)
    except Exception as error:
        result = _failure_result(args, error)
        _write_result(args.output_json, result)
        raise
    _write_result(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
