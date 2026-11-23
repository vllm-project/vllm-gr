#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run paired cold/hit OneRec beam-search measurements through GRLLM offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from benchmarks.metrics import distribution_ms


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_model(model: str, revision: str | None) -> tuple[Path, str]:
    """Use the same model files for the tokenizer, constraint compiler and GRLLM."""
    path = Path(model)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model, revision=revision or None))
    path = path.resolve()
    if (
        path.parent.name == "snapshots"
        and len(path.name) == 40
        and set(path.name) <= set("0123456789abcdef")
    ):
        if Path(model).is_dir() and revision and revision != path.name:
            raise ValueError(f"model snapshot {path.name} does not match {revision}")
        return path, path.name

    # A local directory name cannot identify its weights. Hash its regular
    # files in pathname order, excluding hidden VCS and download metadata.
    files = sorted(
        file
        for file in path.rglob("*")
        if not any(
            part.startswith(".") or part == "__pycache__" for part in file.relative_to(path).parts
        )
        and file.is_file()
    )
    if not files:
        raise ValueError(f"empty model directory: {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(file_sha256(file)))
    identity = f"sha256:{digest.hexdigest()}"
    if revision and revision != identity:
        raise ValueError(f"local model identity {identity} does not match {revision}")
    return path, identity


def selection_digests(ids: list[str], tokens: list[list[int]]) -> dict[str, str]:
    """Digest the ordered, pre-encoded prompts actually handed to the engine."""

    def digest(value: object) -> str:
        return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()

    return {
        "sample_ids_sha256": digest(ids),
        "workload_sha256": digest(list(zip(ids, tokens, strict=True))),
    }


def sample_prompts(
    dataset_path: Path, task: str, tokenizer: Any, input_length: int, count: int
) -> tuple[list[str], list[list[int]], int]:
    """Select original parquet rows and pre-encode the exact prompts used by the engine."""
    import pandas as pd

    from benchmarks.open_one_rec.open_one_rec_loader import get_loader

    # Legacy sample parquet files discard the source indices on write.
    # Select from the original index, including files with non-default indices.
    source_index = pd.read_parquet(dataset_path, columns=["metadata"]).index
    if count > len(source_index):
        raise ValueError(f"requested {count} {task} samples, source contains {len(source_index)}")
    selected = source_index.to_series()
    if count < len(selected):
        selected = selected.sample(n=count, random_state=42)
    loader = get_loader(
        task,
        str(dataset_path.parent.parent),
        tokenizer=tokenizer,
        custom_input_len=input_length - 1,
    )
    records = loader.load_data(split="test", sample_ids=selected.tolist())
    if len(records) != count:
        raise ValueError(f"requested {count} {task} samples, dataset returned {len(records)}")
    ids = list(records)
    random.Random(0).shuffle(ids)
    prompts = [tokenizer.encode(records[row]["prompt"], add_special_tokens=False) for row in ids]
    return (
        [f"{task}_{row}" for row in ids],
        prompts,
        int(loader.task_config["generation_config"]["max_new_tokens"]),
    )


def runtime_source() -> dict[str, str | None]:
    import vllm_gr

    package = Path(vllm_gr.__file__).resolve()
    # A wheel outside the measured checkout cannot be stamped with that
    # checkout's SHA. Direct standalone runs can still record its origin.
    root = package.parent.parent
    sha = None
    if (root / ".git").exists():
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    return {"package_path": str(package), "git_sha": sha}


def resolve_beam_search(llm: Any) -> Any:
    """Select the public Beam method exposed by the loaded runtime.

    Older checkouts expose ``beam_search_v1``. Newer checkouts expose
    ``beam_search``. Prefer the older public method when it is available.
    """
    method = getattr(llm, "beam_search_v1", None)
    return method if method is not None else llm.beam_search


def prefix_cache_counters(llm: Any) -> tuple[int, int] | None:
    """Read queried and reused token totals for the single offline engine."""
    queries = "vllm:prefix_cache_queries"
    hits = "vllm:prefix_cache_hits"
    counters: dict[str, int] = {}
    try:
        for metric in llm.get_metrics():
            if metric.name not in (queries, hits):
                continue
            # Multiple engines or an incompatible snapshot cannot identify the request.
            if metric.name in counters or type(metric.value) is not int or metric.value < 0:
                return None
            counters[metric.name] = metric.value
        return counters[queries], counters[hits]
    except (AttributeError, KeyError, TypeError, AssertionError, RuntimeError):
        # Historical runtimes may not expose these counters or enable their reader.
        return None


def measure_e2e(llm: Any, prompt_token_ids: list[int], params: Any) -> dict[str, Any]:
    """Measure the production trend with no internal monkeypatches or profiler.

    Time one ``beam_search`` call through public output decoding. V1 owns
    Prefill/Decode inside EngineCore, so this frontend reports E2E only.
    """
    before = prefix_cache_counters(llm)
    started = time.perf_counter_ns()
    outputs = resolve_beam_search(llm)(
        [{"prompt_token_ids": prompt_token_ids}], params, concurrency_limit=1, output_mode="full"
    )
    for output in outputs:
        for sequence in output.sequences:
            _ = sequence.text
    finished = time.perf_counter_ns()
    after = prefix_cache_counters(llm)
    cached_tokens = None
    if before is not None and after is not None:
        queries, hits = after[0] - before[0], after[1] - before[1]
        # GR adds the SID begin token to the engine prompt before cache lookup.
        # A matching query delta distinguishes a real zero from stale counters.
        prompt_length = len(prompt_token_ids) + int(params.begin_token is not None)
        if queries == prompt_length and 0 <= hits <= queries:
            cached_tokens = hits
    if len(outputs) != 1:
        raise RuntimeError(f"expected one offline output, received {len(outputs)}")
    output = outputs[0]
    if len(output.sequences) != params.beam_width:
        raise ValueError(f"expected {params.beam_width} beams, received {len(output.sequences)}")
    output_tokens = 0
    for beam in output.sequences:
        # Full BeamSearchSequence.tokens includes the submitted prompt, and
        # its suffix includes the SID begin/end tokens. Text was materialized
        # inside the timer; bookkeeping stays outside it.
        length = len(beam.tokens) - len(prompt_token_ids)
        if length != params.max_tokens:
            raise ValueError(f"expected {params.max_tokens} generated tokens, received {length}")
        output_tokens += length
    return {
        "e2e_ms": (finished - started) / 1_000_000.0,
        "cached_tokens": cached_tokens,
        "returned_beams": len(output.sequences),
        "aggregate_output_tokens": output_tokens,
    }


@contextmanager
def create_v1_llm(
    args: argparse.Namespace, max_tokens: int, artifact_digests: dict[str, str] | None = None
) -> Iterator[Any]:
    """Keep the canonical artifact alive until all workers have shut down."""
    from vllm_gr.constraints.artifact_builder import build_single_trie_artifact
    from vllm_gr.entrypoints.gr import GRLLM

    steps = max_tokens - 2
    if steps not in (1, 2, 3):
        raise ValueError("Beam V1 requires 3/4/5 public output tokens including SID boundaries")
    with TemporaryDirectory(prefix="gr-ab-v1-") as directory:
        artifact = Path(directory) / "constraints.safetensors"
        digest, tokenizer_digest = build_single_trie_artifact(
            Path(args.model), args.catalog.resolve(), artifact
        )
        if artifact_digests is not None:
            artifact_digests.update(artifact=digest, tokenizer=tokenizer_digest)
        config = {
            "worker_decision": True,
            "beam_width": [args.beam_width] * steps,
            "max_decode_steps": steps,
            "constraint_mode": "table",
            "constraint_table": {
                "backend": "cuda",
                "max_top_k": args.beam_width,
                "path": str(artifact.resolve()),
                "format": "constraint_table_v1",
                "digests": {"artifact": digest, "tokenizer": tokenizer_digest},
            },
            "candidate_width": [args.beam_width] * steps,
        }
        model_len = max(8192, args.input_length + max_tokens)
        with GRLLM(
            model=args.model,
            trust_remote_code=True,
            attention_config={"backend": "CUSTOM"},
            gr=config,
            async_scheduling=True,
            disable_log_stats=False,
            max_num_seqs=args.beam_width,
            max_logprobs=args.beam_width,
            max_num_batched_tokens=max(16384, model_len),
            max_model_len=model_len,
            gpu_memory_utilization=0.90,
        ) as llm:
            yield llm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory containing the OneRec dataset"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Constraint table triples document for constrained beam decoding.",
    )
    parser.add_argument("--model", default="OpenOneRec/OneRec-1.7B")
    parser.add_argument("--model-revision", help="Hugging Face revision or local model identity")
    parser.add_argument("--task", default="video")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument(
        "--diagnostic-prompts",
        type=int,
        default=0,
        help="Retired legacy stage diagnostics; accepted for CLI compatibility and ignored.",
    )
    parser.add_argument("--beam-width", type=int, required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument(
        "--beam-engine-driven",
        default=None,
        metavar="{auto,true,false}",
        help="Retired legacy control; only auto is accepted for CLI compatibility",
    )
    args = parser.parse_args()
    if args.beam_engine_driven not in (None, "auto"):
        parser.error("--beam-engine-driven was retired; offline benchmarks use Beam V1")
    if args.diagnostic_prompts:
        print("Legacy frontend stage diagnostics were retired; reporting V1 E2E only.", flush=True)
    if (
        min(args.num_prompts, args.beam_width) < 1
        or args.input_length < 2
        or args.warmup_requests < 0
    ):
        parser.error(
            "num-prompts and beam-width must be positive; input-length must be at least 2 "
            "(one token is reserved for <|sid_begin|>); warmup must be non-negative"
        )
    try:
        if not args.catalog.is_file():
            parser.error(f"--catalog is not a regular file: {args.catalog}")
        catalog_sha256 = file_sha256(args.catalog)
    except OSError as exc:
        parser.error(f"cannot read --catalog {args.catalog}: {exc}")
    dataset_path = (args.data_dir / args.task / f"{args.task}_test.parquet").absolute()
    try:
        dataset_sha256 = file_sha256(dataset_path)
    except OSError as exc:
        parser.error(f"cannot read dataset under --data-dir {args.data_dir}: {exc}")

    import torch
    import vllm
    from transformers import AutoTokenizer

    import vllm_gr
    from vllm_gr.sampling_params import BeamSearchParams

    started_at = datetime.now(UTC)
    source = runtime_source()
    model_id = args.model
    model_path, model_revision = resolve_model(model_id, args.model_revision)
    args.model = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sample_size = max(args.num_prompts, args.warmup_requests)
    request_ids, prompt_token_ids_list, max_tokens = sample_prompts(
        dataset_path, args.task, tokenizer, args.input_length, sample_size
    )
    params = BeamSearchParams(
        beam_width=args.beam_width, max_tokens=max_tokens, temperature=0.0, ignore_eos=True
    )
    params.begin_token = "<|sid_begin|>"
    params.end_token = "<|sid_end|>"
    # GR beam search prepends begin_token after the dataset has already shaped
    # the prompt to input_length. Reserve both that token and the full requested
    # generation budget at the 8K/10K boundaries.
    engine_max_model_len = max(8192, args.input_length + max_tokens)

    miss_samples: list[dict[str, Any]] = []
    hit_samples: list[dict[str, Any]] = []
    artifact_digests: dict[str, str] = {}
    with create_v1_llm(args, max_tokens, artifact_digests) as llm:
        effective = llm.llm_engine.vllm_config
        prefix_caching = effective.cache_config.enable_prefix_caching
        chunked_prefill = effective.scheduler_config.enable_chunked_prefill
        for prompt_token_ids in prompt_token_ids_list[: args.warmup_requests]:
            if not llm.reset_prefix_cache():
                raise RuntimeError("prefix cache reset failed during warmup")
            measure_e2e(llm, prompt_token_ids, params)

        for index, prompt_token_ids in enumerate(
            prompt_token_ids_list[: args.num_prompts], start=1
        ):
            if not llm.reset_prefix_cache():
                raise RuntimeError(f"prefix cache reset failed before measured request {index}")
            miss_samples.append(measure_e2e(llm, prompt_token_ids, params))
            hit_samples.append(measure_e2e(llm, prompt_token_ids, params))

    finished_at = datetime.now(UTC)
    fields = (
        "e2e_ms",
        "prefill_ms",
        "decode_ms",
        "total_beam_ms",
    )
    raw = {
        "schema_version": "vllm-gr.offline.raw.v3",
        "execution_mode": "offline",
        "api": "GRLLM.beam_search",
        "measurement_mode": "beam-search-v1-e2e",
        "instrumentation": {
            "canonical": "top-level perf_counter_ns only; no profiler or internal monkeypatch",
            "diagnostic": "unavailable: legacy frontend stage loop retired",
            "canonical_contaminated_by_diagnostic": bool(
                os.environ.get("VLLM_GR_LIGHTWEIGHT_TIMING") == "1"
            ),
        },
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "model": model_id,
        "model_revision": model_revision,
        "model_path": args.model,
        "runtime": {
            "torch_version": torch.__version__,
            "vllm_version": vllm.__version__,
            "vllm_gr_version": vllm_gr.__version__,
        },
        "runtime_source": source,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "selection": selection_digests(
            request_ids[: args.num_prompts],
            prompt_token_ids_list[: args.num_prompts],
        ),
        "warmup_workload_sha256": selection_digests(
            request_ids[: args.warmup_requests],
            prompt_token_ids_list[: args.warmup_requests],
        )["workload_sha256"],
        "task": args.task,
        "num_prompts": args.num_prompts,
        "warmup_requests": args.warmup_requests,
        "beam_width": args.beam_width,
        "input_length": args.input_length,
        "max_tokens": max_tokens,
        "max_concurrency": 1,
        "completed": len(miss_samples),
        "failed": 0,
        "input_lens": [len(ids) for ids in prompt_token_ids_list[: args.num_prompts]],
        "output_lens": [int(sample["aggregate_output_tokens"]) for sample in miss_samples],
        "duration_seconds": sum(float(sample["e2e_ms"]) for sample in miss_samples) / 1000.0,
        "pair_duration_seconds": sum(
            float(miss["e2e_ms"]) + float(hit["e2e_ms"])
            for miss, hit in zip(miss_samples, hit_samples, strict=True)
        )
        / 1000.0,
        "miss": {
            "e2e_ms": [float(sample["e2e_ms"]) for sample in miss_samples],
            "cached_tokens": [sample["cached_tokens"] for sample in miss_samples],
        },
        "hit": {
            "e2e_ms": [float(sample["e2e_ms"]) for sample in hit_samples],
            "cached_tokens": [sample["cached_tokens"] for sample in hit_samples],
        },
        "returned_beams": [int(sample["returned_beams"]) for sample in miss_samples],
        "aggregate_output_tokens": sum(
            int(sample["aggregate_output_tokens"]) for sample in miss_samples
        ),
        "distributions": {
            f"e2e_ms_{cache_state}": distribution_ms(
                [float(sample["e2e_ms"]) for sample in samples]
            )
            for cache_state, samples in (("miss", miss_samples), ("hit", hit_samples))
        },
        "diagnostic": {
            "num_prompts": 0,
            "miss": {field: [] for field in fields},
            "hit": {field: [] for field in fields},
            "distributions": {},
            "phase_definition": {"version": "retired-legacy-frontend"},
        },
        "phase_definition": {
            "version": "vllm-gr-beam-v1-e2e-v1",
            "e2e": "GRLLM.beam_search call through all public beam text materialization",
            "cache_protocol": "reset cache and measure two identical prompts with reuse counters",
            "internal_stages": "not measured by this frontend E2E benchmark",
        },
        "engine_config": {
            "attention_backend": "CUSTOM",
            "beam_graph_enabled": True,
            "beam_max_width": args.beam_width,
            "max_logprobs": args.beam_width,
            "max_num_seqs": args.beam_width,
            "max_num_batched_tokens": max(16384, engine_max_model_len),
            "max_model_len": engine_max_model_len,
            "scheduling_policy": "fcfs",
            "gpu_memory_utilization": 0.90,
            "catalog_path": str(args.catalog),
            "catalog_sha256": catalog_sha256,
            "constraint_artifact_sha256": artifact_digests["artifact"].removeprefix("sha256:"),
            "tokenizer_sha256": artifact_digests["tokenizer"].removeprefix("sha256:"),
            "constraint_backend": "canonical_cuda",
            "single_request": True,
            "async_scheduling": True,
            "enable_prefix_caching": prefix_caching,
            "enable_chunked_prefill": chunked_prefill,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "beam_width": args.beam_width,
                "input_length": args.input_length,
                "e2e_miss_p50_ms": raw["distributions"]["e2e_ms_miss"]["p50"],
                "e2e_hit_p50_ms": raw["distributions"]["e2e_ms_hit"]["p50"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
