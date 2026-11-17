#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run paired cold/hit OneRec beam-search measurements through GRLLM offline."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def distribution_ms(values: list[float]) -> dict[str, float | str]:
    """Summarize millisecond samples using the same percentiles as the dashboard."""
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(float(value) for value in values)

    def percentile(percent: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percent / 100.0
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "unit": "ms",
    }


def measure_e2e(llm: Any, prompt_token_ids: list[int], params: Any) -> dict[str, Any]:
    """Measure the production trend with no internal monkeypatches or profiler.

    Time one ``beam_search_v1`` call through public output decoding. V1 owns
    Prefill/Decode inside EngineCore, so this frontend reports E2E only.
    """
    started = time.perf_counter_ns()
    outputs = llm.beam_search_v1(
        [{"prompt_token_ids": prompt_token_ids}], params, concurrency_limit=1, output_mode="full"
    )
    for output in outputs:
        for sequence in output.sequences:
            _ = sequence.text
    finished = time.perf_counter_ns()
    if len(outputs) != 1:
        raise RuntimeError(f"expected one offline output, received {len(outputs)}")
    output = outputs[0]
    return {
        "e2e_ms": (finished - started) / 1_000_000.0,
        "returned_beams": len(output.sequences),
        "aggregate_output_tokens": len(output.sequences) * int(params.max_tokens),
    }


@contextmanager
def create_v1_llm(args: argparse.Namespace, max_tokens: int) -> Iterator[Any]:
    """Keep the canonical artifact alive until all workers have shut down."""
    from examples.offline_inference.beam_search.offline_beam_search_v1 import (
        build_artifact,
    )
    from vllm_gr.entrypoints.gr import GRLLM

    steps = max_tokens - 2
    if steps not in (1, 2, 3):
        raise ValueError("Beam V1 requires 3/4/5 public output tokens including SID boundaries")
    with TemporaryDirectory(prefix="gr-ab-v1-") as directory:
        artifact = Path(directory) / "constraints.safetensors"
        digest, tokenizer_digest = build_artifact(
            Path(args.model), args.catalog.resolve(), artifact
        )
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
    parser.add_argument("--data-dir", type=Path, default=Path("/opt/vllm-gr/data"))
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(__file__).resolve().parent / "constraints" / "video_constraint_triples.json",
        help="Constraint table triples document for constrained beam decoding.",
    )
    parser.add_argument("--model", default="OpenOneRec/OneRec-1.7B")
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

    from transformers import AutoTokenizer

    from benchmarks.open_one_rec.open_one_rec_dataset import OneRecDataset
    from vllm_gr.sampling_params import BeamSearchParams

    started_at = datetime.now(UTC)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sample_size = max(args.num_prompts, args.warmup_requests)
    dataset = OneRecDataset(
        dataset_path=str(args.data_dir),
        task_types=[args.task],
        tokenizer=tokenizer,
        custom_input_len=args.input_length - 1,
        sample_size=sample_size,
    )
    requests = dataset.sample(
        num_requests=sample_size,
        tokenizer=tokenizer,
        no_oversample=True,
    )
    if len(requests) < sample_size:
        raise RuntimeError(f"requested {sample_size} samples, dataset returned {len(requests)}")

    # Pre-encode prompts outside the timed region so GRLLM.beam_search_v1's
    # _preprocess_cmpl does not re-encode inside the perf_counter window.
    prompt_token_ids_list = [
        tokenizer.encode(request.prompt, add_special_tokens=False) for request in requests
    ]

    max_tokens = int(requests[0].expected_output_len)
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
    with create_v1_llm(args, max_tokens) as llm:
        effective = llm.llm_engine.vllm_config
        prefix_caching = effective.cache_config.enable_prefix_caching
        chunked_prefill = effective.scheduler_config.enable_chunked_prefill
        for prompt_token_ids in prompt_token_ids_list[: args.warmup_requests]:
            if not llm.reset_prefix_cache():
                raise RuntimeError("prefix cache reset failed during warmup")
            measure_e2e(llm, prompt_token_ids, params)

        measured_requests = requests[: args.num_prompts]
        for index, (request, prompt_token_ids) in enumerate(
            zip(measured_requests, prompt_token_ids_list[: args.num_prompts]), start=1
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
        "schema_version": "vllm-gr.offline.raw.v2",
        "execution_mode": "offline",
        "api": "GRLLM.beam_search_v1",
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
        "model": args.model,
        "task": args.task,
        "num_prompts": args.num_prompts,
        "warmup_requests": args.warmup_requests,
        "beam_width": args.beam_width,
        "input_length": args.input_length,
        "max_tokens": max_tokens,
        "max_concurrency": 1,
        "completed": len(miss_samples),
        "failed": 0,
        "input_lens": [int(request.prompt_len) for request in measured_requests],
        "output_lens": [int(sample["aggregate_output_tokens"]) for sample in miss_samples],
        "duration_seconds": sum(float(sample["e2e_ms"]) for sample in miss_samples) / 1000.0,
        "pair_duration_seconds": sum(
            float(miss["e2e_ms"]) + float(hit["e2e_ms"])
            for miss, hit in zip(miss_samples, hit_samples, strict=True)
        )
        / 1000.0,
        "miss": {"e2e_ms": [float(sample["e2e_ms"]) for sample in miss_samples]},
        "hit": {"e2e_ms": [float(sample["e2e_ms"]) for sample in hit_samples]},
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
            "e2e": "GRLLM.beam_search_v1 call through all public beam text materialization",
            "cache_protocol": "paired cold miss followed by identical warm hit",
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
