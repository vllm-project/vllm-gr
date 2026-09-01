# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-model CUDA P->D latency benchmark for the OneRec Beam workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mode", choices=("legacy", "fast", "compare"), required=True)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--profile-dir", type=Path)
    return parser.parse_args()


def _summary(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "std_ms": float(np.std(values)),
    }


def _set_fast_path(worker: object, enabled: bool) -> bool:
    """Benchmark-only callable RPC; requires trusted local pickle transport."""

    runner = getattr(worker, "model_runner")
    runner._enable_cuda_prefill_postprocess_fast_path = bool(enabled)
    return bool(runner._enable_cuda_prefill_postprocess_fast_path)


def main() -> None:
    args = _parse_args()
    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("runs must be positive and warmups non-negative")

    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.plugin import initialize_runtime
    from vllm_gr.sampling_params import BeamSearchParams

    width = 128
    input_len = 1024
    sid_steps = 3
    initialize_runtime()
    params = BeamSearchParams(
        beam_width=width,
        max_tokens=5,
        temperature=0.0,
        ignore_eos=False,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
    )
    profiler_config = None
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile_dir.resolve()),
            "torch_profiler_with_stack": True,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": False,
            "torch_profiler_use_gzip": False,
            "ignore_frontend": True,
        }
    with GRLLM(
        model=args.model,
        trust_remote_code=True,
        max_logprobs=width,
        catalog_path=args.catalog,
        constraint_backend="constraint_table",
        beam_max_width=width,
        beam_max_decode_steps=sid_steps,
        attention_config={"backend": "CUSTOM"},
        max_num_seqs=1,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.7,
        async_scheduling=False,
        profiler_config=profiler_config,
    ) as llm:
        tokenizer = llm.get_tokenizer()
        source_ids = tokenizer.encode(
            args.prompt.read_text(encoding="utf-8"), add_special_tokens=False
        )
        target = input_len - 1
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise RuntimeError("newline must encode to exactly one token")
        prompt_ids = (
            source_ids[-target:]
            if len(source_ids) >= target
            else [int(newline_ids[0])] * (target - len(source_ids)) + source_ids
        )
        prompts = [{"prompt_token_ids": prompt_ids}]
        end_token_id = tokenizer.convert_tokens_to_ids("<|sid_end|>")

        def run_one() -> tuple[float, str]:
            if not llm.reset_prefix_cache():
                raise RuntimeError("prefix-cache reset was rejected")
            started = time.perf_counter_ns()
            output = llm.beam_search(prompts, params)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            sequences = output[0].sequences
            if len(sequences) != width or any(
                len(sequence.tokens) != input_len + sid_steps + 1
                or sequence.tokens[-1] != end_token_id
                for sequence in sequences
            ):
                raise AssertionError("OneRec output violated the B1/L1024/BW128/SID3 contract")
            canonical = [[int(token) for token in sequence.tokens] for sequence in sequences]
            digest = hashlib.sha256(
                json.dumps(canonical, separators=(",", ":")).encode()
            ).hexdigest()
            return elapsed_ms, digest

        modes = ("legacy", "fast") if args.mode == "compare" else (args.mode,)
        samples: dict[str, list[float]] = {mode: [] for mode in modes}
        digests: dict[str, set[str]] = {mode: set() for mode in modes}
        for mode in modes:
            enabled = mode == "fast"
            reply = llm.llm_engine.collective_rpc(_set_fast_path, args=(enabled,))
            if reply != [enabled]:
                raise RuntimeError(f"failed to set fast path to {enabled}: {reply}")
            for _ in range(args.warmups):
                run_one()
        for iteration in range(args.runs):
            ordered_modes = modes if iteration % 2 == 0 else tuple(reversed(modes))
            for mode in ordered_modes:
                enabled = mode == "fast"
                reply = llm.llm_engine.collective_rpc(_set_fast_path, args=(enabled,))
                if reply != [enabled]:
                    raise RuntimeError(f"failed to set fast path to {enabled}: {reply}")
                elapsed_ms, digest = run_one()
                samples[mode].append(elapsed_ms)
                digests[mode].add(digest)
        if args.profile_dir is not None:
            llm.start_profile("cuda_prefill_postprocess")
            try:
                run_one()
            finally:
                llm.stop_profile()

    result = {
        "mode": args.mode,
        "contract": {
            "batch_size": 1,
            "input_len": input_len,
            "beam_width": width,
            "sid_steps": sid_steps,
            "temperature": 0.0,
            "prefix_cache_reset": True,
            "async_scheduling": False,
        },
        "latency": {mode: _summary(values) for mode, values in samples.items()},
        "output_digests": {mode: sorted(values) for mode, values in digests.items()},
    }
    if args.mode == "compare":
        legacy_mean = float(result["latency"]["legacy"]["mean_ms"])
        fast_mean = float(result["latency"]["fast"]["mean_ms"])
        result["delta_mean_ms"] = fast_mean - legacy_mean
        result["improvement_percent"] = (legacy_mean - fast_mean) / legacy_mean * 100
        result["outputs_identical"] = digests["legacy"] == digests["fast"]
    print("P_D_REAL_MODEL_BENCHMARK")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
