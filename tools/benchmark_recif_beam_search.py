#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extended RECIF benchmark: batch sizes + multi-beam + memory + throughput + input lengths."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from examples.offline_inference.beam_search.recif_utils import (  # noqa: E402
    build_prefix,
)
from vllm_gr.entrypoints.gr import GRLLM  # noqa: E402
from vllm_gr.models.recif import RECIF_MAX_LOGPROBS  # noqa: E402
from vllm_gr.sampling_params import BeamSearchParams  # noqa: E402

SID_MASK = (1 << 42) - 1


def parse_history_line(line: str) -> list[int]:
    return [int(x.strip()) for x in line.split(",") if x.strip()]


def build_random_histories(num_sids: int, num_seqs: int, seed: int) -> list[list[int]]:
    """Generate random histories of given sid count with a fixed seed (reproducible)."""
    rng = random.Random(seed)
    return [[rng.randrange(0, SID_MASK + 1) for _ in range(num_sids)] for _ in range(num_seqs)]


def iter_batches(histories: list[list[int]], batch_size: int):
    """Yield history chunks of at most ``batch_size`` (last chunk may be smaller)."""
    for start in range(0, len(histories), batch_size):
        yield histories[start : start + batch_size]


def benchmark_single_config(
    llm: GRLLM,
    histories: list[list[int]],
    beam_width: int,
    batch_size: int,
    iterations: int,
    warmup: int,
    input_length: int | None = None,
) -> dict:
    """Benchmark one (beam_width, batch_size, input_length) config.

    Requests are grouped into ``beam_search`` calls of ``batch_size`` prompts
    each. Latency metrics are per-batch wall time (one ``beam_search`` call),
    which equals per-request latency when ``batch_size == 1``.
    """
    params = BeamSearchParams(
        beam_width=beam_width,
        max_tokens=3,
        temperature=0.0,
        ignore_eos=True,
    )

    # Warmup
    for _ in range(warmup):
        for batch in iter_batches(histories, batch_size):
            prompts = [{"prompt_token_ids": build_prefix(h)} for h in batch]
            llm.beam_search(prompts, params)

    # Reset CUDA memory stats for accurate peak measurement
    torch.accelerator.reset_peak_memory_stats()

    # Benchmark
    latencies_ms = []
    total_samples = len(histories) * iterations
    total_tokens_generated = 0  # 3 tokens per sample

    t_start = time.perf_counter()
    for _ in range(iterations):
        for batch in iter_batches(histories, batch_size):
            prompts = [{"prompt_token_ids": build_prefix(h)} for h in batch]
            t0 = time.perf_counter()
            outputs = llm.beam_search(prompts, params)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000)

            # Count generated tokens (always 3 for RECIF)
            for h, out in zip(batch, outputs):
                top = out.sequences[0]
                total_tokens_generated += len(top.tokens) - len(build_prefix(h))
    t_end = time.perf_counter()

    latencies_ms.sort()
    n = len(latencies_ms)

    def _pct(p: float) -> float:
        idx = min(int(n * p), n - 1)
        return latencies_ms[idx]

    elapsed_sec = t_end - t_start
    peak_mem_mb = torch.accelerator.max_memory_allocated() / 1024 / 1024

    result = {
        "input_length": input_length,
        "beam_width": beam_width,
        "batch_size": batch_size,
        "iterations": iterations,
        "warmup": warmup,
        "num_histories": len(histories),
        "total_samples": total_samples,
        "elapsed_sec": round(elapsed_sec, 3),
        "throughput_samples_per_sec": round(total_samples / elapsed_sec, 2),
        "throughput_tokens_per_sec": round(total_tokens_generated / elapsed_sec, 2),
        "peak_gpu_memory_mb": round(peak_mem_mb, 1),
        "latency_ms": {
            "avg": round(statistics.mean(latencies_ms), 2),
            "min": round(latencies_ms[0], 2),
            "max": round(latencies_ms[-1], 2),
            "p50": round(_pct(0.50), 2),
            "p95": round(_pct(0.95), 2),
            "p99": round(_pct(0.99), 2),
            "std": round(statistics.stdev(latencies_ms), 2) if n > 1 else 0.0,
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extended RECIF benchmark (input lengths + beam widths + batch sizes)"
    )
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--history", type=str, help="Single history: '1,2,3'")
    parser.add_argument("--history_file", type=Path, help="Batch file: one history per line")
    parser.add_argument(
        "--history_dir",
        type=Path,
        help="Directory with per-length history files (name=<num_sids>.txt, each line one "
        "history). Loads each file and labels it by 4x sid count = token length. "
        "Mutually exclusive with --history/--history_file/--input_lengths.",
    )
    parser.add_argument(
        "--input_lengths",
        type=str,
        help="Comma-separated input token lengths for auto-generated random data, "
        "e.g. '512,1024,2048,4096,8192,10240'. Each length must be divisible by 4 "
        "(4 tokens per RECIF sid). Mutually exclusive with --history/--history_file.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Seed for random input generation (reproducible)",
    )
    parser.add_argument(
        "--num_seqs",
        type=int,
        default=1,
        help="Number of random sequences generated per input length",
    )
    parser.add_argument(
        "--beam_widths",
        type=str,
        default="64,128,256,512",
        help="Comma-separated beam widths to test",
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1",
        help="Comma-separated batch sizes to test (requests per beam_search call), "
        "e.g. '1,8,16,32'. Latency metrics are per batch.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=None,
        help="Engine max_num_seqs. Defaults to max(batch_sizes) * max(beam_widths), "
        "i.e. the worst-case number of forked beam sequences in one batch.",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--attention-backend", choices=("AUTO", "CUSTOM"), default="CUSTOM")
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument("--max_latency_p95_ms", type=float)
    parser.add_argument("--max_latency_avg_ms", type=float)
    parser.add_argument("--min_samples_per_sec", type=float)
    args = parser.parse_args()

    # Load histories: --history_dir / --input_lengths / --history_file / --history
    histories_by_length: dict[int | None, list[list[int]]] = {}
    if args.history_dir:
        if args.history or args.history_file or args.input_lengths:
            print(
                "Error: --history_dir is mutually exclusive with --history/--history_file/--input_lengths",
                file=sys.stderr,
            )
            return 2
        if not args.history_dir.is_dir():
            print(f"Error: --history_dir {args.history_dir} is not a directory", file=sys.stderr)
            return 2
        for f in sorted(args.history_dir.glob("*.txt")):
            num_sids = int(f.stem)
            histories = [parse_history_line(line) for line in f.read_text().strip().splitlines()]
            # label = token length = 4x sid count
            histories_by_length[num_sids * 4] = histories
            print(
                f"[Benchmark] Loaded {len(histories)} histories x {num_sids} sids "
                f"(= {num_sids * 4} tokens) from {f}"
            )
    elif args.input_lengths:
        if args.history or args.history_file:
            print(
                "Error: --input_lengths is mutually exclusive with --history/--history_file",
                file=sys.stderr,
            )
            return 2
        input_lengths = [int(x.strip()) for x in args.input_lengths.split(",")]
        for i, length in enumerate(input_lengths):
            if length % 4 != 0:
                print(
                    f"Error: input length {length} must be divisible by 4 (4 tokens per RECIF sid)",
                    file=sys.stderr,
                )
                return 2
            num_sids = length // 4
            histories_by_length[length] = build_random_histories(
                num_sids, args.num_seqs, args.random_seed + i
            )
        print(
            f"[Benchmark] Generated random histories for input lengths {input_lengths} "
            f"(seed={args.random_seed}, seqs/length={args.num_seqs})"
        )
    elif args.history_file:
        histories_by_length[None] = [
            parse_history_line(line) for line in args.history_file.read_text().strip().splitlines()
        ]
        print(
            f"[Benchmark] Loaded {len(histories_by_length[None])} histories from {args.history_file}"
        )
    elif args.history:
        histories_by_length[None] = [parse_history_line(args.history)]
        print(f"[Benchmark] Using single history: {histories_by_length[None][0]}")
    else:
        print("Error: provide --input_lengths or --history or --history_file", file=sys.stderr)
        return 2

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    if not batch_sizes or any(b < 1 for b in batch_sizes):
        print("Error: --batch_sizes must contain positive integers", file=sys.stderr)
        return 2
    print(f"[Benchmark] Batch sizes to test: {batch_sizes}")
    if args.num_seqs % max(batch_sizes) != 0:
        print(
            f"[Benchmark] Note: num_seqs={args.num_seqs} is not a multiple of the largest batch "
            f"size {max(batch_sizes)}; the last batch of each round will be partial.",
            file=sys.stderr,
        )

    attention_kwargs = (
        {"attention_config": {"backend": "CUSTOM"}} if args.attention_backend == "CUSTOM" else {}
    )

    beam_widths = [int(x.strip()) for x in args.beam_widths.split(",")]
    max_num_seqs = args.max_num_seqs or (max(batch_sizes) * max(beam_widths))

    # Load model once
    print(f"[Benchmark] Loading model from {args.model_path} (max_num_seqs={max_num_seqs})")
    t0 = time.perf_counter()
    llm = GRLLM(
        model=str(args.model_path),
        max_logprobs=RECIF_MAX_LOGPROBS,
        skip_tokenizer_init=True,
        max_num_seqs=max_num_seqs,
        logprobs_mode="processed_logprobs",
        **attention_kwargs,
    )
    load_time_ms = (time.perf_counter() - t0) * 1000
    print(f"[Benchmark] Model loaded in {load_time_ms:.1f} ms")

    # Benchmark each (input_length, beam_width, batch_size) combination
    all_results = []

    for input_length in sorted(
        histories_by_length.keys(), key=lambda x: x if x is not None else -1
    ):
        histories = histories_by_length[input_length]
        label = str(input_length) if input_length is not None else "explicit"
        for bw in beam_widths:
            for bs in batch_sizes:
                print(
                    f"\n[Benchmark] === input_length={label}, beam_width={bw}, batch_size={bs}, "
                    f"num_histories={len(histories)} ==="
                )
                result = benchmark_single_config(
                    llm=llm,
                    histories=histories,
                    beam_width=bw,
                    batch_size=bs,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    input_length=input_length,
                )
                result["model_load_time_ms"] = round(load_time_ms, 1)
                all_results.append(result)

                # Print single result
                lat = result["latency_ms"]
                print(f"  Samples/sec     : {result['throughput_samples_per_sec']}")
                print(f"  Tokens/sec      : {result['throughput_tokens_per_sec']}")
                print(f"  Peak GPU memory : {result['peak_gpu_memory_mb']} MB")
                print(f"  Avg latency     : {lat['avg']} ms")
                print(f"  P95 latency     : {lat['p95']} ms")
                print(f"  P99 latency     : {lat['p99']} ms")

    # Summary comparison table
    print("\n" + "=" * 100)
    print("[Benchmark] Comparison Table")
    print("=" * 100)
    header = (
        f"{'Batch':>6} | {'InLen':>10} | {'Beam':>6} | {'Samples/s':>10} | {'Tokens/s':>10} "
        f"| {'PeakMem(MB)':>12} | {'Avg(ms)':>9} | {'P95(ms)':>9} | {'P99(ms)':>9}"
    )
    print(header)
    print("-" * 100)
    for r in all_results:
        lat = r["latency_ms"]
        in_len = str(r["input_length"]) if r["input_length"] is not None else "explicit"
        print(
            f"{r['batch_size']:>6} | {in_len:>10} | {r['beam_width']:>6} | {r['throughput_samples_per_sec']:>10.2f} | "
            f"{r['throughput_tokens_per_sec']:>10.2f} | {r['peak_gpu_memory_mb']:>12.1f} | "
            f"{lat['avg']:>9.2f} | {lat['p95']:>9.2f} | {lat['p99']:>9.2f}"
        )
    print("=" * 100)

    # Save JSON
    if args.output_json:
        args.output_json.write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"\n[Benchmark] JSON saved to {args.output_json}")

    # Save CSV
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "batch_size",
                    "input_length",
                    "beam_width",
                    "samples_per_sec",
                    "tokens_per_sec",
                    "peak_mem_mb",
                    "avg_ms",
                    "p95_ms",
                    "p99_ms",
                ]
            )
            for r in all_results:
                lat = r["latency_ms"]
                in_len = r["input_length"] if r["input_length"] is not None else ""
                writer.writerow(
                    [
                        r["batch_size"],
                        in_len,
                        r["beam_width"],
                        r["throughput_samples_per_sec"],
                        r["throughput_tokens_per_sec"],
                        r["peak_gpu_memory_mb"],
                        lat["avg"],
                        lat["p95"],
                        lat["p99"],
                    ]
                )
        print(f"[Benchmark] CSV saved to {args.output_csv}")

    # CI gate
    failed = False
    for r in all_results:
        lat = r["latency_ms"]
        in_len = r["input_length"] if r["input_length"] is not None else "explicit"
        bw = r["beam_width"]
        bs = r["batch_size"]
        if args.min_samples_per_sec and r["throughput_samples_per_sec"] < args.min_samples_per_sec:
            print(
                f"FAILED: batch={bs}, input_length={in_len}, beam={bw} throughput "
                f"{r['throughput_samples_per_sec']} samples/s < {args.min_samples_per_sec} samples/s",
                file=sys.stderr,
            )
            failed = True
        if args.max_latency_avg_ms and lat["avg"] > args.max_latency_avg_ms:
            print(
                f"FAILED: batch={bs}, input_length={in_len}, beam={bw} avg {lat['avg']} ms > {args.max_latency_avg_ms} ms",
                file=sys.stderr,
            )
            failed = True
        if args.max_latency_p95_ms and lat["p95"] > args.max_latency_p95_ms:
            print(
                f"FAILED: batch={bs}, input_length={in_len}, beam={bw} p95 {lat['p95']} ms > {args.max_latency_p95_ms} ms",
                file=sys.stderr,
            )
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
