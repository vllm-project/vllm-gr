# SPDX-License-Identifier: Apache-2.0
"""Run one isolated NPU async-resident beam-search batch.

Run each batch geometry in a separate Python process. Each invocation writes a
JSON artifact containing every generated token sequence from the final round,
cumulative score, a stable token hash, and per-round elapsed time. The final
round is profiled. This keeps engine/graph state isolated between geometries
and makes the artifacts suitable for later comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT = REPO_ROOT / "tests/resources/single_one_rec_prompt.txt"


def _default_model() -> str | None:
    return os.getenv("GR_B2_MODEL") or os.getenv("VLLM_GR_TEST_MODEL")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_default_model())
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="active request batch size (1-128)",
    )
    parser.add_argument("--engine-batch-capacity", type=int)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--execution", choices=("graph", "eager"), default="graph")
    parser.add_argument("--begin-token", default="<|sid_begin|>")
    parser.add_argument("--end-token", default="<|sid_end|>")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=20480)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def _sequence_score(sequence: Any) -> float:
    score = getattr(sequence, "cumulative_logprob", None)
    if score is None:
        score = getattr(sequence, "cum_logprob", None)
    if score is None:
        raise RuntimeError("async beam output is missing its cumulative score")
    return float(score)


def _token_hash(sequences: list[dict[str, Any]]) -> str:
    payload = [sequence["tokens"] for sequence in sequences]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.model:
        raise SystemExit("--model or GR_B2_MODEL/VLLM_GR_TEST_MODEL is required")
    if not args.prompt.is_file():
        raise SystemExit(f"prompt file does not exist: {args.prompt}")
    if not 1 <= args.batch_size <= 128:
        raise SystemExit("--batch-size must be between 1 and 128")
    if args.engine_batch_capacity is None:
        args.engine_batch_capacity = args.batch_size
    if args.engine_batch_capacity < args.batch_size:
        raise SystemExit("engine batch capacity must be >= active batch size")
    if args.beam_width < 1:
        raise SystemExit("--beam-width must be positive")
    precomputed = int(bool(args.begin_token)) + int(bool(args.end_token))
    if args.max_tokens <= precomputed:
        raise SystemExit("--max-tokens must leave at least one generated beam token")
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    if args.max_num_seqs < args.engine_batch_capacity:
        raise SystemExit("--max-num-seqs must cover engine batch capacity")
    if args.max_num_batched_tokens < args.engine_batch_capacity * args.beam_width:
        raise SystemExit("--max-num-batched-tokens must cover B*W decode rows")


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)

    # Import only after argument validation so malformed invocations do not
    # initialize the NPU runtime.
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    prompt_text = args.prompt.read_text(encoding="utf-8")
    prompts = [{"prompt": prompt_text} for _ in range(args.batch_size)]
    params = BeamSearchParams(
        beam_width=args.beam_width,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        ignore_eos=True,
        begin_token=args.begin_token or None,
        end_token=args.end_token or None,
    )
    decode_steps = args.max_tokens - int(bool(args.begin_token)) - int(
        bool(args.end_token)
    )
    enforce_eager = args.execution == "eager"
    output_path = args.output or Path(
        f"test-results/08_async_b{args.batch_size}_{args.execution}.json"
    )
    profile_dir = args.profile_dir or Path(
        "test-results"
    ) / f"08_async_b{args.batch_size}_{args.execution}_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = profile_dir.resolve()

    print("TEST_CONFIG:", flush=True)
    print("  scheduling=async", flush=True)
    print("  resident_request=true", flush=True)
    print("  decode_execution=xllm", flush=True)
    print(f"  execution_mode={args.execution}", flush=True)
    print(f"  active_batch_size={args.batch_size}", flush=True)
    print(f"  engine_batch_capacity={args.engine_batch_capacity}", flush=True)
    print(f"  beam_width={args.beam_width}", flush=True)
    print(f"  max_tokens={args.max_tokens}", flush=True)
    print(f"  decode_steps={decode_steps}", flush=True)
    print(f"  runs={args.runs}", flush=True)
    print(f"  profiled_round={args.runs}", flush=True)
    print(f"  model={args.model}", flush=True)
    print(f"  prompt={args.prompt}", flush=True)
    print(f"  profile_dir={profile_dir}", flush=True)

    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "max_logprobs": args.beam_width,
        "beam_graph_enabled": not enforce_eager,
        "beam_max_width": args.beam_width,
        "beam_max_decode_steps": decode_steps,
        "beam_max_batch_size": args.engine_batch_capacity,
        "beam_async": True,
        "attention_config": {"backend": "CUSTOM"},
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "async_scheduling": True,
        "enforce_eager": enforce_eager,
        "additional_config": {"vllm_gr_npu_decode_execution": "xllm"},
        "profiler_config": {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_with_stack": True,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": False,
            "torch_profiler_use_gzip": False,
            "ignore_frontend": True,
        },
    }
    if args.gpu_memory_utilization is not None:
        engine_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    round_elapsed_seconds: list[float] = []
    outputs = None
    with GRLLM(**engine_kwargs) as llm:
        for round_index in range(args.runs):
            profiled = round_index == args.runs - 1
            if profiled:
                llm.start_profile(
                    f"async_b{args.batch_size}_{args.execution}_round_{round_index + 1}"
                )
            try:
                started = time.perf_counter()
                outputs = llm.async_beam_search(
                    prompts,
                    params,
                    concurrency_limit=args.batch_size,
                )
                elapsed = time.perf_counter() - started
            finally:
                if profiled:
                    llm.stop_profile()
            round_elapsed_seconds.append(elapsed)
            print(
                f"round={round_index + 1}/{args.runs} "
                f"elapsed_seconds={elapsed:.6f} profiled={profiled}",
                flush=True,
            )

    assert outputs is not None
    if len(outputs) != args.batch_size:
        raise RuntimeError(
            f"expected {args.batch_size} outputs, received {len(outputs)}"
        )

    begin_count = int(bool(args.begin_token))
    end_count = int(bool(args.end_token))
    items: list[dict[str, Any]] = []
    for item_index, output in enumerate(outputs):
        if len(output.sequences) != args.beam_width:
            raise RuntimeError(
                f"item {item_index}: expected {args.beam_width} beams, "
                f"received {len(output.sequences)}"
            )
        sequences = []
        for rank, sequence in enumerate(output.sequences):
            all_tokens = [int(token) for token in sequence.tokens]
            if len(all_tokens) < args.max_tokens:
                raise RuntimeError(
                    f"item {item_index}, rank {rank}: output has only "
                    f"{len(all_tokens)} tokens, expected at least {args.max_tokens}"
                )
            generated = all_tokens[-args.max_tokens :]
            recommendation_end = len(generated) - end_count if end_count else None
            sequences.append(
                {
                    "rank": rank,
                    "tokens": generated,
                    "recommendation_tokens": generated[
                        begin_count:recommendation_end
                    ],
                    "score": _sequence_score(sequence),
                }
            )
        item = {
            "beam_count": len(sequences),
            "token_sha256": _token_hash(sequences),
            "sequences": sequences,
        }
        items.append(item)
        print(
            f"item={item_index} beams={item['beam_count']} "
            f"hash={item['token_sha256']}",
            flush=True,
        )

    result = {
        "model": args.model,
        "prompt": str(args.prompt),
        "scheduling": "async",
        "resident_request": True,
        "decode_execution": "xllm",
        "execution_mode": args.execution,
        "active_batch_size": args.batch_size,
        "engine_batch_capacity": args.engine_batch_capacity,
        "beam_width": args.beam_width,
        "max_tokens": args.max_tokens,
        "decode_steps": decode_steps,
        "runs": args.runs,
        "profiled_round": args.runs,
        "profile_dir": str(profile_dir),
        "round_elapsed_seconds": round_elapsed_seconds,
        "elapsed_seconds": round_elapsed_seconds[-1],
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"mean_elapsed_seconds: "
        f"{sum(round_elapsed_seconds) / len(round_elapsed_seconds):.6f}",
        flush=True,
    )
    print(f"result_file: {output_path}", flush=True)
    print(f"profile_dir: {profile_dir}", flush=True)
    print("ASYNC_RESIDENT_RUN_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
