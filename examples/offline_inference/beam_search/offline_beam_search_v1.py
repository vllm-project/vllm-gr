#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run or benchmark offline Beam Search V1 with a canonical Constraint Table."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

os.environ.setdefault("USE_TF", "0")


BEGIN_TOKEN = "<|sid_begin|>"
END_TOKEN = "<|sid_end|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Constraint triples JSON produced by tools/build_constraint_triples.py",
    )
    parser.add_argument(
        "--constraint-backend",
        choices=("cuda", "npu"),
        default="cuda",
        help="Constraint Table device backend (CUDA or NPU)",
    )
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument(
        "--execution",
        choices=("eager", "graph"),
        default="eager",
        help="NPU Decode execution mode; CUDA keeps its existing graph behavior",
    )
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--generated-steps", type=int, choices=(1, 2, 3), default=3)
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
        help="Save benchmark summary and per-run timings (requires --repeat)",
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
    if args.output_json is not None and args.repeat is None:
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


def benchmark_beam_search(
    llm: Any, args: argparse.Namespace, prompts: list[Any], params: Any
) -> tuple[list[Any], dict[str, Any]]:
    for iteration in range(args.warmup_runs):
        llm.beam_search_v1(
            isolate_prompts(prompts, f"warmup-{iteration}"), params, output_mode="full"
        )

    latencies_ms: list[float] = []
    returned_beams: list[int] = []
    outputs: list[Any] = []
    for iteration in range(args.repeat):
        isolated = isolate_prompts(prompts, f"benchmark-{iteration}")
        started = perf_counter()
        outputs = llm.beam_search_v1(isolated, params, output_mode="full")
        latencies_ms.append((perf_counter() - started) * 1000.0)
        if len(outputs) != len(prompts):
            raise RuntimeError(f"expected {len(prompts)} request outputs, got {len(outputs)}")
        # Count candidates outside timing; do not tokenize/decode or print here.
        returned_beams.append(sum(len(output.sequences) for output in outputs))

    ordered = sorted(latencies_ms)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    total_seconds = sum(latencies_ms) / 1000.0
    report: dict[str, Any] = {
        "model": str(args.model.resolve()),
        "catalog": str(args.catalog.resolve()),
        "prompt": str(args.prompt.resolve()),
        "constraint_backend": args.constraint_backend,
        "execution": args.execution,
        "batch_size": args.batch_size,
        "beam_width": args.beam_width,
        "input_len": args.input_len,
        "generated_steps": args.generated_steps,
        "length_penalty": args.length_penalty,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "warmup_runs": args.warmup_runs,
        "repeat": args.repeat,
        "prefix_caching": "enabled; isolated salt per row and run (cold Prefill)",
        "timing_scope": "beam_search_v1 API call until output objects return; excludes engine startup, "
        "warmup, pre-call prompt preparation and post-call result inspection/printing",
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
    return outputs, report


def run_beam_search(
    llm: Any, args: argparse.Namespace, prompts: list[Any], params: Any
) -> list[Any]:
    if args.profile_dir is None:
        outputs: list[Any] = llm.beam_search_v1(prompts, params, output_mode="full")
        return outputs

    def run_once(iteration: int) -> list[Any]:
        isolated = isolate_prompts(prompts, f"profile-{iteration}")
        result: list[Any] = llm.beam_search_v1(isolated, params, output_mode="full")
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


def run(args: argparse.Namespace, artifact_path: Path) -> None:
    from vllm.inputs import tokens_input

    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    is_npu = args.constraint_backend == "npu"

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
    vllm_gr_config = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": beam_config,
        "constraint_table": {
            "enabled": True,
            "backend": args.constraint_backend,
            "max_top_k": args.beam_width,
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
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        max_model_len=args.max_model_len,
        # The V1 input processor appends the begin token to every prompt, so a
        # B-wide Prefill schedules B * (input_len + 1) tokens, not B * input_len.
        max_num_batched_tokens=max(args.max_model_len, args.batch_size * (args.input_len + 1)),
        max_num_seqs=max(args.beam_width, args.batch_size),
        max_logprobs=args.beam_width,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=is_npu and args.execution == "eager",
        **profiler_engine_kwargs(args),
    ) as llm:
        tokenizer = llm.get_tokenizer()
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
            outputs, benchmark = benchmark_beam_search(llm, args, request_prompts, params)
        else:
            outputs = run_beam_search(llm, args, request_prompts, params)
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


def main() -> None:
    args = parse_args()
    for path in (args.model, args.catalog, args.prompt):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.input_len + args.generated_steps + 1 > args.max_model_len:
        raise ValueError("prompt and generated tokens exceed --max-model-len")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.print_beams <= 0:
        raise ValueError("--print-beams must be positive")

    with tempfile.TemporaryDirectory(prefix="vllm-gr-beam-v1-") as directory:
        artifact_path = Path(directory).resolve() / "constraint_table.safetensors"
        run(args, artifact_path)


if __name__ == "__main__":
    main()
