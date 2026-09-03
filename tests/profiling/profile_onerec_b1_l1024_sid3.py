#!/usr/bin/env python3
"""Profile one B=1, L=1024, SID=3, BW=128 OneRec-1.7B request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Keep the frontend, EngineCore, Worker and ModelRunner in one process so one
# Kineto trace contains the complete CPU and GPU timeline.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("USE_TF", "0")

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/home/z00469465/OneRec/OneRec-1.7B")
DEFAULT_CATALOG = Path("/home/z00469465/OneRec/generated_data/video_constraint_triples.json")
DEFAULT_PROMPT = REPO_ROOT / "tests/resources/single_one_rec_prompt.txt"

INPUT_LEN = 1024
BEAM_WIDTH = 128
SID_STEPS = 3
MAX_TOKENS = SID_STEPS + 2  # <|sid_begin|>, three SID tokens, <|sid_end|>


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disabled by default so FULL graph replay is tested without prefix lookup.",
    )
    return parser.parse_args()


def make_prompt(tokenizer: Any, prompt_path: Path) -> tuple[list[int], dict[str, Any]]:
    source = tokenizer.encode(prompt_path.read_text(encoding="utf-8"), add_special_tokens=False)
    submitted_len = INPUT_LEN - 1  # GRLLM appends <|sid_begin|>.
    if len(source) >= submitted_len:
        token_ids = source[-submitted_len:]
        adjustment = "left_truncate"
    else:
        newline = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline) != 1:
            raise RuntimeError(f"newline must map to one token, got {newline}")
        token_ids = [int(newline[0])] * (submitted_len - len(source)) + source
        adjustment = "left_pad_newline"

    if len(token_ids) + 1 != INPUT_LEN:
        raise AssertionError("model input must contain exactly 1024 tokens")
    return token_ids, {
        "source_tokens": len(source),
        "submitted_tokens": len(token_ids),
        "sid_begin_tokens_appended_by_grllm": 1,
        "model_input_tokens": INPUT_LEN,
        "adjustment": adjustment,
    }


def validate_output(output: Any, end_token_id: int) -> dict[str, Any]:
    if len(output) != 1:
        raise AssertionError(f"expected one request, got {len(output)}")
    sequences = output[0].sequences
    if len(sequences) != BEAM_WIDTH:
        raise AssertionError(f"expected {BEAM_WIDTH} beams, got {len(sequences)}")

    expected_len = INPUT_LEN + SID_STEPS + 1
    lengths = {len(sequence.tokens) for sequence in sequences}
    if lengths != {expected_len}:
        raise AssertionError(f"unexpected sequence lengths: {sorted(lengths)}")
    if any(sequence.tokens[-1] != end_token_id for sequence in sequences):
        raise AssertionError("not every beam ends with <|sid_end|>")

    sid_tokens = [list(sequence.tokens[INPUT_LEN:-1]) for sequence in sequences]
    if any(len(tokens) != SID_STEPS for tokens in sid_tokens):
        raise AssertionError("not every beam contains exactly three SID tokens")
    digest_rows = [
        {"tokens": sequence.tokens[-4:], "score": float(sequence.cum_logprob)}
        for sequence in sequences
    ]
    digest = hashlib.sha256(
        json.dumps(digest_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "beam_count": len(sequences),
        "best_beam_sid_token_ids": sid_tokens[0],
        "result_sha256": digest,
    }


def collective_rpc(llm: Any, method: str) -> Any:
    engine_core = llm.llm_engine.engine_core
    owner = engine_core if hasattr(engine_core, "collective_rpc") else llm.llm_engine
    replies = owner.collective_rpc(method)
    if len(replies) != 1:
        raise RuntimeError(f"expected one reply from {method}, got {replies}")
    return replies[0]


def install_profile_ranges(llm: Any, persistent_supported: bool) -> None:
    core_client = llm.llm_engine.engine_core
    core = getattr(core_client, "engine_core", core_client)
    runner = core.model_executor.driver_worker.model_runner

    def wrap(obj: Any, method: str, label: str) -> None:
        original = getattr(obj, method)

        def traced(*args: Any, **kwargs: Any) -> Any:
            with torch.autograd.profiler.record_function(label):
                return original(*args, **kwargs)

        setattr(obj, method, traced)

    wrap(core.scheduler, "schedule", "gr_schedule")
    wrap(runner, "_prepare_inputs", "gr_prepare_inputs")
    wrap(runner, "execute_model", "gr_execute_model")
    wrap(runner, "sample_tokens", "gr_sample_tokens")
    wrap(core.scheduler, "update_from_output", "gr_update_from_output")

    if persistent_supported:
        from vllm_gr.v1.worker.gr_model_runner_state import GRModelRunnerState

        original_final = GRModelRunnerState.make_final_output

        def traced_final(state: GRModelRunnerState) -> dict[str, Any]:
            with torch.autograd.profiler.record_function("gr_final_result_d2h"):
                return original_final(state)

        GRModelRunnerState.make_final_output = traced_final


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 1:
        raise ValueError("--warmup-runs must be positive")
    for path in (args.model, args.catalog, args.prompt):
        if not path.exists():
            raise FileNotFoundError(path)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir or REPO_ROOT / "trace" / f"onerec_b1_l1024_sid3_bw128_{timestamp}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    trace_path = output_dir / "onerec_b1_l1024_sid3_bw128.pt.trace.json"

    from vllm_gr.config import GRConfig
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    gr_fields = {field.name for field in fields(GRConfig)}
    persistent_supported = {
        "persistent_request",
        "persistent_runner_state",
    }.issubset(gr_fields)
    engine_kwargs: dict[str, Any] = {}
    if persistent_supported:
        engine_kwargs.update(
            persistent_request=True,
            persistent_runner_state=True,
        )

    params = BeamSearchParams(
        beam_width=BEAM_WIDTH,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        ignore_eos=False,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
    )

    with GRLLM(
        model=str(args.model.resolve()),
        max_model_len=40960,
        trust_remote_code=True,
        max_logprobs=BEAM_WIDTH,
        catalog_path=str(args.catalog.resolve()),
        constraint_backend="constraint_table",
        beam_graph_enabled=True,
        beam_max_width=BEAM_WIDTH,
        beam_max_decode_steps=SID_STEPS,
        async_scheduling=True,
        attention_config={"backend": "CUSTOM"},
        max_num_seqs=BEAM_WIDTH,
        enable_prefix_caching=args.enable_prefix_caching,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        **engine_kwargs,
    ) as llm:
        tokenizer = llm.get_tokenizer()
        token_ids, prompt_info = make_prompt(tokenizer, args.prompt)
        prompts = [{"prompt_token_ids": token_ids}]
        end_token_id = tokenizer.convert_tokens_to_ids("<|sid_end|>")

        initial_graph = collective_rpc(llm, "get_gr_beam_graph_runtime_stats")
        if BEAM_WIDTH not in initial_graph["captured_widths"]:
            raise RuntimeError(f"BW{BEAM_WIDTH} FULL graph was not captured")

        for _ in range(args.warmup_runs):
            if args.enable_prefix_caching and not llm.reset_prefix_cache():
                raise RuntimeError("prefix-cache reset failed")
            output = llm.beam_search(prompts, params)
            torch.cuda.synchronize()
            validate_output(output, end_token_id)

        install_profile_ranges(llm, persistent_supported)
        if args.enable_prefix_caching and not llm.reset_prefix_cache():
            raise RuntimeError("prefix-cache reset failed")
        before_graph = collective_rpc(llm, "get_gr_beam_graph_runtime_stats")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        ) as profiler:
            with torch.autograd.profiler.record_function("onerec_profiled_request"):
                output = llm.beam_search(prompts, params)
            torch.cuda.synchronize()
        profiled_ms = (time.perf_counter() - start) * 1000

        output_info = validate_output(output, end_token_id)
        after_graph = collective_rpc(llm, "get_gr_beam_graph_runtime_stats")
        state_stats = None
        memory_stats = None
        if persistent_supported:
            state_stats = collective_rpc(llm, "get_gr_model_runner_state_stats")
            memory_stats = collective_rpc(llm, "get_gr_memory_stats")
            if int(state_stats["active"]) != 0:
                raise RuntimeError(f"persistent runner state leaked: {state_stats}")
        graph_delta = int(after_graph["full_dispatches"]) - int(before_graph["full_dispatches"])
        if graph_delta != SID_STEPS - 1:
            raise RuntimeError(f"expected two FULL graph replays, got {graph_delta}")

        profiler.export_chrome_trace(str(trace_path))
        (output_dir / "profiler_table.txt").write_text(
            profiler.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total", row_limit=300
            )
            + "\n",
            encoding="utf-8",
        )
        result = {
            "scenario": {
                "batch_size": 1,
                "beam_width": BEAM_WIDTH,
                "model_input_tokens": INPUT_LEN,
                "sid_steps": SID_STEPS,
                "prefix_caching": args.enable_prefix_caching,
                "async_scheduling": True,
                "persistent_request": persistent_supported,
                "persistent_runner_state": persistent_supported,
                "cuda_graph": "FULL exact width",
            },
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "prompt": prompt_info,
            "output": output_info,
            "profiled_request_ms": round(profiled_ms, 6),
            "beam_graph": {
                "before": before_graph,
                "after": after_graph,
                "full_dispatch_delta": graph_delta,
            },
            "runner_state": state_stats,
            "gr_memory": memory_stats,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "trace": str(trace_path),
        }

    result["trace_bytes"] = trace_path.stat().st_size
    result["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    (output_dir / "run.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
