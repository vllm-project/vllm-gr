# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sequential text-to-text V1 latency benchmark for the OneRec CLI."""

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

from tools.AB_Tests.run_offline_benchmark import distribution_ms


def run_v1_benchmark(args, tokenizer, requests):
    import vllm_gr
    from examples.offline_inference.beam_search.offline_beam_search_v1 import (
        build_artifact,
    )
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    if args.num_prompts < 1 or args.warmup_requests < 0 or args.n < 1:
        raise ValueError("invalid request, warmup or beam count")
    if len(requests) < max(args.num_prompts, args.warmup_requests):
        raise ValueError("dataset returned fewer requests than requested")
    if not args.constraint_triples or not args.output_json:
        raise ValueError("V1 requires --constraint-triples and --output-json")
    max_tokens = args.output_len or int(requests[0].expected_output_len)
    steps = max_tokens - 2  # Public max_tokens includes SID begin/end.
    if steps not in (1, 2, 3):
        raise ValueError("V1 --output-len must be 3, 4 or 5 (including SID begin/end)")
    prompts = [request.prompt for request in requests]
    lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) + 1 for prompt in prompts]
    if args.custom_input_len and any(length != args.custom_input_len + 1 for length in lengths):
        raise ValueError("text round-trip changed the requested input length")
    model_len = args.max_model_len or 8192
    if max(lengths) + steps + 1 > model_len:
        raise ValueError("input plus output exceeds max_model_len")
    params = BeamSearchParams(
        beam_width=args.n,
        max_tokens=max_tokens,
        temperature=0.0,
        ignore_eos=True,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
        include_stop_str_in_output=True,
        length_penalty=1.0,
    )
    engine_config = dict(
        async_scheduling=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        max_model_len=model_len,
        max_num_batched_tokens=args.max_num_batched_tokens or model_len,
        max_num_seqs=args.n,
        max_logprobs=args.n,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        seed=0,
    )
    samples = []
    with tempfile.TemporaryDirectory(prefix="onerec-v1-") as directory:
        artifact = Path(directory) / "constraints.safetensors"
        artifact_digest, tokenizer_digest = build_artifact(
            Path(args.model).resolve(), Path(args.constraint_triples).resolve(), artifact
        )
        config = {
            "schema_version": 1,
            "attention_backend": "CUSTOM",
            "beam": {
                "execution_mode": "v1",
                "graph_enabled": True,
                "worker_decision": True,
                "max_width": args.n,
                "max_decode_steps": steps,
            },
            "constraint_table": {
                "enabled": True,
                "backend": "cuda",
                "max_top_k": args.n,
                "path": str(artifact),
                "format": "constraint_table_v1",
                "artifact_digest": artifact_digest,
                "tokenizer_digest": tokenizer_digest,
            },
        }
        with GRLLM(
            model=args.model, trust_remote_code=True, vllm_gr_config=config, **engine_config
        ) as llm:
            for index in range(args.warmup_requests + args.num_prompts):
                measured = index >= args.warmup_requests
                row = index - args.warmup_requests if measured else index
                if not llm.reset_prefix_cache():
                    raise RuntimeError(f"prefix cache reset failed before request {index}")
                start = time.perf_counter_ns()
                outputs = llm.beam_search_v1([{"prompt": prompts[row]}], params)
                # V1 materializes tokens/text lazily: force every beam before stopping the timer.
                for output in outputs:
                    for sequence in output.sequences:
                        _ = sequence.text
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                if len(outputs) != 1 or len(outputs[0].sequences) != args.n:
                    raise RuntimeError("unexpected output/beam count")
                beams = outputs[0].sequences
                if any(len(s.tokens) != lengths[row] + steps + 1 for s in beams):
                    raise RuntimeError("unexpected input/SID output length")
                if measured:
                    samples.append(
                        {
                            "e2e_ms": elapsed_ms,
                            "beams": [
                                {
                                    "tokens": list(s.tokens[-max_tokens:]),
                                    "score": float(s.cum_logprob),
                                }
                                for s in beams
                            ],
                        }
                    )
    latencies = [sample["e2e_ms"] for sample in samples]
    repo = Path(vllm_gr.__file__).resolve().parents[1]
    result = {
        "api": "GRLLM.beam_search_v1",
        "commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": args.model,
        "dataset_path": args.dataset_path,
        "num_prompts": args.num_prompts,
        "warmup_requests": args.warmup_requests,
        "beam_width": args.n,
        "generated_steps": steps,
        "engine_config": engine_config,
        "input_lengths_including_sid_begin": lengths[: args.num_prompts],
        "workload_sha256": hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False).encode()
        ).hexdigest(),
        "artifact_digest": artifact_digest,
        "tokenizer_digest": tokenizer_digest,
        "protocol": "text input encoding through all beam text decoding; single request; "
        "prefix cache reset before every request outside timer; warmup excluded",
        "percentile_method": "linear interpolation at (n - 1) * quantile",
        "cache_miss_e2e_ms": distribution_ms(latencies),
        "requests_per_second": len(samples) * 1000 / sum(latencies),
        "samples": samples,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["cache_miss_e2e_ms"]), flush=True)
