# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Manual TP=1 V1 full/slice validation; includes separate diagnostic passes."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("USE_TF", "0")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--constraint-triples", required=True)
    p.add_argument(
        "--strict-bf16",
        action="store_true",
        help="Disable BF16 reduced-precision reduction in the validation worker",
    )
    p.add_argument("--mode", choices=("eager", "graph"), required=True)
    p.add_argument("--projection", choices=("full", "slice"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        p.error("repeat must be positive and warmup must be nonnegative")
    os.environ["REVIEW_STRICT_BF16"] = "1" if args.strict_bf16 else "0"
    os.environ["REVIEW_PROJECTION"] = args.projection
    from transformers import AutoTokenizer

    from benchmarks.open_one_rec.open_one_rec_dataset import OneRecDataset
    from examples.offline_inference.beam_search.offline_beam_search_v1 import (
        build_artifact,
    )
    from tools.AB_Tests.run_offline_benchmark import distribution_ms
    from vllm_gr.entrypoints.beam_search_v1.output_processor import (
        BeamSearchOutputProcessor,
    )
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    model = Path(args.model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    dataset = OneRecDataset(
        dataset_path=args.dataset_path,
        task_types=["video"],
        tokenizer=tokenizer,
        custom_input_len=1023,
        sample_size=max(args.repeat, args.warmup),
    )
    requests = dataset.sample(
        tokenizer=tokenizer, num_requests=max(args.repeat, args.warmup), no_oversample=True
    )
    prompts = [r.prompt for r in requests]
    assert all(len(tokenizer.encode(x, add_special_tokens=False)) == 1023 for x in prompts)
    artifact = Path(args.output).resolve().with_suffix(".safetensors")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    digest, tokenizer_digest = build_artifact(
        model, Path(args.constraint_triples).resolve(), artifact
    )
    cfg = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "graph_enabled": args.mode == "graph",
            "worker_decision": True,
            "max_width": 128,
            "max_decode_steps": 3,
        },
        "constraint_table": {
            "enabled": True,
            "backend": "cuda",
            "path": str(artifact),
            "format": "constraint_table_v1",
            "artifact_digest": digest,
            "tokenizer_digest": tokenizer_digest,
            "max_top_k": 128,
        },
    }
    last: dict[str, Any] = {}
    original = BeamSearchOutputProcessor.process_outputs

    def capture(self: Any, call: Any, request: Any, result: Any) -> Any:
        last["result"] = result
        return original(self, call, request, result)

    BeamSearchOutputProcessor.process_outputs = capture
    params = BeamSearchParams(
        beam_width=128,
        max_tokens=5,
        temperature=0.0,
        ignore_eos=True,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
        include_stop_str_in_output=True,
        length_penalty=1.0,
    )

    def request(llm: Any, index: int) -> dict[str, Any]:
        assert llm.reset_prefix_cache()
        start = time.perf_counter_ns()
        outputs = llm.beam_search_v1([{"prompt": prompts[index]}], params, output_mode="full")
        for s in outputs[0].sequences:
            _ = s.text
        elapsed = (time.perf_counter_ns() - start) / 1e6
        seq = last["result"].output.items[0].sequences
        # Public V1 outputs omit the Worker-internal parent beam history.
        assert len(seq) == 128 and all(
            len(s.token_ids) == 3 and not s.parent_beam_indices for s in seq
        )
        return {
            "e2e_ms": elapsed,
            "beams": [
                {
                    "tokens": list(s.token_ids),
                    "parents": list(s.parent_beam_indices),
                    "score": s.cumulative_logprob,
                }
                for s in seq
            ],
        }

    with GRLLM(
        model=str(model),
        trust_remote_code=True,
        vllm_gr_config=cfg,
        async_scheduling=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        max_logprobs=128,
        gpu_memory_utilization=0.9,
        enforce_eager=args.mode == "eager",
        seed=0,
        worker_cls="vocab_slice_review_worker.ReviewWorker",
    ) as llm:
        for i in range(args.warmup):
            request(llm, i)
        llm.collective_rpc("review_control", args=("measure",))
        samples = [request(llm, i) for i in range(args.repeat)]
        memory = llm.collective_rpc("review_control", args=("report",))[0]
        print("OFFICIAL", distribution_ms([s["e2e_ms"] for s in samples]), flush=True)
        llm.collective_rpc("review_control", args=("diagnostic",))
        for i in range(args.warmup):
            request(llm, i)
        llm.collective_rpc("review_control", args=("diagnostic",))
        for i in range(args.repeat):
            request(llm, i)
        diagnostic = llm.collective_rpc("review_control", args=("report",))[0]
    data = {
        "parameters": vars(args),
        "workload_sha256": hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False).encode()
        ).hexdigest(),
        "artifact_digest": digest,
        "tokenizer_digest": tokenizer_digest,
        "e2e_ms": distribution_ms([s["e2e_ms"] for s in samples]),
        "samples": samples,
        "memory": memory,
        "diagnostic": diagnostic,
    }
    Path(args.output).write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps({"mode": args.mode, "projection": args.projection, "e2e_ms": data["e2e_ms"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
