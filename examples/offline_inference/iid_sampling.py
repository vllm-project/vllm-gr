# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Sample an unordered bag of OneRec SIDs in a single CUDA GR session."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig

from vllm_gr import GRLLM
from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--table", type=Path, required=True, help="Compiled single-trie safetensors"
    )
    parser.add_argument("--prompt", default="Recommend the next video.")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling = GenerationConfig.from_pretrained(args.model)
    width = args.num_samples
    gr = {
        "beam_width": [width] * 3,
        "candidate_width": [width] * 3,
        "max_decode_steps": 3,
        "worker_decision": True,
        "beam_batch_sizes": [1],
        "beam_graph_enabled": False,
        "constraint_mode": "table",
        "constraint_table": {
            "backend": "cuda",
            "max_top_k": width,
            "format": "constraint_table_v1",
            "path": str(args.table.resolve()),
            "digests": {
                "artifact": "sha256:" + hashlib.sha256(args.table.read_bytes()).hexdigest(),
                "tokenizer": compute_tokenizer_digest(tokenizer),
            },
        },
    }
    with GRLLM(
        model=args.model,
        gr=gr,
        attention_config={"backend": "CUSTOM"},
        async_scheduling=True,
        max_logprobs=width,
        max_num_seqs=width,
        enable_prefix_caching=True,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    ) as engine:
        result = engine.sample_sids(
            tokenizer.encode(args.prompt),
            width,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            seed=args.seed,
        )
    print(
        json.dumps(
            {
                "rows_per_stage": result.rows_per_stage,
                "samples": [
                    {"sid_tokens": tokenizer.convert_ids_to_tokens(list(sid)), "count": count}
                    for sid, count in result.counts.items()
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
