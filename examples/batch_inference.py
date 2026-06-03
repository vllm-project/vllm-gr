#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Batch inference example for vllm-GR."""

import argparse
import os
import time
from typing import TYPE_CHECKING, List

from runtime_utils import add_repo_root_to_path, add_runtime_args, describe_runtime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
add_repo_root_to_path(__file__)

if TYPE_CHECKING:
    from vllm_gr import GRPrompt

# Demo model path - relative to examples directory
DEMO_MODEL_PATH = os.path.join(SCRIPT_DIR, "demo-model")


def generate_sample_prompts(
    num_users: int, seq_length: int, num_candidates: int
) -> List["GRPrompt"]:
    """Generate sample GRPrompt requests for testing.

    Uses the default demo-model feature names:
    - UIH: item_id, timestamp
    - Candidates: item_candidate_id, query_time

    All feature values are pre-processed embedding indices and must satisfy
    0 <= value < num_embeddings (10000 for the demo-model item_embedding table).
    """
    prompts: List[GRPrompt] = []

    for i in range(num_users):
        prompt: GRPrompt = {
            "user_id": f"user_{i:05d}",
            "user_history": [
                {
                    "item_id": (i * 100 + j) % 10000,
                    "timestamp": 1609459200 + j * 3600,
                }
                for j in range(seq_length)
            ],
            "candidates": [
                {
                    "item_candidate_id": 5000 + (i * 30 + k) % 5000,
                    "query_time": 1609459200 + i * 600,
                }
                for k in range(num_candidates)
            ],
        }
        prompts.append(prompt)

    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run batch recommendation inference with vllm-GR",
    )
    add_runtime_args(parser, default_model=DEMO_MODEL_PATH)
    parser.add_argument("--num-users", type=int, default=10, help="Number of users to generate.")
    parser.add_argument("--seq-length", type=int, default=20, help="History length per user.")
    parser.add_argument("--num-candidates", type=int, default=30, help="Candidates per user.")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for recommend_batch()."
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=64, help="Maximum user history sequence length."
    )
    parser.add_argument(
        "--max-candidates", type=int, default=50, help="Maximum number of candidate items."
    )
    return parser


def main():
    """Run batch recommendation inference."""
    args = build_parser().parse_args()
    from vllm_gr import GRLLM, GRRequestOutput

    print("=" * 60)
    print(" vllm-GR Batch Inference Example")
    print("=" * 60)

    # 1. Initialize GRLLM
    #
    # Model Loading Modes:
    # - Production: GRLLM(model="path/to/pretrained_model")
    #   -> Loads config.json and weights from the directory
    #
    # - Demo mode: GRLLM(model="path/to/demo-model", allow_missing_weights=True)
    #   -> Loads config.json for architecture parameters
    #   -> Uses randomly initialized weights (explicit opt-in only)
    #   -> Useful for validating the complete inference pipeline

    print("\n[Model Path]")
    print(f"   Using model: {args.model}")

    llm = GRLLM(
        model=args.model,
        max_seq_len=args.max_seq_len,
        max_candidates=args.max_candidates,
        allow_missing_weights=True,
    )
    active_device = describe_runtime(expected_device=args.expect_device)

    print("\n[GRLLM Initialized]")
    print(f"   Engine: {llm.engine_class.__name__}")
    print(f"   Device path: {active_device}")
    print("   Model: HSTU with randomly initialized weights (explicit demo mode)")

    # 2. Test parameters
    num_users = args.num_users
    seq_length = args.seq_length
    num_candidates = args.num_candidates

    print("\n[Batch Test Parameters]")
    print(f"   Number of users: {num_users}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Candidates per user: {num_candidates}")

    # 3. Generate GRPrompt requests
    prompts = generate_sample_prompts(num_users, seq_length, num_candidates)
    print(f"\n[Generated {len(prompts)} GRPrompt requests]")

    # 4. Method 1: Use recommend() with all prompts at once
    print("\n[Method 1: recommend() with all prompts]")
    print("   Running HSTU model inference...")
    start_time = time.time()
    outputs: List[GRRequestOutput] = llm.recommend(
        prompts=prompts,
        use_tqdm=False,
    )
    method1_time = (time.time() - start_time) * 1000
    print(f"   Time: {method1_time:.2f}ms")
    print(f"   Outputs: {len(outputs)}")

    # 5. Method 2: Use recommend_batch() for large batches
    print(f"\n[Method 2: recommend_batch() with batch_size={args.batch_size}]")
    start_time = time.time()
    outputs_batch: List[GRRequestOutput] = llm.recommend_batch(
        prompts=prompts,
        batch_size=args.batch_size,
    )
    method2_time = (time.time() - start_time) * 1000
    print(f"   Time: {method2_time:.2f}ms")
    print(f"   Outputs: {len(outputs_batch)}")

    # 6. Display results for all users
    print("\n[Results By User]")
    print("   (Note: Scores are from explicitly enabled random-init demo mode)")
    for i, output in enumerate(outputs):
        print(f"\n   User {i}: {output.user_id}")
        print(f"     Request ID: {output.request_id}")
        print("     Candidate scores:")
        for item in output.recommendations:
            display_line = f"item_{item.item_id} score={item.score:.3f}"
            if item.task_scores:
                task_suffix = "  ".join(
                    f"{task_name}={task_score:.3f}"
                    for task_name, task_score in item.task_scores.items()
                )
                display_line = f"{display_line}  {task_suffix}"
            print(f"       {display_line}")

    # 7. Summary statistics
    print("\n[Performance Summary]")
    print(
        f"   Method 1 (recommend):       {method1_time:.2f}ms total, {method1_time / num_users:.2f}ms/request"
    )
    print(
        f"   Method 2 (recommend_batch): {method2_time:.2f}ms total, {method2_time / num_users:.2f}ms/request"
    )
    print(f"   Throughput: {num_users / (method1_time / 1000):.2f} requests/sec")

    # 8. Verify output structure
    print("\n[Output Structure Verification]")
    sample = outputs[0]
    print(f"   Type: {type(sample).__name__}")
    print("   Attributes: request_id, user_id, recommendations, finished, metrics")
    print(f"   Recommendation type: {type(sample.recommendations[0]).__name__}")
    print("   Item attributes: item_id, score, features, task_scores")

    # 9. Serialization example
    print("\n[Serialization Example]")
    output_dict = outputs[0].to_dict()
    print(f"   Keys: {list(output_dict.keys())}")
    print(f"   First recommendation: {output_dict['recommendations'][0]}")

    # 10. Cleanup
    llm.shutdown()

    print("\n" + "=" * 60)
    print(" Batch example completed successfully!")
    print(f" Full HSTU model forward pass was executed on {active_device}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
