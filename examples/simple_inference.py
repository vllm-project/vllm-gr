#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Simple inference example for vllm-GR."""

import argparse

# Enable verbose logging
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

from runtime_utils import (  # noqa: E402
    add_repo_root_to_path,
    add_runtime_args,
    describe_runtime,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
add_repo_root_to_path(__file__)

# Demo model path - relative to examples directory
DEMO_MODEL_PATH = os.path.join(SCRIPT_DIR, "demo-model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple vllm-GR recommendation example",
    )
    add_runtime_args(parser, default_model=DEMO_MODEL_PATH)
    parser.add_argument(
        "--max-seq-len", type=int, default=64, help="Maximum user history sequence length."
    )
    parser.add_argument(
        "--max-candidates", type=int, default=50, help="Maximum number of candidate items."
    )
    return parser


def main():
    """Run a simple recommendation inference."""
    args = build_parser().parse_args()
    from vllm_gr import GRLLM, GRPrompt

    print("=" * 60)
    print(" vllm-GR Simple Inference Example")
    print("=" * 60)

    # 1. Initialize GRLLM
    #
    # Model architecture (layers, heads, embedding_dim, etc.) comes from
    # the model directory's config.json -- following the vLLM convention.
    # GRLLM only accepts runtime/serving parameters (max_seq_len, candidate limits, etc.).

    print("\n[Model Path]")
    print(f"   Using model: {args.model}")

    with GRLLM(
        model=args.model,
        max_seq_len=args.max_seq_len,
        max_candidates=args.max_candidates,
        allow_missing_weights=True,
    ) as llm:
        active_device = describe_runtime(expected_device=args.expect_device)

        print("\n[Configuration]")
        print(f"   Max sequence length: {llm.max_seq_len}")
        print(f"   Max candidates: {llm.max_candidates}")
        print(f"   Engine: {llm.engine_class.__name__}")
        print(f"   Device path: {active_device}")
        print("   Model: HSTU with randomly initialized weights (explicit demo mode)")

        # 2. Prepare a recommendation request using GRPrompt format
        # GRPrompt is the standard input format for vllm-GR.
        # Keep this aligned with examples/demo-model/config.json:
        #   UIH keys: item_id, timestamp
        #   candidate keys: item_candidate_id, query_time
        prompt: GRPrompt = {
            "user_id": "user_alice_12345",
            # User's interaction history
            "user_history": [
                {"item_id": 1001, "timestamp": 1609459200},  # The_Matrix
                {"item_id": 1002, "timestamp": 1609545600},  # Inception
                {"item_id": 1003, "timestamp": 1609632000},  # Interstellar
                {"item_id": 1004, "timestamp": 1609718400},  # Blade_Runner_2049
                {"item_id": 1005, "timestamp": 1609804800},  # Dune
                {"item_id": 1006, "timestamp": 1609891200},  # Arrival
                {"item_id": 1007, "timestamp": 1609977600},  # Ex_Machina
            ],
            # Candidate items to score (keys use model-internal candidate names)
            "candidates": [
                {"item_candidate_id": 2001, "query_time": 1610000000},
                {"item_candidate_id": 2002, "query_time": 1610000000},
                {"item_candidate_id": 2003, "query_time": 1610000000},
                {"item_candidate_id": 2004, "query_time": 1610000000},
                {"item_candidate_id": 2005, "query_time": 1610000000},
                {"item_candidate_id": 2006, "query_time": 1610000000},
                {"item_candidate_id": 2007, "query_time": 1610000000},
                {"item_candidate_id": 2008, "query_time": 1610000000},
                {"item_candidate_id": 2009, "query_time": 1610000000},
                {"item_candidate_id": 2010, "query_time": 1610000000},
                {"item_candidate_id": 2011, "query_time": 1610000000},
                {"item_candidate_id": 2012, "query_time": 1610000000},
                {"item_candidate_id": 2013, "query_time": 1610000000},
                {"item_candidate_id": 2014, "query_time": 1610000000},
                {"item_candidate_id": 2015, "query_time": 1610000000},
            ],
        }

        print("\n[Request Details]")
        print(f"   User: {prompt['user_id']}")
        print(f"   History length: {len(prompt['user_history'])}")
        print(f"   Candidates: {len(prompt['candidates'])}")

        # 3. Get candidate scores
        # This runs the full HSTU model forward pass:
        # GRLLM.recommend() -> GRLLMEngine -> EngineCore -> HSTUForRecommendation -> DlrmHSTU
        print("\n[Running HSTU model inference...]")
        outputs = llm.recommend(
            prompts=[prompt],
            use_tqdm=False,
        )

        # 4. Display results
        output = outputs[0]

        print("\n[Recommendation Results]")
        print(f"   Request ID: {output.request_id}")
        print(f"   User ID: {output.user_id}")
        print(f"   Finished: {output.finished}")
        print(f"   Returned scores for all candidates: {len(output.recommendations)}")
        print("   (Note: Scores are from explicitly enabled random-init demo mode)")

        print("\n   Candidate score list:")
        for item in output.recommendations:
            display_line = f"item_{item.item_id:<10} score={item.score:.4f}"
            if item.task_scores:
                task_suffix = "  ".join(
                    f"{task_name}={task_score:.4f}"
                    for task_name, task_score in item.task_scores.items()
                )
                display_line = f"{display_line}  {task_suffix}"
            print(f"   {display_line}")

        # 5. Access additional properties
        print("\n[Output Summary]")
        print(f"   First 5 item IDs: {output.item_ids[:5]}")
        print(f"   First 5 scores: {[f'{s:.3f}' for s in output.scores[:5]]}")
        print(f"   Best-scoring candidate (derived client-side): {output.top_recommendation}")

        # Convert to dict for JSON serialization
        output_dict = output.to_dict()
        print(f"   Serializable keys: {list(output_dict.keys())}")

    print("\n" + "=" * 60)
    print(" Example completed successfully!")
    print(f" Full HSTU model forward pass was executed on {active_device}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
