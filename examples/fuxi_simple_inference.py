#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Simple FuXi-alpha inference example for vllm-GR."""

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
FUXI_DEMO_MODEL_PATH = os.path.join(SCRIPT_DIR, "fuxi-demo-model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple FuXi-alpha recommendation example",
    )
    add_runtime_args(parser, default_model=FUXI_DEMO_MODEL_PATH)
    parser.add_argument(
        "--max-seq-len", type=int, default=64, help="Maximum user history sequence length."
    )
    parser.add_argument(
        "--max-candidates", type=int, default=50, help="Maximum number of candidate items."
    )
    return parser


def main():
    """Run a simple FuXi-alpha recommendation inference."""
    args = build_parser().parse_args()
    from vllm_gr import GRLLM, GRPrompt

    print("=" * 60)
    print(" vllm-GR FuXi-alpha Simple Inference Example")
    print("=" * 60)

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
        print("   Model: FuXi-alpha with randomly initialized weights (explicit demo mode)")

        # Keep this aligned with examples/fuxi-demo-model/config.json:
        #   history keys: item_id, timestamp
        #   candidate keys: item_id
        prompt: GRPrompt = {
            "user_id": "user_fuxi_demo",
            "user_history": [
                {"item_id": 1001, "timestamp": 1609459200},
                {"item_id": 1002, "timestamp": 1609545600},
                {"item_id": 1003, "timestamp": 1609632000},
                {"item_id": 1004, "timestamp": 1609718400},
                {"item_id": 1005, "timestamp": 1609804800},
            ],
            "candidates": [
                {"item_id": 2001},
                {"item_id": 2002},
                {"item_id": 2003},
                {"item_id": 2004},
                {"item_id": 2005},
            ],
        }

        print("\n[Request Details]")
        print(f"   User: {prompt['user_id']}")
        print(f"   History length: {len(prompt['user_history'])}")
        print(f"   Candidates: {len(prompt['candidates'])}")

        # GRLLM.recommend() -> GRLLMEngine -> EngineCore -> FuXiForRecommendation
        print("\n[Running FuXi-alpha model inference...]")
        outputs = llm.recommend(
            prompts=[prompt],
            use_tqdm=False,
        )

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

        print("\n[Output Summary]")
        print(f"   First 5 item IDs: {output.item_ids[:5]}")
        print(f"   First 5 scores: {[f'{s:.3f}' for s in output.scores[:5]]}")
        print(f"   Best-scoring candidate (derived client-side): {output.top_recommendation}")

        output_dict = output.to_dict()
        print(f"   Serializable keys: {list(output_dict.keys())}")

    print("\n" + "=" * 60)
    print(" Example completed successfully!")
    print(f" Full FuXi-alpha model forward pass was executed on {active_device}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
