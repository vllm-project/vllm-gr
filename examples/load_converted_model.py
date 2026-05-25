#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Verify converted DLRM_v3 checkpoint by loading it and running inference.

This script validates the full pipeline:
  1. Train a DLRM_v3 model with Meta's generative-recommenders (movielens-1m)
  2. Convert the checkpoint:
       python scripts/convert_gr_checkpoint.py \
           --input-path <training_checkpoint_dir> \
           --output-dir <converted_model_dir> \
           --dataset movielens-1m
  3. Run this script to verify the converted weights load correctly:
       python examples/load_converted_model.py \
           --model <converted_model_dir>

The converted model directory must contain:
    config.json        — model architecture + feature configuration
    model.safetensors  — converted weights with "hstu." prefix

The GRPrompt uses movielens-1m feature names to match the training dataset.
For a lightweight demo without real weights, use simple_inference.py instead.
"""

import argparse
import glob
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from runtime_utils import add_repo_root_to_path, add_runtime_args, describe_runtime

add_repo_root_to_path(__file__)


def check_model_dir(model_path: str) -> dict:
    """Validate model directory and report its contents."""
    info = {
        "path": model_path,
        "has_config": False,
        "has_weights": False,
        "weight_files": [],
    }

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    info["has_config"] = os.path.exists(
        os.path.join(model_path, "config.json")
    )

    weight_files = []
    for pattern in ["*.safetensors", "*.bin", "*.pt", "*.pth"]:
        weight_files.extend(glob.glob(os.path.join(model_path, pattern)))

    if weight_files:
        weight_files = sorted(os.path.basename(path) for path in weight_files)
        info["has_weights"] = True
        info["weight_files"] = weight_files
        info["weight_size_mb"] = sum(
            os.path.getsize(os.path.join(model_path, filename))
            for filename in weight_files
        ) / (1024 * 1024)

    return info


def run_inference(model_path: str, expected_device: str | None = None):
    """Load converted model and run a movielens-1m recommendation request."""
    from vllm_gr import GRLLM, GRPrompt

    info = check_model_dir(model_path)

    print("=" * 70)
    print(" vllm-GR: Verify Converted DLRM_v3 Weights (movielens-1m)")
    print("=" * 70)
    print(f"\n  Model path:   {info['path']}")
    print(f"  config.json:  {'found' if info['has_config'] else 'MISSING'}")
    if info["has_weights"]:
        print(f"  Weights:      {', '.join(info['weight_files'])} "
              f"({info['weight_size_mb']:.1f} MB)")
    else:
        raise FileNotFoundError(
            f"No weight file found in {model_path}. "
            "Run convert_gr_checkpoint.py first."
        )

    if not info["has_config"]:
        raise FileNotFoundError(
            f"config.json not found in {model_path}. "
            "Run convert_gr_checkpoint.py with --dataset to generate it."
        )

    llm = GRLLM(
        model=model_path,
        max_seq_len=128,
        max_candidates=10,
    )
    active_device = describe_runtime(expected_device=expected_device)
    print("\n  Model loaded successfully!")

    # GRPrompt with movielens-1m feature names.
    #
    # user_history fields (from hstu_uih_feature_names):
    #   movie_id         — MovieLens item ID (primary sequence ID)
    #   movie_rating     — user's rating (1–5)
    #   action_timestamp — Unix timestamp of the interaction
    #   item_weights     — action weight bitmask (1=click, 2=rate, 4=..., etc.)
    #   dummy_watch_time — placeholder for watch-time (0 for movielens)
    #
    # candidates fields (from hstu_candidate_feature_names):
    #   item_movie_id       — candidate MovieLens item ID
    #   item_query_time     — query timestamp for all candidates
    #   item_dummy_watchtime — placeholder (0)
    #   item_movie_rating   — zeroed in inference (candidates_watchtime_feature)
    #   item_action_weights — zeroed in inference (candidates_weight_feature)
    #
    # context fields (from contextual_feature_to_max_length):
    #   user_id, sex, age_group, occupation, zip_code
    #   These are user-level features appended to the UIH KJT for embedding lookup.
    prompt: GRPrompt = {
        "user_id": "user_42",
        "user_history": [
            {"movie_id": 1,    "movie_rating": 5, "action_timestamp": 978300760, "item_weights": 1, "dummy_watch_time": 0},
            {"movie_id": 260,  "movie_rating": 4, "action_timestamp": 978301398, "item_weights": 1, "dummy_watch_time": 0},
            {"movie_id": 1196, "movie_rating": 5, "action_timestamp": 978302174, "item_weights": 2, "dummy_watch_time": 0},
            {"movie_id": 1210, "movie_rating": 4, "action_timestamp": 978302983, "item_weights": 1, "dummy_watch_time": 0},
            {"movie_id": 2028, "movie_rating": 5, "action_timestamp": 978303390, "item_weights": 4, "dummy_watch_time": 0},
        ],
        "candidates": [
            {"item_movie_id": 3114, "item_query_time": 978310000, "item_dummy_watchtime": 0, "item_movie_rating": 0, "item_action_weights": 0},
            {"item_movie_id": 2571, "item_query_time": 978310000, "item_dummy_watchtime": 0, "item_movie_rating": 0, "item_action_weights": 0},
            {"item_movie_id": 1580, "item_query_time": 978310000, "item_dummy_watchtime": 0, "item_movie_rating": 0, "item_action_weights": 0},
            {"item_movie_id": 1270, "item_query_time": 978310000, "item_dummy_watchtime": 0, "item_movie_rating": 0, "item_action_weights": 0},
            {"item_movie_id": 593,  "item_query_time": 978310000, "item_dummy_watchtime": 0, "item_movie_rating": 0, "item_action_weights": 0},
        ],
        "context": {
            "user_id": 42,
            "sex": 1,
            "age_group": 25,
            "occupation": 7,
            "zip_code": 10001,
        },
    }

    print(f"\n  Running inference...")
    print(f"  User history:  {len(prompt['user_history'])} interactions")
    print(f"  Candidates:    {len(prompt['candidates'])} items")
    print(f"  Context:       {list(prompt['context'].keys())}")
    print(f"  Device path:   {active_device}")

    outputs = llm.recommend(prompts=[prompt], use_tqdm=False)
    output = outputs[0]

    print(f"\n  Candidate scores ({len(output.recommendations)}):")
    for item in output.recommendations:
        display_line = f"movie_{item.item_id} score={item.score:.6f}"
        if item.task_scores:
            task_suffix = "  ".join(
                f"{task_name}={task_score:.6f}"
                for task_name, task_score in item.task_scores.items()
            )
            display_line = f"{display_line}  {task_suffix}"
        print(f"    {display_line}")

    llm.shutdown()
    print("\n" + "=" * 70)
    print(" Verification completed successfully!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Verify converted DLRM_v3 checkpoint (movielens-1m)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/convert_gr_checkpoint.py \\\n"
            "      --input-path /path/to/training_checkpoint \\\n"
            "      --output-dir /path/to/converted_model \\\n"
            "      --dataset movielens-1m\n"
            "\n"
            "  python examples/load_converted_model.py \\\n"
            "      --model /path/to/converted_model\n"
            "\n"
            "For a lightweight demo without real weights, "
            "use simple_inference.py instead."
        ),
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to converted model directory "
             "(with config.json + converted weight files)",
    )
    add_runtime_args(parser, include_model=False)
    args = parser.parse_args()

    run_inference(args.model, expected_device=args.expect_device)


if __name__ == "__main__":
    main()
