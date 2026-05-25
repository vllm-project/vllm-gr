#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Dataset-specific inference examples for vllm-GR.

Demonstrates how to build GRPrompt requests for each DLRM_v3 dataset
supported by Meta's generative-recommenders:

  - MovieLens-1M  : Movie rating prediction with rich user demographics
  - KuaiRand-1K   : Short-video recommendation with 8 engagement tasks
  - Streaming     : Large-scale content streaming with category features
  - Synthetic     : Synthetic dataset for development / CI

Each dataset uses its own feature names, embedding tables, and merge
mappings.  The config files under examples/configs/<dataset>/ mirror
the production-scale configs from Meta's DLRM_v3 but with smaller
dimensions for fast demo execution.

Usage:
    python dataset_inference.py                   # runs all datasets
    python dataset_inference.py movielens-1m      # single dataset
    python dataset_inference.py kuairand-1k streaming
"""

import os
import time
import argparse
from typing import TYPE_CHECKING, List, Dict, Any

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from runtime_utils import add_repo_root_to_path, add_runtime_args, describe_runtime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
add_repo_root_to_path(__file__)

if TYPE_CHECKING:
    from vllm_gr import GRPrompt, GRRequestOutput


# =========================================================================
# Dataset-specific GRPrompt builders
# =========================================================================

def build_movielens_1m_prompts() -> Dict[str, Any]:
    """MovieLens-1M: movie rating prediction with user demographics.

    UIH features:  movie_id, movie_rating, action_timestamp,
                   item_weights, dummy_watch_time
    Candidates:    item_movie_id, item_movie_rating, item_query_time,
                   item_action_weights, item_dummy_watchtime
    Context:       user_id, sex, age_group, occupation, zip_code
    Task:          rating

    NOTE: All feature values must be pre-processed embedding indices
    (0 <= value < num_embeddings).  In the original Meta pipeline,
    pd.Categorical().cat.codes converts raw values (e.g. zip_code
    "48067") into sequential IDs (e.g. 567).
    """
    prompts: List["GRPrompt"] = [
        {
            "user_id": "ml_user_001",
            "user_history": [
                {"movie_id": 1193, "movie_rating": 5, "action_timestamp": 978300760,
                 "item_weights": 1, "dummy_watch_time": 0},
                {"movie_id": 661,  "movie_rating": 3, "action_timestamp": 978302109,
                 "item_weights": 1, "dummy_watch_time": 0},
                {"movie_id": 914,  "movie_rating": 3, "action_timestamp": 978301968,
                 "item_weights": 2, "dummy_watch_time": 0},
                {"movie_id": 3408, "movie_rating": 4, "action_timestamp": 978300275,
                 "item_weights": 1, "dummy_watch_time": 0},
                {"movie_id": 2355, "movie_rating": 5, "action_timestamp": 978824291,
                 "item_weights": 4, "dummy_watch_time": 0},
            ],
            "candidates": [
                {"item_movie_id": 1,   "item_movie_rating": 0, "item_query_time": 978830000, "item_action_weights": 0, "item_dummy_watchtime": 0},
                {"item_movie_id": 48,  "item_movie_rating": 0, "item_query_time": 978830000, "item_action_weights": 0, "item_dummy_watchtime": 0},
                {"item_movie_id": 150, "item_movie_rating": 0, "item_query_time": 978830000, "item_action_weights": 0, "item_dummy_watchtime": 0},
                {"item_movie_id": 260, "item_movie_rating": 0, "item_query_time": 978830000, "item_action_weights": 0, "item_dummy_watchtime": 0},
                {"item_movie_id": 527, "item_movie_rating": 0, "item_query_time": 978830000, "item_action_weights": 0, "item_dummy_watchtime": 0},
            ],
            "context": {
                "user_id": 1, "sex": 0, "age_group": 3,
                "occupation": 7, "zip_code": 567,
            },
        },
    ]
    return {
        "dataset": "movielens-1m",
        "config_dir": os.path.join(SCRIPT_DIR, "configs", "movielens-1m"),
        "prompts": prompts,
        "uih_feature_names": [
            "movie_id", "movie_rating", "action_timestamp",
            "item_weights", "dummy_watch_time",
        ],
        "candidate_feature_names": [
            "item_movie_id", "item_movie_rating", "item_query_time",
            "item_action_weights", "item_dummy_watchtime",
        ],
        "primary_id": "movie_id",
    }


def build_kuairand_1k_prompts() -> Dict[str, Any]:
    """KuaiRand-1K: short-video engagement with 8 tasks.

    UIH features:  video_id, action_timestamp, action_weight, watch_time
    Candidates:    item_video_id, item_action_weight,
                   item_target_watchtime, item_query_time
    Tasks:         is_click, is_like, is_follow, is_comment,
                   is_forward, is_hate, long_view, is_profile_enter
    """
    prompts: List["GRPrompt"] = [
        {
            "user_id": "kr_user_001",
            "user_history": [
                {"video_id": 101, "action_timestamp": 1640000000,
                 "action_weight": 1, "watch_time": 30000},
                {"video_id": 203, "action_timestamp": 1640003600,
                 "action_weight": 2, "watch_time": 15000},
                {"video_id": 55,  "action_timestamp": 1640007200,
                 "action_weight": 4, "watch_time": 60000},
                {"video_id": 789, "action_timestamp": 1640010800,
                 "action_weight": 1, "watch_time": 5000},
                {"video_id": 412, "action_timestamp": 1640014400,
                 "action_weight": 8, "watch_time": 45000},
                {"video_id": 67,  "action_timestamp": 1640018000,
                 "action_weight": 2, "watch_time": 25000},
            ],
            "candidates": [
                {"item_video_id": 1001, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 1640020000},
                {"item_video_id": 1002, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 1640020000},
                {"item_video_id": 1003, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 1640020000},
                {"item_video_id": 1004, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 1640020000},
                {"item_video_id": 1005, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 1640020000},
                {"item_video_id": 1006, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 1640020000},
                {"item_video_id": 1007, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 1640020000},
                {"item_video_id": 1008, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 1640020000},
                {"item_video_id": 1009, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 1640020000},
                {"item_video_id": 1010, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 1640020000},
            ],
            "context": {
                "user_id": 42, "user_active_degree": 3,
                "follow_user_num_range": 5, "fans_user_num_range": 4,
                "friend_user_num_range": 2, "register_days_range": 6,
            },
        },
    ]
    return {
        "dataset": "kuairand-1k",
        "config_dir": os.path.join(SCRIPT_DIR, "configs", "kuairand-1k"),
        "prompts": prompts,
        "uih_feature_names": [
            "video_id", "action_timestamp", "action_weight", "watch_time",
        ],
        "candidate_feature_names": [
            "item_video_id", "item_action_weight",
            "item_target_watchtime", "item_query_time",
        ],
        "primary_id": "video_id",
    }


def build_streaming_prompts() -> Dict[str, Any]:
    """Streaming: large-scale content with category features.

    UIH features:  item_id, item_rating, action_timestamp,
                   item_weights, dummy_watch_time, item_category_id
    Candidates:    item_candidate_id, item_candidate_rating,
                   item_query_time, item_action_weights,
                   item_dummy_watchtime, item_candidate_category_id
    Task:          rating
    """
    prompts: List["GRPrompt"] = [
        {
            "user_id": "stream_user_001",
            "user_history": [
                {"item_id": 500, "item_rating": 4, "action_timestamp": 1650000000,
                 "item_weights": 1, "dummy_watch_time": 0, "item_category_id": 3},
                {"item_id": 501, "item_rating": 5, "action_timestamp": 1650003600,
                 "item_weights": 2, "dummy_watch_time": 0, "item_category_id": 7},
                {"item_id": 502, "item_rating": 3, "action_timestamp": 1650007200,
                 "item_weights": 1, "dummy_watch_time": 0, "item_category_id": 3},
                {"item_id": 503, "item_rating": 4, "action_timestamp": 1650010800,
                 "item_weights": 4, "dummy_watch_time": 0, "item_category_id": 12},
            ],
            "candidates": [
                {"item_candidate_id": 2001, "item_candidate_rating": 0, "item_query_time": 1650020000, "item_action_weights": 0, "item_dummy_watchtime": 0, "item_candidate_category_id": 3},
                {"item_candidate_id": 2002, "item_candidate_rating": 0, "item_query_time": 1650020000, "item_action_weights": 0, "item_dummy_watchtime": 0, "item_candidate_category_id": 7},
                {"item_candidate_id": 2003, "item_candidate_rating": 0, "item_query_time": 1650020000, "item_action_weights": 0, "item_dummy_watchtime": 0, "item_candidate_category_id": 3},
                {"item_candidate_id": 2004, "item_candidate_rating": 0, "item_query_time": 1650020000, "item_action_weights": 0, "item_dummy_watchtime": 0, "item_candidate_category_id": 12},
                {"item_candidate_id": 2005, "item_candidate_rating": 0, "item_query_time": 1650020000, "item_action_weights": 0, "item_dummy_watchtime": 0, "item_candidate_category_id": 5},
                {"item_candidate_id": 2006, "item_candidate_rating": 0, "item_query_time": 1650020000, "item_action_weights": 0, "item_dummy_watchtime": 0, "item_candidate_category_id": 7},
                {"item_candidate_id": 2007, "item_candidate_rating": 0, "item_query_time": 1650020000, "item_action_weights": 0, "item_dummy_watchtime": 0, "item_candidate_category_id": 3},
                {"item_candidate_id": 2008, "item_candidate_rating": 0, "item_query_time": 1650020000, "item_action_weights": 0, "item_dummy_watchtime": 0, "item_candidate_category_id": 12},
            ],
            "context": {"user_id": 100},
        },
    ]
    return {
        "dataset": "streaming",
        "config_dir": os.path.join(SCRIPT_DIR, "configs", "streaming"),
        "prompts": prompts,
        "uih_feature_names": [
            "item_id", "item_rating", "action_timestamp",
            "item_weights", "dummy_watch_time", "item_category_id",
        ],
        "candidate_feature_names": [
            "item_candidate_id", "item_candidate_rating",
            "item_query_time", "item_action_weights",
            "item_dummy_watchtime", "item_candidate_category_id",
        ],
        "primary_id": "item_id",
    }


def build_synthetic_prompts() -> Dict[str, Any]:
    """Synthetic preset for development and CI testing.

    UIH features:  uih_post_id, uih_action_time, uih_weight,
                   uih_owner_id, uih_watchtime, uih_surface_type,
                   uih_video_length
    Candidates:    item_post_id, item_owner_id, item_surface_type,
                   item_video_length, item_action_weight,
                   item_target_watchtime, item_query_time
    Context:       viewer_id, dummy_contexual
    Task:          vvp100
    """
    prompts: List["GRPrompt"] = [
        {
            "user_id": "synthetic_user_001",
            "user_history": [
                {"uih_post_id": 10, "uih_action_time": 1000, "uih_weight": 1,
                 "uih_owner_id": 100, "uih_watchtime": 500,
                 "uih_video_length": 60, "uih_surface_type": 1},
                {"uih_post_id": 20, "uih_action_time": 2000, "uih_weight": 2,
                 "uih_owner_id": 200, "uih_watchtime": 300,
                 "uih_video_length": 120, "uih_surface_type": 2},
                {"uih_post_id": 30, "uih_action_time": 3000, "uih_weight": 1,
                 "uih_owner_id": 100, "uih_watchtime": 800,
                 "uih_video_length": 90, "uih_surface_type": 1},
            ],
            "candidates": [
                {"item_post_id": 50, "item_owner_id": 500, "item_surface_type": 0, "item_video_length": 0, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 4000},
                {"item_post_id": 60, "item_owner_id": 600, "item_surface_type": 0, "item_video_length": 0, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 4000},
                {"item_post_id": 70, "item_owner_id": 500, "item_surface_type": 0, "item_video_length": 0, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 4000},
                {"item_post_id": 80, "item_owner_id": 700, "item_surface_type": 0, "item_video_length": 0, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 4000},
                {"item_post_id": 90, "item_owner_id": 600, "item_surface_type": 0, "item_video_length": 0, "item_action_weight": 0, "item_target_watchtime": 0, "item_query_time": 4000},
            ],
            "context": {"viewer_id": 42, "dummy_contexual": 1},
        },
    ]
    return {
        "dataset": "debug",
        "config_dir": os.path.join(SCRIPT_DIR, "configs", "debug"),
        "prompts": prompts,
        "uih_feature_names": [
            "uih_post_id", "uih_action_time", "uih_weight",
            "uih_owner_id", "uih_watchtime", "uih_surface_type",
            "uih_video_length",
        ],
        "candidate_feature_names": [
            "item_post_id", "item_owner_id",
            "item_surface_type", "item_video_length",
            "item_action_weight", "item_target_watchtime", "item_query_time",
        ],
        "primary_id": "uih_post_id",
    }


DATASET_BUILDERS = {
    "movielens-1m": build_movielens_1m_prompts,
    "kuairand-1k": build_kuairand_1k_prompts,
    "streaming": build_streaming_prompts,
    "debug": build_synthetic_prompts,
}


# =========================================================================
# Runner
# =========================================================================

def run_dataset(spec: Dict[str, Any], args: argparse.Namespace) -> None:
    """Run inference for a single dataset specification."""
    from vllm_gr import GRLLM, GRRequestOutput
    dataset = spec["dataset"]
    config_dir = spec["config_dir"]
    prompts = spec["prompts"]

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset}")
    print(f"  Config:  {config_dir}")
    print(f"{'=' * 60}")

    print(f"\n  UIH features:       {spec['uih_feature_names']}")
    print(f"  Candidate features: {spec['candidate_feature_names']}")
    print(f"  Primary ID:         {spec['primary_id']}")

    print(f"\n  Initializing GRLLM...")
    with GRLLM(
        model=config_dir,
        max_seq_len=args.max_seq_len,
        max_candidates=args.max_candidates,
        allow_missing_weights=True,
    ) as llm:
        active_device = describe_runtime(expected_device=args.expect_device)

        print(f"  Running inference ({len(prompts)} request(s))...")
        start = time.time()
        outputs: List[GRRequestOutput] = llm.recommend(
            prompts=prompts,
            use_tqdm=False,
        )
        elapsed_ms = (time.time() - start) * 1000

        for output in outputs:
            print(f"\n  User: {output.user_id}")
            print(
                "  Returned scores for all candidates "
                f"({len(output.recommendations)}):"
            )
            pid = spec["primary_id"]
            for item in output.recommendations:
                display_line = f"{pid}={item.item_id} score={item.score:.4f}"
                if item.task_scores:
                    task_suffix = "  ".join(
                        f"{task_name}={task_score:.4f}"
                        for task_name, task_score in item.task_scores.items()
                    )
                    display_line = f"{display_line}  {task_suffix}"
                print(f"    {display_line}")

        print("\n  Output contract: no top-k truncation is applied in the example.")
        print("  Caller-side ranking / filtering should be done downstream.")
        print("  Demo path: config-only example run with explicit random-init opt-in.")
        print(f"  Device path: {active_device}")
        print(f"  Latency: {elapsed_ms:.1f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Run DLRM_v3 inference with different datasets",
    )
    add_runtime_args(parser, include_model=False)
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DATASET_BUILDERS.keys()),
        choices=sorted(DATASET_BUILDERS.keys()),
        help=f"Datasets to run. Choices: {list(DATASET_BUILDERS.keys())}",
    )
    parser.add_argument("--max-seq-len", type=int, default=64, help="Maximum user history sequence length.")
    parser.add_argument("--max-candidates", type=int, default=50, help="Maximum number of candidate items.")
    args = parser.parse_args()

    print("=" * 60)
    print("  vllm-GR Dataset Inference Examples")
    print("  Supported datasets from Meta DLRM_v3:")
    for name in DATASET_BUILDERS:
        marker = " *" if name in args.datasets else ""
        print(f"    - {name}{marker}")
    print("=" * 60)

    for name in args.datasets:
        spec = DATASET_BUILDERS[name]()
        run_dataset(spec, args)

    print(f"\n{'=' * 60}")
    print(f"  All done! Ran {len(args.datasets)} dataset(s).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
