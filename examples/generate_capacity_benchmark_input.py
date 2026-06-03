#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate high-diversity benchmark JSON for capacity scenario.

This script creates a benchmark input file with:
- 2048 unique user_history records (to support warm_unique_requests up to 2048)
- 64 unique candidates records
- Wide value ranges to minimize prefix overlap across different requests

Usage
=====

1. Generate the default capacity benchmark input file:

    python examples/generate_capacity_benchmark_input.py

    Output: examples/prefix_cache_benchmark.capacity_input.json (~395 KB)

2. Use the generated file with the capacity benchmark:

    python examples/prefix_cache_scaling_benchmark.py \
        --model /workspace/ckpt/movielens_1m/800_truncated_new/converted \
        --input-json ./examples/prefix_cache_benchmark.capacity_input.json \
        --expect-device npu \
        --modes both \
        --scenario capacity \
        --sequence-lengths 1024 \
        --max-uih-len 8192 \
        --max-candidates 256 \
        --gpu-memory-utilization 0.08 \
        --warm-unique-requests 512 \
        --probe-oldest-requests 128 \
        --revisit-rounds 4

Why This File Is Needed
========================

The original sample input (prefix_cache_benchmark.sample_input.json) has only
120 user_history templates. When warm_unique_requests exceeds 120, the
_build_unique_records() function in prefix_cache_scaling_benchmark.py cycles
through the template pool, causing different warm requests to share prefix
blocks. This produces inflated HBM hit rates (~84%) that don't reflect true
cache behavior.

This generator creates 2048 unique user_history records with wide value ranges
(movie_id: 1001-3048, action_timestamp step: 31, item_weights: 1-32), ensuring
that each warm request selects a distinct combination of records and the warm
phase produces near-zero HBM hits, as expected theoretically.

Customization
=============

Run with command-line arguments to override defaults:

    python examples/generate_capacity_benchmark_input.py --user-history-count 4096

Or edit the constants in main() directly. Key parameters:

    user_history_count  - Number of unique user_history records (default: 2048)
                          Should be >= max warm_unique_requests value used in benchmarks.
    candidates_count    - Number of unique candidates records (default: 64)

To change value generation patterns, edit generate_user_history_records() and
generate_candidates_records(). Key parameters:

    movie_id            - Starts at 1001, increments by 1 per record
    action_timestamp    - Starts at 1700000001, increments by 31 per record
    item_weights        - Cycles through [1, 2, 4, 8, 16, 32]
    dummy_watch_time    - Starts at 20, increments by 1 per record (never cycles)

Arguments
---------

--user-history-count (default: 2048)
    Number of unique user_history records to generate. Should be at least as
    large as the --warm-unique-requests value used in the benchmark. For example,
    if running with --warm-unique-requests 512, the default 2048 is sufficient.
    For larger warm pools (e.g., 1024+), increase this value accordingly.

--candidates-count (default: 64)
    Number of unique candidates records to generate. More candidates increase
    the diversity of the candidate portion of each request.

--output (default: examples/prefix_cache_benchmark.capacity_input.json)
    Path to write the generated JSON file. Defaults to a file alongside this
    script in the examples/ directory.

Output Format
=============

The generated JSON follows the same schema as prefix_cache_benchmark.sample_input.json:

    {
      "meta": { "dataset": "movielens-1m", "description": "..." },
      "cases": [
        {
          "case_id": "capacity_diverse_u1",
          "user_id": "u_1",
          "context": { "user_id": 1, "sex": 1, "age_group": 3, ... },
          "user_history": [ { "movie_id": 1001, ... }, ... ],
          "candidates": [ { "item_movie_id": 5001, ... }, ... ]
        }
      ]
    }
"""

import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_user_history_records(count: int) -> list[dict]:
    """Generate diverse user_history records."""
    records = []
    for i in range(count):
        movie_id = 1001 + i
        movie_rating = [3, 4, 5][i % 3]
        action_timestamp = 1700000001 + i * 31
        item_weights = [1, 2, 4, 8, 16, 32][i % 6]
        dummy_watch_time = 20 + i
        records.append(
            {
                "movie_id": movie_id,
                "movie_rating": movie_rating,
                "action_timestamp": action_timestamp,
                "item_weights": item_weights,
                "dummy_watch_time": dummy_watch_time,
            }
        )
    return records


def generate_candidates_records(count: int) -> list[dict]:
    """Generate diverse candidates records."""
    records = []
    for i in range(count):
        item_movie_id = 5001 + i * 7
        item_movie_rating = [3, 4, 5][i % 3]
        item_query_time = 1700002000 + i * 100
        item_action_weights = [1, 2, 4, 8, 16][i % 5]
        item_dummy_watchtime = 19 + i * 3
        records.append(
            {
                "item_movie_id": item_movie_id,
                "item_movie_rating": item_movie_rating,
                "item_query_time": item_query_time,
                "item_action_weights": item_action_weights,
                "item_dummy_watchtime": item_dummy_watchtime,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Generate high-diversity benchmark JSON for capacity scenario.",
    )
    parser.add_argument(
        "--user-history-count",
        type=int,
        default=2048,
        help="Number of unique user_history records (default: 2048).",
    )
    parser.add_argument(
        "--candidates-count",
        type=int,
        default=64,
        help="Number of unique candidates records (default: 64).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: examples/prefix_cache_benchmark.capacity_input.json).",
    )
    args = parser.parse_args()

    user_history_count = args.user_history_count
    candidates_count = args.candidates_count
    output_path = args.output or os.path.join(
        SCRIPT_DIR, "prefix_cache_benchmark.capacity_input.json"
    )

    user_history = generate_user_history_records(user_history_count)
    candidates = generate_candidates_records(candidates_count)

    benchmark_data = {
        "meta": {
            "dataset": "movielens-1m",
            "description": (
                f"High-diversity template for capacity benchmark. "
                f"user_history contains {user_history_count} unique records with wide value ranges "
                f"to minimize prefix overlap across warm-unique-requests. "
                f"candidates contains {candidates_count} unique records."
            ),
        },
        "cases": [
            {
                "case_id": "capacity_diverse_u1",
                "user_id": "u_1",
                "context": {
                    "user_id": 1,
                    "sex": 1,
                    "age_group": 3,
                    "occupation": 5,
                    "zip_code": 1201,
                },
                "user_history": user_history,
                "candidates": candidates,
            },
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(f"Generated: {output_path}")
    print(f"  - user_history records: {user_history_count}")
    print(f"  - candidates records: {candidates_count}")
    print(f"  - File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
