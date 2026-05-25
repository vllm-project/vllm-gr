# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for replay-style prefix-cache benchmark input generation."""

from __future__ import annotations

import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from examples.build_prefix_cache_benchmark_input import build_replay_payload


class TestBuildPrefixCacheBenchmarkInput(unittest.TestCase):

    def test_build_replay_payload_preserves_context_and_history(self):
        payload = {
            "meta": {"dataset": "movielens-1m"},
            "cases": [
                {
                    "case_id": "case_000001",
                    "user_id": "u-1",
                    "context": {"user_id": 1, "sex": 1},
                    "user_history": [
                        {
                            "movie_id": 101,
                            "movie_rating": 5,
                            "action_timestamp": 1001,
                            "item_weights": 1,
                            "dummy_watch_time": 10,
                        },
                        {
                            "movie_id": 102,
                            "movie_rating": 4,
                            "action_timestamp": 1002,
                            "item_weights": 2,
                            "dummy_watch_time": 20,
                        },
                    ],
                    "candidates": [
                        {
                            "item_movie_id": 201,
                            "item_movie_rating": 5,
                            "item_query_time": 2001,
                            "item_action_weights": 1,
                            "item_dummy_watchtime": 30,
                        },
                        {
                            "item_movie_id": 202,
                            "item_movie_rating": 4,
                            "item_query_time": 2002,
                            "item_action_weights": 2,
                            "item_dummy_watchtime": 40,
                        },
                        {
                            "item_movie_id": 203,
                            "item_movie_rating": 3,
                            "item_query_time": 2003,
                            "item_action_weights": 4,
                            "item_dummy_watchtime": 50,
                        },
                    ],
                }
            ],
        }

        replay = build_replay_payload(
            payload,
            max_seed_cases=1,
            variants_per_case=4,
            keep_candidate_prefix=1,
        )

        self.assertEqual(replay["meta"]["generated_num_cases"], 4)
        cases = replay["cases"]
        self.assertEqual(cases[0]["case_id"], "case_000001__replay_01")
        self.assertEqual(cases[1]["case_id"], "case_000001__replay_02")
        self.assertEqual(cases[2]["prefix_cache_variant"]["variant_kind"], "candidate_tail_rotated")

        for case in cases:
            self.assertEqual(case["user_id"], "u-1")
            self.assertEqual(case["context"], payload["cases"][0]["context"])
            self.assertEqual(case["user_history"], payload["cases"][0]["user_history"])

        self.assertEqual(
            cases[0]["candidates"],
            payload["cases"][0]["candidates"],
        )
        self.assertEqual(
            cases[1]["candidates"],
            payload["cases"][0]["candidates"],
        )
        self.assertEqual(cases[2]["candidates"][0], payload["cases"][0]["candidates"][0])
        self.assertNotEqual(cases[2]["candidates"][1:], payload["cases"][0]["candidates"][1:])


if __name__ == "__main__":
    unittest.main()
