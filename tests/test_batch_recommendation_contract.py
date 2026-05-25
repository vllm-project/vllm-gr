# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Core contract tests for multi-request recommendation boundaries."""

from __future__ import annotations

import unittest

try:
    from tests.smoke_test_utils import smoke_harness
except ImportError:
    from smoke_test_utils import smoke_harness


def _build_prompt(user_id: str, candidate_ids: list[int]) -> dict[str, object]:
    return {
        "user_id": user_id,
        "user_history": [
            {"item_id": 1},
            {"item_id": 2},
        ],
        "candidates": [
            {"item_candidate_id": candidate_id}
            for candidate_id in candidate_ids
        ],
    }


class TestBatchRecommendationContract(unittest.TestCase):

    def _assert_output_boundaries(
        self,
        outputs,
        prompts: list[dict[str, object]],
    ) -> None:
        self.assertEqual(len(outputs), len(prompts))
        self.assertEqual(
            [output.user_id for output in outputs],
            [prompt["user_id"] for prompt in prompts],
        )

        outputs_by_user = {output.user_id: output for output in outputs}
        self.assertEqual(len(outputs_by_user), len(prompts))

        for prompt in prompts:
            user_id = prompt["user_id"]
            output = outputs_by_user[user_id]
            candidates = prompt["candidates"]
            expected_item_ids = [
                candidate["item_candidate_id"]
                for candidate in candidates
            ]

            self.assertTrue(output.finished)
            self.assertEqual(output.user_id, user_id)
            self.assertEqual(output.item_ids, expected_item_ids)
            self.assertEqual(len(output.recommendations), len(candidates))
            self.assertEqual(
                [item.features for item in output.recommendations],
                [None] * len(candidates),
            )

    def test_recommend_multi_prompt_preserves_request_boundaries(self):
        prompts = [
            _build_prompt("u1", [101, 102, 103]),
            _build_prompt("u2", [201, 202, 203]),
            _build_prompt("u3", [301, 302, 303]),
        ]

        for device_type in ("cuda", "npu"):
            with self.subTest(device_type=device_type):
                with smoke_harness(device_type) as harness:
                    llm = harness.create_llm()
                    outputs = llm.recommend(prompts, use_tqdm=False)

                    self._assert_output_boundaries(outputs, prompts)
                    self.assertGreaterEqual(
                        harness.state["trace"].count(("hstu", device_type)),
                        len(prompts),
                    )

                    llm.shutdown()

if __name__ == "__main__":
    unittest.main()
