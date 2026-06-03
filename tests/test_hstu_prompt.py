# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for inputs/hstu/prompt.py — HSTUPrompt."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import unittest

from vllm_gr.inputs.hstu.prompt import HSTUPrompt


class TestHSTUPrompt(unittest.TestCase):
    """HSTUPrompt extends GRPromptBase with HSTU-specific fields."""

    def test_inherits_gr_prompt_base(self) -> None:
        # TypedDict inheritance is structural: parent fields appear in
        # the child's __annotations__ and __required_keys__.
        # Verify that GRPromptBase fields are present in HSTUPrompt.
        self.assertIn("user_id", HSTUPrompt.__annotations__)
        self.assertIn("model_type", HSTUPrompt.__annotations__)
        # user_id is required in both GRPromptBase and HSTUPrompt
        self.assertIn("user_id", HSTUPrompt.__required_keys__)

    def test_required_keys(self) -> None:
        self.assertEqual(
            HSTUPrompt.__required_keys__,
            frozenset({"user_id", "user_history", "candidates"}),
        )

    def test_optional_keys(self) -> None:
        self.assertEqual(
            HSTUPrompt.__optional_keys__,
            frozenset({"model_type", "context", "full_uih_keys"}),
        )

    def test_valid_prompt(self) -> None:
        prompt: HSTUPrompt = {
            "user_id": 42,
            "user_history": [{"movie_id": 1}],
            "candidates": [{"item_movie_id": 10}],
        }
        self.assertEqual(prompt["user_id"], 42)
        self.assertEqual(len(prompt["candidates"]), 1)

    def test_prompt_with_all_optional(self) -> None:
        prompt: HSTUPrompt = {
            "user_id": "u1",
            "model_type": "hstu",
            "user_history": [],
            "candidates": [],
            "context": {"age": 25},
            "full_uih_keys": ["movie_id", "rating"],
        }
        self.assertEqual(prompt["model_type"], "hstu")
        self.assertEqual(prompt["full_uih_keys"], ["movie_id", "rating"])

    def test_data_py_unchanged(self) -> None:
        """GRPrompt in inputs/data.py should have no relationship to HSTUPrompt."""
        from vllm_gr.inputs.data import GRPrompt

        # TypedDicts don't support issubclass; verify that GRPrompt does
        # NOT have HSTUPrompt's full_uih_keys field (they're independent).
        self.assertNotIn("full_uih_keys", GRPrompt.__annotations__)
        # GRPrompt should still have its original keys
        self.assertIn("user_history", GRPrompt.__annotations__)
        self.assertIn("candidates", GRPrompt.__annotations__)


if __name__ == "__main__":
    unittest.main()
