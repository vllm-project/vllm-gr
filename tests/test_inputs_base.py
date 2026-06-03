# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for inputs/base.py — GRPromptBase and GRParsedInput."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import unittest

from vllm_gr.inputs.base import GRParsedInput, GRPromptBase


class TestGRPromptBase(unittest.TestCase):
    """GRPromptBase is a TypedDict with only user_id + model_type."""

    def test_required_keys(self) -> None:
        self.assertEqual(GRPromptBase.__required_keys__, frozenset({"user_id"}))

    def test_optional_keys(self) -> None:
        self.assertEqual(GRPromptBase.__optional_keys__, frozenset({"model_type"}))

    def test_dict_instance(self) -> None:
        prompt: GRPromptBase = {"user_id": 42}
        self.assertIsInstance(prompt, dict)

    def test_dict_with_model_type(self) -> None:
        prompt: GRPromptBase = {"user_id": "u1", "model_type": "hstu"}
        self.assertEqual(prompt["model_type"], "hstu")


class TestGRParsedInput(unittest.TestCase):
    """GRParsedInput is a dataclass with model-agnostic fields."""

    def test_default_construction(self) -> None:
        parsed = GRParsedInput()
        self.assertEqual(parsed.user_id, "")
        self.assertEqual(parsed.schema_fingerprint, "")
        self.assertEqual(parsed.raw_features, {})
        self.assertEqual(parsed.metadata, {})
        self.assertEqual(parsed.original_prompt, {})

    def test_construction_with_args(self) -> None:
        parsed = GRParsedInput(
            user_id="u1",
            schema_fingerprint="fp",
            raw_features={"x": 1},
            metadata={"history_len": 5},
            original_prompt={"user_id": "u1"},
        )
        self.assertEqual(parsed.user_id, "u1")
        self.assertEqual(parsed.raw_features, {"x": 1})
        self.assertEqual(parsed.metadata["history_len"], 5)

    def test_independent_defaults(self) -> None:
        """Ensure mutable defaults are not shared across instances."""
        p1 = GRParsedInput()
        p2 = GRParsedInput()
        p1.raw_features["key"] = "val"
        self.assertNotIn("key", p2.raw_features)

    def test_import_from_package(self) -> None:
        """Verify re-export through inputs/__init__.py."""
        from vllm_gr.inputs import GRParsedInput as PI
        from vllm_gr.inputs import GRPromptBase as PB

        self.assertIs(PB, GRPromptBase)
        self.assertIs(PI, GRParsedInput)


if __name__ == "__main__":
    unittest.main()
