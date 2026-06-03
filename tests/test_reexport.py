# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for re-export compatibility after module reorganization."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import unittest


class TestHstuPromptIdentityReexport(unittest.TestCase):
    """Old path vllm_gr.inputs.hstu_prompt_identity should still work."""

    def test_old_path_import(self) -> None:
        from vllm_gr.inputs.hstu_prompt_identity import (
            build_hstu_cache_salt,
            build_hstu_prompt_token_ids,
        )

        self.assertTrue(callable(build_hstu_prompt_token_ids))
        self.assertTrue(callable(build_hstu_cache_salt))

    def test_new_path_import(self) -> None:
        from vllm_gr.inputs.hstu.prompt_identity import (
            build_hstu_cache_salt,
            build_hstu_prompt_token_ids,
        )

        self.assertTrue(callable(build_hstu_prompt_token_ids))
        self.assertTrue(callable(build_hstu_cache_salt))

    def test_same_objects(self) -> None:
        from vllm_gr.inputs.hstu.prompt_identity import build_hstu_cache_salt as new_fn
        from vllm_gr.inputs.hstu_prompt_identity import build_hstu_cache_salt as old_fn

        self.assertIs(old_fn, new_fn)


class TestGrSchemaReexport(unittest.TestCase):
    """Old path vllm_gr.gr_schema should still work."""

    def test_old_path_import(self) -> None:
        from vllm_gr.gr_schema import (
            attach_schema_fingerprint,
            build_schema_fingerprint,
            extract_features,
        )

        self.assertTrue(callable(attach_schema_fingerprint))
        self.assertTrue(callable(build_schema_fingerprint))
        self.assertTrue(callable(extract_features))

    def test_new_path_import(self) -> None:
        from vllm_gr.inputs.hstu.schema import (
            extract_features,
        )

        self.assertTrue(callable(extract_features))

    def test_same_objects(self) -> None:
        from vllm_gr.gr_schema import extract_features as old_fn
        from vllm_gr.inputs.hstu.schema import extract_features as new_fn

        self.assertIs(old_fn, new_fn)

    def test_functional_equivalence(self) -> None:
        """Verify the re-exported function still works correctly."""
        from vllm_gr.gr_schema import extract_features

        keys, values, lengths = extract_features(
            records=[{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            feature_names=["a", "b"],
            group_name="test",
        )
        self.assertEqual(keys, ["a", "b"])
        self.assertEqual(values, [[1, 3], [2, 4]])
        self.assertEqual(lengths, [2, 2])


class TestAdapterImports(unittest.TestCase):
    """Verify existing adapter code still imports correctly."""

    def test_feature_extractor_import(self) -> None:
        """adapters/hstu/feature_extractor imports from vllm_gr.gr_schema."""
        from vllm_gr.adapters.hstu.feature_extractor import (
            extract_uih_features,
        )

        self.assertTrue(callable(extract_uih_features))

    def test_adapter_import(self) -> None:
        """adapters/hstu/adapter imports from hstu_prompt_identity."""
        from vllm_gr.adapters.hstu.adapter import HSTUModelAdapter

        self.assertTrue(callable(HSTUModelAdapter))


if __name__ == "__main__":
    unittest.main()
