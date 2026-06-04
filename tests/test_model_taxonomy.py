# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Unit tests for GR vs LLM model taxonomy classification."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_model_taxonomy():
    """Load taxonomy module without pulling optional config package deps."""
    module_path = (
        Path(ROOT) / "vllm_gr" / "config" / "model_taxonomy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_gr_model_taxonomy_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_taxonomy = _load_model_taxonomy()
ModelCategory = _taxonomy.ModelCategory
classify_model = _taxonomy.classify_model
is_gr_category = _taxonomy.is_gr_category
is_llm_category = _taxonomy.is_llm_category


class TestClassifyModel(unittest.TestCase):

    def test_explicit_llm_aliases(self):
        for family in ("llm", "pangu", "pangu_llm"):
            with self.subTest(family=family):
                self.assertEqual(
                    classify_model(model_family=family),
                    ModelCategory.LLM,
                )

    def test_explicit_gr_families(self):
        for family in ("hstu", "fuxi"):
            with self.subTest(family=family):
                self.assertEqual(
                    classify_model(model_family=family),
                    ModelCategory.GR,
                )

    def test_pangu_config_architecture(self):
        config = {
            "model_type": "PanguEmbedded",
            "architectures": ["PanguEmbeddedForCausalLM"],
        }
        self.assertEqual(
            classify_model(
                model_family="panguembedded",
                local_config=config,
            ),
            ModelCategory.LLM,
        )

    def test_hstu_config_without_explicit_family(self):
        config = {"model_type": "hstu", "architectures": ["HSTUEmbeddingModel"]}
        self.assertEqual(
            classify_model(model_family=None, local_config=config),
            ModelCategory.GR,
        )

    def test_unresolved_family_with_hstu_config_no_defaults_family(self):
        """Matches test_engine: family None + hstu config still GR engine path."""
        config = {"model_type": "hstu"}
        self.assertEqual(
            classify_model(model_family=None, local_config=config),
            ModelCategory.GR,
        )

    def test_unknown_config_defaults_to_llm(self):
        config = {
            "model_type": "SomeFutureLM",
            "architectures": ["SomeFutureForCausalLM"],
        }
        self.assertEqual(
            classify_model(model_family="somefuturelm", local_config=config),
            ModelCategory.LLM,
        )

    def test_no_model_bootstrap_is_gr(self):
        self.assertEqual(
            classify_model(model_family="hstu", local_config=None),
            ModelCategory.GR,
        )

    def test_category_helpers(self):
        self.assertTrue(is_gr_category(ModelCategory.GR))
        self.assertTrue(is_llm_category("llm"))
        self.assertFalse(is_gr_category("llm"))


if __name__ == "__main__":
    unittest.main()
