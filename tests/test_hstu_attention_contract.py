# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestHSTUAttentionContract(unittest.TestCase):
    def test_hstu_config_is_not_attention_free(self) -> None:
        from vllm_gr.config.hstu_config import HSTUConfig

        config = HSTUConfig()

        self.assertFalse(HSTUConfig.is_attention_free)
        self.assertFalse(config.is_attention_free)

    def test_hstu_model_declares_attention_backend(self) -> None:
        from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

        self.assertFalse(HSTUForRecommendation.is_attention_free)

    def test_example_configs_declare_attention_backend(self) -> None:
        config_paths = [
            ROOT / "examples" / "demo-model" / "config.json",
            *sorted((ROOT / "examples" / "configs").glob("*/config.json")),
        ]

        for config_path in config_paths:
            with self.subTest(config=str(config_path)):
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertIs(config_data["is_attention_free"], False)


if __name__ == "__main__":
    unittest.main()
