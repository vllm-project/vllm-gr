# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import unittest
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "vllm_gr" / "config" / "dataset_configs.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dataset_config_module():
    spec = importlib.util.spec_from_file_location(
        "dataset_config_parity_test_module",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDatasetConfigParity(unittest.TestCase):

    def test_streaming_matches_golden_overrides(self):
        module = _load_dataset_config_module()
        cfg = module.get_hstu_config_overrides("streaming")

        self.assertEqual(
            cfg["candidates_watchtime_feature_name"],
            "item_candidate_rating",
        )
        self.assertEqual(cfg["action_embedding_init_std"], 5.0)
        self.assertEqual(cfg["max_num_candidates_inference"], 2048)

    def test_streaming_embedding_tables_keep_category_initializer_bounds(self):
        module = _load_dataset_config_module()
        tables = module.get_embedding_table_overrides("streaming")
        table_by_name = {table["name"]: table for table in tables}

        self.assertEqual(table_by_name["item_category_id"]["weight_init_max"], 1.0)
        self.assertEqual(table_by_name["item_category_id"]["weight_init_min"], -1.0)

    def test_movielens_large_variants_match_golden_inference_limit(self):
        module = _load_dataset_config_module()

        cfg_13b = module.get_hstu_config_overrides("movielens-13b")
        cfg_18b = module.get_hstu_config_overrides("movielens-18b")

        self.assertEqual(cfg_13b["max_num_candidates_inference"], 2048)
        self.assertEqual(cfg_18b["max_num_candidates_inference"], 2048)

    def test_large_table_datasets_raise_hash_size_to_match_embedding_scale(self):
        module = _load_dataset_config_module()

        self.assertEqual(
            module.get_hstu_config_overrides("streaming")["hash_size"],
            module.HASH_SIZE_1B,
        )
        self.assertEqual(
            module.get_hstu_config_overrides("movielens-13b")["hash_size"],
            module.HASH_SIZE_1B,
        )
        self.assertEqual(
            module.get_hstu_config_overrides("movielens-1m")["hash_size"],
            module.HASH_SIZE,
        )

    def test_unknown_dataset_fails_fast(self):
        module = _load_dataset_config_module()

        with self.assertRaisesRegex(ValueError, "Unsupported dataset"):
            module.get_hstu_config_overrides("stream")

        with self.assertRaisesRegex(ValueError, "Unsupported dataset"):
            module.get_embedding_table_overrides("stream")


class TestHSTUConfigConsistency(unittest.TestCase):

    @staticmethod
    def _minimal_hstu_kwargs():
        return {
            "user_embedding_feature_names": ["item_id"],
            "item_embedding_feature_names": ["item_candidate_id"],
            "uih_post_id_feature_name": "item_id",
            "hstu_uih_feature_names": ["item_id"],
            "hstu_candidate_feature_names": ["item_candidate_id"],
            "embedding_tables": [
                {
                    "name": "item_embedding",
                    "embedding_dim": 8,
                    "num_embeddings": 16,
                    "feature_names": ["item_id", "item_candidate_id"],
                },
            ],
        }

    def test_supported_datasets_can_all_be_materialized_as_hstu_config(self):
        from vllm_gr.config.hstu_config import HSTUConfig

        module = _load_dataset_config_module()
        for dataset in module.SUPPORTED_DATASETS:
            with self.subTest(dataset=dataset):
                overrides = module.get_hstu_config_overrides(dataset)
                tables = module.get_embedding_table_overrides(dataset)
                if tables:
                    overrides["embedding_tables"] = tables
                HSTUConfig(**overrides)

    def test_example_configs_are_hstu_config_consistent(self):
        from vllm_gr.config.hstu_config import HSTUConfig

        config_paths = [
            ROOT / "examples" / "demo-model" / "config.json",
            *sorted((ROOT / "examples" / "configs").glob("*/config.json")),
        ]

        for config_path in config_paths:
            with self.subTest(config=str(config_path)):
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
                HSTUConfig(**config_data)

    def test_contextual_min_lengths_must_be_declared_in_max_lengths(self):
        from vllm_gr.config.hstu_config import HSTUConfig

        with self.assertRaisesRegex(
            ValueError,
            "contextual_feature_to_min_uih_length keys must also be present",
        ):
            HSTUConfig(
                user_embedding_feature_names=["item_id", "user_id"],
                item_embedding_feature_names=["item_candidate_id"],
                uih_post_id_feature_name="item_id",
                hstu_uih_feature_names=["item_id", "user_id", "session_id"],
                hstu_candidate_feature_names=["item_candidate_id"],
                embedding_tables=[
                    {
                        "name": "item_embedding",
                        "embedding_dim": 8,
                        "num_embeddings": 16,
                        "feature_names": ["item_id", "item_candidate_id"],
                    },
                    {
                        "name": "user_embedding",
                        "embedding_dim": 8,
                        "num_embeddings": 16,
                        "feature_names": ["user_id"],
                    },
                ],
                contextual_feature_to_max_length={"user_id": 1},
                contextual_feature_to_min_uih_length={"session_id": 5},
            )

    def test_task_types_default_to_binary_classification(self):
        from vllm_gr.config.hstu_config import HSTUConfig
        from vllm_gr.config.multitask_types import MultitaskTaskType

        config = HSTUConfig(
            task_names=["is_click", "is_like"],
            **self._minimal_hstu_kwargs(),
        )

        self.assertEqual(
            config.task_types,
            [
                MultitaskTaskType.BINARY_CLASSIFICATION,
                MultitaskTaskType.BINARY_CLASSIFICATION,
            ],
        )

    def test_task_types_serialize_back_to_strings(self):
        from vllm_gr.config.hstu_config import HSTUConfig

        config = HSTUConfig(
            task_names=["watch_time"],
            task_types=["regression"],
            **self._minimal_hstu_kwargs(),
        )

        self.assertEqual(
            config.to_dict()["task_types"],
            ["regression"],
        )

    def test_task_types_must_align_with_task_names(self):
        from vllm_gr.config.hstu_config import HSTUConfig

        with self.assertRaisesRegex(ValueError, "same length as task_names"):
            HSTUConfig(
                task_names=["is_click", "is_like"],
                task_types=["binary_classification"],
                **self._minimal_hstu_kwargs(),
            )

    def test_task_types_must_be_known(self):
        from vllm_gr.config.hstu_config import HSTUConfig

        with self.assertRaisesRegex(ValueError, "must be one of"):
            HSTUConfig(
                task_names=["watch_time"],
                task_types=["ranking"],
                **self._minimal_hstu_kwargs(),
            )


if __name__ == "__main__":
    unittest.main()
