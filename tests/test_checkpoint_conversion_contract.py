# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import builtins
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "convert_gr_checkpoint.py"
DATASET_CONFIGS_PATH = ROOT / "vllm_gr" / "config" / "dataset_configs.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_fake_checkpoint_module() -> None:
    fake_checkpoint_module = types.ModuleType("vllm_gr.models.dlrm_v3.checkpoint")
    fake_checkpoint_module.is_sparse_key = lambda *args, **kwargs: False
    fake_checkpoint_module.load_dense_state_dict = lambda *args, **kwargs: None
    fake_checkpoint_module.load_sparse_checkpoint = lambda *args, **kwargs: None
    return fake_checkpoint_module


class TestCheckpointConversionContract(unittest.TestCase):
    def test_dataset_choices_follow_supported_dataset_registry(self) -> None:
        fake_checkpoint_module = _make_fake_checkpoint_module()

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_choices_test_module",
                MODULE_PATH,
            )

        dataset_module = _load_module(
            "checkpoint_conversion_dataset_configs_test_module",
            DATASET_CONFIGS_PATH,
        )

        self.assertEqual(
            module.DATASET_CHOICES,
            tuple(dataset_module.SUPPORTED_DATASETS),
        )

    def test_sparse_embedding_dtype_follows_table_config(self) -> None:
        fake_checkpoint_module = _make_fake_checkpoint_module()

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_contract_test_module",
                MODULE_PATH,
            )

        config = SimpleNamespace(
            embedding_table_data_type="fp16",
            embedding_tables=[
                {"name": "item_id", "data_type": "bf16"},
                {"name": "user_id"},
            ],
        )

        item_dtype = module._resolve_sparse_tensor_target_dtype(
            config,
            "_embedding_collection.embeddings.item_id.weight",
            torch.ones(2, dtype=torch.float32),
        )
        user_dtype = module._resolve_sparse_tensor_target_dtype(
            config,
            "_embedding_collection.embeddings.user_id.weight",
            torch.ones(2, dtype=torch.float32),
        )
        int_dtype = module._resolve_sparse_tensor_target_dtype(
            config,
            "_embedding_collection.embeddings.user_id.offsets",
            torch.ones(2, dtype=torch.int64),
        )

        self.assertEqual(item_dtype, torch.bfloat16)
        self.assertEqual(user_dtype, torch.float16)
        self.assertEqual(int_dtype, torch.int64)

    def test_convert_fails_when_sparse_weights_exist_without_config(self) -> None:
        fake_checkpoint_module = _make_fake_checkpoint_module()

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_sparse_fail_test_module",
                MODULE_PATH,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            (input_dir / "sparse").mkdir(parents=True)

            with mock.patch.object(module, "load_dense_weights", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "requires config.json"):
                    module.convert(
                        input_path=str(input_dir),
                        output_dir=str(output_dir),
                        dense_only=False,
                        output_format="pt",
                    )

    def test_load_sparse_weights_fails_when_model_exposes_no_sparse_params(self) -> None:
        fake_checkpoint_module = _make_fake_checkpoint_module()

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_sparse_mismatch_test_module",
                MODULE_PATH,
            )

        fake_config_module = types.ModuleType("vllm_gr.config.hstu_config")
        fake_model_module = types.ModuleType("vllm_gr.model_executor.models.hstu_vllm")

        class _FakeHSTUConfig:
            @classmethod
            def from_pretrained(cls, _config_dir) -> None:
                return object()

        class _FakeInnerModel:
            @staticmethod
            def state_dict() -> None:
                return {}

        class _FakeHSTUForRecommendation:
            def __init__(self, config) -> None:
                self.hstu = _FakeInnerModel()

        fake_config_module.HSTUConfig = _FakeHSTUConfig
        fake_model_module.HSTUForRecommendation = _FakeHSTUForRecommendation

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "checkpoint"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "sparse").mkdir()

            with mock.patch.dict(
                sys.modules,
                {
                    "vllm_gr.config.hstu_config": fake_config_module,
                    "vllm_gr.model_executor.models.hstu_vllm": fake_model_module,
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "Sparse checkpoint/config mismatch"):
                    module.load_sparse_weights(
                        checkpoint_path=str(checkpoint_dir),
                        config_dir=str(root),
                        dense_weights={},
                    )

    def test_convert_fails_when_safetensors_requested_but_missing(self) -> None:
        fake_checkpoint_module = _make_fake_checkpoint_module()

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_safetensors_fail_test_module",
                MODULE_PATH,
            )

        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "safetensors.torch" or (
                name == "safetensors" and fromlist and "torch" in fromlist
            ):
                raise ImportError("missing safetensors")
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            with mock.patch.object(module, "load_dense_weights", return_value={}):
                with mock.patch("builtins.__import__", side_effect=_fake_import):
                    with self.assertRaisesRegex(RuntimeError, "safetensors.*not installed"):
                        module.convert(
                            input_path=str(input_dir),
                            output_dir=str(output_dir),
                            dense_only=True,
                            output_format="safetensors",
                        )

    def test_crop_sparse_weights_only_trims_embedding_weight_tensors(self) -> None:
        fake_checkpoint_module = _make_fake_checkpoint_module()

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_crop_test_module",
                MODULE_PATH,
            )

        sparse_weights = {
            "_embedding_collection.embeddings.item_id.weight": torch.arange(
                12, dtype=torch.float32
            ).reshape(3, 4),
            "_embedding_collection.embeddings.item_id.offsets": torch.tensor(
                [0, 3], dtype=torch.int64
            ),
            "some_dense_tensor": torch.ones(2, 2, dtype=torch.float32),
        }

        cropped = module._crop_sparse_weights(
            sparse_weights,
            source_embedding_dim=4,
            crop_embedding_dim=2,
        )

        self.assertEqual(
            tuple(cropped["_embedding_collection.embeddings.item_id.weight"].shape),
            (3, 2),
        )
        torch.testing.assert_close(
            cropped["_embedding_collection.embeddings.item_id.weight"],
            sparse_weights["_embedding_collection.embeddings.item_id.weight"][..., :2],
        )
        self.assertTrue(
            torch.equal(
                cropped["_embedding_collection.embeddings.item_id.offsets"],
                sparse_weights["_embedding_collection.embeddings.item_id.offsets"],
            )
        )
        self.assertTrue(
            torch.equal(
                cropped["some_dense_tensor"],
                sparse_weights["some_dense_tensor"],
            )
        )

    def test_crop_sparse_weights_trims_embedding_row_count(self) -> None:
        fake_checkpoint_module = _make_fake_checkpoint_module()

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_crop_rows_test_module",
                MODULE_PATH,
            )

        sparse_weights = {
            "_embedding_collection.embeddings.item_id.weight": torch.arange(
                20, dtype=torch.float32
            ).reshape(5, 4),
        }

        cropped = module._crop_sparse_weights(
            sparse_weights,
            source_embedding_dim=4,
            crop_embedding_dim=None,
            source_table_num_embeddings={"item_id": 5},
            crop_num_embeddings=3,
        )

        self.assertEqual(
            tuple(cropped["_embedding_collection.embeddings.item_id.weight"].shape),
            (3, 4),
        )
        torch.testing.assert_close(
            cropped["_embedding_collection.embeddings.item_id.weight"],
            sparse_weights["_embedding_collection.embeddings.item_id.weight"][:3],
        )

    def test_write_target_config_preserves_contextual_feature_order_from_source(self) -> None:
        fake_checkpoint_module = _make_fake_checkpoint_module()

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_contextual_order_test_module",
                MODULE_PATH,
            )

        class _FakeSourceConfig:
            contextual_feature_to_max_length = {
                "user_id": 1,
                "sex": 1,
                "age_group": 1,
                "occupation": 1,
                "zip_code": 1,
            }
            contextual_feature_to_min_uih_length = {
                "user_id": 20,
                "sex": 20,
                "age_group": 20,
                "occupation": 20,
                "zip_code": 20,
            }

        class _FakeTargetConfig:
            def to_dict(self) -> None:
                return {
                    "model_type": "hstu",
                    "contextual_feature_to_max_length": {
                        "age_group": 1,
                        "occupation": 1,
                        "sex": 1,
                        "user_id": 1,
                        "zip_code": 1,
                    },
                    "contextual_feature_to_min_uih_length": {
                        "age_group": 20,
                        "occupation": 20,
                        "sex": 20,
                        "user_id": 20,
                        "zip_code": 20,
                    },
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            module._write_target_config(
                output_dir=str(output_dir),
                target_config=_FakeTargetConfig(),
                source_config=_FakeSourceConfig(),
            )

            with open(output_dir / "config.json", "r", encoding="utf-8") as f:
                config_payload = json.load(f)

            self.assertEqual(
                list(config_payload["contextual_feature_to_max_length"].keys()),
                ["user_id", "sex", "age_group", "occupation", "zip_code"],
            )
            self.assertEqual(
                list(config_payload["contextual_feature_to_min_uih_length"].keys()),
                ["user_id", "sex", "age_group", "occupation", "zip_code"],
            )

    def test_convert_split_weights_writes_dense_and_embedding_files(self) -> None:
        fake_checkpoint_module = _make_fake_checkpoint_module()

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_split_test_module",
                MODULE_PATH,
            )

        class _FakeSourceConfig:
            hstu_embedding_table_dim = 8
            embedding_tables = [
                {
                    "name": "item_id",
                    "embedding_dim": 8,
                    "num_embeddings": 3,
                    "feature_names": ["item_id"],
                },
            ]

            def to_dict(self) -> None:
                return {
                    "model_type": "hstu",
                    "hstu_embedding_table_dim": 8,
                    "embedding_tables": [
                        {
                            "name": "item_id",
                            "embedding_dim": 8,
                            "num_embeddings": 3,
                            "feature_names": ["item_id"],
                        }
                    ],
                    "task_names": ["rating"],
                    "task_types": ["binary_classification"],
                }

        class _FakeTargetConfig(_FakeSourceConfig):
            def __init__(self, cropped_num_embeddings: int) -> None:
                self.hstu_embedding_table_dim = 8
                self.embedding_tables = [
                    {
                        "name": "item_id",
                        "embedding_dim": 8,
                        "num_embeddings": cropped_num_embeddings,
                        "feature_names": ["item_id"],
                    }
                ]

            def to_dict(self) -> None:
                return {
                    "model_type": "hstu",
                    "hstu_embedding_table_dim": self.hstu_embedding_table_dim,
                    "embedding_tables": self.embedding_tables,
                    "task_names": ["rating"],
                    "task_types": ["binary_classification"],
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "sparse").mkdir()

            with (
                mock.patch.object(
                    module,
                    "load_dense_weights",
                    return_value={"dense.weight": torch.ones(2, 2, dtype=torch.float32)},
                ),
                mock.patch.object(
                    module,
                    "_materialize_source_config",
                    return_value=str(root / "source_cfg"),
                ),
                mock.patch.object(
                    module,
                    "_load_hstu_config",
                    return_value=_FakeSourceConfig(),
                ),
                mock.patch.object(
                    module,
                    "_build_target_hstu_config",
                    side_effect=lambda source_config,
                    crop_embedding_dim,
                    crop_num_embeddings=None: _FakeTargetConfig(crop_num_embeddings or 3),
                ),
                mock.patch.object(
                    module,
                    "load_sparse_weights",
                    return_value={
                        "_embedding_collection.embeddings.item_id.weight": torch.arange(
                            24, dtype=torch.float32
                        ).reshape(3, 8),
                    },
                ),
                mock.patch.object(
                    module,
                    "_init_single_process_distributed",
                    return_value=None,
                ),
                mock.patch.object(
                    module,
                    "_destroy_distributed",
                    return_value=None,
                ),
            ):
                module.convert(
                    input_path=str(input_dir),
                    output_dir=str(output_dir),
                    config_json=str(root / "dummy_config.json"),
                    dense_only=False,
                    output_format="pt",
                    crop_num_embeddings=2,
                    split_weights=True,
                )

            dense_path = output_dir / "dense.pt"
            embedding_path = output_dir / "embedding.pt"
            config_path = output_dir / "config.json"
            meta_path = output_dir / "conversion_meta.json"

            self.assertTrue(dense_path.exists())
            self.assertTrue(embedding_path.exists())
            self.assertFalse((output_dir / "model.pt").exists())
            self.assertTrue(config_path.exists())
            self.assertTrue(meta_path.exists())

            dense_state = torch.load(dense_path, map_location="cpu", weights_only=False)
            embedding_state = torch.load(
                embedding_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(tuple(dense_state["hstu.dense.weight"].shape), (2, 2))
            self.assertEqual(
                tuple(
                    embedding_state["hstu._embedding_collection.embeddings.item_id.weight"].shape
                ),
                (2, 8),
            )

            with open(config_path, "r", encoding="utf-8") as f:
                config_payload = json.load(f)
            self.assertEqual(config_payload["hstu_embedding_table_dim"], 8)
            self.assertEqual(
                config_payload["embedding_tables"][0]["num_embeddings"],
                2,
            )

            with open(meta_path, "r", encoding="utf-8") as f:
                meta_payload = json.load(f)
            self.assertEqual(meta_payload["dense_artifact"], "dense.pt")
            self.assertEqual(meta_payload["embedding_artifact"], "embedding.pt")
            self.assertEqual(meta_payload["source_embedding_dim"], 8)
            self.assertEqual(meta_payload["target_embedding_dim"], 8)
            self.assertEqual(meta_payload["crop_num_embeddings"], 2)
            self.assertEqual(
                meta_payload["embedding_table_num_embeddings"]["item_id"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
