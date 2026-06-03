# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "vllm_gr" / "model_executor" / "models" / "embedding_collection_patch.py"


def _load_module(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeFeature:
    def __init__(self, values: torch.Tensor, lengths: torch.Tensor) -> None:
        self._values = values
        self._lengths = lengths

    def values(self) -> torch.Tensor:
        return self._values

    def lengths(self) -> torch.Tensor:
        return self._lengths


class _FakeInputKJT:
    def __init__(self, feature_map) -> None:
        self._feature_map = feature_map

    def to_dict(self) -> None:
        return self._feature_map


class _FakeJaggedTensor:
    def __init__(self, values, lengths, weights=None) -> None:
        self.values = values
        self.lengths = lengths
        self.weights = weights


class _FakeEmbeddingModule:
    def __init__(self, num_embeddings: int) -> None:
        self.weight = torch.zeros(num_embeddings, 2)
        self.last_input = None

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        self.last_input = input
        return input.unsqueeze(-1).expand(-1, 2).to(torch.float32)


class TestEmbeddingCollectionPatch(unittest.TestCase):
    def test_patch_allows_in_range_indices(self) -> None:
        module = _load_module(
            "embedding_collection_patch_test_module",
            MODULE_PATH,
        )

        fake_jagged_module = types.ModuleType("torchrec.sparse.jagged_tensor")
        fake_tensor_dict_module = types.ModuleType("torchrec.sparse.tensor_dict")
        fake_jagged_module.JaggedTensor = _FakeJaggedTensor
        fake_tensor_dict_module.maybe_td_to_kjt = lambda features, _: features

        embedding_module = _FakeEmbeddingModule(num_embeddings=4)
        embedding_collection = types.SimpleNamespace(
            embeddings={"item_id": embedding_module},
            _feature_names=[["item_id"]],
            _embedding_names_by_table=[["item_id"]],
            _need_indices=False,
        )
        features = _FakeInputKJT(
            {
                "item_id": _FakeFeature(
                    values=torch.tensor([0, 1, 3], dtype=torch.int64),
                    lengths=torch.tensor([3], dtype=torch.int64),
                ),
            }
        )

        with mock.patch.dict(
            sys.modules,
            {
                "torchrec.sparse.jagged_tensor": fake_jagged_module,
                "torchrec.sparse.tensor_dict": fake_tensor_dict_module,
            },
        ):
            outputs = module.ec_patched_forward_wo_embedding_copy(
                embedding_collection,
                features,
            )

        torch.testing.assert_close(
            embedding_module.last_input, features.to_dict()["item_id"].values()
        )
        self.assertIn("item_id", outputs)
        self.assertTrue(torch.equal(outputs["item_id"].lengths, torch.tensor([3])))

    def test_patch_rejects_out_of_range_indices(self) -> None:
        module = _load_module(
            "embedding_collection_patch_test_module_strict",
            MODULE_PATH,
        )

        fake_jagged_module = types.ModuleType("torchrec.sparse.jagged_tensor")
        fake_tensor_dict_module = types.ModuleType("torchrec.sparse.tensor_dict")
        fake_jagged_module.JaggedTensor = _FakeJaggedTensor
        fake_tensor_dict_module.maybe_td_to_kjt = lambda features, _: features

        embedding_module = _FakeEmbeddingModule(num_embeddings=4)
        embedding_collection = types.SimpleNamespace(
            embeddings={"item_id": embedding_module},
            _feature_names=[["item_id"]],
            _embedding_names_by_table=[["item_id"]],
            _need_indices=False,
        )
        features = _FakeInputKJT(
            {
                "item_id": _FakeFeature(
                    values=torch.tensor([-3, 1, 9], dtype=torch.int64),
                    lengths=torch.tensor([3], dtype=torch.int64),
                ),
            }
        )

        with mock.patch.dict(
            sys.modules,
            {
                "torchrec.sparse.jagged_tensor": fake_jagged_module,
                "torchrec.sparse.tensor_dict": fake_tensor_dict_module,
            },
        ):
            with self.assertRaisesRegex(
                ValueError,
                "out-of-range ids",
            ):
                module.ec_patched_forward_wo_embedding_copy(
                    embedding_collection,
                    features,
                )
        self.assertIsNone(embedding_module.last_input)


if __name__ == "__main__":
    unittest.main()
