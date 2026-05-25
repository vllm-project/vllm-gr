# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "vllm_gr"
    / "models"
    / "dlrm_v3"
    / "modules"
    / "dlrm_hstu.py"
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_fake_common_module():
    fake_common = types.ModuleType("vllm_gr.models.dlrm_v3.common")

    class _HammerModule(torch.nn.Module):
        pass

    fake_common.HammerModule = _HammerModule
    fake_common.fx_infer_max_len = lambda lengths: int(lengths.max().item()) if lengths.numel() else 0
    fake_common.fx_mark_length_features = lambda lengths: (lengths[0],)
    fake_common.init_mlp_weights_optional_bias = lambda module: module
    fake_common.set_static_max_seq_lens = lambda lengths: None
    return fake_common


def _make_placeholder_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class TestDlrmHSTUEmbeddingPadding(unittest.TestCase):

    def _load_dlrm_hstu_module(self):
        fake_torchrec_embedding_configs = _make_placeholder_module(
            "torchrec.modules.embedding_configs",
            EmbeddingConfig=object,
        )
        fake_torchrec_embedding_modules = _make_placeholder_module(
            "torchrec.modules.embedding_modules",
            EmbeddingCollection=object,
        )
        fake_torchrec_jagged_tensor = _make_placeholder_module(
            "torchrec.sparse.jagged_tensor",
            KeyedJaggedTensor=object,
        )
        fake_hstu_transducer = _make_placeholder_module(
            "vllm_gr.models.dlrm_v3.modules.hstu_transducer",
            HSTUTransducer=object,
        )
        fake_multitask_module = _make_placeholder_module(
            "vllm_gr.models.dlrm_v3.modules.multitask_module",
            DefaultMultitaskModule=object,
            TaskConfig=object,
        )
        fake_positional_encoder = _make_placeholder_module(
            "vllm_gr.models.dlrm_v3.modules.positional_encoder",
            HSTUPositionalEncoder=object,
        )
        fake_postprocessors = _make_placeholder_module(
            "vllm_gr.models.dlrm_v3.modules.postprocessors",
            LayerNormPostprocessor=object,
            TimestampLayerNormPostprocessor=object,
        )
        fake_preprocessors = _make_placeholder_module(
            "vllm_gr.models.dlrm_v3.modules.preprocessors",
            ContextualPreprocessor=object,
        )
        fake_stu = _make_placeholder_module(
            "vllm_gr.models.dlrm_v3.modules.stu",
            STU=object,
            STULayer=object,
            STULayerConfig=object,
            STUStack=object,
        )
        fake_jagged_ops = _make_placeholder_module(
            "vllm_gr.ops.jagged_tensors",
            concat_2D_jagged=lambda **kwargs: None,
        )
        fake_layer_norm = _make_placeholder_module(
            "vllm_gr.ops.layer_norm",
            LayerNorm=torch.nn.Identity,
            SwishLayerNorm=torch.nn.Identity,
        )
        fake_utils = _make_placeholder_module(
            "vllm_gr.utils",
            ReliableOps=SimpleNamespace(
                asynchronous_complete_cumsum=lambda lengths: lengths.cumsum(dim=0)
            ),
        )

        with mock.patch.dict(
            sys.modules,
            {
                "torchrec.modules.embedding_configs": fake_torchrec_embedding_configs,
                "torchrec.modules.embedding_modules": fake_torchrec_embedding_modules,
                "torchrec.sparse.jagged_tensor": fake_torchrec_jagged_tensor,
                "vllm_gr.models.dlrm_v3.common": _make_fake_common_module(),
                "vllm_gr.models.dlrm_v3.modules.hstu_transducer": fake_hstu_transducer,
                "vllm_gr.models.dlrm_v3.modules.multitask_module": fake_multitask_module,
                "vllm_gr.models.dlrm_v3.modules.positional_encoder": fake_positional_encoder,
                "vllm_gr.models.dlrm_v3.modules.postprocessors": fake_postprocessors,
                "vllm_gr.models.dlrm_v3.modules.preprocessors": fake_preprocessors,
                "vllm_gr.models.dlrm_v3.modules.stu": fake_stu,
                "vllm_gr.ops.jagged_tensors": fake_jagged_ops,
                "vllm_gr.ops.layer_norm": fake_layer_norm,
                "vllm_gr.utils": fake_utils,
            },
        ):
            return _load_module(
                "dlrm_hstu_padding_test_module",
                MODULE_PATH,
            )

    def test_pad_embedding_to_runtime_dim_appends_zero_columns(self):
        module = self._load_dlrm_hstu_module()

        model = module.DlrmHSTU.__new__(module.DlrmHSTU)
        torch.nn.Module.__init__(model)
        model._hstu_configs = SimpleNamespace(hstu_embedding_table_dim=5)

        embedding = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        padded = model._pad_embedding_to_runtime_dim(
            feature_name="item_id",
            embedding=embedding,
        )

        self.assertEqual(tuple(padded.shape), (1, 5))
        torch.testing.assert_close(padded[:, :3], embedding)
        torch.testing.assert_close(
            padded[:, 3:],
            torch.zeros((1, 2), dtype=torch.float32),
        )

    def test_pad_embedding_to_runtime_dim_rejects_oversized_embedding(self):
        module = self._load_dlrm_hstu_module()

        model = module.DlrmHSTU.__new__(module.DlrmHSTU)
        torch.nn.Module.__init__(model)
        model._hstu_configs = SimpleNamespace(hstu_embedding_table_dim=2)

        with self.assertRaisesRegex(ValueError, "exceeds runtime expectation"):
            model._pad_embedding_to_runtime_dim(
                feature_name="item_id",
                embedding=torch.ones((1, 3), dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
