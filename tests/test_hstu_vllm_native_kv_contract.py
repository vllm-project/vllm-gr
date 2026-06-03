# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

import importlib
import importlib.machinery
import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch


@pytest.fixture(scope="class")
def setup_fake_torchrec_modules():
    """
    Fixture to set up fake torchrec modules only for tests that require them.
    Cleans up after the test completes.
    """
    # Store original modules if they exist
    original_modules = {}
    modules_to_set = {
        "torchrec",
        "torchrec.modules",
        "torchrec.modules.embedding_configs",
        "torchrec.modules.embedding_modules",
        "torchrec.sparse",
        "torchrec.sparse.jagged_tensor",
    }
    for module_name in modules_to_set:
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]

    # Set up fake modules
    fake_torchrec = types.ModuleType("torchrec")
    fake_torchrec_modules = types.ModuleType("torchrec.modules")
    fake_embedding_configs = types.ModuleType("torchrec.modules.embedding_configs")
    fake_embedding_modules = types.ModuleType("torchrec.modules.embedding_modules")
    fake_torchrec_sparse = types.ModuleType("torchrec.sparse")
    fake_jagged_tensor = types.ModuleType("torchrec.sparse.jagged_tensor")

    class _EmbeddingConfig:
        pass

    class _EmbeddingCollection:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class _KeyedJaggedTensor:
        pass

    fake_torchrec.__spec__ = importlib.machinery.ModuleSpec("torchrec", loader=None)
    fake_torchrec_modules.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.modules", loader=None
    )
    fake_embedding_configs.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.modules.embedding_configs", loader=None
    )
    fake_embedding_modules.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.modules.embedding_modules", loader=None
    )
    fake_torchrec_sparse.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.sparse", loader=None
    )
    fake_jagged_tensor.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.sparse.jagged_tensor", loader=None
    )
    fake_embedding_configs.EmbeddingConfig = _EmbeddingConfig
    fake_embedding_modules.EmbeddingCollection = _EmbeddingCollection
    fake_jagged_tensor.KeyedJaggedTensor = _KeyedJaggedTensor
    sys.modules.update(
        {
            "torchrec": fake_torchrec,
            "torchrec.modules": fake_torchrec_modules,
            "torchrec.modules.embedding_configs": fake_embedding_configs,
            "torchrec.modules.embedding_modules": fake_embedding_modules,
            "torchrec.sparse": fake_torchrec_sparse,
            "torchrec.sparse.jagged_tensor": fake_jagged_tensor,
        }
    )

    yield

    # Clean up: remove fake modules
    for module_name in modules_to_set:
        if module_name in original_modules:
            sys.modules[module_name] = original_modules[module_name]
        else:
            sys.modules.pop(module_name, None)


@contextmanager
def _patched_gr_npu_model_runner():
    fake_vllm_ascend = types.ModuleType("vllm_ascend")
    fake_vllm_ascend_worker = types.ModuleType("vllm_ascend.worker")
    fake_vllm_ascend_runner = types.ModuleType("vllm_ascend.worker.model_runner_v1")

    class _FakeNPUModelRunner:
        def __init__(self, vllm_config, device):
            self.vllm_config = vllm_config
            self.model_config = getattr(vllm_config, "model_config", None)
            self.compilation_config = getattr(vllm_config, "compilation_config", None)
            self.device = device
            self.attn_groups = []
            self.model = None

        def load_model(self, *args, **kwargs):
            del args, kwargs
            return None

        def get_kv_cache_spec(self):
            return {}

        def initialize_kv_cache(self, kv_cache_config):
            self._initialized_kv_cache_config = kv_cache_config
            return None

        def _build_attention_metadata(self, *args, **kwargs):
            self._build_attention_metadata_called = (args, kwargs)
            return {"base_called": True}

    fake_vllm_ascend.__spec__ = importlib.machinery.ModuleSpec(
        "vllm_ascend", loader=None
    )
    fake_vllm_ascend_worker.__spec__ = importlib.machinery.ModuleSpec(
        "vllm_ascend.worker", loader=None
    )
    fake_vllm_ascend_runner.__spec__ = importlib.machinery.ModuleSpec(
        "vllm_ascend.worker.model_runner_v1", loader=None
    )
    fake_vllm_ascend.register_connector = lambda *args, **kwargs: None
    fake_vllm_ascend.register_model_loader = lambda *args, **kwargs: None
    fake_vllm_ascend.register_service_profiling = lambda *args, **kwargs: None
    fake_vllm_ascend_runner.NPUModelRunner = _FakeNPUModelRunner

    with mock.patch.dict(
        sys.modules,
        {
            "vllm_ascend": fake_vllm_ascend,
            "vllm_ascend.worker": fake_vllm_ascend_worker,
            "vllm_ascend.worker.model_runner_v1": fake_vllm_ascend_runner,
        },
    ):
        sys.modules.pop("vllm_gr.worker.gr_npu_model_runner", None)
        gr_npu_model_runner_module = importlib.import_module(
            "vllm_gr.worker.gr_npu_model_runner"
        )
        with mock.patch.object(
            gr_npu_model_runner_module,
            "resolve_model_type",
            return_value="hstu",
        ), mock.patch.object(
            gr_npu_model_runner_module,
            "create_adapter",
            return_value=SimpleNamespace(payload_type=None),
        ):
            yield gr_npu_model_runner_module.GRNPUModelRunner

        sys.modules.pop("vllm_gr.worker.gr_npu_model_runner", None)

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_gr.attention.backends.hstu_attn import (
    HSTUAttentionMetadata,
    HSTUAttentionMetadataBuilder,
    _build_hstu_token_positions,
)
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.model_executor.layers.hstu_attention_layer import (
    HSTUKVCacheAttentionLayer,
)
from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation
from vllm_gr.models.dlrm_v3.modules import stu as stu_module
from vllm_gr.models.dlrm_v3.modules.hstu_transducer import HSTUTransducer
from vllm_gr.models.dlrm_v3.modules.stu import STULayer, STULayerConfig
from vllm_gr.ops.hstu_attention import delta_hstu_mha


def _dummy_vllm_config(block_size: int = 16):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


@pytest.mark.usefixtures("setup_fake_torchrec_modules")
class TestHSTUVllmNativeKVContract(unittest.TestCase):

    @staticmethod
    def _make_hstu_model_for_metadata_test():
        model = HSTUForRecommendation.__new__(HSTUForRecommendation)
        torch.nn.Module.__init__(model)
        model.enable_kv_cache = True
        model._hstu_kv_cache_layers = [SimpleNamespace(layer_name="layer0")]
        model.config = SimpleNamespace(
            contextual_feature_to_max_length={"ctx_a": 2, "ctx_b": 3}
        )
        return model

    @staticmethod
    def _make_attention_metadata_for_coverage_test(
        *,
        slot_count: int,
        computed_tokens: int,
        history_len: int = 6,
        contextual_seq_len: int = 5,
        num_targets: int = 6,
    ) -> HSTUAttentionMetadata:
        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, slot_count], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, slot_count], dtype=torch.int32),
            seq_lens=torch.tensor([slot_count], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([slot_count], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([computed_tokens], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=slot_count,
            max_query_len=slot_count,
            max_seq_len=slot_count,
            block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
            slot_mapping=torch.arange(slot_count, dtype=torch.int64),
            causal=True,
        )
        return HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor([num_targets], dtype=torch.int32),
            history_lengths=torch.tensor([history_len], dtype=torch.int32),
            contextual_seq_len=contextual_seq_len,
            layer_names=["layer0"],
            cache_layers=None,
        )

    @staticmethod
    def _build_native_kv_test_fixture(
        *,
        seq_len: int,
        history_len: int,
        num_targets: int,
        contextual_seq_len: int = 0,
    ) -> tuple[STULayer, HSTUKVCacheAttentionLayer, HSTUAttentionMetadata, torch.Tensor]:
        native_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        native_layer.eval()

        cache_layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=2,
            head_size=2,
            head_size_v=3,
            dtype=torch.float32,
            vllm_config=None,
            enable_kv_cache=True,
        )
        runtime_kv_cache = torch.zeros((1, 16, 2, 5), dtype=torch.float32)
        cache_layer.get_runtime_kv_cache = (
            lambda kv_cache=None, _kv=runtime_kv_cache: _kv if kv_cache is None else kv_cache
        )

        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, seq_len], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.arange(seq_len, dtype=torch.int64),
            causal=True,
        )
        metadata = HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor([num_targets], dtype=torch.int32),
            history_lengths=torch.tensor([history_len], dtype=torch.int32),
            contextual_seq_len=contextual_seq_len,
            layer_names=["hstu_kv_cache_layers.0"],
            cache_layers=[cache_layer],
        )
        x = torch.randn(seq_len, 6, dtype=torch.float32)
        return native_layer, cache_layer, metadata, x

    def test_gr_engine_args_expand_max_model_len_to_full_visible_sequence(self) -> None:
        args = GREngineArgs(
            model=None,
            max_seq_len=128,
            max_candidates=256,
            hf_overrides={
                "contextual_feature_to_max_length": {
                    "user_id": 1,
                    "sex": 1,
                    "age_group": 1,
                    "occupation": 1,
                    "zip_code": 1,
                }
            },
            allow_missing_weights=True,
        )

        self.assertEqual(args.max_model_len, 389)
        self.assertIsInstance(args.hf_overrides, dict)
        self.assertEqual(args.hf_overrides["model_max_length"], 389)

    def test_cache_layer_reports_full_attention_spec(self) -> None:
        layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=4,
            head_size=32,
            head_size_v=64,
            dtype="bfloat16",
            vllm_config=None,
            enable_kv_cache=True,
        )

        spec = layer.get_kv_cache_spec(_dummy_vllm_config())

        self.assertIsInstance(spec, FullAttentionSpec)
        self.assertEqual(spec.block_size, 16)
        self.assertEqual(spec.num_kv_heads, 4)
        self.assertEqual(spec.head_size, 32)
        self.assertEqual(spec.head_size_v, 64)
        self.assertEqual(spec.dtype, torch.bfloat16)

    def test_metadata_builder_splits_prefix_and_delta_positions(self) -> None:
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=4,
            head_size_v=8,
            dtype=torch.bfloat16,
        )
        builder = HSTUAttentionMetadataBuilder(
            spec,
            ["hstu_kv_cache_layers.0"],
            _dummy_vllm_config(),
            torch.device("cpu"),
        )
        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 5, 8], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 5, 8], dtype=torch.int32),
            seq_lens=torch.tensor([5, 3], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([5, 3], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0, 0], dtype=torch.int32),
            num_reqs=2,
            num_actual_tokens=8,
            max_query_len=5,
            max_seq_len=5,
            block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
            slot_mapping=torch.tensor([10, 11, 12, 13, 14, 20, 21, 22], dtype=torch.int64),
            causal=True,
        )

        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
        metadata.num_targets = torch.tensor([2, 2], dtype=torch.int32)
        metadata.history_lengths = torch.tensor([3, 1], dtype=torch.int32)
        metadata.contextual_seq_len = 0
        metadata.reset_derived_cache()

        prefix_positions, delta_positions, full_positions = _build_hstu_token_positions(
            metadata,
            device=torch.device("cpu"),
        )

        self.assertEqual(prefix_positions.tolist(), [0, 1, 2, 5])
        self.assertEqual(delta_positions.tolist(), [3, 4, 6, 7])
        self.assertEqual(full_positions.tolist(), list(range(8)))
        self.assertEqual(
            metadata.get_slot_mapping("prefix", device=torch.device("cpu")).tolist(),
            [10, 11, 12, 20],
        )
        self.assertEqual(
            metadata.get_slot_mapping("delta", device=torch.device("cpu")).tolist(),
            [13, 14, 21, 22],
        )

    def test_attention_impl_delta_path_matches_full_delta_attention(self) -> None:
        layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=1,
            head_size=2,
            head_size_v=3,
            dtype=torch.float32,
            vllm_config=None,
            enable_kv_cache=True,
        )
        kv_cache = torch.zeros((1, 4, 1, 5), dtype=torch.float32)
        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=3,
            max_query_len=3,
            max_seq_len=3,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
            causal=True,
        )
        metadata = HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor([1], dtype=torch.int32),
            history_lengths=torch.tensor([2], dtype=torch.int32),
            contextual_seq_len=0,
            layer_names=["hstu_kv_cache_layers.0"],
            cache_layers=[layer],
        )

        prefix_k = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
        prefix_v = torch.tensor(
            [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
            dtype=torch.float32,
        )
        delta_q = torch.tensor([[[0.5, 1.5]]], dtype=torch.float32)
        delta_k = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)
        delta_v = torch.tensor([[[7.0, 8.0, 9.0]]], dtype=torch.float32)

        layer.write_to_paged_cache(
            prefix_k,
            prefix_v,
            common.slot_mapping[:2],
            kv_cache=kv_cache,
        )
        output = layer(
            query=delta_q,
            key=delta_k,
            value=delta_v,
            kv_cache=kv_cache,
            attn_metadata=metadata,
            attn_mode="delta",
        )

        expected_k = torch.cat([prefix_k, delta_k], dim=0)
        expected_v = torch.cat([prefix_v, delta_v], dim=0)
        expected = delta_hstu_mha(
            max_seq_len=3,
            alpha=layer.scale,
            delta_q=delta_q,
            k=expected_k,
            v=expected_v,
            seq_offsets=torch.tensor([0, 3], dtype=torch.int64),
            num_targets=metadata.num_targets,
            contextual_seq_len=0,
        )

        torch.testing.assert_close(output, expected)

    def test_prefix_mode_matches_full_forward_prefix_outputs(self) -> None:
        full_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        full_layer.eval()
        native_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        native_layer.load_state_dict(full_layer.state_dict())
        native_layer.eval()

        seq_len = 7
        history_len = 4
        num_targets = 3
        x = torch.randn(seq_len, 6, dtype=torch.float32)

        full_output = full_layer(
            x=x,
            x_lengths=torch.tensor([seq_len], dtype=torch.int64),
            x_offsets=torch.tensor([0, seq_len], dtype=torch.int64),
            max_seq_len=seq_len,
            num_targets=torch.tensor([num_targets], dtype=torch.int64),
        )

        cache_layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=2,
            head_size=2,
            head_size_v=3,
            dtype=torch.float32,
            vllm_config=None,
            enable_kv_cache=True,
        )
        runtime_kv_cache = torch.zeros((1, 16, 2, 5), dtype=torch.float32)
        cache_layer.get_runtime_kv_cache = (
            lambda kv_cache=None, _kv=runtime_kv_cache: _kv if kv_cache is None else kv_cache
        )

        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, seq_len], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.arange(seq_len, dtype=torch.int64),
            causal=True,
        )
        metadata = HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor([num_targets], dtype=torch.int32),
            history_lengths=torch.tensor([history_len], dtype=torch.int32),
            contextual_seq_len=0,
            layer_names=["hstu_kv_cache_layers.0"],
            cache_layers=[cache_layer],
        )

        prefix_output = native_layer(
            x=x[:history_len],
            x_lengths=torch.tensor([history_len], dtype=torch.int64),
            x_offsets=torch.tensor([0, history_len], dtype=torch.int64),
            max_seq_len=history_len,
            num_targets=torch.tensor([0], dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="prefix",
        )

        torch.testing.assert_close(
            prefix_output,
            full_output[:history_len],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_batched_variable_target_counts_match_full_forward(self) -> None:
        full_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        full_layer.eval()
        native_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        native_layer.load_state_dict(full_layer.state_dict())
        native_layer.eval()

        seq_lens = [7, 11]
        history_lens = [4, 6]
        num_targets = [3, 5]
        total_seq_len = sum(seq_lens)
        x = torch.randn(total_seq_len, 6, dtype=torch.float32)

        full_output = full_layer(
            x=x,
            x_lengths=torch.tensor(seq_lens, dtype=torch.int64),
            x_offsets=torch.tensor([0, 7, 18], dtype=torch.int64),
            max_seq_len=max(seq_lens),
            num_targets=torch.tensor(num_targets, dtype=torch.int64),
        )

        cache_layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=2,
            head_size=2,
            head_size_v=3,
            dtype=torch.float32,
            vllm_config=None,
            enable_kv_cache=True,
        )
        runtime_kv_cache = torch.zeros((2, 16, 2, 5), dtype=torch.float32)
        cache_layer.get_runtime_kv_cache = (
            lambda kv_cache=None, _kv=runtime_kv_cache: _kv if kv_cache is None else kv_cache
        )

        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 7, 18], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 7, 18], dtype=torch.int32),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            _seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0, 0], dtype=torch.int32),
            num_reqs=2,
            num_actual_tokens=total_seq_len,
            max_query_len=max(seq_lens),
            max_seq_len=max(seq_lens),
            block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
            slot_mapping=torch.tensor(
                list(range(seq_lens[0])) + [16 + i for i in range(seq_lens[1])],
                dtype=torch.int64,
            ),
            causal=True,
        )
        metadata = HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor(num_targets, dtype=torch.int32),
            history_lengths=torch.tensor(history_lens, dtype=torch.int32),
            contextual_seq_len=0,
            layer_names=["hstu_kv_cache_layers.0"],
            cache_layers=[cache_layer],
        )

        prefix_x = torch.cat(
            [
                x[0:history_lens[0]],
                x[seq_lens[0]:seq_lens[0] + history_lens[1]],
            ],
            dim=0,
        )
        delta_x = torch.cat(
            [
                x[history_lens[0]:seq_lens[0]],
                x[seq_lens[0] + history_lens[1]:total_seq_len],
            ],
            dim=0,
        )
        prefix_output = native_layer(
            x=prefix_x,
            x_lengths=torch.tensor(history_lens, dtype=torch.int64),
            x_offsets=torch.tensor([0, 4, 10], dtype=torch.int64),
            max_seq_len=max(history_lens),
            num_targets=torch.tensor([0, 0], dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="prefix",
        )
        delta_output = native_layer.cached_forward(
            delta_x=delta_x,
            num_targets=torch.tensor(num_targets, dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="delta",
        )

        combined = torch.cat(
            [
                prefix_output[0:history_lens[0]],
                delta_output[0:num_targets[0]],
                prefix_output[history_lens[0]:sum(history_lens)],
                delta_output[num_targets[0]:sum(num_targets)],
            ],
            dim=0,
        )

        torch.testing.assert_close(
            combined,
            full_output,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_contextual_prefix_metadata_matches_full_forward_semantics(self) -> None:
        full_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        full_layer.eval()
        native_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        native_layer.load_state_dict(full_layer.state_dict())
        native_layer.eval()

        seq_len = 9
        contextual_seq_len = 2
        history_len = 4
        num_targets = 3
        prefix_len = contextual_seq_len + history_len
        x = torch.randn(seq_len, 6, dtype=torch.float32)

        full_output = full_layer(
            x=x,
            x_lengths=torch.tensor([seq_len], dtype=torch.int64),
            x_offsets=torch.tensor([0, seq_len], dtype=torch.int64),
            max_seq_len=seq_len,
            num_targets=torch.tensor([num_targets], dtype=torch.int64),
        )

        cache_layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=2,
            head_size=2,
            head_size_v=3,
            dtype=torch.float32,
            vllm_config=None,
            enable_kv_cache=True,
        )
        runtime_kv_cache = torch.zeros((1, 16, 2, 5), dtype=torch.float32)
        cache_layer.get_runtime_kv_cache = (
            lambda kv_cache=None, _kv=runtime_kv_cache: _kv if kv_cache is None else kv_cache
        )

        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, seq_len], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.arange(seq_len, dtype=torch.int64),
            causal=True,
        )
        metadata = HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor([num_targets], dtype=torch.int32),
            history_lengths=torch.tensor([history_len], dtype=torch.int32),
            contextual_seq_len=contextual_seq_len,
            layer_names=["hstu_kv_cache_layers.0"],
            cache_layers=[cache_layer],
        )

        prefix_output = native_layer(
            x=x[:prefix_len],
            x_lengths=torch.tensor([prefix_len], dtype=torch.int64),
            x_offsets=torch.tensor([0, prefix_len], dtype=torch.int64),
            max_seq_len=prefix_len,
            num_targets=torch.tensor([0], dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="prefix",
        )
        delta_output = native_layer.cached_forward(
            delta_x=x[prefix_len:],
            num_targets=torch.tensor([num_targets], dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="delta",
        )

        torch.testing.assert_close(
            torch.cat([prefix_output, delta_output], dim=0),
            full_output,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_native_prefix_projection_uses_full_sequence_shape(self) -> None:
        seq_len = 9
        history_len = 6
        num_targets = 3
        native_layer, cache_layer, metadata, x = self._build_native_kv_test_fixture(
            seq_len=seq_len,
            history_len=history_len,
            num_targets=num_targets,
        )

        captured_rows: list[int] = []
        original_uqvk = stu_module.hstu_compute_uqvk

        def _capture_uqvk(*args, **kwargs):
            captured_rows.append(int(kwargs["x"].shape[0]))
            return original_uqvk(*args, **kwargs)

        with mock.patch.object(stu_module, "hstu_compute_uqvk", side_effect=_capture_uqvk):
            native_layer(
                x=x[:history_len],
                x_lengths=torch.tensor([history_len], dtype=torch.int64),
                x_offsets=torch.tensor([0, history_len], dtype=torch.int64),
                max_seq_len=history_len,
                num_targets=torch.tensor([0], dtype=torch.int64),
                hstu_attn_metadata=metadata,
                hstu_cache_layer=cache_layer,
                hstu_attention_mode="prefix",
            )

        self.assertEqual(captured_rows, [seq_len])

    def test_native_delta_projection_uses_full_sequence_shape(self) -> None:
        seq_len = 9
        history_len = 6
        num_targets = 3
        native_layer, cache_layer, metadata, x = self._build_native_kv_test_fixture(
            seq_len=seq_len,
            history_len=history_len,
            num_targets=num_targets,
        )

        native_layer(
            x=x[:history_len],
            x_lengths=torch.tensor([history_len], dtype=torch.int64),
            x_offsets=torch.tensor([0, history_len], dtype=torch.int64),
            max_seq_len=history_len,
            num_targets=torch.tensor([0], dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="prefix",
        )

        captured_rows: list[int] = []
        original_uqvk = stu_module.hstu_compute_uqvk

        def _capture_uqvk(*args, **kwargs):
            captured_rows.append(int(kwargs["x"].shape[0]))
            return original_uqvk(*args, **kwargs)

        with mock.patch.object(stu_module, "hstu_compute_uqvk", side_effect=_capture_uqvk):
            native_layer.cached_forward(
                delta_x=x[history_len:],
                num_targets=torch.tensor([num_targets], dtype=torch.int64),
                hstu_attn_metadata=metadata,
                hstu_cache_layer=cache_layer,
                hstu_attention_mode="delta",
            )

        self.assertEqual(captured_rows, [seq_len])

    def test_prefix_hit_coverage_falls_back_without_warning(self) -> None:
        model = self._make_hstu_model_for_metadata_test()
        metadata = self._make_attention_metadata_for_coverage_test(
            slot_count=1,
            computed_tokens=21,
        )
        fake_forward_context = SimpleNamespace(attn_metadata={"layer0": metadata})

        with mock.patch(
            "vllm.forward_context.is_forward_context_available",
            return_value=True,
        ), mock.patch(
            "vllm.forward_context.get_forward_context",
            return_value=fake_forward_context,
        ), mock.patch(
            "vllm_gr.model_executor.models.hstu_vllm.logger.debug_once"
        ) as debug_once, mock.patch(
            "vllm_gr.model_executor.models.hstu_vllm.logger.warning_once"
        ) as warning_once:
            result = model._get_hstu_attention_metadata(
                uih_seq_lengths=torch.tensor([11], dtype=torch.int32),
                num_targets=torch.tensor([6], dtype=torch.int32),
            )

        self.assertIsNone(result)
        debug_once.assert_called_once()
        warning_once.assert_not_called()

    def test_incomplete_coverage_keeps_warning(self) -> None:
        model = self._make_hstu_model_for_metadata_test()
        metadata = self._make_attention_metadata_for_coverage_test(
            slot_count=1,
            computed_tokens=0,
        )
        fake_forward_context = SimpleNamespace(attn_metadata={"layer0": metadata})

        with mock.patch(
            "vllm.forward_context.is_forward_context_available",
            return_value=True,
        ), mock.patch(
            "vllm.forward_context.get_forward_context",
            return_value=fake_forward_context,
        ), mock.patch(
            "vllm_gr.model_executor.models.hstu_vllm.logger.debug_once"
        ) as debug_once, mock.patch(
            "vllm_gr.model_executor.models.hstu_vllm.logger.warning_once"
        ) as warning_once:
            result = model._get_hstu_attention_metadata(
                uih_seq_lengths=torch.tensor([11], dtype=torch.int32),
                num_targets=torch.tensor([6], dtype=torch.int32),
            )

        self.assertIsNone(result)
        debug_once.assert_not_called()
        warning_once.assert_called_once()

    def test_npu_runner_falls_back_to_hstu_layers_for_kv_cache_spec(self) -> None:
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=8,
            head_size_v=16,
            dtype=torch.float32,
        )
        layer = SimpleNamespace(
            layer_name="hstu_kv_cache_layers.0",
            get_kv_cache_spec=lambda _vllm_config: spec,
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    enable_kv_cache=True,
                    contextual_feature_to_max_length={},
                )
            ),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )

        with _patched_gr_npu_model_runner() as gr_npu_model_runner_cls:
            runner = gr_npu_model_runner_cls(vllm_config, torch.device("cpu"))
            runner.model = SimpleNamespace(_hstu_kv_cache_layers=[layer])

            kv_cache_spec = runner.get_kv_cache_spec()

        self.assertEqual(kv_cache_spec, {"hstu_kv_cache_layers.0": spec})
        self.assertIs(
            vllm_config.compilation_config.static_forward_context["hstu_kv_cache_layers.0"],
            layer,
        )

    def test_builder_derives_contextual_seq_len_from_config(self) -> None:
        common_metadata = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=4,
            block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
            slot_mapping=torch.zeros(4, dtype=torch.int64),
            _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
            max_query_len=4,
            max_seq_len=4,
        )
        builder = HSTUAttentionMetadataBuilder(
            FullAttentionSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=8,
                head_size_v=16,
                dtype=torch.float32,
            ),
            ["layer0"],
            SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        contextual_feature_to_max_length={"user_id": 1, "sex": 1},
                    )
                ),
                speculative_config=None,
                parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            ),
            torch.device("cpu"),
        )
        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common_metadata)

        torch.testing.assert_close(
            metadata.num_targets.to(dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )
        torch.testing.assert_close(
            metadata.history_lengths.to(dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )
        self.assertEqual(metadata.contextual_seq_len, 2)

    def test_model_syncs_hstu_attention_metadata_from_runtime_lengths(self) -> None:
        model = self._make_hstu_model_for_metadata_test()
        metadata = self._make_attention_metadata_for_coverage_test(
            slot_count=22,
            computed_tokens=0,
            history_len=0,
            contextual_seq_len=0,
            num_targets=0,
        )
        fake_forward_context = SimpleNamespace(attn_metadata={"layer0": metadata})

        with mock.patch(
            "vllm.forward_context.is_forward_context_available",
            return_value=True,
        ), mock.patch(
            "vllm.forward_context.get_forward_context",
            return_value=fake_forward_context,
        ):
            result = model._get_hstu_attention_metadata(
                uih_seq_lengths=torch.tensor([11], dtype=torch.int32),
                num_targets=torch.tensor([6], dtype=torch.int32),
            )

        self.assertIs(result, metadata)
        torch.testing.assert_close(
            metadata.num_targets.to(dtype=torch.int64),
            torch.tensor([6], dtype=torch.int64),
        )
        torch.testing.assert_close(
            metadata.history_lengths.to(dtype=torch.int64),
            torch.tensor([11], dtype=torch.int64),
        )
        self.assertEqual(metadata.contextual_seq_len, 5)

    def test_npu_runner_uses_generic_kv_init_for_hstu(self) -> None:
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="hstu"),
            ),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        with _patched_gr_npu_model_runner() as gr_npu_model_runner_cls:
            runner = gr_npu_model_runner_cls(vllm_config, torch.device("cpu"))
            runner.model = SimpleNamespace(_hstu_kv_cache_layers=[object()])
            runner.attn_groups = []
            runner.cross_layers_kv_cache = None
            runner.cross_layers_attn_backend = None
            runner.may_add_encoder_only_layers_to_kv_cache_config = mock.Mock()
            runner.maybe_add_kv_sharing_layers_to_kv_cache_groups = mock.Mock()
            runner.may_reinitialize_input_batch = mock.Mock()

            kv_cache_config = SimpleNamespace()
            fake_kernel_block_sizes = [16]
            fake_kv_caches = {"hstu_kv_cache_layers.0": object()}

            fake_gpu_model_runner_module = types.ModuleType(
                "vllm.v1.worker.gpu_model_runner"
            )
            fake_worker_utils_module = types.ModuleType("vllm.v1.worker.utils")

            class _FakeGPUModelRunner:
                initialize_attn_backend = mock.Mock()
                _prepare_kernel_block_sizes = mock.Mock(
                    return_value=fake_kernel_block_sizes
                )
                initialize_metadata_builders = mock.Mock()
                _allocate_kv_cache_tensors = mock.Mock(return_value={"raw": object()})
                _reshape_kv_cache_tensors = mock.Mock(return_value=fake_kv_caches)

            fake_gpu_model_runner_module.GPUModelRunner = _FakeGPUModelRunner
            fake_worker_utils_module.bind_kv_cache = mock.Mock()

            with mock.patch.dict(
                sys.modules,
                {
                    "vllm.v1.worker.gpu_model_runner": fake_gpu_model_runner_module,
                    "vllm.v1.worker.utils": fake_worker_utils_module,
                },
            ):
                runner.initialize_kv_cache(kv_cache_config)

        runner.may_add_encoder_only_layers_to_kv_cache_config.assert_called_once()
        runner.maybe_add_kv_sharing_layers_to_kv_cache_groups.assert_called_once()
        _FakeGPUModelRunner.initialize_attn_backend.assert_called_once()
        _FakeGPUModelRunner._prepare_kernel_block_sizes.assert_called_once()
        _FakeGPUModelRunner.initialize_metadata_builders.assert_called_once()
        runner.may_reinitialize_input_batch.assert_called_once()
        _FakeGPUModelRunner._allocate_kv_cache_tensors.assert_called_once()
        _FakeGPUModelRunner._reshape_kv_cache_tensors.assert_called_once()
        fake_worker_utils_module.bind_kv_cache.assert_called_once()

    def test_hstu_transducer_falls_back_to_full_compute_when_cached_path_fails(
        self,
    ) -> None:
        transducer = HSTUTransducer.__new__(HSTUTransducer)
        torch.nn.Module.__init__(transducer)

        seq_lengths = torch.tensor([5], dtype=torch.int64)
        seq_offsets = torch.tensor([0, 5], dtype=torch.int64)
        seq_timestamps = torch.arange(5, dtype=torch.int64)
        seq_embeddings = torch.randn(5, 4)
        num_targets = torch.tensor([2], dtype=torch.int64)
        seq_payloads = {"payload": torch.tensor([1], dtype=torch.int64)}
        cached_failure = RuntimeError(
            "The expanded size of the tensor (22) must match the existing size (0)"
        )
        full_result = torch.randn_like(seq_embeddings)
        postprocess_result = (None, torch.randn(2, 4))

        transducer._preprocess = mock.Mock(
            return_value=(
                5,
                3,
                2,
                seq_lengths,
                seq_offsets,
                seq_timestamps,
                seq_embeddings,
                num_targets,
                seq_payloads,
            )
        )
        transducer._can_use_vllm_kv_cache = mock.Mock(return_value=True)
        transducer._cached_path_has_consistent_inputs = mock.Mock(return_value=True)
        transducer._hstu_compute_cached = mock.Mock(side_effect=cached_failure)
        transducer._should_fallback_cached_compute = mock.Mock(return_value=True)
        transducer._hstu_compute = mock.Mock(return_value=full_result)
        transducer._postprocess = mock.Mock(return_value=postprocess_result)

        candidate_embeddings, full_embeddings = transducer.forward(
            max_uih_len=3,
            max_targets=2,
            total_uih_len=3,
            total_targets=2,
            seq_lengths=seq_lengths,
            seq_embeddings=seq_embeddings,
            seq_timestamps=seq_timestamps,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
            hstu_attn_metadata=object(),
        )

        self.assertIs(candidate_embeddings, postprocess_result[1])
        self.assertIs(full_embeddings, postprocess_result[0])
        transducer._hstu_compute_cached.assert_called_once()
        transducer._cached_path_has_consistent_inputs.assert_called_once_with(
            seq_lengths=seq_lengths,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            hstu_attn_metadata=mock.ANY,
        )
        transducer._should_fallback_cached_compute.assert_called_once_with(
            cached_failure,
            seq_embeddings=seq_embeddings,
        )
        transducer._hstu_compute.assert_called_once_with(
            max_seq_len=5,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            hstu_attn_metadata=None,
        )

    def test_hstu_transducer_disables_cached_path_on_embedding_length_mismatch(
        self,
    ) -> None:
        transducer = HSTUTransducer.__new__(HSTUTransducer)
        torch.nn.Module.__init__(transducer)

        seq_lengths = torch.tensor([5], dtype=torch.int64)
        seq_offsets = torch.tensor([0, 5], dtype=torch.int64)
        seq_timestamps = torch.arange(5, dtype=torch.int64)
        seq_embeddings = torch.randn(3, 4)
        num_targets = torch.tensor([2], dtype=torch.int64)
        seq_payloads = {"payload": torch.tensor([1], dtype=torch.int64)}
        full_result = torch.randn(5, 4)
        postprocess_result = (None, torch.randn(2, 4))
        metadata = SimpleNamespace(
            common_attn_metadata=SimpleNamespace(
                slot_mapping=torch.zeros(5, dtype=torch.int64),
                _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int64),
            )
        )

        transducer._preprocess = mock.Mock(
            return_value=(
                5,
                3,
                2,
                seq_lengths,
                seq_offsets,
                seq_timestamps,
                seq_embeddings,
                num_targets,
                seq_payloads,
            )
        )
        transducer._can_use_vllm_kv_cache = mock.Mock(return_value=True)
        transducer._hstu_compute_cached = mock.Mock()
        transducer._hstu_compute = mock.Mock(return_value=full_result)
        transducer._postprocess = mock.Mock(return_value=postprocess_result)

        candidate_embeddings, full_embeddings = transducer.forward(
            max_uih_len=3,
            max_targets=2,
            total_uih_len=3,
            total_targets=2,
            seq_lengths=seq_lengths,
            seq_embeddings=seq_embeddings,
            seq_timestamps=seq_timestamps,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
            hstu_attn_metadata=metadata,
        )

        self.assertIs(candidate_embeddings, postprocess_result[1])
        self.assertIs(full_embeddings, postprocess_result[0])
        transducer._hstu_compute_cached.assert_not_called()
        transducer._hstu_compute.assert_called_once_with(
            max_seq_len=5,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            hstu_attn_metadata=metadata,
        )

    def test_hstu_transducer_falls_back_when_cached_compute_returns_empty_shape(
        self,
    ) -> None:
        transducer = HSTUTransducer.__new__(HSTUTransducer)
        torch.nn.Module.__init__(transducer)

        seq_lengths = torch.tensor([5], dtype=torch.int64)
        seq_offsets = torch.tensor([0, 5], dtype=torch.int64)
        seq_timestamps = torch.arange(5, dtype=torch.int64)
        seq_embeddings = torch.randn(5, 4).to("cpu")
        num_targets = torch.tensor([2], dtype=torch.int64)
        seq_payloads = {"payload": torch.tensor([1], dtype=torch.int64)}
        cached_result = torch.empty(0, 4)
        full_result = torch.randn(5, 4)
        postprocess_result = (None, torch.randn(2, 4))

        transducer._preprocess = mock.Mock(
            return_value=(
                5,
                3,
                2,
                seq_lengths,
                seq_offsets,
                seq_timestamps,
                seq_embeddings,
                num_targets,
                seq_payloads,
            )
        )
        transducer._can_use_vllm_kv_cache = mock.Mock(return_value=True)
        transducer._cached_path_has_consistent_inputs = mock.Mock(return_value=True)
        transducer._hstu_compute_cached = mock.Mock(return_value=cached_result)
        transducer._should_fallback_cached_compute = mock.Mock(return_value=True)
        transducer._hstu_compute = mock.Mock(return_value=full_result)
        transducer._postprocess = mock.Mock(return_value=postprocess_result)

        candidate_embeddings, full_embeddings = transducer.forward(
            max_uih_len=3,
            max_targets=2,
            total_uih_len=3,
            total_targets=2,
            seq_lengths=seq_lengths,
            seq_embeddings=seq_embeddings,
            seq_timestamps=seq_timestamps,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
            hstu_attn_metadata=object(),
        )

        self.assertIs(candidate_embeddings, postprocess_result[1])
        self.assertIs(full_embeddings, postprocess_result[0])
        transducer._hstu_compute_cached.assert_called_once()
        transducer._should_fallback_cached_compute.assert_called_once()
        transducer._hstu_compute.assert_called_once_with(
            max_seq_len=5,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            hstu_attn_metadata=None,
        )
