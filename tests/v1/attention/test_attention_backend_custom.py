# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.helpers.paths import REPO_ROOT

ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestCustomAttentionBackendMarker(unittest.TestCase):
    """Verify CUSTOM backend marker persistence across CLI/API paths."""

    @classmethod
    def setUpClass(cls):
        """Apply patches so EngineArgs.create_engine_config is wrapped."""
        from vllm_gr.engine.arg_utils_gr import patch_add_cli_args

        patch_add_cli_args()

    def _make_fake_config(self, backend=None, marker=None):
        """Build a lightweight VllmConfig-like object for mocking."""
        from vllm.config import AttentionConfig, VllmConfig

        attention_config = AttentionConfig()
        if backend is not None:
            attention_config.backend = backend
        elif backend is None:
            attention_config.backend = None

        config = VllmConfig(attention_config=attention_config)
        # Model execution is mocked; retain the GR runtime contract on CPU hosts.
        config.scheduler_config.async_scheduling = True
        config.scheduler_config.enable_chunked_prefill = False
        config.scheduler_config.long_prefill_token_threshold = 0
        if marker is not None:
            config.additional_config["vllm_gr"] = {
                "gr": {},
                "attention_backend": marker,
            }
        return config

    @mock.patch("vllm_gr.engine.arg_utils_gr.origin_create_engine_config")
    def test_attention_backend_cli_custom_persists_marker(self, mock_origin_create):
        """--attention-backend CUSTOM sets marker and restores backend."""
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.engine.arg_utils_gr import create_engine_config

        # Ascend clears the native backend during VllmConfig creation.
        fake_config = self._make_fake_config()
        mock_origin_create.return_value = fake_config

        class FakeSelf:
            _gr_selected = True
            attention_backend = "CUSTOM"
            attention_config = None

        result = create_engine_config(FakeSelf())

        from vllm_gr.config import GRAttentionBackend, GRConfig

        assert GRConfig.from_vllm_config(result).attention_backend is GRAttentionBackend.CUSTOM
        assert result.attention_config.backend == AttentionBackendEnum.CUSTOM

    @mock.patch("vllm_gr.engine.arg_utils_gr.origin_create_engine_config")
    def test_attention_config_backend_custom_persists_marker(self, mock_origin_create):
        """Restore the explicit CUSTOM backend after Ascend clears it."""
        from vllm.config import AttentionConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.engine.arg_utils_gr import create_engine_config

        fake_config = self._make_fake_config()
        mock_origin_create.return_value = fake_config

        class FakeSelf:
            _gr_selected = True
            attention_backend = None
            attention_config = AttentionConfig(backend=AttentionBackendEnum.CUSTOM)

        result = create_engine_config(FakeSelf())

        from vllm_gr.config import GRAttentionBackend, GRConfig

        assert GRConfig.from_vllm_config(result).attention_backend is GRAttentionBackend.CUSTOM
        assert result.attention_config.backend == AttentionBackendEnum.CUSTOM

    @mock.patch("vllm_gr.engine.arg_utils_gr.origin_create_engine_config")
    def test_non_custom_backend_does_not_set_marker(self, mock_origin_create):
        """Non-CUSTOM backend leaves the GR attention marker unset."""
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.config import GRConfig
        from vllm_gr.engine.arg_utils_gr import create_engine_config

        fake_config = self._make_fake_config(backend=AttentionBackendEnum.FLASH_ATTN)
        mock_origin_create.return_value = fake_config

        class FakeSelf:
            attention_backend = "FLASH_ATTN"
            attention_config = None

        result = create_engine_config(FakeSelf())

        assert GRConfig.from_vllm_config(result).attention_backend is None
        assert result.attention_config.backend is AttentionBackendEnum.FLASH_ATTN


class TestCustomBackendMarkerSerialization(unittest.TestCase):
    """Verify marker survives pickle round-trip."""

    def test_marker_survives_pickle(self):
        """Pickle preserves both backend=CUSTOM and the marker."""
        from vllm.config import AttentionConfig, VllmConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.config import GRAttentionBackend, GRConfig
        from vllm_gr.config.parser import parse_gr_config

        config = VllmConfig(attention_config=AttentionConfig())
        config.attention_config.backend = AttentionBackendEnum.CUSTOM
        GRConfig(attention_backend=GRAttentionBackend.CUSTOM).write_to_vllm_config(config)

        _pickle = importlib.import_module("pickle")
        serialized = _pickle.dumps(config)
        restored = _pickle.loads(serialized)

        assert restored.attention_config.backend == AttentionBackendEnum.CUSTOM
        assert parse_gr_config(restored).attention_backend is GRAttentionBackend.CUSTOM


class TestNpuDefaultBlockSize(unittest.TestCase):
    """Guard the vLLM 16 / vLLM-Ascend 128 default mismatch."""

    @staticmethod
    def _config(*, user_specified=False, is_hybrid=False):
        return SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=16,
                user_specified_block_size=user_specified,
            ),
            model_config=SimpleNamespace(is_hybrid=is_hybrid),
        )

    def test_npu_uses_128_when_block_size_was_not_explicit(self):
        from vllm_gr.platforms.npu.config import NPUConfigService

        config = self._config()
        NPUConfigService().apply_native_defaults(config)

        assert config.cache_config.block_size == 128

    def test_npu_preserves_explicit_block_size(self):
        from vllm_gr.platforms.npu.config import NPUConfigService

        config = self._config(user_specified=True)
        NPUConfigService().apply_native_defaults(config)

        assert config.cache_config.block_size == 16

    def test_npu_preserves_hybrid_model_block_size(self):
        from vllm_gr.platforms.npu.config import NPUConfigService

        config = self._config(is_hybrid=True)
        NPUConfigService().apply_native_defaults(config)

        assert config.cache_config.block_size == 16

    def test_gpu_keeps_vllm_default(self):
        from vllm_gr.platforms.cuda.config import CudaConfigService

        config = self._config()
        CudaConfigService().apply_native_defaults(config)

        assert config.cache_config.block_size == 16


class TestBeamAttentionVllmApiContract(unittest.TestCase):
    """Pin the vLLM 0.30 attention APIs used only on accelerator paths."""

    def test_backend_accepts_native_configuration_validation(self):
        import torch
        from vllm.platforms.interface import DeviceCapability

        from vllm_gr.v1.attention.backends import beam_attn

        execution = SimpleNamespace(is_available=lambda: True)
        with (
            mock.patch.object(beam_attn, "CUDA_DEVICE", True),
            mock.patch.object(beam_attn, "NPU_DEVICE", False),
            mock.patch.object(beam_attn, "_get_platform_execution_module", return_value=execution),
        ):
            reasons = beam_attn.BeamAttentionBackend.validate_configuration(
                head_size=128,
                dtype=torch.bfloat16,
                kv_cache_dtype="auto",
                block_size=16,
                use_mla=False,
                has_sink=False,
                use_sparse=False,
                use_mm_prefix=False,
                use_per_head_quant_scales=False,
                device_capability=DeviceCapability(8, 9),
                attn_type="decoder",
            )
        assert reasons == []

    def test_backend_pins_ascend_kv_cache_shape(self):
        """vLLM 0.30 dropped the base-class method; the NPU runner calls it."""
        from vllm import envs as vllm_envs

        from vllm_gr.v1.attention.backends import beam_attn

        def shape_for(layout):
            with mock.patch.object(vllm_envs, "VLLM_KV_CACHE_LAYOUT", layout):
                return beam_attn.BeamAttentionBackend.get_kv_cache_shape(8, 128, 2, 64)

        # Default (unset) and the token-major NHD view.
        assert shape_for(None) == (2, 8, 128, 2, 64)
        assert shape_for("NHD") == (2, 8, 128, 2, 64)
        # Head-major layouts switch the block/head axes, matching vllm-ascend.
        assert shape_for("HND") == (2, 8, 2, 128, 64)
        assert shape_for("LBHNC") == (2, 8, 2, 128, 64)

    def test_impl_accepts_vllm_sinks_keyword(self):
        """Attention forwards extra implementation arguments as ``sinks``."""
        import inspect

        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl

        parameters = inspect.signature(BeamAttentionImpl.__init__).parameters
        assert "sinks" in parameters
        assert "running_sinks" not in parameters

    def test_fa3_scheduler_uses_num_heads_kv_keyword(self):
        """FA3 requires the scheduler keyword ``num_heads_kv``."""
        from types import SimpleNamespace

        import torch

        import vllm_gr.v1.attention.backends.beam_attn_metadata_tensors as beam_metadata_tensors
        from vllm_gr.v1.attention.backends import beam_attn_metadata

        builder = object.__new__(beam_attn_metadata.BeamAttentionMetadataBuilder)
        builder.aot_schedule = True
        builder.cache_config = SimpleNamespace(cache_dtype="auto")
        builder.kv_cache_dtype = torch.bfloat16
        builder.num_heads_q = 16
        builder.num_kv_heads = 4
        builder.headdim = 128
        builder.block_size = 16
        builder.aot_sliding_window = (-1, -1)

        with mock.patch.object(
            beam_attn_metadata,
            "get_scheduler_metadata",
            return_value={},
        ) as scheduler:
            beam_metadata_tensors.get_schedule(
                builder._tensor_context(),
                batch_size=2,
                cu_query_lens=torch.tensor([0, 1, 2], dtype=torch.int32),
                max_query_len=1,
                seqlens=torch.tensor([8, 8], dtype=torch.int32),
                max_seq_len=8,
                causal=True,
                max_num_splits=0,
                scheduler=beam_attn_metadata.get_scheduler_metadata,
            )

        kwargs = scheduler.call_args.kwargs
        assert kwargs["num_heads_kv"] == 4
        assert "num_kv_heads" not in kwargs

    def test_batch_invariant_forces_single_flash_attention_split(self):
        from vllm_gr.v1.attention.backends import beam_attn_gpu

        with mock.patch.object(beam_attn_gpu.envs, "VLLM_BATCH_INVARIANT", True):
            assert beam_attn_gpu._num_splits(0) == 1
            assert beam_attn_gpu._num_splits(7) == 1
        with mock.patch.object(beam_attn_gpu.envs, "VLLM_BATCH_INVARIANT", False):
            assert beam_attn_gpu._num_splits(0) == 0
            assert beam_attn_gpu._num_splits(7) == 7


if __name__ == "__main__":
    unittest.main()


pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _isolate_backend_plan_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise configuration policy independently of a native backend install."""
    from vllm_gr.platforms import planning

    monkeypatch.setattr(planning, "admit_backend_plan", lambda config: None)
