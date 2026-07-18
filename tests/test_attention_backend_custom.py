# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
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
        if marker is not None:
            config.additional_config["vllm_gr_attention_backend"] = marker
        return config

    @mock.patch("vllm_gr.engine.arg_utils_gr.origin_create_engine_config")
    def test_attention_backend_cli_custom_persists_marker(self, mock_origin_create):
        """--attention-backend CUSTOM sets marker and restores backend."""
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.engine.arg_utils_gr import create_engine_config

        # Simulate: platform reset backend to None inside original create_engine_config
        fake_config = self._make_fake_config(backend=None)
        mock_origin_create.return_value = fake_config

        class FakeSelf:
            attention_backend = "CUSTOM"
            attention_config = None

        result = create_engine_config(FakeSelf())

        assert result.additional_config.get("vllm_gr_attention_backend") == "CUSTOM"
        assert result.attention_config.backend == AttentionBackendEnum.CUSTOM

    @mock.patch("vllm_gr.engine.arg_utils_gr.origin_create_engine_config")
    def test_attention_config_backend_custom_persists_marker(self, mock_origin_create):
        """--attention-config backend=CUSTOM sets marker and restores backend."""
        from vllm.config import AttentionConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.engine.arg_utils_gr import create_engine_config

        fake_config = self._make_fake_config(backend=None)
        mock_origin_create.return_value = fake_config

        class FakeSelf:
            attention_backend = None
            attention_config = AttentionConfig(backend=AttentionBackendEnum.CUSTOM)

        result = create_engine_config(FakeSelf())

        assert result.additional_config.get("vllm_gr_attention_backend") == "CUSTOM"
        assert result.attention_config.backend == AttentionBackendEnum.CUSTOM

    @mock.patch("vllm_gr.engine.arg_utils_gr.origin_create_engine_config")
    def test_non_custom_backend_does_not_set_marker(self, mock_origin_create):
        """Non-CUSTOM backend leaves additional_config untouched."""
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.engine.arg_utils_gr import create_engine_config

        fake_config = self._make_fake_config(backend=AttentionBackendEnum.FLASH_ATTN)
        mock_origin_create.return_value = fake_config

        class FakeSelf:
            attention_backend = "FLASH_ATTN"
            attention_config = None

        result = create_engine_config(FakeSelf())

        assert "vllm_gr_attention_backend" not in result.additional_config
        assert result.attention_config.backend is not AttentionBackendEnum.CUSTOM

    @mock.patch("vllm_gr.engine.arg_utils_gr.origin_create_engine_config")
    def test_catalog_path_persists_in_additional_config(self, mock_origin_create):
        """GR-only model options must not be injected into ModelConfig."""
        from vllm_gr.engine.arg_utils_gr import create_engine_config

        fake_config = self._make_fake_config()
        mock_origin_create.return_value = fake_config

        class FakeSelf:
            attention_backend = None
            attention_config = None
            catalog_path = "/tmp/catalog.json"

        result = create_engine_config(FakeSelf())

        assert result.additional_config["vllm_gr_catalog_path"] == "/tmp/catalog.json"


class TestCustomBackendMarkerSerialization(unittest.TestCase):
    """Verify marker survives pickle round-trip."""

    def test_marker_survives_pickle(self):
        """Pickle preserves both backend=CUSTOM and the marker."""
        from vllm.config import AttentionConfig, VllmConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        config = VllmConfig(attention_config=AttentionConfig())
        config.attention_config.backend = AttentionBackendEnum.CUSTOM
        config.additional_config["vllm_gr_attention_backend"] = "CUSTOM"

        _pickle = importlib.import_module("pickle")
        serialized = _pickle.dumps(config)
        restored = _pickle.loads(serialized)

        assert restored.attention_config.backend == AttentionBackendEnum.CUSTOM
        assert restored.additional_config.get("vllm_gr_attention_backend") == "CUSTOM"

    def test_marker_restores_custom_backend_selection(self):
        """When backend=None in child but marker present, selector still returns Beam."""
        import vllm.v1.attention.selector as selector_mod
        from vllm.config import AttentionConfig, VllmConfig

        # Make sure the patch is applied
        from vllm_gr.patch import _apply_ascend_attention_backend_patch
        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

        _apply_ascend_attention_backend_patch(force=True)

        # Simulate child process: config deserialized with backend=None but marker present
        child_config = VllmConfig(attention_config=AttentionConfig())
        child_config.attention_config.backend = None
        child_config.additional_config["vllm_gr_attention_backend"] = "CUSTOM"

        # Temporarily bind as current config so get_current_vllm_config_or_none works
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(child_config):
            import torch
            from vllm.v1.attention.selector import AttentionSelectorConfig

            selector_cfg = AttentionSelectorConfig(
                head_size=128,
                dtype=torch.bfloat16,
                kv_cache_dtype=None,
                block_size=128,
            )
            result = selector_mod._cached_get_attn_backend(None, selector_cfg)

        assert result is BeamAttentionBackend


class TestNativePlatformBackendFallback(unittest.TestCase):
    """Verify backend=None without marker delegates to native platform backend."""

    def test_backend_none_without_marker_does_not_return_beam(self):
        """Without marker, backend=None resolves to native platform backend."""
        import vllm.v1.attention.selector as selector_mod
        from vllm.config import AttentionConfig, VllmConfig

        from vllm_gr.patch import _apply_ascend_attention_backend_patch

        _apply_ascend_attention_backend_patch(force=True)

        # Config with backend=None and NO marker
        config = VllmConfig(attention_config=AttentionConfig())
        config.attention_config.backend = None
        # additional_config empty (no marker)

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(config):
            import torch
            from vllm.v1.attention.selector import AttentionSelectorConfig

            # Force cache miss by using a different head_size
            selector_cfg = AttentionSelectorConfig(
                head_size=64,
                dtype=torch.float16,
                kv_cache_dtype=None,
                block_size=128,
            )
            result = selector_mod._cached_get_attn_backend(None, selector_cfg)

        # Result should NOT be BeamAttentionBackend; it delegates to original.
        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

        assert result is not BeamAttentionBackend


class TestBeamAttentionVllmApiContract(unittest.TestCase):
    """Pin the vLLM 0.22 attention APIs used only on accelerator paths."""

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

        import vllm_gr.v1.attention.backends.beam_attn as beam_attn

        builder = object.__new__(beam_attn.BeamAttentionMetadataBuilder)
        builder.aot_schedule = True
        builder.cache_config = SimpleNamespace(cache_dtype="auto")
        builder.kv_cache_dtype = torch.bfloat16
        builder.num_heads_q = 16
        builder.num_kv_heads = 4
        builder.headdim = 128
        builder.block_size = 16
        builder.aot_sliding_window = (-1, -1)

        with mock.patch.object(beam_attn, "get_scheduler_metadata", return_value={}) as scheduler:
            builder._get_schedule(
                batch_size=2,
                cu_query_lens=torch.tensor([0, 1, 2], dtype=torch.int32),
                max_query_len=1,
                seqlens=torch.tensor([8, 8], dtype=torch.int32),
                max_seq_len=8,
                causal=True,
                max_num_splits=0,
            )

        kwargs = scheduler.call_args.kwargs
        assert kwargs["num_heads_kv"] == 4
        assert "num_kv_heads" not in kwargs


try:
    from vllm_gr.patch import _is_npu_platform

    _IS_NPU = _is_npu_platform()
except Exception:
    _IS_NPU = False


@unittest.skipIf(_IS_NPU, "NPU platform: NPUWorker is expected to be present")
class TestNpuWorkerPatchSkippedOnNonNpu(unittest.TestCase):
    """Verify non-NPU environments do not import or patch NPUWorker."""

    def test_patch_npu_worker_returns_zero_without_npu(self):
        """_patch_npu_worker_init_device returns 0 when NPUWorker absent."""
        from vllm_gr.patch import _patch_npu_worker_init_device

        # On a standard CUDA/CPU install vllm_ascend is not importable.
        patch_count = _patch_npu_worker_init_device()
        assert patch_count == 0

    def test_npu_worker_module_not_imported_on_non_npu(self):
        """The vllm_ascend.worker.worker module is not present in sys.modules."""
        # On an NPU machine vllm_ascend itself may already be loaded by the
        # platform runtime; we only care that NPUWorker is not imported / patched.
        assert "vllm_ascend.worker.worker" not in sys.modules

    def test_re_apply_patches_does_not_import_npu_worker(self):
        """Running re_apply_patches on non-NPU does not pull in NPUWorker."""

        from vllm_gr.patch import re_apply_patches

        before = set(sys.modules.keys())
        re_apply_patches()
        after = set(sys.modules.keys())

        _ = after - before
        assert "vllm_ascend.worker.worker" not in sys.modules


if __name__ == "__main__":
    unittest.main()
