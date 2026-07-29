# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
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
        from vllm_gr.engine.arg_utils_gr import _apply_npu_default_block_size

        config = self._config()
        with mock.patch(
            "vllm.platforms.current_platform",
            SimpleNamespace(device_type="npu"),
        ):
            _apply_npu_default_block_size(config)

        assert config.cache_config.block_size == 128

    def test_npu_preserves_explicit_block_size(self):
        from vllm_gr.engine.arg_utils_gr import _apply_npu_default_block_size

        config = self._config(user_specified=True)
        with mock.patch(
            "vllm.platforms.current_platform",
            SimpleNamespace(device_type="npu"),
        ):
            _apply_npu_default_block_size(config)

        assert config.cache_config.block_size == 16

    def test_npu_preserves_hybrid_model_block_size(self):
        from vllm_gr.engine.arg_utils_gr import _apply_npu_default_block_size

        config = self._config(is_hybrid=True)
        with mock.patch(
            "vllm.platforms.current_platform",
            SimpleNamespace(device_type="npu"),
        ):
            _apply_npu_default_block_size(config)

        assert config.cache_config.block_size == 16

    def test_gpu_keeps_vllm_default(self):
        from vllm_gr.engine.arg_utils_gr import _apply_npu_default_block_size

        config = self._config()
        with mock.patch(
            "vllm.platforms.current_platform",
            SimpleNamespace(device_type="cuda"),
        ):
            _apply_npu_default_block_size(config)

        assert config.cache_config.block_size == 16


class TestNativePlatformBackendFallback(unittest.TestCase):
    """Verify backend=None without marker delegates to native platform backend."""

    def test_backend_none_without_marker_does_not_return_beam(self):
        """Without marker, backend=None delegates to the native selector."""
        import vllm.v1.attention.selector as selector_mod
        from vllm.config import set_current_vllm_config

        import vllm_gr.patch as gr_patch

        gr_patch._apply_ascend_attention_backend_patch(force=True)

        class FakeNativeBackend:
            """Stand-in for the platform backend without importing device ops."""

            @staticmethod
            def get_required_kv_cache_layout():
                return None

        # The original selector is cached. Clear it so this test exercises the
        # delegation path while resolve_obj_by_qualname is patched below.
        gr_patch._ASCEND_ORIGINAL_CACHED_GET_ATTN.cache_clear()

        # Only additional_config is read by the GR wrapper. Keeping this fake
        # config minimal avoids initializing a real accelerator platform in a
        # unit test whose contract is delegation.
        config = mock.Mock(additional_config={})
        fake_platform = mock.Mock(device_name="test-device")
        fake_platform.get_attn_backend_cls.return_value = "tests.FakeNativeBackend"

        with (
            set_current_vllm_config(config),
            mock.patch("vllm.platforms.current_platform", fake_platform),
            mock.patch.object(
                selector_mod,
                "resolve_obj_by_qualname",
                return_value=FakeNativeBackend,
            ) as mock_resolve,
        ):
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

        # The native platform still chooses its backend qualname, but the unit
        # test does not import the complete CUDA/NPU backend dependency graph.
        fake_platform.get_attn_backend_cls.assert_called_once()
        mock_resolve.assert_called_once()
        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

        assert result is FakeNativeBackend
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

        import vllm_gr.v1.attention.backends.beam_attn_metadata as beam_attn_metadata

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


class TestPatchApplicationOrder(unittest.TestCase):
    """Pin the import-safe ordering required by vllm-ascend."""

    def test_npu_worker_is_initialized_before_beam_runner_patches(self):
        import vllm_gr.patch as gr_patch

        call_order = []

        with (
            mock.patch.object(
                gr_patch,
                "_apply_ascend_attention_backend_patch",
                side_effect=lambda force=False: call_order.append("attention") or 1,
            ),
            mock.patch.object(
                gr_patch,
                "_patch_npu_worker_init_device",
                side_effect=lambda: call_order.append("npu_worker") or 1,
            ),
            mock.patch.object(
                gr_patch,
                "_apply_gr_beam_patches",
                side_effect=lambda force=False: call_order.append("beam") or 1,
            ),
        ):
            patch_count = gr_patch._apply_patches(force=True)

        assert patch_count == 3
        assert call_order == ["attention", "npu_worker", "beam"]


if __name__ == "__main__":
    unittest.main()
