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
    """Verify selector delegates correctly for backend=CUSTOM with/without marker."""

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

    def test_backend_custom_with_marker_returns_beam(self):
        """backend=CUSTOM with vllm-gr marker returns BeamAttentionBackend."""
        import vllm.v1.attention.selector as selector_mod
        from vllm.config import AttentionConfig, VllmConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.patch import _apply_ascend_attention_backend_patch

        _apply_ascend_attention_backend_patch(force=True)

        config = VllmConfig(attention_config=AttentionConfig())
        config.attention_config.backend = AttentionBackendEnum.CUSTOM
        config.additional_config["vllm_gr_attention_backend"] = "CUSTOM"

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(config):
            import torch
            from vllm.v1.attention.selector import AttentionSelectorConfig

            selector_cfg = AttentionSelectorConfig(
                head_size=48,
                dtype=torch.float16,
                kv_cache_dtype=None,
                block_size=128,
            )
            result = selector_mod._cached_get_attn_backend(
                AttentionBackendEnum.CUSTOM, selector_cfg
            )

        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

        assert result is BeamAttentionBackend

    @mock.patch("vllm.v1.attention.selector.resolve_obj_by_qualname")
    @mock.patch("vllm.platforms.current_platform.get_attn_backend_cls")
    def test_backend_custom_without_marker_delegates_to_registered_backend(
        self, mock_get_cls, mock_resolve
    ):
        """backend=CUSTOM without marker delegates to the registered class."""
        import vllm.v1.attention.selector as selector_mod
        from vllm.config import AttentionConfig, VllmConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.patch import _apply_ascend_attention_backend_patch

        _apply_ascend_attention_backend_patch(force=True)

        class FakeCustomBackend:
            @staticmethod
            def get_required_kv_cache_layout():
                return None

            @staticmethod
            def get_name():
                return "FakeCustomBackend"

        mock_get_cls.return_value = "fake.path.FakeCustomBackend"
        mock_resolve.return_value = FakeCustomBackend

        config = VllmConfig(attention_config=AttentionConfig())
        config.attention_config.backend = AttentionBackendEnum.CUSTOM
        # additional_config empty (no marker)

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(config):
            import torch
            from vllm.v1.attention.selector import AttentionSelectorConfig

            selector_cfg = AttentionSelectorConfig(
                head_size=96,
                dtype=torch.float16,
                kv_cache_dtype=None,
                block_size=128,
            )
            result = selector_mod._cached_get_attn_backend(
                AttentionBackendEnum.CUSTOM, selector_cfg
            )

        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

        assert result is not BeamAttentionBackend
        assert result is FakeCustomBackend


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
