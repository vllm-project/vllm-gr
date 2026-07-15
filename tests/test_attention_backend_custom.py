# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for isolating vllm-gr Beam from vLLM's CUSTOM slot."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_gr.v1.attention.backend_selector import (
    VLLM_GR_ATTENTION_BACKEND_KEY,
    VLLM_GR_BEAM_ATTENTION_BACKEND,
    is_beam_attention_enabled,
)

PATCH_PATH = Path(__file__).parents[1] / "vllm_gr" / "patch.py"


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _load_patch_module(monkeypatch: pytest.MonkeyPatch, *, vllm_config, beam_backend):
    """Load patch.py against a minimal vLLM runtime without GPU/NPU imports."""
    vllm = _package("vllm")
    vllm_v1 = _package("vllm.v1")
    vllm_v1_attention = _package("vllm.v1.attention")
    vllm_v1_backends = _package("vllm.v1.attention.backends")
    vllm_config_module = types.ModuleType("vllm.config")
    vllm_config_module.get_current_vllm_config_or_none = lambda: vllm_config

    selector_module = types.ModuleType("vllm.v1.attention.selector")
    calls = []

    def original_selector(backend, attn_selector_config):
        calls.append((backend, attn_selector_config))
        return "third-party-custom-backend"

    selector_module._cached_get_attn_backend = original_selector

    platform_module = types.ModuleType("vllm.platforms")
    platform_module.current_platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=9, minor=0)
    )
    backend_utils_module = types.ModuleType("vllm.v1.attention.backends.utils")
    layouts = []
    backend_utils_module.set_kv_cache_layout = layouts.append

    beam_module = types.ModuleType("vllm_gr.v1.attention.backends.beam_attn")
    beam_module.BeamAttentionBackend = beam_backend
    ops_module = types.ModuleType("vllm_gr.ops.npu.split_qkv_rmsnorm_rope")
    ops_module.apply_split_qkv_rmsnorm_rope_patch = lambda force=False: 0

    modules = {
        "vllm": vllm,
        "vllm.v1": vllm_v1,
        "vllm.v1.attention": vllm_v1_attention,
        "vllm.v1.attention.backends": vllm_v1_backends,
        "vllm.config": vllm_config_module,
        "vllm.v1.attention.selector": selector_module,
        "vllm.platforms": platform_module,
        "vllm.v1.attention.backends.utils": backend_utils_module,
        "vllm_gr.v1.attention.backends.beam_attn": beam_module,
        "vllm_gr.ops.npu.split_qkv_rmsnorm_rope": ops_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("_vllm_gr_patch_test", PATCH_PATH)
    assert spec is not None and spec.loader is not None
    patch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch_module)
    assert patch_module._apply_attention_selector_patch() == 1
    return selector_module, calls, layouts


def _load_arg_utils_module(monkeypatch: pytest.MonkeyPatch):
    vllm = _package("vllm")
    vllm_config = types.ModuleType("vllm.config")
    vllm_config.ModelConfig = type("ModelConfig", (), {})
    vllm_engine = _package("vllm.engine")
    vllm_arg_utils = types.ModuleType("vllm.engine.arg_utils")

    class EngineArgs:
        @staticmethod
        def add_cli_args(parser):
            return parser

        def create_model_config(self):
            return SimpleNamespace()

        def create_engine_config(self):
            return SimpleNamespace(additional_config=self.additional_config)

        @classmethod
        def from_cli_args(cls, _args):
            return SimpleNamespace(additional_config={})

    vllm_arg_utils.EngineArgs = EngineArgs
    vllm_arg_utils.get_kwargs = lambda _cls: {"catalog_path": {"default": None}}
    vllm_argparse = types.ModuleType("vllm.utils.argparse_utils")
    vllm_argparse.FlexibleArgumentParser = object
    for name, module in {
        "vllm": vllm,
        "vllm.config": vllm_config,
        "vllm.engine": vllm_engine,
        "vllm.engine.arg_utils": vllm_arg_utils,
        "vllm.utils.argparse_utils": vllm_argparse,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).parents[1] / "vllm_gr" / "engine" / "arg_utils_gr.py"
    spec = importlib.util.spec_from_file_location("_vllm_gr_arg_utils_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _BeamBackend:
    @classmethod
    def validate_configuration(cls, **_kwargs):
        return []

    @classmethod
    def get_required_kv_cache_layout(cls):
        return "NHD"

    @staticmethod
    def get_name():
        return "BEAM"


def _selector_config():
    return SimpleNamespace(
        _asdict=lambda: {
            "head_size": 128,
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "block_size": 16,
            "use_mla": False,
            "has_sink": False,
            "use_sparse": False,
            "use_mm_prefix": False,
            "attn_type": "decoder",
        }
    )


def test_marker_is_explicit_and_serializable() -> None:
    assert not is_beam_attention_enabled(None)
    assert not is_beam_attention_enabled(SimpleNamespace(additional_config={}))
    assert not is_beam_attention_enabled(
        SimpleNamespace(additional_config={VLLM_GR_ATTENTION_BACKEND_KEY: "beam"})
    )


def test_beam_cli_marker_does_not_change_generic_custom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_arg_utils_module(monkeypatch)
    generic_custom_args = SimpleNamespace(catalog_path=None, vllm_gr_beam_attn=False)
    generic_custom_engine_args = module.from_cli_args(None, generic_custom_args)
    assert module.create_engine_config(generic_custom_engine_args).additional_config == {}

    beam_args = SimpleNamespace(catalog_path=None, vllm_gr_beam_attn=True)
    beam_engine_args = module.from_cli_args(None, beam_args)
    beam_config = module.create_engine_config(beam_engine_args)
    assert beam_config.additional_config == {
        VLLM_GR_ATTENTION_BACKEND_KEY: VLLM_GR_BEAM_ATTENTION_BACKEND
    }
    assert is_beam_attention_enabled(
        SimpleNamespace(
            additional_config={VLLM_GR_ATTENTION_BACKEND_KEY: VLLM_GR_BEAM_ATTENTION_BACKEND}
        )
    )


def test_custom_without_marker_delegates_to_original_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This represents a third-party backend already registered under CUSTOM.
    config = SimpleNamespace(
        additional_config={}, attention_config=SimpleNamespace(backend="CUSTOM")
    )
    selector, calls, layouts = _load_patch_module(
        monkeypatch, vllm_config=config, beam_backend=_BeamBackend
    )

    assert (
        selector._cached_get_attn_backend("CUSTOM", _selector_config())
        == "third-party-custom-backend"
    )
    assert len(calls) == 1
    assert layouts == []


def test_beam_marker_selects_beam_after_npu_backend_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``backend=None`` models the serialised NPU child config after Ascend reset.
    config = SimpleNamespace(
        additional_config={VLLM_GR_ATTENTION_BACKEND_KEY: VLLM_GR_BEAM_ATTENTION_BACKEND},
        attention_config=SimpleNamespace(backend=None),
    )
    selector, calls, layouts = _load_patch_module(
        monkeypatch, vllm_config=config, beam_backend=_BeamBackend
    )

    assert selector._cached_get_attn_backend(None, _selector_config()) is _BeamBackend
    assert calls == []
    assert layouts == ["NHD"]


def test_explicit_beam_failure_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class InvalidBeamBackend(_BeamBackend):
        @classmethod
        def validate_configuration(cls, **_kwargs):
            return ["dtype not supported"]

    config = SimpleNamespace(
        additional_config={VLLM_GR_ATTENTION_BACKEND_KEY: VLLM_GR_BEAM_ATTENTION_BACKEND}
    )
    selector, calls, _ = _load_patch_module(
        monkeypatch, vllm_config=config, beam_backend=InvalidBeamBackend
    )

    with pytest.raises(RuntimeError, match="Beam attention was explicitly requested"):
        selector._cached_get_attn_backend("CUSTOM", _selector_config())
    assert calls == []


def test_beam_layout_failure_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        additional_config={VLLM_GR_ATTENTION_BACKEND_KEY: VLLM_GR_BEAM_ATTENTION_BACKEND}
    )
    selector, calls, _ = _load_patch_module(
        monkeypatch, vllm_config=config, beam_backend=_BeamBackend
    )

    def fail_layout(_layout):
        raise ValueError("layout is unavailable")

    sys.modules["vllm.v1.attention.backends.utils"].set_kv_cache_layout = fail_layout
    with pytest.raises(RuntimeError, match="Beam attention was explicitly requested"):
        selector._cached_get_attn_backend(None, _selector_config())
    assert calls == []


def test_beam_backend_does_not_register_custom_globally() -> None:
    source = (
        Path(__file__).parents[1] / "vllm_gr" / "v1" / "attention" / "backends" / "beam_attn.py"
    ).read_text()

    assert "register_backend" not in source
    assert "AttentionBackendEnum" not in source


try:
    from vllm_gr.patch import _is_npu_platform

    _IS_NPU = _is_npu_platform()
except Exception:
    _IS_NPU = False


@unittest.skipIf(_IS_NPU, "NPU platform: NPUWorker is expected to be present")
class TestNpuWorkerPatchSkippedOnNonNpu(unittest.TestCase):
    """Preserve upstream coverage for the NPU worker initialization patch."""

    def test_patch_npu_worker_returns_zero_without_npu(self):
        from vllm_gr.patch import _patch_npu_worker_init_device

        assert _patch_npu_worker_init_device() == 0

    def test_npu_worker_module_not_imported_on_non_npu(self):
        assert "vllm_ascend.worker.worker" not in sys.modules

    def test_re_apply_patches_does_not_import_npu_worker(self):
        from vllm_gr.patch import re_apply_patches

        re_apply_patches()
        assert "vllm_ascend.worker.worker" not in sys.modules
