# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for plugin.register() → activate() → Layer 1/2 paths.

These tests exercise the real call-chain that vLLM uses at startup, with
heavy dependencies (patch, config, model registry) mocked at ``sys.modules``
level so that the import chain is never triggered.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from vllm_gr.runtime.activation import _reset_activation_state


def _install_mock_module(dotted_name: str) -> types.ModuleType:
    """Insert a mock module into ``sys.modules`` and return it."""
    mod = types.ModuleType(dotted_name)
    mod.__name__ = dotted_name
    mod.__package__ = dotted_name.rsplit(".", 1)[0] if "." in dotted_name else dotted_name
    sys.modules[dotted_name] = mod
    return mod


@pytest.fixture(autouse=True)
def _isolate():
    """Reset activation state and clean up mock modules around each test."""
    _reset_activation_state()
    saved_env = os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)
    yield
    _reset_activation_state()
    if saved_env is not None:
        os.environ["_VLLM_GR_RUNTIME_SELECTION"] = saved_env
    else:
        os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)


# ── Test: register() without ENV is metadata-only ────────────────────


class TestRegisterMetadataOnly:
    def test_register_without_env_does_not_activate(self) -> None:
        """register() with no ENV must not call activate()."""
        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_re_apply = MagicMock(return_value=0)
        mock_patch_mod.re_apply_patches = mock_re_apply  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_mod.ensure_hstu_config_registered = MagicMock(return_value=True)  # type: ignore[attr-defined]

        with (
            patch("vllm_gr.plugin.register_gr_model_architectures", return_value=True) as mock_models,
            patch("vllm_gr.plugin._register_adapter_families") as mock_adapters,
        ):
            from vllm_gr.plugin import register
            register()

        mock_re_apply.assert_not_called()
        mock_models.assert_not_called()
        mock_adapters.assert_not_called()

        sys.modules.pop("vllm_gr.patch", None)
        sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)

    def test_register_without_env_does_not_patch(self) -> None:
        """After register() with no ENV, is_patched() must be False."""
        from vllm_gr.plugin import register
        register()

        from vllm_gr.patch import is_patched
        assert is_patched() is False


# ── Test: register() with ENV activates via the propagated selection ──


class TestRegisterEnvActivation:
    def test_register_with_env_native_does_not_patch(self) -> None:
        """register() with ENV containing native selection → no patches."""
        from vllm_gr._runtime_selection import RuntimeSelection

        native = RuntimeSelection(family="native")
        os.environ["_VLLM_GR_RUNTIME_SELECTION"] = native.serialize()

        with patch("vllm_gr.adapters.registry.register_adapter_layer1_only"):
            from vllm_gr.plugin import register
            register()

        from vllm_gr.runtime.activation import _LAST_STATE
        assert _LAST_STATE.get("patch_count") == 0

    def test_register_with_env_gr_recommend_patches(self) -> None:
        """register() with ENV containing gr_recommend → patches applied."""
        from vllm_gr._runtime_selection import RuntimeSelection

        selection = RuntimeSelection(family="gr_recommend", model_arch="hstu")
        os.environ["_VLLM_GR_RUNTIME_SELECTION"] = selection.serialize()

        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_re_apply = MagicMock(return_value=3)
        mock_patch_mod.re_apply_patches = mock_re_apply  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_mod.ensure_hstu_config_registered = MagicMock(return_value=True)  # type: ignore[attr-defined]

        with (
            patch("vllm_gr.plugin.register_gr_model_architectures", return_value=True),
            patch("vllm_gr.plugin._register_adapter_families"),
        ):
            from vllm_gr.plugin import register
            register()

        mock_re_apply.assert_called_once()

        sys.modules.pop("vllm_gr.patch", None)
        sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)


# ── Test: Layer 2 is not blocked by prior Layer 1 ────────────────────


class TestLayer2NotBlocked:
    def test_adapter_layer2_imported_after_activation(self) -> None:
        """When needs_adapter_layer2=True, adapter.py must be imported."""
        from vllm_gr.adapters.registry import _reset as registry_reset

        registry_reset()
        for key in list(sys.modules):
            if key.startswith("vllm_gr.adapters.hstu"):
                sys.modules.pop(key, None)

        adapter_calls: list[str] = []
        original_import = __import__("importlib").import_module

        def _tracking_import_module(name: str):
            if name.startswith("vllm_gr.adapters."):
                adapter_calls.append(name)
            return original_import(name)

        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_patch_mod.re_apply_patches = MagicMock(return_value=1)  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_mod.ensure_hstu_config_registered = MagicMock(return_value=True)  # type: ignore[attr-defined]

        with (
            patch("vllm_gr.plugin.register_gr_model_architectures", return_value=True),
            patch("vllm_gr.plugin.import_module", side_effect=_tracking_import_module),
            patch("vllm_gr.adapters.registry.importlib.import_module", side_effect=_tracking_import_module),
        ):
            from vllm_gr._runtime_selection import RuntimeSelection
            from vllm_gr._static_descriptors import FAMILY_GR_RECOMMEND, MODEL_ARCH_HSTU
            from vllm_gr.runtime.patch_plan import build_patch_plan
            from vllm_gr.runtime.activation import activate

            selection = RuntimeSelection(
                family=FAMILY_GR_RECOMMEND,
                model_arch=MODEL_ARCH_HSTU,
            )
            plan = build_patch_plan(selection)
            activate(plan)

        assert "vllm_gr.adapters.hstu" in adapter_calls, \
            f"Layer 1 import missing from {adapter_calls}"
        assert "vllm_gr.adapters.hstu.adapter" in adapter_calls, \
            f"Layer 2 import missing from {adapter_calls}"

        sys.modules.pop("vllm_gr.patch", None)
        sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)
        from vllm_gr.adapters.hstu import register_family  # noqa: F401
        registry_reset()
        import vllm_gr.adapters.hstu  # re-trigger register_family

    def test_layer2_imported_even_after_layer1_preregistration(self) -> None:
        """Layer 1 pre-registration (via arg_utils) must not block Layer 2."""
        from vllm_gr.adapters.registry import (
            _reset as registry_reset,
            get_registered_families,
        )

        registry_reset()
        for key in list(sys.modules):
            if key.startswith("vllm_gr.adapters.hstu"):
                sys.modules.pop(key, None)

        adapter_calls: list[str] = []
        original_import = __import__("importlib").import_module

        def _tracking_import_module(name: str):
            if name.startswith("vllm_gr.adapters."):
                adapter_calls.append(name)
            return original_import(name)

        # Step 1: Layer 1 pre-registration (as arg_utils does)
        import vllm_gr.adapters.hstu  # triggers register_family()
        assert "hstu" in get_registered_families()

        adapter_calls.clear()

        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_patch_mod.re_apply_patches = MagicMock(return_value=1)  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_mod.ensure_hstu_config_registered = MagicMock(return_value=True)  # type: ignore[attr-defined]

        with (
            patch("vllm_gr.plugin.register_gr_model_architectures", return_value=True),
            patch("vllm_gr.plugin.import_module", side_effect=_tracking_import_module),
            patch("vllm_gr.adapters.registry.importlib.import_module", side_effect=_tracking_import_module),
        ):
            from vllm_gr._runtime_selection import RuntimeSelection
            from vllm_gr._static_descriptors import FAMILY_GR_RECOMMEND, MODEL_ARCH_HSTU
            from vllm_gr.runtime.patch_plan import build_patch_plan
            from vllm_gr.runtime.activation import activate

            selection = RuntimeSelection(
                family=FAMILY_GR_RECOMMEND,
                model_arch=MODEL_ARCH_HSTU,
            )
            plan = build_patch_plan(selection)
            activate(plan)

        assert "vllm_gr.adapters.hstu" not in adapter_calls, \
            f"Layer 1 should have been skipped (already registered): {adapter_calls}"
        assert "vllm_gr.adapters.hstu.adapter" in adapter_calls, \
            f"Layer 2 import missing from {adapter_calls}"

        sys.modules.pop("vllm_gr.patch", None)
        sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)
        registry_reset()
        for key in list(sys.modules):
            if key.startswith("vllm_gr.adapters.hstu"):
                sys.modules.pop(key, None)
        import vllm_gr.adapters.hstu  # re-register for other tests


# ── Test: create_engine_config() explicit activation path ─────────────


def _install_vllm_mocks():
    """Install lightweight vLLM mocks into sys.modules.

    This lets ``vllm_gr.engine`` and ``vllm_gr.engine.arg_utils`` be
    imported without triggering the real vLLM / TensorFlow chain.
    Returns a set of dotted names for cleanup.
    """
    # -- vllm.platforms --
    platforms_mod = _install_mock_module("vllm.platforms")
    platforms_mod.current_platform = types.SimpleNamespace(device_type="cpu")

    # -- vllm.v1.engine (needed by vllm_gr/engine/__init__.py) --
    v1_engine_mod = _install_mock_module("vllm.v1.engine")
    v1_engine_mod.EngineCoreRequest = type("EngineCoreRequest", (), {})

    # -- vllm.engine.arg_utils (EngineArgs / AsyncEngineArgs) --
    _install_mock_module("vllm.engine")
    vllm_arg_utils = _install_mock_module("vllm.engine.arg_utils")

    # EngineArgs must be a proper dataclass so that GREngineArgs' @dataclass
    # decorator can inherit its fields (model, hf_overrides, etc).
    from dataclasses import dataclass as _dc, field as _dc_field
    from typing import Optional as _Opt

    @_dc
    class _FakeEngineArgs:
        model: str = ""
        hf_overrides: _Opt[object] = _dc_field(default=None)
        tensor_parallel_size: int = 1
        gpu_memory_utilization: float = 0.9
        trust_remote_code: bool = False
        max_model_len: int | None = None
        skip_tokenizer_init: bool = False
        runner: str = ""
        enforce_eager: bool = False
        disable_hybrid_kv_cache_manager: bool = False
        worker_cls: str = ""
        dtype: str = "auto"
        load_format: str | None = None
        # Minimal fields that GRCommonEngineArgsMixin.__post_init__ accesses.
        def __post_init__(self):
            pass
        def create_engine_config(self, usage_context=None, headless=False):
            return None

    @_dc
    class _FakeAsyncEngineArgs:
        model: str = ""
        hf_overrides: _Opt[object] = _dc_field(default=None)
        def __post_init__(self):
            pass
        @classmethod
        def from_cli_args(cls, args):
            return cls()

    vllm_arg_utils.EngineArgs = _FakeEngineArgs
    vllm_arg_utils.AsyncEngineArgs = _FakeAsyncEngineArgs

    # -- vLLM modules used by vllm_gr.entrypoints.gr --
    _install_mock_module("vllm.entrypoints")
    vllm_llm = _install_mock_module("vllm.entrypoints.llm")
    vllm_llm.LLM = type("LLM", (), {})

    _install_mock_module("vllm.usage")
    usage_lib = _install_mock_module("vllm.usage.usage_lib")
    usage_lib.UsageContext = types.SimpleNamespace(LLM_CLASS="llm")

    _install_mock_module("vllm.utils")
    counter_mod = _install_mock_module("vllm.utils.counter")
    counter_mod.Counter = type("Counter", (), {"__init__": lambda self: None})

    pooling_mod = _install_mock_module("vllm.pooling_params")
    pooling_mod.PoolingParams = type("PoolingParams", (), {})
    outputs_mod = _install_mock_module("vllm.outputs")
    outputs_mod.PoolingOutput = type("PoolingOutput", (), {})
    outputs_mod.PoolingRequestOutput = type("PoolingRequestOutput", (), {})

    # Avoid importing the real GR engine/npu adapter in the entrypoint test.
    gr_engine_mod = _install_mock_module("vllm_gr.engine.gr_engine")
    gr_engine_mod.GRLLMEngine = types.SimpleNamespace(from_engine_args=MagicMock())
    npu_adapter_mod = _install_mock_module("vllm_gr.utils.npu_adapter")
    npu_adapter_mod.release_accelerator_memory = MagicMock()

    # -- vllm_gr.engine (needs the mock EngineCoreRequest above) --
    # Clear any cached state so a fresh import picks up the mocks.
    for key in list(sys.modules):
        if key.startswith("vllm_gr.engine") and key != "vllm_gr.engine.gr_engine":
            sys.modules.pop(key, None)

    return {
        "vllm.platforms", "vllm.v1.engine", "vllm.engine",
        "vllm.engine.arg_utils", "vllm.entrypoints",
        "vllm.entrypoints.llm", "vllm.usage", "vllm.usage.usage_lib",
        "vllm.utils", "vllm.utils.counter", "vllm.pooling_params",
        "vllm.outputs",
        "vllm_gr.engine.gr_engine", "vllm_gr.utils.npu_adapter",
    }


def _cleanup_vllm_mocks(keys):
    for key in keys:
        sys.modules.pop(key, None)
    # Also clear any vllm_gr.engine submodules that were imported.
    for key in list(sys.modules):
        if key.startswith("vllm_gr.engine"):
            sys.modules.pop(key, None)
        if key.startswith("vllm_gr.entrypoints.gr"):
            sys.modules.pop(key, None)


class TestCreateEngineConfigActivation:
    def test_create_engine_config_activates_gr_recommend(self) -> None:
        """GREngineArgs.create_engine_config() calls activate(gr_recommend).

        Installs lightweight vLLM mocks so that the real
        ``GRCommonEngineArgsMixin.create_engine_config()`` can be exercised
        end-to-end without importing TensorFlow / broken numpy.
        """
        mock_keys = _install_vllm_mocks()

        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_re_apply = MagicMock(return_value=11)
        mock_patch_mod.re_apply_patches = mock_re_apply  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_mod.ensure_hstu_config_registered = MagicMock(return_value=True)  # type: ignore[attr-defined]

        try:
            with (
                patch("vllm_gr.plugin.register_gr_model_architectures", return_value=True) as mock_models,
                patch("vllm_gr.plugin._register_adapter_families") as mock_adapters,
            ):
                from vllm_gr.runtime.activation import _ACTIVATED
                assert not _ACTIVATED, "Should not be activated before create_engine_config"

                from vllm_gr.engine.arg_utils import GREngineArgs

                args = GREngineArgs(model="dummy", model_family="hstu")
                args.create_engine_config()

            # Assert all four activation steps were called.
            mock_re_apply.assert_called_once()
            mock_config_mod.ensure_hstu_config_registered.assert_called_once()
            mock_models.assert_called_once()
            mock_adapters.assert_called_once()

            from vllm_gr.runtime.activation import _ACTIVATED
            assert _ACTIVATED, "Should be activated after create_engine_config"
        finally:
            _cleanup_vllm_mocks(mock_keys)
            sys.modules.pop("vllm_gr.patch", None)
            sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)

    def test_grllm_activates_before_engine_args_init(self) -> None:
        """GRLLM must register HSTU config before vLLM can read model config."""
        events: list[str] = []
        mock_keys = _install_vllm_mocks()

        vllm_arg_utils = sys.modules["vllm.engine.arg_utils"]
        fake_engine_args_cls = vllm_arg_utils.EngineArgs

        def _fake_engine_post_init(self):
            events.append("engine_args_post_init")

        fake_engine_args_cls.__post_init__ = _fake_engine_post_init

        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_patch_mod.re_apply_patches = MagicMock(
            side_effect=lambda: events.append("patch") or 1
        )  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_mod.ensure_hstu_config_registered = MagicMock(
            side_effect=lambda: events.append("hstu_config") or True
        )  # type: ignore[attr-defined]

        try:
            with (
                patch("vllm_gr.plugin.register_gr_model_architectures", return_value=True),
                patch("vllm_gr.plugin._register_adapter_families"),
            ):
                from vllm_gr.entrypoints.gr import GRLLM

                GRLLM(model="dummy", model_family="hstu", allow_missing_weights=True)

            assert events.index("hstu_config") < events.index("engine_args_post_init")
        finally:
            _cleanup_vllm_mocks(mock_keys)
            sys.modules.pop("vllm_gr.patch", None)
            sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)
