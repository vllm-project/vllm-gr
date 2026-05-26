# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for activate() behavior."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from vllm_gr._static_descriptors import (
    CAPABILITY_GR_BEAM,
    FAMILY_GR_RECOMMEND,
    FAMILY_LLM,
    FAMILY_NATIVE,
    MODEL_ARCH_HSTU,
)
from vllm_gr._runtime_errors import GRInconsistencyError
from vllm_gr.runtime.activation import activate, _reset_activation_state
from vllm_gr.runtime.patch_plan import PatchPlan


# ── Helpers ──────────────────────────────────────────────────────

def _native_plan() -> PatchPlan:
    return PatchPlan(
        family=FAMILY_NATIVE,
        capabilities=frozenset(),
        model_arch=None,
        needs_core_patches=False,
        needs_hstu_config=False,
        needs_model_registry=False,
        needs_adapter_layer2=False,
    )


def _gr_recommend_plan() -> PatchPlan:
    return PatchPlan(
        family=FAMILY_GR_RECOMMEND,
        capabilities=frozenset(),
        model_arch=MODEL_ARCH_HSTU,
        needs_core_patches=True,
        needs_hstu_config=True,
        needs_model_registry=True,
        needs_adapter_layer2=True,
    )


def _llm_beam_plan() -> PatchPlan:
    return PatchPlan(
        family=FAMILY_LLM,
        capabilities=frozenset({CAPABILITY_GR_BEAM}),
        model_arch=None,
        needs_core_patches=False,
        needs_hstu_config=False,
        needs_model_registry=False,
        needs_adapter_layer2=False,
    )


def _install_mock_module(dotted_name: str) -> types.ModuleType:
    """Insert a mock module into sys.modules and return it."""
    mod = types.ModuleType(dotted_name)
    mod.__name__ = dotted_name
    mod.__package__ = dotted_name.rsplit(".", 1)[0] if "." in dotted_name else dotted_name
    sys.modules[dotted_name] = mod
    return mod


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset activation state before each test."""
    _reset_activation_state()
    yield
    _reset_activation_state()


# ── Tests ────────────────────────────────────────────────────────


class TestActivateNative:
    def test_native_no_patches(self) -> None:
        """activate(native_plan) should not call re_apply_patches."""
        with patch("vllm_gr.adapters.registry.register_adapter_layer1_only"):
            state = activate(_native_plan())

        assert state["patch_count"] == 0
        assert state["config_registered"] is True
        assert state["models_registered"] is True

    def test_native_is_patched_remains_false(self) -> None:
        """After activate(native_plan), is_patched() should still be False."""
        previous_patch_module = sys.modules.pop("vllm_gr.patch", None)
        with patch("vllm_gr.adapters.registry.register_adapter_layer1_only"):
            activate(_native_plan())

        try:
            from vllm_gr.patch import is_patched
            assert is_patched() is False
        finally:
            sys.modules.pop("vllm_gr.patch", None)
            if previous_patch_module is not None:
                sys.modules["vllm_gr.patch"] = previous_patch_module


class TestActivateGrRecommend:
    def test_gr_recommend_triggers_all(self) -> None:
        """activate(gr_recommend_plan) triggers patch + config + model + adapter."""
        events: list[str] = []
        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_patch_fn = MagicMock(side_effect=lambda: events.append("patch") or 42)
        mock_patch_mod.re_apply_patches = mock_patch_fn  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_fn = MagicMock(side_effect=lambda: events.append("config") or True)
        mock_config_mod.ensure_hstu_config_registered = mock_config_fn  # type: ignore[attr-defined]

        with (
            patch("vllm_gr.plugin.register_gr_model_architectures", return_value=True) as mock_models,
            patch("vllm_gr.plugin._register_adapter_families") as mock_adapters,
        ):
            state = activate(_gr_recommend_plan())

        mock_patch_fn.assert_called_once()
        mock_config_fn.assert_called_once()
        mock_models.assert_called_once()
        mock_adapters.assert_called_once()
        assert state["patch_count"] == 42
        assert state["config_registered"] is True
        assert state["models_registered"] is True
        assert events.index("config") < events.index("patch")

        sys.modules.pop("vllm_gr.patch", None)
        sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)


class TestActivateLlmBeam:
    def test_llm_beam_no_patches(self) -> None:
        """activate(llm_beam_plan) does not trigger patches or registrations."""
        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_re_apply = MagicMock(return_value=0)
        mock_patch_mod.re_apply_patches = mock_re_apply  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_mod.ensure_hstu_config_registered = MagicMock(return_value=True)  # type: ignore[attr-defined]

        with (
            patch("vllm_gr.plugin.register_gr_model_architectures", return_value=True) as mock_models,
            patch("vllm_gr.plugin._register_adapter_families") as mock_adapters,
            patch("vllm_gr.adapters.registry.register_adapter_layer1_only"),
        ):
            state = activate(_llm_beam_plan())

        mock_re_apply.assert_not_called()
        mock_models.assert_not_called()
        mock_adapters.assert_not_called()
        assert state["patch_count"] == 0

        sys.modules.pop("vllm_gr.patch", None)
        sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)


class TestActivateIdempotent:
    def test_idempotent_native(self) -> None:
        """Calling activate(native_plan) twice does not error."""
        with patch("vllm_gr.adapters.registry.register_adapter_layer1_only"):
            state1 = activate(_native_plan())
            state2 = activate(_native_plan())

        assert state1 == state2

    def test_idempotent_gr_recommend(self) -> None:
        """activate() is fully idempotent — second call returns cached state
        and does NOT re-invoke any patch/config/model/adapter function."""
        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_re_apply = MagicMock(return_value=1)
        mock_patch_mod.re_apply_patches = mock_re_apply  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_mod.ensure_hstu_config_registered = MagicMock(return_value=True)  # type: ignore[attr-defined]

        with (
            patch("vllm_gr.plugin.register_gr_model_architectures", return_value=True) as mock_models,
            patch("vllm_gr.plugin._register_adapter_families") as mock_adapters,
        ):
            state1 = activate(_gr_recommend_plan())
            state2 = activate(_gr_recommend_plan())

        assert state1 == state2
        # Each function called exactly once — not re-invoked on second call.
        mock_re_apply.assert_called_once()
        mock_models.assert_called_once()
        mock_adapters.assert_called_once()

        sys.modules.pop("vllm_gr.patch", None)
        sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)


class TestActivateConfigFailure:
    def test_config_failure_raises(self) -> None:
        """If config registration fails, activate raises RuntimeError."""
        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_patch_mod.re_apply_patches = MagicMock(return_value=1)  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_mod.ensure_hstu_config_registered = MagicMock(return_value=False)  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="HSTU config registration"):
            activate(_gr_recommend_plan())

        sys.modules.pop("vllm_gr.patch", None)
        sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)

    def test_model_failure_raises(self) -> None:
        """If model registration fails, activate raises RuntimeError."""
        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_patch_mod.re_apply_patches = MagicMock(return_value=1)  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_mod.ensure_hstu_config_registered = MagicMock(return_value=True)  # type: ignore[attr-defined]

        with (
            patch("vllm_gr.plugin.register_gr_model_architectures", return_value=False),
            pytest.raises(RuntimeError, match="model architecture registration"),
        ):
            activate(_gr_recommend_plan())

        sys.modules.pop("vllm_gr.patch", None)
        sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)


class TestActivatePlanConflict:
    def test_conflicting_plan_raises(self) -> None:
        """activate(native) then activate(gr_recommend) must raise."""
        with patch("vllm_gr.adapters.registry.register_adapter_layer1_only"):
            activate(_native_plan())

        with pytest.raises(GRInconsistencyError, match="conflicting plan"):
            activate(_gr_recommend_plan())

    def test_same_plan_is_idempotent(self) -> None:
        """activate(native) then activate(native) must return cached state."""
        with patch("vllm_gr.adapters.registry.register_adapter_layer1_only"):
            state1 = activate(_native_plan())
            state2 = activate(_native_plan())

        assert state1 is state2

    def test_reverse_conflict_also_raises(self) -> None:
        """activate(gr_recommend) then activate(native) must raise."""
        mock_patch_mod = _install_mock_module("vllm_gr.patch")
        mock_patch_mod.re_apply_patches = MagicMock(return_value=1)  # type: ignore[attr-defined]

        mock_config_mod = _install_mock_module("vllm_gr.config.hstu_vllm_registration")
        mock_config_mod.ensure_hstu_config_registered = MagicMock(return_value=True)  # type: ignore[attr-defined]

        with (
            patch("vllm_gr.plugin.register_gr_model_architectures", return_value=True),
            patch("vllm_gr.plugin._register_adapter_families"),
        ):
            activate(_gr_recommend_plan())

        with pytest.raises(GRInconsistencyError, match="conflicting plan"):
            activate(_native_plan())

        sys.modules.pop("vllm_gr.patch", None)
        sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)
