# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for PatchPlan construction and build_patch_plan()."""

from __future__ import annotations

import pytest

from vllm_gr._runtime_selection import RuntimeSelection
from vllm_gr._static_descriptors import (
    CAPABILITY_GR_BEAM,
    FAMILY_GR_RECOMMEND,
    FAMILY_LLM,
    FAMILY_NATIVE,
    MODEL_ARCH_HSTU,
)
from vllm_gr.runtime.patch_plan import PatchPlan, build_patch_plan


class TestPatchPlanFrozen:
    def test_frozen_dataclass(self) -> None:
        plan = PatchPlan(
            family=FAMILY_NATIVE,
            capabilities=frozenset(),
            model_arch=None,
            needs_core_patches=False,
            needs_hstu_config=False,
            needs_model_registry=False,
            needs_adapter_layer2=False,
        )
        with pytest.raises(AttributeError):
            plan.family = "changed"  # type: ignore[misc]


class TestBuildPatchPlanNative:
    def test_native_all_false(self) -> None:
        selection = RuntimeSelection(family=FAMILY_NATIVE)
        plan = build_patch_plan(selection)

        assert plan.family == FAMILY_NATIVE
        assert plan.capabilities == frozenset()
        assert plan.model_arch is None
        assert plan.needs_core_patches is False
        assert plan.needs_hstu_config is False
        assert plan.needs_model_registry is False
        assert plan.needs_adapter_layer2 is False


class TestBuildPatchPlanGrRecommend:
    def test_gr_recommend_all_true(self) -> None:
        selection = RuntimeSelection(
            family=FAMILY_GR_RECOMMEND,
            model_arch=MODEL_ARCH_HSTU,
        )
        plan = build_patch_plan(selection)

        assert plan.family == FAMILY_GR_RECOMMEND
        assert plan.model_arch == MODEL_ARCH_HSTU
        assert plan.needs_core_patches is True
        assert plan.needs_hstu_config is True
        assert plan.needs_model_registry is True
        assert plan.needs_adapter_layer2 is True


class TestBuildPatchPlanLlm:
    def test_llm_gr_beam_no_core_patches(self) -> None:
        """llm+gr_beam must not trigger HSTU core type replacement."""
        selection = RuntimeSelection(
            family=FAMILY_LLM,
            capabilities=frozenset({CAPABILITY_GR_BEAM}),
        )
        plan = build_patch_plan(selection)

        assert plan.family == FAMILY_LLM
        assert CAPABILITY_GR_BEAM in plan.capabilities
        assert plan.needs_core_patches is False
        assert plan.needs_hstu_config is False
        assert plan.needs_model_registry is False
        assert plan.needs_adapter_layer2 is False

    def test_llm_without_capability_no_patches(self) -> None:
        """llm without gr_beam — no patches, no HSTU."""
        selection = RuntimeSelection(
            family=FAMILY_LLM,
            capabilities=frozenset({"some_other_cap"}),
        )
        plan = build_patch_plan(selection)

        assert plan.family == FAMILY_LLM
        assert plan.needs_core_patches is False
        assert plan.needs_hstu_config is False


class TestBuildPatchPlanAcceptsRuntimeSelection:
    def test_from_deserialized_selection(self) -> None:
        original = RuntimeSelection(
            family=FAMILY_GR_RECOMMEND,
            model_arch=MODEL_ARCH_HSTU,
        )
        serialized = original.serialize()
        restored = RuntimeSelection.deserialize(serialized)

        plan = build_patch_plan(restored)
        assert plan.family == FAMILY_GR_RECOMMEND
        assert plan.needs_core_patches is True

    def test_native_from_default_selection(self) -> None:
        from vllm_gr._runtime_selection import normalize_runtime_selection, RuntimeIntentInput

        selection = normalize_runtime_selection(RuntimeIntentInput())
        plan = build_patch_plan(selection)
        assert plan.family == FAMILY_NATIVE
        assert plan.needs_core_patches is False
