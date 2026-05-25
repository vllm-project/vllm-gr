# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for static runtime descriptors."""

from __future__ import annotations

import importlib
import sys

from vllm_gr._static_descriptors import (
    CAPABILITIES,
    CAPABILITY_FORBIDDEN_FAMILIES,
    CAPABILITY_GR_BEAM,
    CAPABILITY_HOST_FAMILY,
    FAMILIES,
    FAMILY_GR_RECOMMEND,
    FAMILY_LLM,
    FAMILY_NATIVE,
    MODEL_ARCH_HSTU,
    MODEL_ARCHS_BY_FAMILY,
)


def test_static_descriptors_are_closed_sets() -> None:
    # All three families are registered
    assert FAMILIES == frozenset({FAMILY_NATIVE, FAMILY_LLM, FAMILY_GR_RECOMMEND})

    # model_arch mapping: hstu lives exclusively under gr_recommend
    assert MODEL_ARCHS_BY_FAMILY[FAMILY_GR_RECOMMEND] == frozenset({MODEL_ARCH_HSTU})
    assert FAMILY_NATIVE not in MODEL_ARCHS_BY_FAMILY
    assert FAMILY_LLM not in MODEL_ARCHS_BY_FAMILY

    # capability host matrix: gr_beam must be hosted by llm
    assert CAPABILITY_HOST_FAMILY[CAPABILITY_GR_BEAM] == FAMILY_LLM
    assert CAPABILITY_GR_BEAM in CAPABILITIES

    # forbidden families: native and gr_recommend cannot carry capabilities
    assert CAPABILITY_FORBIDDEN_FAMILIES == frozenset(
        {FAMILY_NATIVE, FAMILY_GR_RECOMMEND}
    )


def test_static_descriptors_module_has_no_heavy_imports() -> None:
    module_name = "vllm_gr._static_descriptors"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    assert module.FAMILIES == FAMILIES
