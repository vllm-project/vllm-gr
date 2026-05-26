# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PatchPlan derivation from RuntimeSelection.

Pure data / pure function — no heavy imports, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from vllm_gr._runtime_selection import RuntimeSelection
from vllm_gr._static_descriptors import (
    FAMILY_GR_RECOMMEND,
    FAMILY_NATIVE,
)


@dataclass(frozen=True)
class PatchPlan:
    """Declarative description of what the activation step should do.

    Built from a :class:`RuntimeSelection` by :func:`build_patch_plan`.
    Consumed by :func:`~vllm_gr.runtime.activation.activate`.
    """

    family: str
    capabilities: frozenset[str]
    model_arch: Optional[str]

    needs_core_patches: bool
    needs_hstu_config: bool
    needs_model_registry: bool
    needs_adapter_layer2: bool


def build_patch_plan(selection: RuntimeSelection) -> PatchPlan:
    """Derive a :class:`PatchPlan` from a validated *selection*.

    Mapping logic
    -------------
    ============= ================ =============== =================== ===================
    family        needs_core_patch needs_hstu_conf needs_model_registr needs_adapter_layer
    ============= ================ =============== =================== ===================
    native        False            False           False               False
    gr_recommend  True             True            True                True
    llm+gr_beam   False            False           False               False
    ============= ================ =============== =================== ===================
    """
    family = selection.family
    caps = selection.capabilities

    if family == FAMILY_NATIVE:
        return PatchPlan(
            family=family,
            capabilities=caps,
            model_arch=selection.model_arch,
            needs_core_patches=False,
            needs_hstu_config=False,
            needs_model_registry=False,
            needs_adapter_layer2=False,
        )

    if family == FAMILY_GR_RECOMMEND:
        return PatchPlan(
            family=family,
            capabilities=caps,
            model_arch=selection.model_arch,
            needs_core_patches=True,
            needs_hstu_config=True,
            needs_model_registry=True,
            needs_adapter_layer2=True,
        )

    # llm+gr_beam and any other combination — no core patches, no HSTU.
    return PatchPlan(
        family=family,
        capabilities=caps,
        model_arch=selection.model_arch,
        needs_core_patches=False,
        needs_hstu_config=False,
        needs_model_registry=False,
        needs_adapter_layer2=False,
    )
