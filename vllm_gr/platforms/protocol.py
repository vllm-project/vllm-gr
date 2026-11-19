# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility imports for providers using the original protocol module."""

from .interface import (
    LEGACY_PROVIDER_API_VERSION,
    PROVIDER_API_VERSION,
    BeamAttentionExecution,
    GRBackendProvider,
    GRPlanningProvider,
    ModelRunnerPatchHooks,
)

__all__ = [
    "LEGACY_PROVIDER_API_VERSION",
    "PROVIDER_API_VERSION",
    "BeamAttentionExecution",
    "GRBackendProvider",
    "GRPlanningProvider",
    "ModelRunnerPatchHooks",
]
