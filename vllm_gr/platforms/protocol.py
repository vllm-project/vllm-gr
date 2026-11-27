# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility imports for providers using the original protocol module."""

from .interface import (
    CONSTRAINT_BACKEND_API_VERSION,
    LEGACY_PROVIDER_API_VERSION,
    PROVIDER_API_VERSION,
    BeamAttentionExecution,
    CanonicalConstraintProvider,
    ConfigProvider,
    GRBackendProvider,
    GRPlanningProvider,
    ModelRunnerPatchHooks,
)

__all__ = [
    "CONSTRAINT_BACKEND_API_VERSION",
    "LEGACY_PROVIDER_API_VERSION",
    "PROVIDER_API_VERSION",
    "BeamAttentionExecution",
    "CanonicalConstraintProvider",
    "ConfigProvider",
    "GRBackendProvider",
    "GRPlanningProvider",
    "ModelRunnerPatchHooks",
]
