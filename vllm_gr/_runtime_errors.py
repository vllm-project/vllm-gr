# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unified runtime selection error types."""

from __future__ import annotations


class GRRuntimeError(Exception):
    """Base class for GR runtime selection / propagation errors."""


class GRCompatibilityError(GRRuntimeError):
    """Family, capability, or model_arch combination is invalid."""


class GRPreconditionError(GRRuntimeError):
    """Required runtime precondition is missing or invalid."""


class GRPropagationError(GRRuntimeError):
    """Cross-process propagation payload is missing or corrupt."""


class GRInconsistencyError(GRRuntimeError):
    """Conflicting inputs or inconsistent activation state."""
