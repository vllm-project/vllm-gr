# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Errors raised by GR platform selection and admission."""


class UnsupportedBackendError(RuntimeError):
    """The selected native platform has no GR integration."""


class AmbiguousBackendError(RuntimeError):
    """More than one platform candidate in the same priority tier matched."""


class GRDeviceRequestError(ValueError):
    """A request cannot run on an otherwise supported device configuration."""
