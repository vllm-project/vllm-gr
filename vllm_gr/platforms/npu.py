# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility imports for the original Ascend provider module."""

from .ascend import NPU_BACKEND, NPUBackendProvider

__all__ = ["NPU_BACKEND", "NPUBackendProvider"]
