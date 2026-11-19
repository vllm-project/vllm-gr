# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Ascend backend provider."""

from .provider import NPU_BACKEND, NPUBackendProvider

__all__ = ["NPU_BACKEND", "NPUBackendProvider"]
