# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""NPU backend provider, implemented on the Ascend platform."""

from .provider import NPU_BACKEND, NPUBackendProvider

__all__ = ["NPU_BACKEND", "NPUBackendProvider"]
