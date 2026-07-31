# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable public API for vllm-gr backend providers."""

from .errors import (
    BackendAlreadyActiveError,
    BackendNotActiveError,
    BackendRegistryError,
    DuplicateBackendRegistrationError,
    IncompatibleBackendAPIError,
    InvalidBackendProviderError,
    UnsupportedBackendError,
)
from .loader import activate_current_backend
from .protocol import BACKEND_API_VERSION, GRBackendProvider
from .registry import (
    activate_backend,
    get_active_backend,
    register_backend,
    resolve_backend,
)

__all__ = [
    "BackendAlreadyActiveError",
    "BackendNotActiveError",
    "BackendRegistryError",
    "BACKEND_API_VERSION",
    "DuplicateBackendRegistrationError",
    "GRBackendProvider",
    "IncompatibleBackendAPIError",
    "InvalidBackendProviderError",
    "UnsupportedBackendError",
    "activate_backend",
    "activate_current_backend",
    "get_active_backend",
    "register_backend",
    "resolve_backend",
]
