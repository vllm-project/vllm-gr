# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable public API for vllm-gr backend providers."""

from .errors import (
    BackendAlreadyActiveError,
    BackendNotActiveError,
    BackendRegistryError,
    DuplicateBackendRegistrationError,
    InvalidBackendProviderError,
    UnsupportedBackendError,
)
from .loader import activate_current_backend, load_backend_plugin
from .protocol import GRBackendProvider
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
    "DuplicateBackendRegistrationError",
    "GRBackendProvider",
    "InvalidBackendProviderError",
    "UnsupportedBackendError",
    "activate_backend",
    "activate_current_backend",
    "get_active_backend",
    "load_backend_plugin",
    "register_backend",
    "resolve_backend",
]
