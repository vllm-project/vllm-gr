# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Errors raised by the vllm-gr backend provider infrastructure."""


class BackendRegistryError(RuntimeError):
    pass


class InvalidBackendProviderError(BackendRegistryError):
    pass


class IncompatibleBackendAPIError(BackendRegistryError):
    pass


class DuplicateBackendRegistrationError(BackendRegistryError):
    pass


class UnsupportedBackendError(BackendRegistryError):
    pass


class BackendNotActiveError(BackendRegistryError):
    pass


class BackendAlreadyActiveError(BackendRegistryError):
    pass
