# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Errors raised by backend provider registration and device request admission."""


class GRDeviceRequestError(ValueError):
    """A request cannot run on an otherwise supported device configuration."""


class BackendRegistryError(RuntimeError):
    pass


class InvalidBackendProviderError(BackendRegistryError):
    pass


class InvalidBackendVersionError(BackendRegistryError):
    pass


class DuplicateBackendRegistrationError(BackendRegistryError):
    pass


class UnsupportedBackendError(BackendRegistryError):
    pass


class AmbiguousBackendError(BackendRegistryError):
    pass


class IncompatibleBackendError(BackendRegistryError):
    pass


class BackendNotActiveError(BackendRegistryError):
    pass


class BackendAlreadyActiveError(BackendRegistryError):
    pass
