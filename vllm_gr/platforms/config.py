# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Provider-owned configuration and engine admission on the host."""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any

from .errors import BackendNotActiveError, UnsupportedBackendError
from .interface import ConfigProvider
from .registry import get_active_backend

if TYPE_CHECKING:
    from vllm_gr.config import GRConfig


class BackendConfigService:
    """Optional v2 policy. An engine plan supplies the hardware implementation."""

    default_worker_decision = False
    supports_constraint_vocab_slicing = False

    def apply_native_defaults(self, config: Any) -> None:
        pass

    def validate_engine(self, config: Any, gr: GRConfig) -> None:
        pass

    def finalize_native_config(self, config: Any, gr: GRConfig) -> None:
        pass


def validate_config_factory(provider: Any) -> None:
    factory = getattr(provider, "create_config_service", None)
    if factory is None:
        return
    if not callable(factory):
        raise ValueError("Backend provider create_config_service must be callable")
    try:
        parameters = signature(factory)
    except (ValueError, TypeError):
        return  # Some native extension callables do not publish a signature.
    try:
        parameters.bind()
    except TypeError as exc:
        raise ValueError(f"Invalid backend factory create_config_service signature: {exc}") from exc


def provider_config_service(provider: Any) -> BackendConfigService:
    if isinstance(provider, ConfigProvider) and callable(provider.create_config_service):
        return provider.create_config_service()
    return BackendConfigService()


def engine_config_service() -> BackendConfigService:
    """Resolve defaults before EngineArgs constructs the plan."""
    from .loader import activate_current_backend, requested_backend_id

    try:
        provider = get_active_backend()
    except BackendNotActiveError:
        try:
            provider = activate_current_backend()
        except UnsupportedBackendError:
            # Ordinary native EngineArgs can also run with GR disabled. Enabled
            # engines still require a matching provider during plan admission.
            if requested_backend_id() is not None:
                raise
            return BackendConfigService()
    return provider_config_service(provider)
