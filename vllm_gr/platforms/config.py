# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Provider-owned configuration and engine admission on the host."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .errors import BackendNotActiveError, UnsupportedBackendError
from .interface import ConfigProvider
from .plan import BACKEND_PLAN_KEY
from .registry import get_active_backend

if TYPE_CHECKING:
    from vllm_gr.config import GRConfig


class BackendConfigService:
    """Optional v2 policy. An engine plan supplies the hardware implementation."""

    default_worker_decision = False
    promote_catalog_tables = False
    supports_constraint_vocab_slicing = False
    supports_sid_stage_vocab = False
    infer_matching_candidate_widths = False

    def startup_inputs(self, gr: GRConfig, admitted: Mapping[str, Any] | None) -> dict[str, Any]:
        """Return provider-owned host options used when constructing the backend plan."""
        return {}

    def apply_native_defaults(self, config: Any) -> None:
        pass

    def validate_engine(self, config: Any, gr: GRConfig) -> None:
        pass

    def finalize_native_config(self, config: Any, gr: GRConfig) -> None:
        pass

    def validate_device(self, config: Any, gr: GRConfig, request: Any) -> None:
        if gr.beam_sid_stage_vocab and not self.supports_sid_stage_vocab:
            raise ValueError(
                "beam.sid_stage_vocab requires a provider with stage vocabulary support"
            )

    def decode_batch_decline_reason(
        self, config: Any, batch_size: int, beam_width: int
    ) -> str | None:
        """Return the backend's reason to run a Decode batch as separate requests."""
        return None

    def validate_request_budget(self, config: Any, request: Any) -> None:
        pass


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


def request_config_service(config: Any) -> BackendConfigService:
    """Use the provider admitted by EngineArgs for frontend, EngineCore and Worker."""
    from .loader import activate_current_backend

    additional = getattr(config, "additional_config", None)
    if isinstance(additional, Mapping) and BACKEND_PLAN_KEY in additional:
        try:
            provider = get_active_backend()
        except BackendNotActiveError:
            provider = activate_current_backend()
        envelope = additional[BACKEND_PLAN_KEY]
        data = envelope.get("plan") if isinstance(envelope, Mapping) else None
        if not isinstance(data, Mapping) or provider.backend_id != data.get("backend_id"):
            raise ValueError("Request config backend differs from the active provider")
        return provider_config_service(provider)
    # Older callers pass native configs constructed before plan admission.
    from .legacy import legacy_config_service

    device = getattr(getattr(config, "device_config", None), "device", None)
    service: BackendConfigService = legacy_config_service(
        None if device is None else str(device).split(":")[0]
    )
    return service
