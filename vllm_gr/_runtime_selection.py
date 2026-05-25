# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RuntimeSelection normalization, validation, and serialization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from vllm_gr._runtime_errors import (
    GRCompatibilityError,
    GRInconsistencyError,
    GRPreconditionError,
    GRPropagationError,
)
from vllm_gr._static_descriptors import (
    CAPABILITIES,
    CAPABILITY_FORBIDDEN_FAMILIES,
    CAPABILITY_GR_BEAM,
    CAPABILITY_HOST_FAMILY,
    DIAG_SOURCE_API,
    DIAG_SOURCE_AUTO,
    DIAG_SOURCE_CLI,
    DIAG_SOURCE_DEFAULT,
    DIAG_SOURCE_ENV,
    DIAG_SOURCES,
    FAMILIES,
    FAMILY_GR_RECOMMEND,
    FAMILY_LLM,
    FAMILY_NATIVE,
    MODEL_ARCH_HSTU,
    MODEL_ARCHS_BY_FAMILY,
    USER_MODE_GR_BEAM,
    USER_MODE_GR_RECOMMEND,
    USER_MODE_NATIVE,
    USER_MODES,
)

CURRENT_SCHEMA_VERSION: int = 1

ENV_VLLM_GR_MODE: str = "VLLM_GR_MODE"

ENV_PROPAGATION_RUNTIME_SELECTION: str = "_VLLM_GR_RUNTIME_SELECTION"
ENV_PROPAGATION_BEAM_CONFIG: str = "_VLLM_GR_BEAM_CONFIG"
ENV_PROPAGATION_ACTIVATED: str = "_VLLM_GR_ACTIVATED"


@dataclass(frozen=True)
class DiagInfo:
    source: str

    def __post_init__(self) -> None:
        if self.source not in DIAG_SOURCES:
            raise GRPreconditionError(
                f"Invalid diagnostics source {self.source!r}; "
                f"expected one of {sorted(DIAG_SOURCES)}"
            )

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DiagInfo:
        source = payload.get("source")
        if not isinstance(source, str):
            raise GRPropagationError("Diagnostics payload missing string 'source'")
        return cls(source=source)


@dataclass(frozen=True)
class RuntimeSelection:
    family: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    model_arch: Optional[str] = None
    diagnostics: DiagInfo = field(default_factory=lambda: DiagInfo(DIAG_SOURCE_DEFAULT))

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise GRCompatibilityError(
                f"Unknown family {self.family!r}; expected one of {sorted(FAMILIES)}"
            )

        unknown_caps = self.capabilities - CAPABILITIES
        if unknown_caps:
            raise GRCompatibilityError(
                f"Unknown capabilities {sorted(unknown_caps)!r}; "
                f"expected subset of {sorted(CAPABILITIES)}"
            )

        if self.family in CAPABILITY_FORBIDDEN_FAMILIES and self.capabilities:
            raise GRCompatibilityError(
                f"Family {self.family!r} cannot carry capabilities "
                f"{sorted(self.capabilities)!r}"
            )

        if self.family == FAMILY_LLM and not self.capabilities:
            raise GRCompatibilityError(
                "family='llm' requires at least one capability; "
                "use family='native' for plain vLLM execution"
            )

        for capability in self.capabilities:
            host = CAPABILITY_HOST_FAMILY.get(capability)
            if host is not None and self.family != host:
                raise GRCompatibilityError(
                    f"Capability {capability!r} requires family={host!r}, "
                    f"got family={self.family!r}"
                )

        allowed_model_archs = MODEL_ARCHS_BY_FAMILY.get(self.family, frozenset())
        if self.model_arch is not None:
            if self.model_arch not in allowed_model_archs:
                raise GRCompatibilityError(
                    f"model_arch={self.model_arch!r} is invalid for "
                    f"family={self.family!r}"
                )
        elif self.family == FAMILY_GR_RECOMMEND:
            raise GRCompatibilityError(
                "family='gr_recommend' requires model_arch='hstu'"
            )

        if self.family != FAMILY_GR_RECOMMEND and self.model_arch is not None:
            raise GRCompatibilityError(
                f"model_arch={self.model_arch!r} is only valid for "
                f"family='gr_recommend'"
            )

    def serialize(self) -> str:
        self.validate()
        payload = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "family": self.family,
            "model_arch": self.model_arch,
            "capabilities": sorted(self.capabilities),
            "diagnostics": self.diagnostics.to_dict(),
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def deserialize(cls, data: str) -> RuntimeSelection:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise GRPropagationError(
                f"RuntimeSelection JSON is invalid: {exc}"
            ) from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeSelection:
        if not isinstance(payload, Mapping):
            raise GRPropagationError(
                "RuntimeSelection payload must be a JSON object"
            )
        schema_version = payload.get("schema_version")
        if schema_version != CURRENT_SCHEMA_VERSION:
            raise GRPropagationError(
                f"Unsupported RuntimeSelection schema_version={schema_version!r}; "
                f"expected {CURRENT_SCHEMA_VERSION}"
            )

        family = payload.get("family")
        if not isinstance(family, str):
            raise GRPropagationError("RuntimeSelection payload missing string 'family'")

        model_arch = payload.get("model_arch")
        if model_arch is not None and not isinstance(model_arch, str):
            raise GRPropagationError(
                "RuntimeSelection payload field 'model_arch' must be a string or null"
            )

        raw_caps = payload.get("capabilities", [])
        if not isinstance(raw_caps, list) or not all(
            isinstance(item, str) for item in raw_caps
        ):
            raise GRPropagationError(
                "RuntimeSelection payload field 'capabilities' must be a string list"
            )

        diagnostics_payload = payload.get("diagnostics", {"source": DIAG_SOURCE_DEFAULT})
        if not isinstance(diagnostics_payload, Mapping):
            raise GRPropagationError(
                "RuntimeSelection payload field 'diagnostics' must be an object"
            )

        selection = cls(
            family=family,
            model_arch=model_arch,
            capabilities=frozenset(raw_caps),
            diagnostics=DiagInfo.from_dict(diagnostics_payload),
        )
        selection.validate()
        return selection


@dataclass(frozen=True)
class GRBeamConfig:
    token_catalog_path: Optional[str] = None

    def validate(self) -> None:
        if self.token_catalog_path is not None and not self.token_catalog_path.strip():
            raise GRPreconditionError("token_catalog_path cannot be empty")

    def serialize(self) -> str:
        self.validate()
        payload = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "token_catalog_path": self.token_catalog_path,
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def deserialize(cls, data: str) -> GRBeamConfig:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise GRPropagationError(f"GRBeamConfig JSON is invalid: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GRBeamConfig:
        if not isinstance(payload, Mapping):
            raise GRPropagationError(
                "GRBeamConfig payload must be a JSON object"
            )
        schema_version = payload.get("schema_version")
        if schema_version != CURRENT_SCHEMA_VERSION:
            raise GRPropagationError(
                f"Unsupported GRBeamConfig schema_version={schema_version!r}; "
                f"expected {CURRENT_SCHEMA_VERSION}"
            )

        token_catalog_path = payload.get("token_catalog_path")
        if token_catalog_path is not None and not isinstance(token_catalog_path, str):
            raise GRPropagationError(
                "GRBeamConfig payload field 'token_catalog_path' must be a string or null"
            )

        config = cls(token_catalog_path=token_catalog_path)
        config.validate()
        return config


@dataclass(frozen=True)
class GRBeamRequestOptions:
    beam_width: int = 1
    begin_token: Optional[str] = None
    end_token: Optional[str] = None

    def validate(self) -> None:
        if self.beam_width < 1:
            raise GRPreconditionError("beam_width must be >= 1")


@dataclass(frozen=True)
class RuntimeIntentInput:
    cli_mode: Optional[str] = None
    api_mode: Optional[str] = None
    env_mode: Optional[str] = None
    use_auto: bool = False
    auto_inferred_family: Optional[str] = None


def parse_vllm_gr_mode(raw_value: Optional[str]) -> Optional[str]:
    """Parse ``VLLM_GR_MODE``; unset/blank -> ``None``; unknown -> fail fast."""
    if raw_value is None:
        return None
    normalized = raw_value.strip().lower()
    if not normalized:
        return None
    if normalized not in USER_MODES:
        raise GRInconsistencyError(
            f"Unknown VLLM_GR_MODE={raw_value!r}; "
            f"expected one of {sorted(USER_MODES)}"
        )
    return normalized


def read_vllm_gr_mode_from_env(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    env = os.environ if environ is None else environ
    return parse_vllm_gr_mode(env.get(ENV_VLLM_GR_MODE))


def selection_from_user_mode(mode: str, *, source: str) -> RuntimeSelection:
    if mode == USER_MODE_NATIVE:
        return RuntimeSelection(
            family=FAMILY_NATIVE,
            diagnostics=DiagInfo(source=source),
        )
    if mode == USER_MODE_GR_RECOMMEND:
        return RuntimeSelection(
            family=FAMILY_GR_RECOMMEND,
            model_arch=MODEL_ARCH_HSTU,
            diagnostics=DiagInfo(source=source),
        )
    if mode == USER_MODE_GR_BEAM:
        return RuntimeSelection(
            family=FAMILY_LLM,
            capabilities=frozenset({CAPABILITY_GR_BEAM}),
            diagnostics=DiagInfo(source=source),
        )
    raise GRInconsistencyError(f"Unsupported user mode {mode!r}")


def normalize_runtime_selection(
    intent: RuntimeIntentInput,
) -> RuntimeSelection:
    """Normalize CLI/API/ENV/auto inputs into a validated RuntimeSelection."""
    explicit_sources: dict[str, str] = {}
    if intent.cli_mode is not None:
        explicit_sources["cli"] = intent.cli_mode
    if intent.api_mode is not None:
        explicit_sources["api"] = intent.api_mode

    if len(explicit_sources) > 1:
        values = set(explicit_sources.values())
        if len(values) > 1:
            detail = ", ".join(
                f"{name}={value!r}" for name, value in explicit_sources.items()
            )
            raise GRInconsistencyError(
                f"Conflicting explicit runtime modes: {detail}"
            )

    if intent.env_mode is not None and explicit_sources:
        env_conflicts = [
            (name, value)
            for name, value in explicit_sources.items()
            if value != intent.env_mode
        ]
        if env_conflicts:
            detail_parts = [
                f"{name}={value!r}" for name, value in explicit_sources.items()
            ]
            detail_parts.append(f"env={intent.env_mode!r}")
            detail = ", ".join(detail_parts)
            raise GRInconsistencyError(f"Conflicting runtime mode: {detail}")

    has_explicit = bool(explicit_sources) or intent.env_mode is not None
    if intent.use_auto and has_explicit:
        raise GRInconsistencyError(
            "--auto cannot be combined with explicit runtime mode selection"
        )

    if intent.use_auto:
        if intent.auto_inferred_family is None:
            raise GRInconsistencyError(
                "--auto was requested but no family could be inferred"
            )
        if intent.auto_inferred_family == FAMILY_GR_RECOMMEND:
            return RuntimeSelection(
                family=FAMILY_GR_RECOMMEND,
                model_arch=MODEL_ARCH_HSTU,
                diagnostics=DiagInfo(source=DIAG_SOURCE_AUTO),
            )
        if intent.auto_inferred_family == FAMILY_NATIVE:
            return RuntimeSelection(
                family=FAMILY_NATIVE,
                diagnostics=DiagInfo(source=DIAG_SOURCE_AUTO),
            )
        if intent.auto_inferred_family == FAMILY_LLM:
            raise GRCompatibilityError(
                "Auto-inferred LLM requires an explicit capability such as "
                "'--gr-beam'"
            )
        raise GRInconsistencyError(
            f"Unsupported auto-inferred family {intent.auto_inferred_family!r}"
        )

    if intent.cli_mode is not None:
        selection = selection_from_user_mode(intent.cli_mode, source=DIAG_SOURCE_CLI)
    elif intent.api_mode is not None:
        selection = selection_from_user_mode(intent.api_mode, source=DIAG_SOURCE_API)
    elif intent.env_mode is not None:
        selection = selection_from_user_mode(intent.env_mode, source=DIAG_SOURCE_ENV)
    else:
        selection = RuntimeSelection(
            family=FAMILY_NATIVE,
            diagnostics=DiagInfo(source=DIAG_SOURCE_DEFAULT),
        )

    selection.validate()
    return selection
