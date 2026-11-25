# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Immutable, import-light data exchanged between engine and worker processes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PLAN_SCHEMA_VERSION = 1
PLAN_API_VERSION = 3
BACKEND_PLAN_KEY = "vllm_gr_backend_plan"


def _json_value(value: Any) -> Any:
    # Do not stringify arbitrary objects: reprs can contain addresses and local
    # device ordinals. Providers must explicitly encode their own typed options.
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping) and all(type(key) is str for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ValueError(f"Backend plan requires finite JSON values, got {type(value).__name__}")


@dataclass(frozen=True, slots=True, init=False)
class PlanData:
    """Deeply immutable JSON object; decoded views are independent copies."""

    canonical_json: str

    def __init__(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise ValueError("Backend plan data must be a JSON object")
        encoded = json.dumps(
            _json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        object.__setattr__(self, "canonical_json", encoded)

    def to_dict(self) -> dict[str, Any]:
        return dict(json.loads(self.canonical_json))


@dataclass(frozen=True, slots=True)
class BackendRequest:
    """Normalized engine inputs; no runtime objects or process-local state."""

    config: PlanData
    native: PlanData
    compatibility: PlanData
    options: PlanData

    def __post_init__(self) -> None:
        for value in (self.config, self.native, self.compatibility, self.options):
            if not isinstance(value, PlanData):
                raise TypeError("BackendRequest fields must be immutable PlanData")

    @property
    def has_constraint_table(self) -> bool:
        constraint = self.config.to_dict().get("constraint")
        return bool(constraint and constraint["mode"] == "table")

    @property
    def uses_custom_attention(self) -> bool:
        return self.config.to_dict().get("attention_backend") == "CUSTOM"

    @property
    def uses_beam_v1(self) -> bool:
        return self.config.to_dict().get("beam_execution_mode") == "v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "native": self.native.to_dict(),
            "compatibility": self.compatibility.to_dict(),
            "options": self.options.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BackendRequest:
        if not isinstance(value, Mapping) or set(value) != {
            "config",
            "native",
            "compatibility",
            "options",
        }:
            raise ValueError("Invalid BackendRequest fields")
        return cls(**{key: PlanData(item) for key, item in value.items()})


@dataclass(frozen=True, slots=True)
class ServiceRequirements:
    """Implementation identities a worker must bind before serving the plan."""

    stage: str | None = None
    attention: str | None = None
    graph: str | None = None
    constraint: str | None = None

    def __post_init__(self) -> None:
        for name, identity in self.to_dict().items():
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f"Backend {name} requirement needs a non-empty implementation ID")

    def to_dict(self) -> dict[str, str]:
        return {
            name: value
            for name in ("stage", "attention", "graph", "constraint")
            if (value := getattr(self, name)) is not None
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServiceRequirements:
        if not isinstance(value, Mapping) or set(value) - {
            "stage",
            "attention",
            "graph",
            "constraint",
        }:
            raise ValueError("Invalid backend service requirements")
        if any(identity is None for identity in value.values()):
            raise ValueError("Omit optional backend requirements without an implementation ID")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class BackendPlan:
    """One admitted engine plan, with extensible backend-owned choices."""

    backend_id: str
    provider_api_version: int
    request: BackendRequest
    choices: PlanData
    plan_api_version: int = PLAN_API_VERSION
    schema_version: int = PLAN_SCHEMA_VERSION
    requirements: ServiceRequirements | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise ValueError("BackendPlan requires a backend_id")
        if type(self.provider_api_version) is not int or self.provider_api_version < 1:
            raise ValueError("BackendPlan requires an integer provider API version")
        if type(self.plan_api_version) is not int or self.plan_api_version != PLAN_API_VERSION:
            raise ValueError("Unsupported backend planning API version")
        if type(self.schema_version) is not int or self.schema_version != PLAN_SCHEMA_VERSION:
            raise ValueError("Unsupported BackendPlan schema version")
        if not isinstance(self.request, BackendRequest) or not isinstance(self.choices, PlanData):
            raise TypeError("BackendPlan requires immutable request and choices")
        if self.requirements is not None and not isinstance(self.requirements, ServiceRequirements):
            raise TypeError("BackendPlan requirements must be immutable ServiceRequirements")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_api_version": self.plan_api_version,
            "provider_api_version": self.provider_api_version,
            "backend_id": self.backend_id,
            "request": self.request.to_dict(),
            "choices": self.choices.to_dict(),
            **(
                {"requirements": self.requirements.to_dict()}
                if self.requirements is not None
                else {}
            ),
        }

    @property
    def required_services(self) -> dict[str, str | None]:
        """Required report keys and pinned IDs. None permits a worker-selected ID."""
        services: dict[str, str | None] = {"runner": None}
        if self.requirements is not None:
            services.update(self.requirements.to_dict())
        else:
            choices = self.choices.to_dict()
            if choices.get("requires_attention"):
                services["attention"] = None
            if choices.get("constraint") == "legacy":
                services["constraint"] = None
        if self.request.uses_custom_attention:
            services.setdefault("attention", None)
        return services

    @property
    def constraint_mode(self) -> str | None:
        if self.requirements is None:
            return self.choices.to_dict().get("constraint")
        if self.requirements.constraint is None:
            return None
        return "canonical" if self.request.has_constraint_table else "legacy"

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(PlanData(self.to_dict()).canonical_json.encode("ascii")).hexdigest()

    def to_wire(self) -> dict[str, Any]:
        return {"plan": self.to_dict(), "fingerprint": self.fingerprint}

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> BackendPlan:
        if not isinstance(value, Mapping) or set(value) != {"plan", "fingerprint"}:
            raise ValueError("Invalid backend plan envelope")
        data = value["plan"]
        fields = {
            "schema_version",
            "plan_api_version",
            "provider_api_version",
            "backend_id",
            "request",
            "choices",
        }
        if not isinstance(data, Mapping) or set(data) not in (fields, fields | {"requirements"}):
            raise ValueError("Invalid backend plan fields")
        plan = cls(
            backend_id=data["backend_id"],
            provider_api_version=data["provider_api_version"],
            plan_api_version=data["plan_api_version"],
            schema_version=data["schema_version"],
            request=BackendRequest.from_dict(data["request"]),
            choices=PlanData(data["choices"]),
            requirements=(
                ServiceRequirements.from_dict(data["requirements"])
                if "requirements" in data
                else None
            ),
        )
        if value["fingerprint"] != plan.fingerprint:
            raise ValueError("Backend plan fingerprint mismatch")
        return plan


def read_backend_plan(config: Any) -> BackendPlan | None:
    additional = getattr(config, "additional_config", None)
    if not isinstance(additional, Mapping) or BACKEND_PLAN_KEY not in additional:
        return None
    return BackendPlan.from_wire(additional[BACKEND_PLAN_KEY])
