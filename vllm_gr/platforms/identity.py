# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Import-light bridge from the native vLLM platform to a GR identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

UNSPECIFIED_PLATFORM = "UNSPECIFIED"


def qualified_class_name(cls: type) -> str:
    """Return ``module.qualname`` without importing or instantiating the class."""
    module = getattr(cls, "__module__", "") or ""
    qualname = getattr(cls, "__qualname__", "") or ""
    return f"{module}.{qualname}".lstrip(".")


@dataclass(frozen=True)
class PlatformIdentity:
    """Immutable description of the native platform vLLM already selected.

    ``device_type`` is tensor placement, not a backend identity: NVIDIA CUDA and
    AMD ROCm both report ``cuda``. Selection therefore also carries the upstream
    platform enum and the qualified class lineage, so a provider can recognize a
    vendor subclass without core carrying a table of known platform classes.

    The record is built from the live ``current_platform`` instance, so reading
    it imports no vendor runtime and no hardware module.
    """

    platform_enum: str
    device_type: str
    device_name: str
    platform_class: str
    platform_lineage: tuple[str, ...]

    def has_lineage(self, qualified_name: str) -> bool:
        """Whether the platform class is, or derives from, ``qualified_name``."""
        return qualified_name in self.platform_lineage

    def describe(self) -> str:
        """Return a one-line identity description for diagnostics."""
        return (
            f"platform={self.platform_enum!r} device_type={self.device_type!r} "
            f"class={self.platform_class!r}"
        )


def _enum_name(platform: Any) -> str:
    """Read the upstream ``PlatformEnum`` member name from a live platform."""
    raw = getattr(platform, "_enum", None)
    if raw is None:
        return UNSPECIFIED_PLATFORM
    if isinstance(raw, str):
        value = raw
    else:
        value = str(getattr(raw, "name", raw))
    normalized = value.strip().upper()
    return normalized or UNSPECIFIED_PLATFORM


def _text(platform: Any, attribute: str) -> str:
    value = getattr(platform, attribute, None)
    return value.strip().lower() if isinstance(value, str) else ""


def platform_identity(platform: Any) -> PlatformIdentity:
    """Build the identity of an already-selected native platform instance.

    Nothing is inferred from installed packages, and availability is never
    probed here: the caller must pass the platform native vLLM already chose.
    """
    platform_class = type(platform)
    return PlatformIdentity(
        platform_enum=_enum_name(platform),
        device_type=_text(platform, "device_type"),
        device_name=_text(platform, "device_name"),
        platform_class=qualified_class_name(platform_class),
        platform_lineage=tuple(qualified_class_name(base) for base in platform_class.__mro__),
    )


def current_platform_identity() -> PlatformIdentity:
    """Return the identity of vLLM's live ``current_platform``.

    Native vLLM remains the only hardware selection authority; this function
    only describes the choice it already made.
    """
    from vllm.platforms import current_platform

    return platform_identity(current_platform)
