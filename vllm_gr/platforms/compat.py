# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Package version checks for provider declarations and built-in runtimes."""

from __future__ import annotations

import logging
import os
from functools import cache
from importlib import metadata

from .errors import InvalidBackendProviderError
from .interface import GRBackendProvider

logger = logging.getLogger("vllm_gr.platforms.compat")

# Exact versions supported by the built-in providers. The CI resolver reads
# these literals without importing the target checkout.
SUPPORTED_VLLM_VERSION = "0.30.0"
SUPPORTED_VLLM_ASCEND_VERSION = "0.30.0rc2"
SUPPORTED_TRITON_ASCEND_VERSION = "3.6.0"

#: When set to a truthy value, an unsatisfied declared range is a hard failure.
#: The default only warns: whether a declared range still holds is a vendor
#: compatibility decision, and failing hard by default would refuse startup on
#: any patch-level runtime bump.
_STRICT_ENV = "VLLM_GR_STRICT_PROVIDER_RANGES"
_TRUTHY = ("1", "true", "yes", "on")


def validate_provider_metadata(provider: GRBackendProvider, origin: str) -> None:
    """Validate the v2 declarations without probing an installed runtime."""
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    for name in ("gr_version_range", "vllm_version_range", "vendor_version_range"):
        if not hasattr(provider, name):
            raise InvalidBackendProviderError(f"Missing required {name}; {origin}")
        value = getattr(provider, name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise InvalidBackendProviderError(f"{name} must be a string or None; {origin}")
        try:
            SpecifierSet(value)
        except InvalidSpecifier as exc:
            raise InvalidBackendProviderError(f"Invalid {name} {value!r}; {origin}") from exc
    if not hasattr(provider, "vendor_distribution"):
        raise InvalidBackendProviderError(f"Missing required vendor_distribution; {origin}")
    vendor = provider.vendor_distribution
    if vendor is not None and (not isinstance(vendor, str) or not vendor.strip()):
        raise InvalidBackendProviderError(f"Invalid vendor_distribution {vendor!r}; {origin}")
    if provider.vendor_version_range is not None and vendor is None:
        raise InvalidBackendProviderError(
            f"vendor_version_range requires vendor_distribution; {origin}"
        )


def strict_ranges_enabled() -> bool:
    """Whether declared range mismatches must fail startup instead of warn."""
    return os.environ.get(_STRICT_ENV, "").strip().lower() in _TRUTHY


def installed_version(distribution: str) -> str | None:
    """Read distribution metadata without importing its runtime modules."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def require_runtime_version(distribution: str, required: str) -> str:
    """Require an exact public runtime version and retain its local suffix."""
    installed = installed_version(distribution)
    if installed is None:
        raise RuntimeError(
            f"vllm-gr requires {distribution} to be installed before runtime initialization"
        )
    # Match the CI resolver's public version comparison for built-in pins.
    if installed.partition("+")[0] != required:
        raise RuntimeError(
            f"Unsupported {distribution} version: found {installed}, required {required}."
        )
    return installed


def check_vllm_runtime() -> dict[str, str]:
    """Check the vLLM runtime supported by the built-in providers."""
    return {"vllm": require_runtime_version("vllm", SUPPORTED_VLLM_VERSION)}


def _specifier_allows(specifier: str, version: str) -> bool:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    try:
        return Version(version) in SpecifierSet(specifier)
    except Exception:
        # An unparsable declared range is a provider defect, not a runtime one.
        raise InvalidBackendProviderError(
            f"Provider declared an unparsable version range {specifier!r}"
        ) from None


def _check_range(
    *,
    label: str,
    declared: str | None,
    distribution: str,
    provider: GRBackendProvider,
    failures: list[str],
) -> None:
    if declared is None:
        return
    installed = installed_version(distribution)
    if installed is None:
        failures.append(
            f"{label} distribution {distribution!r} is not installed "
            f"(provider {provider.backend_id!r} declares {declared!r})"
        )
        return
    if not _specifier_allows(declared, installed):
        failures.append(
            f"{label} version {installed} is outside the range {declared!r} "
            f"declared by provider {provider.backend_id!r}"
        )


def check_declared_ranges(provider: GRBackendProvider) -> tuple[str, ...]:
    """Return every declared-range violation for one provider.

    Empty means all declared ranges hold. ``None`` declares no constraint.
    """
    failures: list[str] = []
    gr_version = installed_version("vllm-gr") or _source_tree_version()

    declared_gr = provider.gr_version_range
    if declared_gr is not None and gr_version is not None:
        if not _specifier_allows(declared_gr, gr_version):
            failures.append(
                f"vllm-gr version {gr_version} is outside the range "
                f"{declared_gr!r} declared by provider {provider.backend_id!r}"
            )

    _check_range(
        label="vLLM",
        declared=provider.vllm_version_range,
        distribution="vllm",
        provider=provider,
        failures=failures,
    )
    if provider.vendor_version_range is not None and provider.vendor_distribution is not None:
        _check_range(
            label="vendor runtime",
            declared=provider.vendor_version_range,
            distribution=provider.vendor_distribution,
            provider=provider,
            failures=failures,
        )
    return tuple(failures)


def _source_tree_version() -> str | None:
    try:
        from vllm_gr.version import __version__
    except Exception:  # pragma: no cover - import-time diagnostics only
        return None
    return str(__version__)


def enforce_declared_ranges(provider: GRBackendProvider) -> None:
    """Raise on violations when strict mode is enabled; otherwise warn once."""
    failures = check_declared_ranges(provider)
    if not failures:
        return
    message = "; ".join(failures)
    if strict_ranges_enabled():
        raise InvalidBackendProviderError(
            f"{_STRICT_ENV} is set and a declared compatibility range failed: {message}"
        )
    _warn_range_once(provider.backend_id, message)


@cache
def _warn_range_once(backend_id: str, message: str) -> None:
    logger.warning(
        "vllm-gr backend %r declares a compatibility range that does not hold: %s. "
        "Set %s=1 to make this a startup failure.",
        backend_id,
        message,
        _STRICT_ENV,
    )
