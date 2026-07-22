# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capability detection for the optional Ascend NPU runtime."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

_SELECTOR_PARAMETERS = ("backend", "attn_selector_config", "num_heads")


@dataclass(frozen=True)
class NPUAttentionCapabilities:
    """Attention capabilities relevant to the CUSTOM backend bridge."""

    is_npu: bool
    native_custom_backend: bool
    selector_api_compatible: bool
    vllm_version: str | None = None
    vllm_ascend_version: str | None = None
    reason: str | None = None

    @property
    def requires_custom_backend_bridge(self) -> bool:
        return self.is_npu and not self.native_custom_backend


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def is_npu_platform() -> bool:
    """Return whether vLLM selected an NPU platform without importing Ascend."""
    try:
        from vllm.platforms import current_platform
    except ImportError:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


def _selector_api_is_compatible(selector_fn) -> bool:
    try:
        parameters = tuple(inspect.signature(selector_fn).parameters)
    except (TypeError, ValueError):
        return False
    return parameters == _SELECTOR_PARAMETERS


def _platform_selects_registered_custom_backend(current_platform) -> bool:
    """Probe whether the platform honors the registered CUSTOM class path."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_gr.v1.attention.backends import (
        BeamAttentionBackend,
        register_beam_attention_backend,
    )

    register_beam_attention_backend()
    expected_path = f"{BeamAttentionBackend.__module__}.{BeamAttentionBackend.__qualname__}"
    selector_config = SimpleNamespace(
        use_mla=False,
        use_sparse=False,
        use_compress=False,
    )
    try:
        selected_path = current_platform.get_attn_backend_cls(
            AttentionBackendEnum.CUSTOM,
            attn_selector_config=selector_config,
            num_heads=None,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return selected_path == expected_path


def detect_npu_attention_capabilities() -> NPUAttentionCapabilities:
    """Detect behavior instead of selecting an implementation by version."""
    versions = {
        "vllm_version": _package_version("vllm"),
        "vllm_ascend_version": _package_version("vllm-ascend"),
    }
    if not is_npu_platform():
        return NPUAttentionCapabilities(
            is_npu=False,
            native_custom_backend=False,
            selector_api_compatible=False,
            reason="vLLM did not select the NPU platform",
            **versions,
        )

    try:
        import vllm.v1.attention.selector as selector_mod
        from vllm.platforms import current_platform
    except ImportError as exc:
        return NPUAttentionCapabilities(
            is_npu=True,
            native_custom_backend=False,
            selector_api_compatible=False,
            reason=f"required vLLM attention API is unavailable: {exc}",
            **versions,
        )

    selector_api_compatible = _selector_api_is_compatible(selector_mod._cached_get_attn_backend)
    native_custom_backend = _platform_selects_registered_custom_backend(current_platform)
    reason = None
    if not native_custom_backend and not selector_api_compatible:
        reason = "vLLM attention selector signature is not supported"
    elif not native_custom_backend:
        reason = "NPU platform does not honor the registered CUSTOM backend"

    return NPUAttentionCapabilities(
        is_npu=True,
        native_custom_backend=native_custom_backend,
        selector_api_compatible=selector_api_compatible,
        reason=reason,
        **versions,
    )
