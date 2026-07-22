# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU-only compatibility bridge for the vLLM-GR CUSTOM attention backend."""

from __future__ import annotations

import logging
import threading
from functools import cache, wraps
from typing import Any

from vllm_gr.platform.npu.capabilities import (
    NPUAttentionCapabilities,
    detect_npu_attention_capabilities,
    is_npu_platform,
)

logger = logging.getLogger(__name__)

GR_ATTENTION_BACKEND_KEY = "vllm_gr_attention_backend"
GR_CUSTOM_ATTENTION_BACKEND = "CUSTOM"


class NPUAttentionCompatibilityError(RuntimeError):
    """Raised when an explicit CUSTOM request cannot be honored safely."""


_INSTALL_LOCK = threading.RLock()
_ORIGINAL_SELECTOR = None
_INSTALLED_SELECTOR = None


def preserve_custom_backend_intent(config: Any, *, requested: bool) -> None:
    """Persist CUSTOM intent only when Ascend may erase the selected backend."""
    if not requested or not is_npu_platform():
        return

    capabilities = detect_npu_attention_capabilities()
    if capabilities.requires_custom_backend_bridge and not capabilities.selector_api_compatible:
        raise NPUAttentionCompatibilityError(_unsupported_api_message(capabilities))

    additional_config = getattr(config, "additional_config", None)
    if not isinstance(additional_config, dict):
        raise ValueError(
            "vLLM-GR NPU attention options require --additional-config to be a JSON object"
        )
    additional_config[GR_ATTENTION_BACKEND_KEY] = GR_CUSTOM_ATTENTION_BACKEND

    attention_config = getattr(config, "attention_config", None)
    if attention_config is not None and getattr(attention_config, "backend", None) is None:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        attention_config.backend = AttentionBackendEnum.CUSTOM


def _unsupported_api_message(capabilities: NPUAttentionCapabilities) -> str:
    versions = (
        f"vLLM={capabilities.vllm_version or 'unknown'}, "
        f"vLLM-Ascend={capabilities.vllm_ascend_version or 'unknown'}"
    )
    return (
        "The requested CUSTOM attention backend cannot be installed safely: "
        f"{capabilities.reason or 'unknown compatibility failure'} ({versions})."
    )


def _marker_requests_custom_backend() -> bool:
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None:
        return False
    additional_config = getattr(config, "additional_config", None)
    return isinstance(additional_config, dict) and (
        additional_config.get(GR_ATTENTION_BACKEND_KEY) == GR_CUSTOM_ATTENTION_BACKEND
    )


def _validate_custom_backend(backend_cls, attn_selector_config) -> None:
    from vllm.platforms import current_platform

    selector_values = attn_selector_config._asdict()
    invalid_reasons = backend_cls.validate_configuration(
        device_capability=current_platform.get_device_capability(),
        **selector_values,
    )
    if invalid_reasons:
        reasons = ", ".join(invalid_reasons)
        raise ValueError(
            "Selected vLLM-GR CUSTOM attention backend is not valid for "
            f"this NPU configuration: {reasons}"
        )


def _apply_required_kv_cache_layout(backend_cls) -> None:
    required_layout = backend_cls.get_required_kv_cache_layout()
    if required_layout is None:
        return
    from vllm.v1.attention.backends.utils import set_kv_cache_layout

    set_kv_cache_layout(required_layout)


def _build_selector_bridge(original_selector):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    @cache
    def select_backend_cached(
        backend,
        attn_selector_config,
        num_heads,
        custom_requested,
    ):
        if not custom_requested:
            return original_selector(
                backend,
                attn_selector_config,
                num_heads=num_heads,
            )

        try:
            from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

            _validate_custom_backend(BeamAttentionBackend, attn_selector_config)
            _apply_required_kv_cache_layout(BeamAttentionBackend)
            return BeamAttentionBackend
        except Exception as exc:
            if isinstance(exc, (NPUAttentionCompatibilityError, ValueError)):
                raise
            raise NPUAttentionCompatibilityError(
                "The requested vLLM-GR CUSTOM attention backend could not be loaded"
            ) from exc

    @wraps(original_selector)
    def select_backend(backend, attn_selector_config, num_heads=None):
        explicitly_requested = backend == AttentionBackendEnum.CUSTOM
        marker_requested = backend is None and _marker_requests_custom_backend()
        return select_backend_cached(
            backend,
            attn_selector_config,
            num_heads,
            explicitly_requested or marker_requested,
        )

    select_backend.cache_clear = select_backend_cached.cache_clear
    select_backend.cache_info = select_backend_cached.cache_info
    return select_backend


def install_npu_attention_backend_compat(*, force: bool = False) -> int:
    """Install the selector bridge only for affected NPU platforms."""
    del force  # Re-entry is intentionally idempotent; never replace another patch.
    capabilities = detect_npu_attention_capabilities()
    if not capabilities.is_npu or capabilities.native_custom_backend:
        return 0
    if not capabilities.selector_api_compatible:
        logger.warning(_unsupported_api_message(capabilities))
        return 0

    import vllm.v1.attention.selector as selector_mod

    global _INSTALLED_SELECTOR
    global _ORIGINAL_SELECTOR
    with _INSTALL_LOCK:
        if selector_mod._cached_get_attn_backend is _INSTALLED_SELECTOR:
            return 0
        if _INSTALLED_SELECTOR is not None:
            raise NPUAttentionCompatibilityError(
                "vLLM attention selector changed after the NPU compatibility "
                "bridge was installed; refusing to overwrite another patch"
            )

        _ORIGINAL_SELECTOR = selector_mod._cached_get_attn_backend
        _INSTALLED_SELECTOR = _build_selector_bridge(_ORIGINAL_SELECTOR)
        selector_mod._cached_get_attn_backend = _INSTALLED_SELECTOR
        logger.info("Installed the vLLM-GR NPU CUSTOM attention compatibility bridge")
        return 1


def uninstall_npu_attention_backend_compat() -> bool:
    """Restore the selector when this module still owns the installed bridge."""
    import vllm.v1.attention.selector as selector_mod

    global _INSTALLED_SELECTOR
    global _ORIGINAL_SELECTOR
    with _INSTALL_LOCK:
        if _INSTALLED_SELECTOR is None:
            return False
        if selector_mod._cached_get_attn_backend is not _INSTALLED_SELECTOR:
            raise NPUAttentionCompatibilityError(
                "Cannot restore the vLLM selector because another patch replaced it"
            )
        selector_mod._cached_get_attn_backend = _ORIGINAL_SELECTOR
        _INSTALLED_SELECTOR = None
        _ORIGINAL_SELECTOR = None
        return True
