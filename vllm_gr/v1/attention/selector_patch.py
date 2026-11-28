# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Beam attention selection and vLLM selector cache isolation.

The selection seam itself is :meth:`GRPlatform.get_gr_attention_backend_cls`; this
module reads the engine marker and keeps engine-dependent Beam decisions
out of the native selector cache. The
shared AttentionBackendEnum.CUSTOM registry slot is never claimed.
"""

from functools import wraps
from typing import Any

from vllm_gr.config import GRAttentionBackend, GRConfig


def _get_current_gr_attention_backend() -> GRAttentionBackend | None:
    from vllm.config import get_current_vllm_config_or_none

    return GRConfig.from_vllm_config(get_current_vllm_config_or_none()).attention_backend


def _gr_attention_is_custom() -> bool:
    """Whether the current engine's GR configuration selects Beam attention."""
    return _get_current_gr_attention_backend() is GRAttentionBackend.CUSTOM


def install_gr_attention_selector() -> None:
    """Keep Beam decisions out of vLLM's engine-independent cache."""
    from vllm.v1.attention import selector

    native = selector._cached_get_attn_backend
    if getattr(native, "_vllm_gr_attention_cache_isolated", False):
        return

    @wraps(native)
    def select(backend: Any, attn_selector_config: Any, num_heads: int | None = None) -> Any:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if _gr_attention_is_custom() and backend in (None, AttentionBackendEnum.CUSTOM):
            from vllm.utils.import_utils import resolve_obj_by_qualname

            from vllm_gr.platforms import current_gr_platform

            path = current_gr_platform.get_gr_attention_backend_cls(
                backend, attn_selector_config, num_heads=num_heads
            )
            return resolve_obj_by_qualname(path)
        return native(backend, attn_selector_config, num_heads=num_heads)

    setattr(select, "_vllm_gr_attention_cache_isolated", True)
    setattr(select, "cache_clear", native.cache_clear)
    selector._cached_get_attn_backend = select


def clear_attention_backend_cache() -> None:
    """Clear native selection results cached for earlier engine configs."""
    from vllm.v1.attention.selector import _cached_get_attn_backend

    _cached_get_attn_backend.cache_clear()
