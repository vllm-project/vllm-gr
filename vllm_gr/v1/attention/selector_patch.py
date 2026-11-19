# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Hardware-neutral CUSTOM attention selection for GR runtime initialization."""

from inspect import isabstract
from typing import Any

from vllm_gr.config import GRAttentionBackend, GRConfig

_ORIGINAL_ATTR = "_vllm_gr_custom_attention_original"


def _get_current_gr_attention_backend() -> GRAttentionBackend | None:
    from vllm.config import get_current_vllm_config_or_none

    return GRConfig.from_vllm_config(get_current_vllm_config_or_none()).attention_backend


def _get_custom_beam_backend(selector_config: Any, *, num_heads: int | None = None) -> type:
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.attention.backend import AttentionBackend

    from vllm_gr.platforms.attention import admission_for_config, selector_admission

    with selector_admission(selector_config, num_heads=num_heads) as admitted:
        select = getattr(admission_for_config(), "get_attn_backend_cls", None)
        if select is None:
            from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

            backend = BeamAttentionBackend
        else:
            backend = select(attn_selector_config=admitted, num_heads=num_heads)
            if isinstance(backend, str):
                backend = resolve_obj_by_qualname(backend)
            if (
                not isinstance(backend, type)
                or not issubclass(backend, AttentionBackend)
                or isabstract(backend)
            ):
                raise TypeError(
                    "CUSTOM attention provider must select a concrete vLLM AttentionBackend class"
                )
        reasons = backend.validate_configuration(
            device_capability=current_platform.get_device_capability(),
            **admitted._asdict(),
        )
    if reasons:
        raise ValueError(f"CUSTOM Beam attention does not support this configuration: {reasons}")
    return backend


def clear_attention_backend_cache() -> None:
    """Clear native selection results while preserving the live GR wrapper."""
    import vllm.v1.attention.selector as selector_mod

    current = selector_mod._cached_get_attn_backend
    native = getattr(current, _ORIGINAL_ATTR, current)
    native.cache_clear()


def apply_custom_attention_backend_patch(force: bool = False) -> int:
    """Preserve CUSTOM intent without importing hardware execution modules.

    ``force`` is accepted only for compatibility with existing callers and has
    no effect. Both values inspect the live selector on every call. Keep an
    installed wrapper; if upstream replaces the target, wrap the new callable
    rather than restoring a stale saved original.
    """
    del force
    try:
        import vllm.v1.attention.selector as selector_mod
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        return 0

    original = selector_mod._cached_get_attn_backend
    if _ORIGINAL_ATTR in getattr(original, "__dict__", {}):
        return 0

    def _patched_cached_get_attn_backend(
        backend: Any,
        attn_selector_config: Any,
        num_heads: int | None = None,
    ) -> Any:
        # Preserve the original CUSTOM import error rather than hiding it.
        if backend == AttentionBackendEnum.CUSTOM:
            return _get_custom_beam_backend(attn_selector_config, num_heads=num_heads)
        if backend is None and _get_current_gr_attention_backend() is GRAttentionBackend.CUSTOM:
            return _get_custom_beam_backend(attn_selector_config, num_heads=num_heads)
        return original(backend, attn_selector_config, num_heads=num_heads)

    setattr(_patched_cached_get_attn_backend, _ORIGINAL_ATTR, original)
    selector_mod._cached_get_attn_backend = _patched_cached_get_attn_backend
    return 1
