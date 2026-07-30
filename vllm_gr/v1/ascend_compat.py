# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility patches for vLLM-Ascend integration points."""

import logging
from typing import Any, cast

from vllm_gr.config import GRAttentionBackend, GRConfig

logger = logging.getLogger(__name__)

_ASCEND_ATTN_BACKEND_PATCH_APPLIED = False
_ASCEND_ORIGINAL_CACHED_GET_ATTN: Any = None
_NPU_WORKER_PATCH_APPLIED = False


def _get_current_gr_attention_backend() -> GRAttentionBackend | None:
    """Read the validated GR backend marker used by the attention selector."""
    from vllm.config import get_current_vllm_config_or_none

    return GRConfig.from_vllm_config(get_current_vllm_config_or_none()).attention_backend


def is_npu_platform() -> bool:
    """Return whether the active vLLM platform is NPU."""
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


def _get_custom_beam_backend() -> type:
    """Return the registered Beam backend and apply its required cache layout."""
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    required_layout = BeamAttentionBackend.get_required_kv_cache_layout()
    if required_layout is not None:
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout(required_layout)
    return cast(type[Any], BeamAttentionBackend)


def apply_ascend_attention_backend_patch(force: bool = False) -> int:
    """Preserve explicit and marker-restored CUSTOM backend selection."""
    global _ASCEND_ATTN_BACKEND_PATCH_APPLIED
    global _ASCEND_ORIGINAL_CACHED_GET_ATTN

    if _ASCEND_ATTN_BACKEND_PATCH_APPLIED and not force:
        return 0

    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        return 0

    try:
        import vllm_gr.v1.attention.backends.beam_attn  # noqa: F401
    except Exception:
        return 0

    if not AttentionBackendEnum.CUSTOM.is_overridden():
        _ASCEND_ATTN_BACKEND_PATCH_APPLIED = True
        return 0

    try:
        import vllm.v1.attention.selector as selector_mod
    except ImportError:
        return 0

    if _ASCEND_ORIGINAL_CACHED_GET_ATTN is None:
        _ASCEND_ORIGINAL_CACHED_GET_ATTN = selector_mod._cached_get_attn_backend
    original_cached_fn = _ASCEND_ORIGINAL_CACHED_GET_ATTN

    def _patched_cached_get_attn_backend(
        backend: Any,
        attn_selector_config: Any,
        num_heads: int | None = None,
    ) -> Any:
        if backend == AttentionBackendEnum.CUSTOM:
            return _get_custom_beam_backend()

        if backend is None and _get_current_gr_attention_backend() is GRAttentionBackend.CUSTOM:
            return _get_custom_beam_backend()

        return original_cached_fn(
            backend,
            attn_selector_config,
            num_heads=num_heads,
        )

    selector_mod._cached_get_attn_backend = _patched_cached_get_attn_backend
    _ASCEND_ATTN_BACKEND_PATCH_APPLIED = True
    return 1


def patch_npu_worker_init_device() -> int:
    """Run vllm-gr patch installation before NPUWorker device initialization."""
    global _NPU_WORKER_PATCH_APPLIED

    if not is_npu_platform() or _NPU_WORKER_PATCH_APPLIED:
        return 0

    try:
        from vllm_ascend.worker.worker import NPUWorker
    except ImportError as exc:
        logger.warning(
            "Failed to import vllm_ascend.worker.worker.NPUWorker; the "
            "vllm-gr delayed NPU patch retry hook was not installed for this "
            "attempt. ImportError: %s",
            exc,
        )
        logger.debug("NPUWorker import traceback", exc_info=True)
        return 0

    original_init_device = NPUWorker.init_device

    def _patched_init_device(self: Any) -> Any:
        from vllm_gr.patch import re_apply_patches

        re_apply_patches()
        return original_init_device(self)

    NPUWorker.init_device = _patched_init_device
    _NPU_WORKER_PATCH_APPLIED = True
    return 1
