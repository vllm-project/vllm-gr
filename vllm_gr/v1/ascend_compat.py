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


def _try_import_npu_worker() -> Any | None:
    """Import the v1 NPUWorker class from vllm-ascend.

    Current vllm-ascend defines ``NPUWorker`` in ``worker.worker_v1``; older
    builds exposed it from ``worker.worker``.  Try both, returning ``None``
    when neither is importable.
    """
    for module_path in (
        "vllm_ascend.worker.worker_v1",
        "vllm_ascend.worker.worker",
    ):
        try:
            module = __import__(module_path, fromlist=["NPUWorker"])
            worker_cls = getattr(module, "NPUWorker", None)
            if worker_cls is not None:
                return worker_cls
        except (ImportError, AttributeError):
            continue
    return None


def patch_npu_worker_init_device() -> int:
    """Run vllm-gr patch installation before NPUWorker device initialization."""
    global _NPU_WORKER_PATCH_APPLIED

    if not is_npu_platform() or _NPU_WORKER_PATCH_APPLIED:
        return 0

    # The v1 NPUWorker lives in worker_v1 in current vllm-ascend; older builds
    # expose it from worker.  Try the v1 path first, then fall back.
    NPUWorker = _try_import_npu_worker()
    if NPUWorker is None:
        logger.warning(
            "Failed to import NPUWorker from vllm_ascend.worker.worker_v1 or "
            "vllm_ascend.worker.worker; the vllm-gr delayed NPU patch retry "
            "hook was not installed for this attempt."
        )
        return 0

    original_init_device = NPUWorker.init_device

    def _patched_init_device(self: Any) -> Any:
        from vllm_gr.patch import re_apply_patches

        re_apply_patches()
        # Mount set_gr_constraint_resources so the driver can inject constraint
        # tables via EngineCore.collective_rpc after init_device returns.
        from vllm_gr.constraints.worker_inject import load_constraint_tables_on_worker
        from vllm_gr.v1.worker.model_runner_patch import (
            install_worker_utility_hooks,
        )

        install_worker_utility_hooks(type(self))
        result = original_init_device(self)
        # Preload constraint tables onto the device at weight-load time so the
        # runtime never does a lazy build/H2D on the first constrained step.
        load_constraint_tables_on_worker(self)
        return result

    NPUWorker.init_device = _patched_init_device
    _NPU_WORKER_PATCH_APPLIED = True
    return 1
