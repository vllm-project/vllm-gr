# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Monkey-patch mechanism for vllm-gr.

After removing the GR adapter / recommend runtime, the only patches applied
here are the ones still used by the OneRec beam-search path and by the NPU
runtime:

- :func:`_apply_gr_beam_patches` — offline / OpenAI beam-search and
  catalog-constrained generation support.

The previous ``_apply_gr_core_patches`` that replaced vLLM core types
(``EngineCoreRequest`` / ``Request`` / ``NewRequestData``) with GR-extended
variants has been removed together with the recommend runtime.
"""

import logging

logger = logging.getLogger("vllm_gr.patch")


def _is_npu_platform() -> bool:
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


_GR_BEAM_PATCHES_APPLIED = False
_ASCEND_ATTN_BACKEND_PATCH_APPLIED = False
_ASCEND_ORIGINAL_CACHED_GET_ATTN = None


def _apply_ascend_attention_backend_patch(force: bool = False) -> int:
    """Preserve CUSTOM attention backend on all platforms.

    The vllm-ascend platform plugin ignores any user-specified attention
    backend (other than FLASH_ATTN) in its ``get_attn_backend_cls()``, so
    ``get_attn_backend`` would return an Ascend-specific backend even when the
    config explicitly requests ``AttentionBackendEnum.CUSTOM``.

    To avoid relying on an environment variable, this patch is applied
    **unconditionally** on every platform.  It is a no-op whenever
    ``backend != AttentionBackendEnum.CUSTOM``, so on GPU the CUDA platform's
    native handling is preserved.  When ``backend == CUSTOM`` the patch returns
    ``BeamAttentionBackend`` directly, bypassing any platform-specific
    ``get_attn_backend_cls``.
    """
    global _ASCEND_ATTN_BACKEND_PATCH_APPLIED
    global _ASCEND_ORIGINAL_CACHED_GET_ATTN

    if _ASCEND_ATTN_BACKEND_PATCH_APPLIED and not force:
        return 0

    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        return 0

    # Ensure BeamAttentionBackend is registered under CUSTOM.
    try:
        import vllm_gr.v1.attention.backends.beam_attn  # noqa: F401
    except Exception:
        return 0

    if not AttentionBackendEnum.CUSTOM.is_overridden():
        _ASCEND_ATTN_BACKEND_PATCH_APPLIED = True
        return 0

    patch_count = 0

    # --- Patch selector._cached_get_attn_backend -----------------------
    try:
        import vllm.v1.attention.selector as selector_mod
    except ImportError:
        return 0

    # Save the original on first call to avoid double-wrapping on re-apply.
    if _ASCEND_ORIGINAL_CACHED_GET_ATTN is None:
        _ASCEND_ORIGINAL_CACHED_GET_ATTN = selector_mod._cached_get_attn_backend

    original_cached_fn = _ASCEND_ORIGINAL_CACHED_GET_ATTN

    def _patched_cached_get_attn_backend(backend, attn_selector_config, num_heads=None):
        """Return BeamAttentionBackend when backend is CUSTOM."""
        if backend == AttentionBackendEnum.CUSTOM:
            from vllm_gr.v1.attention.backends.beam_attn import (
                BeamAttentionBackend,
            )

            # Replicate the KV cache layout adjustment from the original
            # selector so that backends requiring a specific layout still
            # get it applied.
            required_layout = BeamAttentionBackend.get_required_kv_cache_layout()
            if required_layout is not None:
                from vllm.v1.attention.backends.utils import (
                    set_kv_cache_layout,
                )

                set_kv_cache_layout(required_layout)

            return BeamAttentionBackend

        # On Ascend NPU the platform plugin resets attention_config.backend to
        # None inside VllmConfig.__post_init__.  The reset happens before the
        # config is serialized to child processes, so EngineCore and Worker
        # processes see backend=None even though the user requested CUSTOM.
        # We store a marker in additional_config (untouched by the platform)
        # and use it as a fallback here.
        if backend is None:
            try:
                from vllm.config import get_current_vllm_config_or_none

                vllm_config = get_current_vllm_config_or_none()
                if (
                    vllm_config is not None
                    and getattr(vllm_config, "additional_config", None)
                    and isinstance(vllm_config.additional_config, dict)
                    and vllm_config.additional_config.get("vllm_gr_attention_backend") == "CUSTOM"
                ):
                    from vllm_gr.v1.attention.backends.beam_attn import (
                        BeamAttentionBackend,
                    )

                    required_layout = BeamAttentionBackend.get_required_kv_cache_layout()
                    if required_layout is not None:
                        from vllm.v1.attention.backends.utils import (
                            set_kv_cache_layout,
                        )

                        set_kv_cache_layout(required_layout)

                    return BeamAttentionBackend
            except Exception:
                pass

        return original_cached_fn(backend, attn_selector_config, num_heads=num_heads)

    selector_mod._cached_get_attn_backend = _patched_cached_get_attn_backend
    patch_count += 1

    _ASCEND_ATTN_BACKEND_PATCH_APPLIED = True
    return patch_count


def _apply_gr_beam_patches(force: bool = False) -> int:
    """Apply beam-search related patches from vllm-gr.

    These patches are independent from the GR core type replacement patches
    and can be safely applied alongside them.
    """
    global _GR_BEAM_PATCHES_APPLIED

    if _GR_BEAM_PATCHES_APPLIED and not force:
        return 0

    patch_count = 0

    try:
        from vllm_gr.logprobs_patch import patch_flat_logprobs

        patch_flat_logprobs()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_flat_logprobs")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_flat_logprobs: %s", exc)

    try:
        from vllm_gr.entrypoints.openai.beam_search_patch import (
            patch_beam_search,
        )

        patch_beam_search()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_beam_search")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_beam_search: %s", exc)

    try:
        from vllm_gr.entrypoints.openai.beam_search_patch import patch_sampling

        patch_sampling()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_sampling")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_sampling: %s", exc)

    try:
        from vllm_gr.engine.arg_utils_gr import patch_add_cli_args

        patch_add_cli_args()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_add_cli_args")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_add_cli_args: %s", exc)

    try:
        from vllm_gr.entrypoints.openai.serving_models import (
            patch_OpenAIServingModels_init,
        )

        patch_OpenAIServingModels_init()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_OpenAIServingModels_init")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_OpenAIServingModels_init: %s", exc)

    try:
        from vllm_gr.entrypoints.openai.beam_search_patch import patch_batch_and_fork

        patch_batch_and_fork()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_batch_and_fork")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_batch_and_fork: %s", exc)

    # Optional: LMCache adapter patch
    try:
        import importlib.util
        from pathlib import Path

        adapter_path = (
            Path(__file__).resolve().parent
            / "3rd_party"
            / "lmcache"
            / "integration"
            / "vllm"
            / "vllm_v1_adapter.py"
        )
        if adapter_path.exists():
            import os
            import sys

            should_patch = False
            if os.getenv("VLLM_GR_LMCACHE_PATCH", "").lower() in (
                "1",
                "true",
                "yes",
                "y",
            ):
                should_patch = True
            elif any("LMCacheConnectorV1" in arg for arg in sys.argv):
                should_patch = True

            if should_patch:
                spec = importlib.util.spec_from_file_location(
                    "vllm_gr_3rd_party_lmcache_integration_vllm_v1_adapter",
                    adapter_path,
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.patch_installed_lmcache_adapter()
                patch_count += 1
                logger.debug("vllm_gr: Applied LMCache adapter patch")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping LMCache adapter patch: %s", exc)

    _GR_BEAM_PATCHES_APPLIED = True

    if patch_count > 0:
        logger.info(
            "vllm_gr: Applied %d beam-search monkey patch(es)",
            patch_count,
        )
    return patch_count


def patch_lmcache_adapter():
    """Apply LMCache adapter patch.

    Public helper used by engine_core_patch.py in child processes where
    only the LMCache monkey patch must be installed, without re-running
    the full patch sequence.
    """
    import importlib.util
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parent
        / "3rd_party"
        / "lmcache"
        / "integration"
        / "vllm"
        / "vllm_v1_adapter.py"
    )
    if not module_path.exists():
        return
    spec = importlib.util.spec_from_file_location(
        "vllm_gr_3rd_party_lmcache_integration_vllm_v1_adapter",
        module_path,
    )
    if spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_installed_lmcache_adapter()


_NPU_WORKER_PATCH_APPLIED = False


def _patch_npu_worker_init_device() -> int:
    """Wrap NPUWorker.init_device to call re_apply_patches() first.

    When vllm-ascend overrides ``worker_cls`` to ``NPUWorker`` (bypassing
    ``GRNPUWorker``), ``re_apply_patches()`` is not called in worker processes.
    NPUWorker instantiates NPUModelRunner inside ``init_device()``, and model
    layer creation calls ``get_attn_backend()``.  We therefore run
    ``re_apply_patches()`` **before** the original ``init_device()`` so the
    selector patch is in place before any attention backend is resolved.
    """
    global _NPU_WORKER_PATCH_APPLIED

    if not _is_npu_platform():
        return 0

    if _NPU_WORKER_PATCH_APPLIED:
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

    def _patched_init_device(self):
        # Apply patches BEFORE original init_device() because NPUWorker
        # instantiates NPUModelRunner inside init_device(), and model
        # layer creation calls get_attn_backend().
        re_apply_patches()
        return original_init_device(self)

    NPUWorker.init_device = _patched_init_device
    _NPU_WORKER_PATCH_APPLIED = True
    return 1


def _apply_patches(force: bool = False) -> int:
    """Apply all runtime patches needed by vllm-gr."""
    ascend_attn_count = _apply_ascend_attention_backend_patch(force=force)
    # Import/patch NPUWorker before the beam patch tries to access
    # NPUModelRunner. vllm-ascend's worker module deliberately initializes
    # vllm_ascend.ops first, avoiding an import-order cycle through device_op.
    npu_worker_count = _patch_npu_worker_init_device()
    beam_count = _apply_gr_beam_patches(force=force)
    return ascend_attn_count + beam_count + npu_worker_count


def re_apply_patches() -> int:
    """Re-scan sys.modules and patch any newly-loaded vLLM modules."""
    return _apply_patches(force=True)


def apply_patches_if_needed() -> int:
    """Apply patches if not yet applied. Safe to call multiple times."""
    return _apply_patches()
