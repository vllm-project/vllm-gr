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

_GR_BEAM_PATCHES_APPLIED = False


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
        from vllm_gr.v1.attention.backends import register_beam_attention_backend

        patch_count += register_beam_attention_backend()
    except Exception as exc:
        logger.debug("vllm_gr: Could not register CUSTOM attention backend: %s", exc)

    try:
        from vllm_gr.logprobs_patch import patch_flat_logprobs

        patch_flat_logprobs()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_flat_logprobs")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_flat_logprobs: %s", exc)

    try:
        from vllm_gr.entrypoints.openai.beam_search_patch import patch_beam_search

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


def _apply_patches(force: bool = False) -> int:
    """Apply all runtime patches needed by vllm-gr."""
    from vllm_gr.platform.npu.runtime import install_npu_runtime_compat

    npu_compat_count = install_npu_runtime_compat(force=force)
    beam_count = _apply_gr_beam_patches(force=force)
    return npu_compat_count + beam_count


def re_apply_patches() -> int:
    """Re-scan sys.modules and patch any newly-loaded vLLM modules."""
    return _apply_patches(force=True)


def apply_patches_if_needed() -> int:
    """Apply patches if not yet applied. Safe to call multiple times."""
    return _apply_patches()
