# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os
import sys
from pathlib import Path

from vllm.logger import init_logger

from vllm_gr.engine.arg_utils import patch_add_cli_args
from vllm_gr.entrypoints.beam_search_patch import patch_offline_beam_search
from vllm_gr.entrypoints.openai.beam_search_patch import (
    patch_batch_and_fork,
    patch_beam_search,
    patch_sampling,
)
from vllm_gr.entrypoints.openai.serving_models import patch_OpenAIServingModels_init

logger = init_logger(__name__)

_PATCHED = False


def _load_lmcache_adapter_patch_module():
    module_path = (
        Path(__file__).resolve().parent
        / "3rd_party"
        / "lmcache"
        / "integration"
        / "vllm"
        / "vllm_v1_adapter.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_gr_3rd_party_lmcache_integration_vllm_v1_adapter",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(
            f"Unable to load vllm-gr lmcache adapter patch module from {module_path}"
        )
    spec.loader.exec_module(module)
    return module


def _patch_lmcache_installed_adapter() -> None:
    _load_lmcache_adapter_patch_module().patch_installed_lmcache_adapter()


def _should_patch_lmcache_adapter() -> bool:
    if os.getenv("VLLM_GR_LMCACHE_PATCH", "").lower() in (
        "1",
        "true",
        "yes",
        "y",
    ):
        return True

    return any("LMCacheConnectorV1" in arg for arg in sys.argv)


def patch_lmcache_adapter() -> None:
    """Apply only the lmcache runtime patch.

    This helper is used in engine child processes where only the lmcache
    monkey patch must be installed, without re-running the full vllm_gr
    startup patch sequence.
    """
    _patch_lmcache_installed_adapter()


def run_patch():
    global _PATCHED
    if _PATCHED:
        logger.info("vllm-gr patches have already been applied.")
        return
    _PATCHED = True
    patch_beam_search()
    patch_sampling()
    patch_add_cli_args()
    patch_OpenAIServingModels_init()
    patch_offline_beam_search()
    patch_batch_and_fork()
    if _should_patch_lmcache_adapter():
        _patch_lmcache_installed_adapter()
