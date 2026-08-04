# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime installer for the bundled LMCache compatibility adapter."""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

from vllm.logger import init_logger

logger = init_logger(__name__)

_ADAPTER_MODULE_NAME = "vllm_gr_3rd_party_lmcache_integration_vllm_v1_adapter"
_LMCACHE_ADAPTER_PATCHED = False


def is_lmcache_adapter_requested(
    environ: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
) -> bool:
    """Return whether this process requested the LMCache compatibility patch."""
    environment = os.environ if environ is None else environ
    arguments = sys.argv if argv is None else argv
    return environment.get("VLLM_GR_LMCACHE_PATCH", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    } or any("LMCacheConnectorV1" in argument for argument in arguments)


def _load_lmcache_adapter() -> ModuleType | None:
    adapter_path = (
        Path(__file__).resolve().parent
        / "3rd_party"
        / "lmcache"
        / "integration"
        / "vllm"
        / "vllm_v1_adapter.py"
    )
    if not adapter_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(_ADAPTER_MODULE_NAME, adapter_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_lmcache_adapter() -> bool:
    """Apply the LMCache adapter and report whether this call installed it."""
    global _LMCACHE_ADAPTER_PATCHED

    if _LMCACHE_ADAPTER_PATCHED:
        return False

    module = _load_lmcache_adapter()
    if module is None:
        return False

    if not module.patch_installed_lmcache_adapter():
        return False

    _LMCACHE_ADAPTER_PATCHED = True
    logger.debug("vllm_gr: Applied LMCache adapter patch")
    return True
