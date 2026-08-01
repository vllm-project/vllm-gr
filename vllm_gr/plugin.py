# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM general plugin entry point for vllm-gr.

Registered via ``pyproject.toml`` under ``[project.entry-points."vllm.general_plugins"]``.
vLLM calls ``load_general_plugins()`` in every process (main, EngineCore subprocess,
Worker subprocess), which discovers and executes this entry point.

This module is the formal runtime initialization boundary for vllm-gr.
After removing the GR adapter / recommend runtime, its only responsibility
is to apply the vLLM monkey patches needed by the OneRec beam-search path
(catalog-constrained generation).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_gr.platforms.protocol import GRBackendProvider

logger = logging.getLogger("vllm_gr.plugin")


def initialize_process_backend() -> GRBackendProvider:
    """Activate and validate the backend for the current process."""
    from vllm_gr._env_check import configure_model_runner_default

    configure_model_runner_default()

    from vllm_gr.platforms import activate_current_backend

    backend = activate_current_backend()

    # CUSTOM Attention registration is process-local, just like backend
    # activation. Register it explicitly for main, EngineCore, and Worker
    # processes before any backend compatibility patch inspects that slot.
    from vllm_gr.v1.attention.backends.beam_attn_registration import (
        register_beam_attention_backend,
    )

    register_beam_attention_backend()
    backend.check_environment()
    return backend


def initialize_runtime() -> dict[str, object]:
    """Initialize the current process for vllm-gr runtime usage.

    Safe to call repeatedly. The underlying patch helper is idempotent, so
    this can be used both by the vLLM plugin loader and by explicit engine
    startup paths in the main process.
    """
    backend = initialize_process_backend()

    from vllm_gr.patch import get_last_beam_patch_report, re_apply_patches

    patch_count = re_apply_patches()

    report = get_last_beam_patch_report()
    state: dict[str, object] = {
        "backend": backend.device_type,
        "patch_count": patch_count,
        "patches": report.statuses() if report is not None else {},
    }
    logger.debug(
        "vllm-gr runtime initialized: backend=%s patches=%d",
        backend.device_type,
        patch_count,
    )
    return state


def register() -> None:
    """Entry point used by vLLM's general plugin loader."""
    state = initialize_runtime()
    logger.debug(
        "vllm-gr plugin loaded (backend=%s, %s patches applied)",
        state["backend"],
        state["patch_count"],
    )
