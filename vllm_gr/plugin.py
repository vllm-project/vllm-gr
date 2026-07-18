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

import logging

logger = logging.getLogger("vllm_gr.plugin")


def initialize_runtime() -> dict[str, int]:
    """Initialize the current process for vllm-gr runtime usage.

    Safe to call repeatedly. The underlying patch helper is idempotent, so
    this can be used both by the vLLM plugin loader and by explicit engine
    startup paths in the main process.
    """
    from vllm_gr._env_check import check_environment, configure_model_runner_default

    configure_model_runner_default()

    from vllm_gr.patch import re_apply_patches

    check_environment()

    patch_count = re_apply_patches()

    state = {"patch_count": patch_count}
    logger.debug("vllm-gr runtime initialized: patches=%d", patch_count)
    return state


def register() -> None:
    """Entry point used by vLLM's general plugin loader."""
    state = initialize_runtime()
    logger.debug("vllm-gr plugin loaded (%d patches applied)", state["patch_count"])
