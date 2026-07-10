# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM general plugin entry point for vllm-gr.

Registered via ``pyproject.toml`` under ``[project.entry-points."vllm.general_plugins"]``.
vLLM calls ``load_general_plugins()`` in every process (main, EngineCore subprocess,
Worker subprocess), which discovers and executes this entry point.

This module is the formal runtime initialization boundary for vllm-gr.
After removing the GR adapter / recommend runtime, its only responsibility
is to apply the vLLM monkey patches needed by the OneRec beam-search path
(catalog-constrained generation) and by the NPU runtime (Yuanrong KV
transfer, paged-KV cache merge, split-QKV-RMSNorm-RoPE fusion).
"""

import logging
import os

logger = logging.getLogger("vllm_gr.plugin")


def initialize_runtime() -> dict[str, int]:
    """Initialize the current process for vllm-gr runtime usage.

    Safe to call repeatedly. The underlying patch helper is idempotent, so
    this can be used both by the vLLM plugin loader and by explicit engine
    startup paths in the main process.
    """
    # vllm-gr's grouped-beam patches currently target the V1 model runner.
    # The V2 runner (default in vLLM 0.22.1+) uses a different internal layout
    # and is not yet supported. Force V1 unless the user explicitly opted in.
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is None:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        logger.debug("vllm-gr: disabled V2 model runner for grouped beam compatibility")
    elif os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "").lower() in ("1", "true", "yes"):
        logger.warning(
            "vllm-gr: VLLM_USE_V2_MODEL_RUNNER is enabled, but vllm-gr's "
            "grouped beam-search patches only support the V1 model runner. "
            "Beam search may fail or produce incorrect logprobs."
        )

    from vllm_gr.patch import re_apply_patches

    patch_count = re_apply_patches()

    state = {"patch_count": patch_count}
    logger.debug("vllm-gr runtime initialized: patches=%d", patch_count)
    return state


def register() -> None:
    """Entry point used by vLLM's general plugin loader."""
    state = initialize_runtime()
    logger.debug("vllm-gr plugin loaded (%d patches applied)", state["patch_count"])
