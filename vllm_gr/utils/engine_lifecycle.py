# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sync engine shutdown helpers for GR and vLLM LLM passthrough paths."""

from __future__ import annotations

from typing import Any

from vllm_gr.logger import init_logger

logger = init_logger(__name__)


def shutdown_sync_engine(engine: Any) -> None:
    """Shut down a sync engine from either GRLLMEngine or vLLM LLMEngine."""
    if engine is None:
        return

    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return

    engine_core = getattr(engine, "engine_core", None)
    if engine_core is not None:
        core_shutdown = getattr(engine_core, "shutdown", None)
        if callable(core_shutdown):
            core_shutdown()
            return

    logger.warning(
        "Engine %s has no shutdown method; resources may not be released",
        type(engine).__name__,
    )


def shutdown_grllm(llm: Any) -> None:
    """Delegate shutdown to ``llm.shutdown()`` when available."""
    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return

    logger.warning(
        "Object %s has no shutdown method; cannot release resources",
        type(llm).__name__,
    )
