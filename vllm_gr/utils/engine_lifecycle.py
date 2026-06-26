# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sync engine shutdown helpers for the vLLM LLM passthrough path."""

from __future__ import annotations

from typing import Any

from vllm_gr.logger import init_logger

logger = init_logger(__name__)


def shutdown_sync_engine(engine: Any) -> None:
    """Shut down a sync engine (vLLM LLMEngine)."""
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
