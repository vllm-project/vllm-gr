# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-neutral helpers for accelerator memory cleanup."""

from __future__ import annotations

from typing import Any

from vllm_gr.logger import init_logger

logger = init_logger(__name__)


def cleanup_torch_backend(backend: Any, *, collect_ipc: bool) -> bool:
    """Best-effort cleanup for one explicitly selected torch backend."""
    if backend is None:
        return False

    is_available = getattr(backend, "is_available", None)
    if callable(is_available):
        try:
            if not is_available():
                return False
        except Exception:
            logger.debug("Backend availability probe failed", exc_info=True)
            return False

    cleaned = False
    for operation_name in ("synchronize", "empty_cache"):
        operation = getattr(backend, operation_name, None)
        if callable(operation):
            try:
                operation()
                cleaned = True
            except Exception:
                logger.debug(
                    "Backend %s failed during cleanup",
                    operation_name,
                    exc_info=True,
                )

    if collect_ipc:
        ipc_collect = getattr(backend, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
                cleaned = True
            except Exception:
                logger.debug("Backend ipc_collect failed during cleanup", exc_info=True)

    return cleaned
