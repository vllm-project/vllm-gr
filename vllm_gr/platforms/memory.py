# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-neutral helpers for Provider-owned accelerator cleanup."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def cleanup_torch_backend(backend: Any, *, collect_ipc: bool) -> bool:
    """Best-effort cleanup for one torch accelerator backend."""
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
    for method_name in ("synchronize", "empty_cache"):
        method = getattr(backend, method_name, None)
        if callable(method):
            try:
                method()
                cleaned = True
            except Exception:
                logger.debug(
                    "Backend %s failed during cleanup",
                    method_name,
                    exc_info=True,
                )

    ipc_collect = getattr(backend, "ipc_collect", None)
    if collect_ipc and callable(ipc_collect):
        try:
            ipc_collect()
            cleaned = True
        except Exception:
            logger.debug("Backend ipc_collect failed during cleanup", exc_info=True)

    return cleaned
