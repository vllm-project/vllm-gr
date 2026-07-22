# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU process initialization for vLLM-GR."""

from __future__ import annotations

import logging
import threading

from vllm_gr.platform.npu.attention_backend import install_npu_attention_backend_compat
from vllm_gr.platform.npu.capabilities import is_npu_platform

logger = logging.getLogger(__name__)

_WORKER_PATCH_LOCK = threading.Lock()
_PATCHED_WORKER_CLASS = None


def _install_npu_worker_attention_hook() -> int:
    """Install the attention bridge before NPUWorker creates its ModelRunner."""
    if not is_npu_platform():
        return 0

    try:
        from vllm_ascend.worker.worker import NPUWorker
    except ImportError as exc:
        logger.warning(
            "Could not install the vLLM-GR NPU worker attention hook: %s",
            exc,
        )
        return 0

    global _PATCHED_WORKER_CLASS
    with _WORKER_PATCH_LOCK:
        if _PATCHED_WORKER_CLASS is NPUWorker:
            return 0
        if _PATCHED_WORKER_CLASS is not None:
            raise RuntimeError("NPUWorker changed after the vLLM-GR attention hook was installed")

        original_init_device = NPUWorker.init_device

        def init_device_with_gr_attention(self):
            install_npu_attention_backend_compat(force=True)
            return original_init_device(self)

        NPUWorker.init_device = init_device_with_gr_attention
        _PATCHED_WORKER_CLASS = NPUWorker
        return 1


def install_npu_runtime_compat(*, force: bool = False) -> int:
    """Install NPU-only runtime compatibility without touching GPU paths."""
    if not is_npu_platform():
        return 0
    attention_count = install_npu_attention_backend_compat(force=force)
    worker_count = _install_npu_worker_attention_hook()
    return attention_count + worker_count
