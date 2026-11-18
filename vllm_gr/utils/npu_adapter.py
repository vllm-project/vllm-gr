# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Device availability and accelerator-memory cleanup helpers."""

from __future__ import annotations

import torch

from vllm_gr.logger import init_logger

logger = init_logger(__name__)


def is_npu_available() -> bool:
    """Return whether torch_npu is installed and the runtime is available."""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def is_cuda_available() -> bool:
    """Return whether CUDA is available."""
    try:
        return torch.cuda.is_available()
    except AttributeError:
        return False


def _cleanup_torch_backend(backend, *, collect_ipc: bool) -> bool:
    """Best-effort cleanup for a torch accelerator backend."""
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
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize):
        try:
            synchronize()
            cleaned = True
        except Exception:
            logger.debug("Backend synchronize failed during cleanup", exc_info=True)

    empty_cache = getattr(backend, "empty_cache", None)
    if callable(empty_cache):
        try:
            empty_cache()
            cleaned = True
        except Exception:
            logger.debug("Backend empty_cache failed during cleanup", exc_info=True)

    if collect_ipc:
        ipc_collect = getattr(backend, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
                cleaned = True
            except Exception:
                logger.debug("Backend ipc_collect failed during cleanup", exc_info=True)

    return cleaned


def release_accelerator_memory() -> None:
    """Release cached CUDA/NPU memory after engine shutdown."""
    npu = getattr(torch, "npu", None)
    try:
        npu_initialized = npu is not None and npu.is_initialized()
    except Exception:
        logger.debug("NPU initialization probe failed during cleanup", exc_info=True)
    else:
        # synchronize() would lazily initialize an unused NPU runtime.
        if npu_initialized:
            _cleanup_torch_backend(npu, collect_ipc=False)
    _cleanup_torch_backend(getattr(torch, "cuda", None), collect_ipc=True)
