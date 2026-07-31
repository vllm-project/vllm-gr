# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Device availability and accelerator-memory cleanup helpers."""

from __future__ import annotations

import torch


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


def release_accelerator_memory() -> None:
    """Release cached CUDA/NPU memory after engine shutdown."""
    from vllm_gr.platforms.memory import cleanup_torch_backend

    cleanup_torch_backend(getattr(torch, "npu", None), collect_ipc=False)
    cleanup_torch_backend(getattr(torch, "cuda", None), collect_ipc=True)
