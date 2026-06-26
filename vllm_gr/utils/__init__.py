# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility helpers for backend detection and safe operator dispatch.

Convenience re-exports for the few symbols that are still consumed via the
``vllm_gr.utils`` top level; everything else should be imported directly from
its defining submodule (``vllm_gr.utils.npu_adapter`` etc.).
"""

from vllm_gr.utils.npu_adapter import (
    is_cuda_available,
    is_npu_available,
    release_accelerator_memory,
)
from vllm_gr.utils.reliable_ops import ReliableOps

__all__ = [
    "ReliableOps",
    "is_cuda_available",
    "is_npu_available",
    "release_accelerator_memory",
]
