# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""NPU Beam attention cache layout."""

from __future__ import annotations

from typing import Any


def restore_npu_cache_pool_layout(tensor: Any, request_width: int) -> Any:
    """Restore ``[B, W, ...]`` before the xllm cache-write operator."""

    if request_width < 1:
        raise ValueError("request_width must be positive")
    if tensor.shape[0] != 1 or tensor.shape[1] == request_width:
        return tensor
    if tensor.shape[1] % request_width:
        raise RuntimeError("flattened NPU pool width is not divisible by Beam width")
    batch = tensor.shape[1] // request_width
    return tensor.view(batch, request_width, *tensor.shape[2:])
