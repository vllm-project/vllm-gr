# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

"""Timestamp positional embedding operator entrypoint.

The public API still accepts ``kernel`` for upstream compatibility, but this
wrapper no longer performs backend dispatch itself. Runtime selection now
happens inside the PyTorch reference path based on device/backend capability:

- CUDA/NPU with working fbgemm jagged-add support use the accelerated op
- CPU or environments without that op keep the dense PyTorch fallback
"""

from __future__ import annotations

from typing import Optional

import torch

from vllm_gr.models.dlrm_v3.common import HammerKernel
from vllm_gr.ops.pytorch.pt_position import (
    pytorch_add_timestamp_positional_embeddings,
)


def add_timestamp_positional_embeddings(
    alpha: float,
    max_seq_len: int,
    max_contextual_seq_len: int,
    position_embeddings_weight: torch.Tensor,
    timestamp_embeddings_weight: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str = "sqrt",
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """Apply timestamp positional embeddings through the compatibility wrapper.

    ``kernel`` is preserved so existing call sites do not break, but it is only
    a compatibility hint in this fork and does not directly choose the backend
    implementation at this layer.
    """
    _ = kernel
    assert time_bucket_fn in ["sqrt", "log"]
    seq_embeddings = seq_embeddings * alpha
    return pytorch_add_timestamp_positional_embeddings(
        seq_embeddings=seq_embeddings,
        seq_offsets=seq_offsets,
        pos_embeddings=position_embeddings_weight,
        ts_embeddings=timestamp_embeddings_weight,
        timestamps=timestamps,
        max_seq_len=max_seq_len,
        max_contextual_seq_len=max_contextual_seq_len,
        seq_lengths=seq_lengths,
        num_targets=num_targets,
        interleave_targets=interleave_targets,
        time_bucket_fn=time_bucket_fn,
    )
