# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.
# Licensed under the Apache License, Version 2.0


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch reference implementation of timestamp positional embeddings."""

from __future__ import annotations

from typing import Optional

import torch

from vllm_gr.models.dlrm_v3.common import (
    dense_to_jagged,
    fx_unwrap_optional_tensor,
    jagged_to_padded_dense,
)
from vllm_gr.utils.npu_adapter import get_device_adapter


@torch.fx.wrap
def _get_col_indices(
    max_seq_len: int,
    max_contextual_seq_len: int,
    max_pos_ind: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
) -> torch.Tensor:
    batch_size = seq_lengths.size(0)
    col_indices = torch.arange(max_seq_len, device=seq_lengths.device).expand(
        batch_size, max_seq_len
    )
    if num_targets is not None:
        if interleave_targets:
            high_inds = seq_lengths - fx_unwrap_optional_tensor(num_targets) * 2
        else:
            high_inds = seq_lengths - fx_unwrap_optional_tensor(num_targets)
        col_indices = torch.clamp(col_indices, max=high_inds.view(-1, 1))
        col_indices = high_inds.view(-1, 1) - col_indices
    else:
        col_indices = seq_lengths.view(-1, 1) - col_indices
    col_indices = col_indices + max_contextual_seq_len
    col_indices = torch.clamp(col_indices, max=max_pos_ind - 1)
    if max_contextual_seq_len > 0:
        col_indices[:, :max_contextual_seq_len] = torch.arange(
            0,
            max_contextual_seq_len,
            device=col_indices.device,
            dtype=col_indices.dtype,
        ).view(1, -1)
    return col_indices


def pytorch_add_timestamp_positional_embeddings(
    seq_embeddings: torch.Tensor,
    seq_offsets: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    max_seq_len: int,
    max_contextual_seq_len: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
) -> torch.Tensor:
    max_pos_ind = pos_embeddings.size(0)
    pos_inds = _get_col_indices(
        max_seq_len=max_seq_len,
        max_contextual_seq_len=max_contextual_seq_len,
        max_pos_ind=max_pos_ind,
        seq_lengths=seq_lengths,
        num_targets=num_targets,
        interleave_targets=interleave_targets,
    )
    batch_size, _ = pos_inds.shape

    num_time_buckets = ts_embeddings.size(1) - 1
    time_bucket_increments = 60.0
    time_bucket_divisor = 1.0
    time_delta = 0

    timestamps = jagged_to_padded_dense(
        values=timestamps.unsqueeze(-1),
        offsets=[seq_offsets],
        max_lengths=[max_seq_len],
        padding_value=0,
    ).squeeze(-1)
    query_time = torch.gather(
        timestamps,
        dim=1,
        index=(seq_lengths - 1).unsqueeze(1).clamp(min=0),
    )
    ts = query_time - timestamps
    ts = ts + time_delta
    ts = ts.clamp(min=1e-6) / time_bucket_increments
    if time_bucket_fn == "log":
        ts = torch.log(ts)
    else:
        ts = torch.sqrt(ts)
    ts = (ts / time_bucket_divisor).clamp(min=0).int()
    ts = torch.clamp(
        ts,
        min=0,
        max=num_time_buckets,
    )

    position_embeddings = torch.index_select(
        pos_embeddings, 0, pos_inds.reshape(-1)
    ).view(batch_size, max_seq_len, -1)
    time_embeddings = torch.index_select(
        ts_embeddings, 0, ts.reshape(-1)
    ).view(batch_size, max_seq_len, -1)
    dense_add = (time_embeddings + position_embeddings).to(seq_embeddings.dtype)

    adapter = get_device_adapter()
    if adapter.should_use_fbgemm(seq_embeddings.device):
        try:
            return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
                seq_embeddings,
                [seq_offsets],
                dense_add,
            )[0]
        except (RuntimeError, AttributeError) as exc:
            raise RuntimeError(
                "torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output "
                f"failed on accelerated device {torch.device(seq_embeddings.device)}. "
                "CUDA/NPU paths require a working fbgemm backend and do not "
                "fall back to the PyTorch reference implementation."
            ) from exc

    dense_embeddings = jagged_to_padded_dense(
        values=seq_embeddings,
        offsets=[seq_offsets],
        max_lengths=[max_seq_len],
        padding_value=0.0,
    )
    if dense_embeddings.dim() == 2:
        dense_embeddings = dense_embeddings.unsqueeze(-1)
    dense_embeddings = dense_embeddings + dense_add
    return dense_to_jagged(dense_embeddings, [seq_offsets])
