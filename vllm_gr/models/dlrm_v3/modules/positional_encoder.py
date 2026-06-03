# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

"""HSTU Positional Encoder"""

from math import sqrt
from typing import Optional

import torch

from vllm_gr.models.dlrm_v3.common import HammerModule
from vllm_gr.ops.position import add_timestamp_positional_embeddings


class HSTUPositionalEncoder(HammerModule):
    """
    Positional encoder for HSTU model.
    Adds position and timestamp embeddings to sequence embeddings.
    """

    def __init__(
        self,
        num_position_buckets: int,
        num_time_buckets: int,
        embedding_dim: int,
        contextual_seq_len: int,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._contextual_seq_len: int = contextual_seq_len

        self._position_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_position_buckets, embedding_dim).uniform_(
                -sqrt(1.0 / num_position_buckets),
                sqrt(1.0 / num_position_buckets),
            ),
        )
        self._timestamp_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_time_buckets + 1, embedding_dim).uniform_(
                -sqrt(1.0 / num_time_buckets),
                sqrt(1.0 / num_time_buckets),
            ),
        )

    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Add positional and timestamp embeddings using golden semantics."""
        return add_timestamp_positional_embeddings(
            alpha=self._embedding_dim**0.5,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=self._contextual_seq_len,
            position_embeddings_weight=self._position_embeddings_weight,
            timestamp_embeddings_weight=self._timestamp_embeddings_weight,
            seq_offsets=seq_offsets,
            seq_lengths=seq_lengths,
            seq_embeddings=seq_embeddings,
            timestamps=seq_timestamps,
            num_targets=num_targets,
            interleave_targets=False,
            kernel=self.hammer_kernel(),
        )
