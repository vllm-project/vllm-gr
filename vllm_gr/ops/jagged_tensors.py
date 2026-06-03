# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

"""
Jagged Tensor Operator Entry Point

Jagged Tensor is the core data structure for handling variable-length sequences.
This fork currently routes concat/split through the PyTorch reference helpers.
The public API still accepts upstream-style dispatch and shape-hint arguments,
but this wrapper no longer chooses among multiple backend-specific
implementations at this layer:

- ``kernel`` is preserved as a compatibility hint for existing call sites
- ``max_seq_len`` and ``total_len_left/right`` remain in the signature for
  upstream compatibility, but are not used to select a specialized kernel here
- the effective concat/split semantics come from
  ``vllm_gr.ops.pytorch.pt_jagged_tensors``
"""

from typing import Optional, Tuple

import torch
from torch.fx._symbolic_trace import is_fx_tracing

from vllm_gr.models.dlrm_v3.common import HammerKernel
from vllm_gr.ops.pytorch.pt_jagged_tensors import (
    pytorch_concat_2D_jagged,
    pytorch_split_2D_jagged,
)


def concat_2D_jagged(
    max_seq_len: int,
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """
    Concatenate two 2D Jagged Tensors

    Used for merging user history sequence and candidate item sequence.
    ``kernel`` and ``max_seq_len`` are preserved for API compatibility, but
    this wrapper currently validates the contract and then dispatches through
    the PyTorch reference implementation only.
    """
    if not is_fx_tracing():
        torch._assert(values_left.dim() == 2, "values_left must be 2D")
        torch._assert(values_right.dim() == 2, "values_right must be 2D")
        torch._assert(
            values_right.shape[1] == values_left.shape[1],
            (
                "values_left shape[1] must be equal to "
                f"values_right shape[1] {values_left.shape[1]} "
                f"vs {values_right.shape[1]}"
            ),
        )
    del max_seq_len, kernel
    return pytorch_concat_2D_jagged(
        values_left=values_left,
        values_right=values_right,
        max_len_left=max_len_left,
        max_len_right=max_len_right,
        offsets_left=offsets_left,
        offsets_right=offsets_right,
    )


def split_2D_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    total_len_left: Optional[int] = None,
    total_len_right: Optional[int] = None,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split 2D Jagged Tensor into two parts

    Used for separating user embeddings and candidate embeddings from merged
    sequence. ``kernel`` plus ``total_len_left/right`` are preserved for API
    compatibility, but this wrapper currently validates the contract and then
    dispatches through the PyTorch reference implementation only.
    """
    if not is_fx_tracing():
        torch._assert(values.dim() == 2, "values must be 2D")
        torch._assert(
            offsets_left is not None or offsets_right is not None,
            "offsets_left and offsets_right cannot be None at the same time",
        )
        if offsets_left is None:
            torch._assert(
                max_len_left is not None,
                "max_len_left must be provided when offsets_left is None",
            )
        if offsets_right is None:
            torch._assert(
                max_len_right is not None,
                "max_len_right must be provided when offsets_right is None",
            )
        if offsets_left is not None and offsets_right is not None:
            torch._assert(
                offsets_left.shape[0] == offsets_right.shape[0],
                "offsets_left shape[0] must be equal to offsets_right shape[0]",
            )
    del total_len_left, total_len_right, kernel
    return pytorch_split_2D_jagged(
        max_seq_len=max_seq_len,
        values=values,
        max_len_left=max_len_left,
        max_len_right=max_len_right,
        offsets_left=offsets_left,
        offsets_right=offsets_right,
    )
