# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

"""PyTorch implementation of HSTU attention"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from vllm_gr.models.dlrm_v3.common import dense_to_jagged, jagged_to_padded_dense


@torch.fx.wrap
def _get_valid_attn_mask(
    device: torch.device,
    causal: bool,
    N: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    """Generate valid attention mask for HSTU"""
    ids = torch.arange(0, N, device=device).view(1, N)
    max_ids = seq_lengths.view(-1, 1, 1)

    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = torch.clamp(ids, min=0)
        max_ids = max_ids - contextual_seq_len + 1

    if num_targets is not None:
        max_ids = max_ids - num_targets.view(-1, 1, 1)
        ids = torch.clamp(ids, max=max_ids)
        row_ids = ids.view(-1, N, 1).expand(-1, N, N)
        col_ids = ids.view(-1, 1, N).expand(-1, N, N)
    else:
        row_ids = ids.view(N, 1).expand(N, N)
        col_ids = row_ids.t()
        row_ids = row_ids.view(1, N, N)
        col_ids = col_ids.view(1, N, N)

    row_col_dist = row_ids - col_ids
    valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)

    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)

    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)

    if max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask,
                torch.logical_or(
                    row_col_dist <= max_attn_len,
                    row_ids >= max_ids - min_full_attn_seq_len,
                ),
            )
        else:
            valid_attn_mask = torch.logical_and(valid_attn_mask, row_col_dist <= max_attn_len)

    if contextual_seq_len > 0:
        valid_attn_mask = torch.logical_or(
            valid_attn_mask, torch.logical_and(row_ids == 0, col_ids < max_ids)
        )

    return valid_attn_mask


def _pad_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad Q, K, V tensors to dense format"""
    L, H, D = q.shape
    V_dim = v.shape[2]

    padded_q = (
        jagged_to_padded_dense(
            values=q.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )

    padded_k = (
        jagged_to_padded_dense(
            values=k.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )

    padded_v = (
        jagged_to_padded_dense(
            values=v.reshape(L, H * V_dim),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, V_dim)
        .transpose(1, 2)
    )

    return padded_q, padded_k, padded_v


@torch.fx.wrap
def pytorch_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    """
    PyTorch implementation of HSTU multi-head attention.

    Args:
        max_seq_len: Maximum sequence length
        alpha: Attention scaling factor
        q, k, v: Query, Key, Value tensors (L, H, D)
        seq_offsets: Sequence offsets (B+1,)
        causal: Use causal attention
        dropout_pr: Dropout probability
        training: Training mode flag
        num_targets: Number of target items per sequence
        max_attn_len: Maximum attention length
        contextual_seq_len: Contextual sequence length
        min_full_attn_seq_len: Minimum full attention sequence length

    Returns:
        Attention output tensor (L, H, V)
    """
    L, H, _ = q.shape
    V_dim = v.shape[2]

    # Pad Q, K, V to dense format
    q, k, v = _pad_qkv(q, k, v, seq_offsets, max_seq_len)

    # Compute attention scores
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len

    # Apply attention mask
    valid_attn_mask = _get_valid_attn_mask(
        device=q.device,
        causal=causal,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=min_full_attn_seq_len,
    )

    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)

    if dropout_pr > 0.0:
        qk_attn = F.dropout(qk_attn, p=dropout_pr, training=training)

    # Compute attention output
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)

    # Convert back to jagged format
    return dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),
        [seq_offsets],
    ).view(L, H, V_dim)


@torch.fx.wrap
def pytorch_cached_hstu_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    """
    PyTorch implementation of cached HSTU attention (for incremental inference).
    """
    L, H, D = delta_q.shape
    _, _, V_dim = v.shape
    B = seq_offsets.size(0) - 1
    if num_targets is None:
        if B == 0:
            return delta_q.new_empty((0, H, V_dim))
        if L % B != 0:
            raise ValueError(
                "pytorch_cached_hstu_mha requires num_targets for variable-size "
                f"delta batches; got total_delta_tokens={L}, batch_size={B}."
            )
        num_targets = torch.full(
            (B,),
            L // B,
            dtype=seq_offsets.dtype,
            device=delta_q.device,
        )
    else:
        num_targets = num_targets.to(device=delta_q.device, dtype=seq_offsets.dtype)

    delta_offsets = torch.zeros(
        B + 1,
        dtype=seq_offsets.dtype,
        device=delta_q.device,
    )
    if B > 0:
        delta_offsets[1:] = torch.cumsum(num_targets, dim=0)
    max_delta_len = int(num_targets.max().item()) if num_targets.numel() > 0 else 0

    delta_q = (
        jagged_to_padded_dense(
            values=delta_q.reshape(L, H * D),
            offsets=[delta_offsets],
            max_lengths=[max_delta_len],
            padding_value=0.0,
        )
        .view(B, max_delta_len, H, D)
        .transpose(1, 2)
    )

    full_k = (
        jagged_to_padded_dense(
            values=k.reshape(-1, H * D),
            offsets=[seq_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        .view(B, -1, H, D)
        .transpose(1, 2)
    )

    full_v = (
        jagged_to_padded_dense(
            values=v.reshape(-1, H * V_dim),
            offsets=[seq_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        .view(B, -1, H, V_dim)
        .transpose(1, 2)
    )

    qk_attn = torch.einsum("bhxa,bhya->bhxy", delta_q, full_k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len

    full_valid_attn_mask = _get_valid_attn_mask(
        device=delta_q.device,
        causal=True,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )

    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    row_ids = (
        seq_lengths.view(-1, 1)
        - num_targets.view(-1, 1)
        + torch.arange(max_delta_len, device=delta_q.device).view(1, -1)
    )
    valid_delta_rows = torch.arange(max_delta_len, device=delta_q.device).view(
        1, -1
    ) < num_targets.view(-1, 1)
    row_ids = torch.clamp(row_ids, min=0, max=max_seq_len - 1)
    valid_attn_mask = full_valid_attn_mask.gather(
        dim=1,
        index=row_ids.unsqueeze(-1).expand(-1, -1, max_seq_len),
    )
    valid_attn_mask = valid_attn_mask * valid_delta_rows.unsqueeze(-1)

    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    attn_output = torch.einsum("bhxd,bhdv->bhxv", qk_attn, full_v)

    return dense_to_jagged(
        attn_output.transpose(1, 2).reshape(B, max_delta_len, H * V_dim),
        [delta_offsets],
    ).view(L, H, V_dim)
