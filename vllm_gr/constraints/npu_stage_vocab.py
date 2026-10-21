# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Private stage-local token numbering for the NPU constrained operator.

The public Artifact and request metadata retain tokenizer-global token IDs.
Only the NPU operator sees the compact tables and prefixes defined here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from vllm_gr.constraints.table import CONSTRAINT_TABLE_TENSOR_NAMES, ConstraintTable


@dataclass(frozen=True, slots=True)
class NPUStageVocab:
    """Own compact tables and maps for three independent stage vocabularies."""

    vocab_size: int
    stage_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    prefix_token_maps: tuple[torch.Tensor, torch.Tensor]
    tables: tuple[torch.Tensor, ...]

    def owned_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            f"stage_table_{name}": tensor
            for name, tensor in zip(CONSTRAINT_TABLE_TENSOR_NAMES, self.tables)
        }
        tensors.update(
            {f"stage_token_ids_{step}": tensor for step, tensor in enumerate(self.stage_token_ids)}
        )
        tensors.update(
            {
                f"stage_prefix_map_{step}": tensor
                for step, tensor in enumerate(self.prefix_token_maps)
            }
        )
        return tensors

    def local_prefixes(self, prefixes: torch.Tensor, current_step: int) -> torch.Tensor:
        """Map active prefix columns, sending absent IDs to the padding slot."""
        local = torch.zeros_like(prefixes)
        padding = self.vocab_size - 1
        for step in range(current_step):
            token_ids = prefixes[:, step].to(torch.int64)
            mapping = self.prefix_token_maps[step]
            in_range = token_ids.ge(0) & token_ids.lt(mapping.numel())
            mapped = mapping[token_ids.clamp(0, mapping.numel() - 1)]
            local[:, step] = torch.where(in_range, mapped, padding)
        return local

    def global_tokens(
        self,
        token_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        """Restore public token IDs; structural padding remains token zero."""
        mapping = self.stage_token_ids[current_step]
        safe_ids = token_ids.to(torch.int64).clamp(0, mapping.numel() - 1)
        mapped = mapping[safe_ids].to(torch.int32)
        return torch.where(valid_mask, mapped, 0).contiguous()


def build_npu_stage_vocab(
    table: ConstraintTable,
    device: torch.device,
    upload: Callable[[NDArray[Any], torch.device], torch.Tensor],
) -> NPUStageVocab | None:
    """Create an optional compact representation without changing the Artifact.

    Independent sorted stage IDs preserve the original within-bucket ordering.
    One extra vocabulary slot is mandatory: the fused NPU kernel rejects a
    first prefix at ``vocab_size - 1``, so that slot can never be a legal token.
    """
    stage_ids = (
        np.unique(table.first_token_ids),
        np.unique(table.prefix1_values),
        np.unique(table.prefix2_values),
    )
    if any(ids.size == 0 for ids in stage_ids):
        return None
    max_stage_size = max(int(ids.size) for ids in stage_ids)
    vocab_size = ((max_stage_size + 1 + 127) // 128) * 128
    if vocab_size >= table.metadata.vocab_size:
        return None

    first, second, third = stage_ids
    degrees = table.prefix1_offsets[first + 1] - table.prefix1_offsets[first]
    offsets = np.full(vocab_size + 1, table.prefix1_values.size, dtype=np.int32)
    offsets[0] = 0
    offsets[1 : first.size + 1] = np.cumsum(degrees, dtype=np.int64).astype(np.int32)
    second_values = np.searchsorted(second, table.prefix1_values).astype(np.int32)
    compact_arrays = (
        np.arange(first.size, dtype=np.int32),
        offsets,
        second_values,
        np.repeat(np.arange(first.size, dtype=np.int64), degrees) * vocab_size
        + second_values.astype(np.int64),
        table.prefix2_value_offsets,
        np.searchsorted(third, table.prefix2_values).astype(np.int32),
    )
    prefix_maps = []
    for ids in (first, second):
        mapping = np.full(table.metadata.vocab_size, vocab_size - 1, dtype=np.int32)
        mapping[ids] = np.arange(ids.size, dtype=np.int32)
        prefix_maps.append(upload(mapping, device))
    return NPUStageVocab(
        vocab_size=vocab_size,
        stage_token_ids=(
            upload(first.astype(np.int64), device),
            upload(second.astype(np.int64), device),
            upload(third.astype(np.int64), device),
        ),
        prefix_token_maps=(prefix_maps[0], prefix_maps[1]),
        tables=tuple(upload(array, device) for array in compact_arrays),
    )


__all__ = ["NPUStageVocab", "build_npu_stage_vocab"]
