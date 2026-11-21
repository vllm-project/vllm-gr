# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Per-Trie tables and stage-local token numbering for the NPU constrained operator.

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


def npu_trie_arrays(table: ConstraintTable, trie_id: int) -> tuple[NDArray[Any], ...]:
    """Rebase one validated root without modifying the canonical artifact."""
    if not 0 <= trie_id < table.metadata.trie_count:
        raise ValueError("trie_id is outside the constraint table")
    if table.metadata.trie_count == 1:
        return tuple(getattr(table, name) for name in CONSTRAINT_TABLE_TENSOR_NAMES)
    vocab = table.metadata.vocab_size
    first_begin, first_end = table.root_offsets[trie_id : trie_id + 2]
    offsets1 = table.prefix1_offsets[trie_id * vocab : (trie_id + 1) * vocab + 1]
    begin1, end1 = int(offsets1[0]), int(offsets1[-1])
    root_key = trie_id * vocab**2
    pair_begin, pair_end = np.searchsorted(table.prefix1_pair_keys, [root_key, root_key + vocab**2])
    offsets2 = table.prefix2_value_offsets[pair_begin : pair_end + 1]
    begin2, end2 = int(offsets2[0]), int(offsets2[-1])
    return (
        table.first_token_ids[first_begin:first_end],
        offsets1 - begin1,
        table.prefix1_values[begin1:end1],
        table.prefix1_pair_keys[pair_begin:pair_end] - root_key,
        offsets2 - begin2,
        table.prefix2_values[begin2:end2],
    )


@dataclass(frozen=True, slots=True)
class NPUStageVocab:
    """Own compact tables and maps for three independent stage vocabularies."""

    vocab_size: int
    stage_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    prefix_token_maps: tuple[torch.Tensor, torch.Tensor]
    tables: tuple[torch.Tensor, ...]
    trie_tables: tuple[tuple[torch.Tensor, ...], ...] = ()

    def tables_for_trie(self, trie_id: int) -> tuple[torch.Tensor, ...]:
        return self.trie_tables[trie_id] if self.trie_tables else self.tables

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
        for trie_id, tables in enumerate(self.trie_tables[1:], 1):
            tensors.update(
                {
                    f"stage_trie_{trie_id}_{name}": tensor
                    for name, tensor in zip(CONSTRAINT_TABLE_TENSOR_NAMES, tables)
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

    # Every root shares these sorted union maps and the same projection shape.
    # Its six tables retain only that root's legal edges. In particular, shared
    # SA/SB token IDs must never combine their SC lists across roots.
    first, second, third = stage_ids
    trie_tables: list[tuple[torch.Tensor, ...]] = []
    prefix_maps: list[torch.Tensor] = []
    try:
        for trie_id in range(table.metadata.trie_count):
            root_first, offsets1, values1, _keys, offsets2, values2 = npu_trie_arrays(
                table, trie_id
            )
            local_first = np.searchsorted(first, root_first).astype(np.int32)
            degrees = offsets1[root_first + 1] - offsets1[root_first]
            offsets = np.zeros(vocab_size + 1, dtype=np.int32)
            offsets[local_first + 1] = degrees
            offsets = np.cumsum(offsets, dtype=np.int64).astype(np.int32)
            second_values = np.searchsorted(second, values1).astype(np.int32)
            compact_arrays = (
                local_first,
                offsets,
                second_values,
                np.repeat(local_first.astype(np.int64), degrees) * vocab_size
                + second_values.astype(np.int64),
                offsets2,
                np.searchsorted(third, values2).astype(np.int32),
            )
            trie_tables.append(tuple(upload(array, device) for array in compact_arrays))
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
            tables=trie_tables[0],
            trie_tables=tuple(trie_tables),
        )
    except BaseException:
        trie_tables.clear()
        prefix_maps.clear()
        raise


__all__ = ["NPUStageVocab", "build_npu_stage_vocab", "npu_trie_arrays"]
