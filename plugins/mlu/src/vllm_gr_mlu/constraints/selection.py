# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Fixed-shape eager selection over device-resident single-Trie tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintOutput,
    build_constraint_output,
)
from vllm_gr.constraints.scoring import score_selected_tokens
from vllm_gr.constraints.table import CONSTRAINT_TABLE_TENSOR_NAMES, ConstraintTable

if TYPE_CHECKING:
    from vllm_gr.constraints.compiler import ConstraintTableResource

# Initial implementation limits, not a declaration of validated hardware capacity.
MAX_ROWS = 128
MAX_TOP_K = 128
LOGIT_DTYPES = (torch.float16, torch.bfloat16)


def require_mlu_device(device: torch.device) -> None:
    if device.type != "mlu":
        raise ValueError(f"MLU constraints require an mlu device, got {device}")


@dataclass(slots=True)
class DeviceConstraintTable:
    tensors: dict[str, torch.Tensor]
    vocab_size: int
    max_degrees: tuple[int, int, int]

    @classmethod
    def upload(
        cls, table: ConstraintTable | ConstraintTableResource, device: torch.device
    ) -> DeviceConstraintTable:
        if isinstance(table, ConstraintTable):
            vocab_size = table.metadata.vocab_size
            max_prefix1 = table.metadata.max_prefix1_degree
            max_prefix2 = table.metadata.max_prefix2_degree
        else:
            vocab_size = table.vocab_size
            max_prefix1 = table.max_prefix1_degree
            max_prefix2 = table.max_prefix2_degree
        return cls(
            {
                name: torch.from_numpy(np.array(getattr(table, name), copy=True)).to(device=device)
                for name in CONSTRAINT_TABLE_TENSOR_NAMES
            },
            vocab_size,
            (int(table.first_token_ids.size), max_prefix1, max_prefix2),
        )

    @property
    def device(self) -> torch.device:
        return self.tensors["first_token_ids"].device

    @property
    def nbytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors.values())

    def _ranges(
        self, prefixes: torch.Tensor, step: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if step == 0:
            values = self.tensors["first_token_ids"]
            starts = torch.zeros(prefixes.shape[0], dtype=torch.long, device=prefixes.device)
            return values, starts, torch.full_like(starts, values.numel())

        first = prefixes[:, 0].long()
        valid = (first >= 0) & (first < self.vocab_size)
        safe_first = first.clamp(0, self.vocab_size - 1)
        if step == 1:
            values = self.tensors["prefix1_values"]
            offsets = self.tensors["prefix1_offsets"]
            slots = safe_first
        else:
            values = self.tensors["prefix2_values"]
            keys = self.tensors["prefix1_pair_keys"]
            if keys.numel() == 0:
                starts = torch.zeros_like(first)
                return values, starts, starts
            second = prefixes[:, 1].long()
            valid = valid & (second >= 0) & (second < self.vocab_size)
            query = safe_first * self.vocab_size + second.clamp(0, self.vocab_size - 1)
            indices = torch.searchsorted(keys, query)
            slots = indices.clamp(max=keys.numel() - 1)
            valid = valid & (indices < keys.numel()) & (keys[slots] == query)
            offsets = self.tensors["prefix2_value_offsets"]

        starts = offsets[slots].long()
        counts = offsets[slots + 1].long() - starts
        return values, starts, counts.masked_fill(~valid, 0)

    @torch.inference_mode()
    def select(
        self,
        logits: torch.Tensor,
        prefixes: torch.Tensor,
        temperatures: torch.Tensor,
        step: int,
        top_k: int,
    ) -> ConstraintOutput:
        values, starts, counts = self._ranges(prefixes, step)
        if values.numel() == 0:
            shape = (logits.shape[0], top_k)
            token_ids = torch.zeros(shape, dtype=torch.int32, device=logits.device)
            valid = torch.zeros(shape, dtype=torch.bool, device=logits.device)
            scores = torch.full(shape, INVALID_LOGIT, dtype=torch.float32, device=logits.device)
        else:
            width = max(top_k, self.max_degrees[step])
            lanes = torch.arange(width, dtype=torch.long, device=logits.device)
            legal = lanes < counts[:, None]
            offsets = (starts[:, None] + lanes).masked_fill(~legal, 0)
            candidates = values[offsets]
            rank_scores = logits.gather(1, candidates.long()).float()
            rank_scores = rank_scores.masked_fill(~legal | rank_scores.isnan(), float("-inf"))
            # Legal lanes precede padding in the CSR range. Stable sorting keeps
            # legal -inf tokens ahead of padding and never uses scores as validity.
            order = rank_scores.argsort(dim=1, descending=True, stable=True)[:, :top_k]
            valid = legal.gather(1, order)
            token_ids = candidates.gather(1, order).masked_fill(~valid, 0)
            scores = score_selected_tokens(logits, token_ids, valid, temperatures)
        return build_constraint_output(
            token_ids.contiguous(),
            scores.contiguous(),
            valid.contiguous(),
            logits=logits,
            top_k=top_k,
        )
