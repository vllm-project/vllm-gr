# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-table backend that gathers only legal candidates from device logits."""

from __future__ import annotations

import numpy as np
import torch

from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintMetadata,
    ConstraintOutput,
    build_constraint_output,
    validate_select_inputs,
)
from vllm_gr.constraints.table import ConstraintTable, ConstraintTableInfo


class CPUConstraintBackend:
    """Keep all six tables on Host; transfer only per-batch IDs and masks."""

    __slots__ = ("_table", "_closed")

    def __init__(self, table: ConstraintTable) -> None:
        self._table: ConstraintTable | None = table
        self._closed = False

    @property
    def info(self) -> ConstraintTableInfo:
        table = self._require_table()
        return table.info

    def _require_table(self) -> ConstraintTable:
        if self._closed or self._table is None:
            raise RuntimeError("CPU constraint backend is closed")
        return self._table

    def _allowed_ids(self, prefixes: np.ndarray, step: int) -> list[np.ndarray]:
        table = self._require_table()
        rows = prefixes.shape[0]
        if step == 0:
            return [table.first_token_ids] * rows
        if step == 1:
            result: list[np.ndarray] = []
            for token_id in prefixes[:, 0]:
                token = int(token_id)
                if not 0 <= token < table.info.vocab_size:
                    result.append(table.prefix1_values[:0])
                    continue
                begin = int(table.prefix1_offsets[token])
                end = int(table.prefix1_offsets[token + 1])
                result.append(table.prefix1_values[begin:end])
            return result

        pair_keys = (
            prefixes[:, 0].astype(np.int64) * table.info.vocab_size
            + prefixes[:, 1].astype(np.int64)
        )
        positions = np.searchsorted(table.prefix1_pair_keys, pair_keys)
        result = []
        for key, position in zip(pair_keys, positions):
            pos = int(position)
            if pos >= len(table.prefix1_pair_keys) or int(table.prefix1_pair_keys[pos]) != int(key):
                result.append(table.prefix2_values[:0])
                continue
            begin = int(table.prefix2_value_offsets[pos])
            end = int(table.prefix2_value_offsets[pos + 1])
            result.append(table.prefix2_values[begin:end])
        return result

    def select(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput:
        table = self._require_table()
        validate_select_inputs(logits, metadata, table.info)

        # Only the small request prefix crosses to Host. The full logits and
        # all six tables remain on their owning side of the boundary.
        prefixes = metadata.prefix_token_ids.detach().to(device="cpu").numpy()
        allowed = self._allowed_ids(prefixes, metadata.current_step)
        candidate_width = max(metadata.top_k, max((len(ids) for ids in allowed), default=0), 1)
        host_ids = np.zeros((logits.shape[0], candidate_width), dtype=np.int64)
        host_mask = np.zeros((logits.shape[0], candidate_width), dtype=np.bool_)
        for row, ids in enumerate(allowed):
            count = len(ids)
            if count:
                host_ids[row, :count] = ids
                host_mask[row, :count] = True

        candidate_ids = torch.from_numpy(host_ids).to(device=logits.device)
        candidate_mask = torch.from_numpy(host_mask).to(device=logits.device)
        candidate_logits = logits.gather(1, candidate_ids).float()
        temperatures = metadata.temperatures.to(device=logits.device).reshape(-1, 1)
        candidate_logits = candidate_logits / temperatures
        masked_logits = candidate_logits.masked_fill(~candidate_mask, float("-inf"))
        normalizer = torch.logsumexp(masked_logits, dim=1, keepdim=True)
        candidate_logprobs = torch.where(
            candidate_mask,
            masked_logits - normalizer,
            torch.full_like(masked_logits, INVALID_LOGIT),
        )
        top_logprobs, top_indices = candidate_logprobs.topk(
            metadata.top_k,
            dim=1,
            largest=True,
            sorted=True,
        )
        top_ids = candidate_ids.gather(1, top_indices)
        return build_constraint_output(
            top_ids.contiguous(),
            top_logprobs.contiguous(),
            logits=logits,
            top_k=metadata.top_k,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._table = None
        self._closed = True
