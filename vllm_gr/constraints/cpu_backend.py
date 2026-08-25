# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-table Backend that gathers only the current batch's candidates."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
    build_constraint_output,
    validate_select_inputs,
)
from vllm_gr.constraints.table import ConstraintTable
from vllm_gr.constraints.validity import valid_mask_from_counts


class CPUConstraintBackend:
    """Keep all six canonical tensors on Host for the Backend lifetime."""

    __slots__ = ("_closed", "_info", "_table")

    def __init__(self, table: ConstraintTable, info: ConstraintBackendInfo) -> None:
        self._table: ConstraintTable | None = table
        self._info = info
        self._closed = False

    @property
    def info(self) -> ConstraintBackendInfo:
        return self._info

    def _require_table(self) -> ConstraintTable:
        if self._closed or self._table is None:
            raise RuntimeError("CPU constraint backend is closed")
        return self._table

    def _allowed_ids(
        self,
        prefixes: NDArray[np.int32],
        current_step: int,
    ) -> list[NDArray[np.int32]]:
        table = self._require_table()
        rows = prefixes.shape[0]
        if current_step == 0:
            return [table.first_token_ids] * rows

        vocab_size = table.metadata.vocab_size
        empty1 = table.prefix1_values[:0]
        if current_step == 1:
            allowed: list[NDArray[np.int32]] = []
            for raw_a in prefixes[:, 0]:
                a = int(raw_a)
                if not 0 <= a < vocab_size:
                    allowed.append(empty1)
                    continue
                begin = int(table.prefix1_offsets[a])
                end = int(table.prefix1_offsets[a + 1])
                allowed.append(table.prefix1_values[begin:end])
            return allowed

        # Validate both components independently before calculating the pair
        # key. Otherwise values such as (-1, vocab + b) can alias (0, b).
        empty2 = table.prefix2_values[:0]
        result: list[NDArray[np.int32]] = []
        for raw_a, raw_b in prefixes[:, :2]:
            a, b = int(raw_a), int(raw_b)
            if not (0 <= a < vocab_size and 0 <= b < vocab_size):
                result.append(empty2)
                continue
            pair_key = a * vocab_size + b
            position = int(np.searchsorted(table.prefix1_pair_keys, pair_key))
            if (
                position >= table.prefix1_pair_keys.size
                or int(table.prefix1_pair_keys[position]) != pair_key
            ):
                result.append(empty2)
                continue
            begin = int(table.prefix2_value_offsets[position])
            end = int(table.prefix2_value_offsets[position + 1])
            result.append(table.prefix2_values[begin:end])
        return result

    def select(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput:
        self._require_table()
        validate_select_inputs(logits, metadata, self._info)

        # Prefixes are the only request data needed on Host. Raw logits never
        # leave their device, and the canonical tensors never leave Host.
        prefixes = metadata.prefix_token_ids.detach().to(device="cpu").numpy()
        allowed = self._allowed_ids(prefixes, metadata.current_step)
        candidate_width = max(
            metadata.top_k,
            max((int(ids.size) for ids in allowed), default=0),
            1,
        )
        host_ids = np.zeros((logits.shape[0], candidate_width), dtype=np.int64)
        host_mask = np.zeros((logits.shape[0], candidate_width), dtype=np.bool_)
        for row, ids in enumerate(allowed):
            count = int(ids.size)
            if count:
                host_ids[row, :count] = ids
                host_mask[row, :count] = True

        candidate_ids = torch.from_numpy(host_ids).to(device=logits.device)
        candidate_mask = torch.from_numpy(host_mask).to(device=logits.device)
        candidate_logits = logits.gather(1, candidate_ids).float()
        temperatures = metadata.temperatures.reshape(-1, 1)
        temperatures = torch.where(
            temperatures == 0,
            torch.ones_like(temperatures),
            temperatures,
        )
        candidate_logits = candidate_logits / temperatures

        # Normalize across every legal candidate before truncating to Top-K.
        masked_logits = candidate_logits.masked_fill(~candidate_mask, float("-inf"))
        log_denom = torch.logsumexp(masked_logits, dim=1, keepdim=True)
        candidate_logprobs = torch.where(
            candidate_mask,
            masked_logits - log_denom,
            torch.full_like(masked_logits, float("-inf")),
        )
        top_logprobs, top_indices = candidate_logprobs.topk(
            metadata.top_k,
            dim=1,
            largest=True,
            sorted=True,
        )
        top_ids = candidate_ids.gather(1, top_indices).to(torch.int32)
        valid_counts = torch.tensor(
            [min(int(ids.size), metadata.top_k) for ids in allowed],
            dtype=torch.int64,
            device=logits.device,
        )
        valid_mask = valid_mask_from_counts(valid_counts, metadata.top_k)
        top_logprobs = torch.where(
            valid_mask,
            top_logprobs,
            torch.full_like(top_logprobs, INVALID_LOGIT),
        )
        return build_constraint_output(
            top_ids.contiguous(),
            top_logprobs.contiguous(),
            valid_mask,
            logits=logits,
            top_k=metadata.top_k,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._table = None
        self._closed = True


__all__ = ["CPUConstraintBackend"]
