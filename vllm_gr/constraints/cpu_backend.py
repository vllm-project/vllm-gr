# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

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
from vllm_gr.constraints.scoring import score_selected_tokens
from vllm_gr.constraints.table import ConstraintTable
from vllm_gr.constraints.validity import valid_mask_from_counts


class CPUConstraintBackend:
    """Keep canonical constraint tables on Host for the Backend lifetime."""

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
        trie_ids: NDArray[np.int32] | None = None,
    ) -> list[NDArray[np.int32]]:
        """Query validated inputs; None IDs use the single-Trie fast path."""
        table = self._require_table()
        rows = prefixes.shape[0]
        if current_step == 0:
            if trie_ids is None:
                return [table.first_token_ids] * rows
            return [
                table.first_token_ids[table.root_offsets[r] : table.root_offsets[r + 1]]
                for r in trie_ids
            ]

        vocab_size = table.metadata.vocab_size
        empty1 = table.prefix1_values[:0]
        if current_step == 1:
            allowed: list[NDArray[np.int32]] = []
            for index, raw_a in enumerate(prefixes[:, 0]):
                a = int(raw_a)
                if not 0 <= a < vocab_size:
                    allowed.append(empty1)
                    continue
                row = a if trie_ids is None else int(trie_ids[index]) * vocab_size + a
                begin = int(table.prefix1_offsets[row])
                end = int(table.prefix1_offsets[row + 1])
                allowed.append(table.prefix1_values[begin:end])
            return allowed

        # Validate both components independently before calculating the pair
        # key. Otherwise values such as (-1, vocab + b) can alias (0, b).
        empty2 = table.prefix2_values[:0]
        result: list[NDArray[np.int32]] = []
        for index, (raw_a, raw_b) in enumerate(prefixes[:, :2]):
            a, b = int(raw_a), int(raw_b)
            if not (0 <= a < vocab_size and 0 <= b < vocab_size):
                result.append(empty2)
                continue
            row = a if trie_ids is None else int(trie_ids[index]) * vocab_size + a
            pair_key = row * vocab_size + b
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
        table = self._require_table()
        validate_select_inputs(logits, metadata, self._info)

        # Copy prefixes and multi-Trie selectors to Host. Raw logits never
        # leave their device, and single-Trie queries need no selector array.
        prefixes = metadata.prefix_token_ids.detach().to(device="cpu").numpy()
        trie_ids = None
        if table.metadata.trie_count > 1:
            assert metadata.trie_ids is not None  # Checked by validate_select_inputs.
            trie_ids = metadata.trie_ids.detach().cpu().numpy()
        allowed = self._allowed_ids(prefixes, metadata.current_step, trie_ids)
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
        # The legal table controls selection only.  Beam scores are normalized
        # over the shared logits domain so different parent rows are comparable.
        # Temperature scaling is intentionally deferred to scoring: every
        # supported temperature is non-negative, so it cannot change Top-K.
        # Sanitize NaN before selection: torch.topk would treat a poisoned
        # candidate as the largest value and promote it, while the fused CUDA
        # kernel sorts NaN candidates last.  Sanitizing here keeps CPU and
        # CUDA selecting the same tokens on poisoned logits.
        candidate_logits = torch.nan_to_num(
            logits.gather(1, candidate_ids).float(), nan=float("-inf")
        )
        masked_logits = candidate_logits.masked_fill(~candidate_mask, float("-inf"))
        _, top_indices = masked_logits.topk(
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
        top_logprobs = score_selected_tokens(
            logits,
            top_ids,
            valid_mask,
            metadata.temperatures,
            invalid_logprob=INVALID_LOGIT,
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
