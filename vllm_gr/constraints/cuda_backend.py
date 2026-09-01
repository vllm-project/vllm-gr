# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Backend owning the canonical six-table device representation."""

from __future__ import annotations

from enum import Enum, auto

import torch

from vllm_gr.constraints.backend import (
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
    build_constraint_output,
    validate_select_inputs,
)
from vllm_gr.ops.cuda.rec_constrained_topk import rec_constrained_topk_cuda


class _BackendState(Enum):
    INITIALIZING = auto()
    READY = auto()
    CLOSED = auto()


class CUDAConstraintBackend:
    """Own GPU-resident canonical tables and dispatch CUDA constrained Top-K."""

    __slots__ = (
        "_first_token_ids",
        "_info",
        "_prefix1_offsets",
        "_prefix1_pair_keys",
        "_prefix1_values",
        "_prefix2_value_offsets",
        "_prefix2_values",
        "_state",
    )

    def __init__(
        self,
        *,
        first_token_ids: torch.Tensor,
        prefix1_offsets: torch.Tensor,
        prefix1_values: torch.Tensor,
        prefix1_pair_keys: torch.Tensor,
        prefix2_value_offsets: torch.Tensor,
        prefix2_values: torch.Tensor,
        info: ConstraintBackendInfo,
    ) -> None:
        self._first_token_ids: torch.Tensor | None = first_token_ids
        self._prefix1_offsets: torch.Tensor | None = prefix1_offsets
        self._prefix1_values: torch.Tensor | None = prefix1_values
        self._prefix1_pair_keys: torch.Tensor | None = prefix1_pair_keys
        self._prefix2_value_offsets: torch.Tensor | None = prefix2_value_offsets
        self._prefix2_values: torch.Tensor | None = prefix2_values
        self._info = info
        self._state = _BackendState.INITIALIZING

    @property
    def info(self) -> ConstraintBackendInfo:
        return self._info

    def _tables(self) -> tuple[torch.Tensor, ...]:
        tables = (
            self._first_token_ids,
            self._prefix1_offsets,
            self._prefix1_values,
            self._prefix1_pair_keys,
            self._prefix2_value_offsets,
            self._prefix2_values,
        )
        if self._state is _BackendState.CLOSED or any(table is None for table in tables):
            raise RuntimeError("CUDA constraint backend is closed")
        return tuple(table for table in tables if table is not None)

    def _invoke(self, logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        table_vocab_size = self._info.table_metadata.vocab_size
        vocab_offset = 0
        expected_logits_vocab_size = table_vocab_size
        if logits.shape[1] != table_vocab_size:
            from vllm_gr.ops.sliced_logits import get_vocab_slice_config

            offset, slice_size, _ = get_vocab_slice_config()
            if offset is None or slice_size is None or logits.shape[1] != slice_size:
                raise ValueError(
                    f"CUDA constraint logits vocab {logits.shape[1]} does not match "
                    f"table vocab_size {table_vocab_size} or the configured vocabulary slice"
                )
            if offset < 0 or offset + slice_size > table_vocab_size:
                raise ValueError("configured CUDA vocabulary slice is outside table vocabulary")
            vocab_offset = offset
            expected_logits_vocab_size = slice_size
        validate_select_inputs(
            logits,
            metadata,
            self._info,
            expected_logits_vocab_size=expected_logits_vocab_size,
        )
        if logits.device.type != "cuda":
            raise ValueError(f"CUDA constraint backend requires CUDA logits, got {logits.device}")
        tables = self._tables()
        for table in tables:
            if table.device != logits.device:
                raise RuntimeError(
                    "all CUDA constraint tables must share logits.device; "
                    f"got table={table.device}, logits={logits.device}"
                )
        token_ids, logprobs, valid_mask, _ = rec_constrained_topk_cuda(
            logits,
            metadata.prefix_token_ids,
            *tables,
            metadata.temperatures,
            metadata.current_step,
            metadata.top_k,
            self._info.table_metadata.max_prefix1_degree,
            self._info.table_metadata.max_prefix2_degree,
            vocab_offset=vocab_offset,
            table_vocab_size=table_vocab_size,
        )
        return build_constraint_output(
            token_ids,
            logprobs,
            valid_mask,
            logits=logits,
            top_k=metadata.top_k,
        )

    def _select_for_readiness(
        self, logits: torch.Tensor, metadata: ConstraintMetadata
    ) -> ConstraintOutput:
        if self._state is not _BackendState.INITIALIZING:
            raise RuntimeError("CUDA readiness can only run while initializing")
        return self._invoke(logits, metadata)

    def _mark_ready(self) -> None:
        if self._state is not _BackendState.INITIALIZING:
            raise RuntimeError("only an initializing CUDA constraint backend can become ready")
        self._state = _BackendState.READY

    def select(self, logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        if self._state is not _BackendState.READY:
            raise RuntimeError("CUDA constraint backend is not ready")
        return self._invoke(logits, metadata)

    def close(self) -> None:
        if self._state is _BackendState.CLOSED:
            return
        self._first_token_ids = None
        self._prefix1_offsets = None
        self._prefix1_values = None
        self._prefix1_pair_keys = None
        self._prefix2_value_offsets = None
        self._prefix2_values = None
        self._state = _BackendState.CLOSED


__all__ = ["CUDAConstraintBackend"]
