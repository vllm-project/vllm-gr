# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU Backend owning the canonical six-table device representation."""

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
from vllm_gr.constraints.validity import (
    constraint_candidate_counts,
    valid_mask_from_counts,
)
from vllm_gr.ops.npu.rec_constrained_topk import NPUConstraintOp


class _BackendState(Enum):
    INITIALIZING = auto()
    READY = auto()
    CLOSED = auto()


class NPUConstraintBackend:
    """Hold only six NPU tensors, diagnostics, one real op, and state."""

    __slots__ = (
        "_first_token_ids",
        "_info",
        "_op",
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
        op: NPUConstraintOp,
    ) -> None:
        self._first_token_ids: torch.Tensor | None = first_token_ids
        self._prefix1_offsets: torch.Tensor | None = prefix1_offsets
        self._prefix1_values: torch.Tensor | None = prefix1_values
        self._prefix1_pair_keys: torch.Tensor | None = prefix1_pair_keys
        self._prefix2_value_offsets: torch.Tensor | None = prefix2_value_offsets
        self._prefix2_values: torch.Tensor | None = prefix2_values
        self._info = info
        self._op: NPUConstraintOp | None = op
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
            raise RuntimeError("NPU constraint backend is closed")
        return tuple(table for table in tables if table is not None)

    def _invoke(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput:
        validate_select_inputs(logits, metadata, self._info)
        tables = self._tables()
        if self._op is None:
            raise RuntimeError("NPU constraint backend is closed")
        for table in tables:
            if table.device != logits.device:
                raise RuntimeError(
                    "all NPU constraint tables must share logits.device; "
                    f"got table={table.device}, logits={logits.device}"
                )

        # The fused operator consumes [B, W, max_steps]. A singleton W is a
        # zero-copy view of the public flat [rows, 3] Backend metadata.
        sequence_group = metadata.prefix_token_ids.unsqueeze(1)
        # The fused kernel's bf16 load path reads stale unified-buffer data on
        # ascend910b and produces garbage logprobs, while its fp32 path is
        # correct; bf16 -> fp32 is a lossless bit extension, so widen first.
        # (Same normalization as vllm_gr.ops.npu.rec_constrained_topk.)
        op_logits = logits.to(torch.float32) if logits.dtype == torch.bfloat16 else logits
        token_ids, logprobs = self._op(
            op_logits,
            sequence_group,
            *tables,
            metadata.temperatures,
            metadata.current_step,
            metadata.top_k,
            self._info.table_metadata.max_prefix1_degree,
            self._info.table_metadata.max_prefix2_degree,
        )
        valid_counts = constraint_candidate_counts(
            metadata.prefix_token_ids,
            tables[0],
            tables[1],
            tables[3],
            tables[4],
            current_step=metadata.current_step,
            vocab_size=self._info.table_metadata.vocab_size,
        )
        return build_constraint_output(
            token_ids,
            logprobs,
            valid_mask_from_counts(valid_counts, metadata.top_k),
            logits=logits,
            top_k=metadata.top_k,
        )

    def _select_for_readiness(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput:
        if self._state is not _BackendState.INITIALIZING:
            raise RuntimeError("NPU readiness can only run while initializing")
        return self._invoke(logits, metadata)

    def _mark_ready(self) -> None:
        if self._state is not _BackendState.INITIALIZING:
            raise RuntimeError("only an initializing NPU constraint backend can become ready")
        self._state = _BackendState.READY

    def select(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput:
        if self._state is _BackendState.CLOSED:
            raise RuntimeError("NPU constraint backend is closed")
        if self._state is not _BackendState.READY:
            raise RuntimeError("NPU constraint backend is not ready")
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
        self._op = None
        self._state = _BackendState.CLOSED


__all__ = ["NPUConstraintBackend"]
