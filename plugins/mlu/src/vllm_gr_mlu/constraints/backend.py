# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Canonical Constraint Table backend for the eager MLU path."""

from __future__ import annotations

import torch

from vllm_gr.config.models import CONSTRAINT_TABLE_FORMAT, ConstraintBackendKind
from vllm_gr.constraints.backend import (
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
    validate_select_inputs,
)
from vllm_gr.constraints.table import ConstraintTable

from .selection import (
    LOGIT_DTYPES,
    MAX_ROWS,
    MAX_TOP_K,
    DeviceConstraintTable,
    require_mlu_device,
)


class MLUConstraintBackend:
    def __init__(
        self, table: DeviceConstraintTable, info: ConstraintBackendInfo, dtype: torch.dtype
    ) -> None:
        self._table: DeviceConstraintTable | None = table
        self._info = info
        self._dtype = dtype

    @property
    def info(self) -> ConstraintBackendInfo:
        return self._info

    def select(self, logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        table = self._table
        if table is None:
            raise RuntimeError("MLU constraint backend is closed")
        if logits.device != table.device:
            raise ValueError(f"MLU logits device {logits.device} must match {table.device}")
        if logits.dtype != self._dtype:
            raise ValueError(f"MLU logits dtype {logits.dtype} must match {self._dtype}")
        validate_select_inputs(logits, metadata, self.info)
        return table.select(
            logits,
            metadata.prefix_token_ids,
            metadata.temperatures,
            metadata.current_step,
            metadata.top_k,
        )

    def close(self) -> None:
        self._table = None


def create_constraint_backend(
    *,
    artifact: ConstraintTable,
    device: torch.device,
    dtype: torch.dtype,
    max_rows: int,
    max_top_k: int,
) -> MLUConstraintBackend:
    require_mlu_device(device)
    if artifact.metadata.format != CONSTRAINT_TABLE_FORMAT or artifact.metadata.trie_count != 1:
        raise ValueError(f"MLU constraints require single-Trie {CONSTRAINT_TABLE_FORMAT}")
    if dtype not in LOGIT_DTYPES:
        raise ValueError(f"MLU constraint logits must be fp16/bf16, got {dtype}")
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS:
        raise ValueError(f"MLU max_rows must be in [1, {MAX_ROWS}], got {max_rows!r}")
    if type(max_top_k) is not int or not 1 <= max_top_k <= MAX_TOP_K:
        raise ValueError(f"MLU max_top_k must be in [1, {MAX_TOP_K}], got {max_top_k!r}")

    table = DeviceConstraintTable.upload(artifact, device)
    info = ConstraintBackendInfo(
        table_metadata=artifact.metadata,
        backend=ConstraintBackendKind.MLU,
        max_top_k=max_top_k,
        max_rows=max_rows,
        host_memory_bytes=0,
        device_memory_bytes=table.nbytes,
    )
    return MLUConstraintBackend(table, info, dtype)
