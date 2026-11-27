# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Legacy Top-K adapter over the same MLU constraint selection primitives."""

from __future__ import annotations

import torch

from vllm_gr.constraints.compiler import (
    ConstraintTableResource,
    validate_resource_tables,
)
from vllm_gr.constraints.execution import ConstrainedTopKTensorResult

from .selection import (
    LOGIT_DTYPES,
    MAX_ROWS,
    MAX_TOP_K,
    DeviceConstraintTable,
    require_mlu_device,
)


class MLUConstrainedTopKExecution:
    def __init__(self, device: str | torch.device | None = None) -> None:
        self._requested_device = "mlu" if device is None else device
        if str(self._requested_device).partition(":")[0] != "mlu":
            raise ValueError(f"MLU constraints require an mlu device, got {device}")
        self._digest: str | None = None
        self._table: DeviceConstraintTable | None = None

    def is_available(self) -> bool:
        try:
            from torch_mlu import mlu
        except (ImportError, OSError):
            return False
        return bool(mlu.is_available())

    def load_tables(self, resource: ConstraintTableResource) -> None:
        if self._digest == resource.digest:
            return
        if self._digest is not None:
            raise ValueError("cannot replace loaded MLU constraint tables (call reset() first)")
        validate_resource_tables(resource)
        if resource.vocab_size <= 0:
            raise ValueError("MLU constraint vocabulary must be positive")
        if not self.is_available():
            raise RuntimeError("torch_mlu and an MLU device are required for constraint execution")
        device = torch.device(self._requested_device)
        require_mlu_device(device)
        self._table = DeviceConstraintTable.upload(resource, device)
        self._digest = resource.digest

    def reset(self) -> None:
        self._table = None
        self._digest = None

    def close(self) -> None:
        self.reset()

    def run(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> ConstrainedTopKTensorResult:
        table = self._table
        if table is None:
            raise RuntimeError("MLU constraint tables not loaded; call load_tables first")
        if type(current_step) is not int or current_step not in (0, 1, 2):
            raise ValueError(f"MLU current_step must be 0, 1, or 2, got {current_step!r}")
        if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"MLU top_k must be in [1, {MAX_TOP_K}], got {top_k!r}")
        if (
            logits.ndim != 2
            or not 1 <= logits.shape[0] <= MAX_ROWS
            or logits.shape[1] != table.vocab_size
            or logits.dtype not in LOGIT_DTYPES
            or not logits.is_contiguous()
        ):
            raise ValueError("MLU logits must be contiguous fp16/bf16 [rows, full vocabulary]")
        if not (logits.device == sequence_group.device == temperatures.device == table.device):
            raise ValueError("MLU constraint inputs must share the table device")
        rows = logits.shape[0]
        if sequence_group.dtype != torch.int32 or sequence_group.ndim != 3:
            raise ValueError("sequence_group must be int32 [batch, width, steps]")
        if current_step == 0 and sequence_group.shape != (rows, 1, 1):
            raise ValueError("MLU Prefill sequence_group must be [rows, 1, 1]")
        batch, width, steps = sequence_group.shape
        if rows != batch * width or steps <= current_step:
            raise ValueError("MLU sequence_group must provide one prefix per logits row")
        if (
            temperatures.dtype != torch.float32
            or temperatures.ndim != 1
            or temperatures.numel() not in (0, 1, rows)
        ):
            raise ValueError("MLU temperatures must be float32 [0], [1] or [rows]")
        result = table.select(
            logits, sequence_group.reshape(rows, steps), temperatures, current_step, top_k
        )
        return ConstrainedTopKTensorResult(result.token_ids, result.logprobs, result.valid_mask)
