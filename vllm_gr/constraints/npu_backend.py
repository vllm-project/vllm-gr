# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU backend owning only six Device tensors, metadata, and one callable."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

import torch

from vllm_gr.constraints.backend import (
    ConstraintMetadata,
    ConstraintOutput,
    build_constraint_output,
    validate_select_inputs,
)
from vllm_gr.constraints.table import ConstraintTableInfo

NPUOp = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def resolve_npu_constraint_op() -> NPUOp:
    """Resolve the production callable once; never return a Torch fallback."""

    module = import_module("vllm_gr.ops.npu.rec_constrained_topk")
    custom_ops = module._custom_ops()
    op = getattr(custom_ops, "rec_constrained_topk", None) if custom_ops is not None else None
    if not callable(op):
        reason = module.unavailable_reason()
        raise RuntimeError(f"NPU rec_constrained_topk is unavailable: {reason}")
    return op


class NPUConstraintBackend:
    __slots__ = (
        "_first_token_ids",
        "_prefix1_offsets",
        "_prefix1_values",
        "_prefix1_pair_keys",
        "_prefix2_value_offsets",
        "_prefix2_values",
        "_info",
        "_op",
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
        info: ConstraintTableInfo,
        op: NPUOp,
    ) -> None:
        self._first_token_ids: torch.Tensor | None = first_token_ids
        self._prefix1_offsets: torch.Tensor | None = prefix1_offsets
        self._prefix1_values: torch.Tensor | None = prefix1_values
        self._prefix1_pair_keys: torch.Tensor | None = prefix1_pair_keys
        self._prefix2_value_offsets: torch.Tensor | None = prefix2_value_offsets
        self._prefix2_values: torch.Tensor | None = prefix2_values
        self._info = info
        self._op: NPUOp | None = op

    @property
    def info(self) -> ConstraintTableInfo:
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
        if self._op is None or any(table is None for table in tables):
            raise RuntimeError("NPU constraint backend is closed")
        return tables  # type: ignore[return-value]

    def select(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput:
        validate_select_inputs(logits, metadata, self._info)
        tables = self._tables()
        if any(table.device != logits.device for table in tables):
            raise RuntimeError("all NPU constraint tables must share logits.device")
        if self._op is None:
            raise RuntimeError("NPU constraint backend is closed")

        prefixes = metadata.prefix_token_ids.to(device=logits.device, dtype=torch.int32)
        # The current ACS ABI accepts [B, W, max_steps]; a singleton W keeps
        # the public Backend contract flat at [rows, max_steps].
        sequence_group = prefixes.unsqueeze(1).contiguous()
        temperatures = metadata.temperatures.to(device=logits.device).contiguous()
        token_ids, logprobs = self._op(
            logits,
            sequence_group,
            *tables,
            temperatures,
            metadata.current_step,
            metadata.top_k,
            self._info.max_prefix1_degree,
            self._info.max_prefix2_degree,
        )
        return build_constraint_output(
            token_ids.contiguous(),
            logprobs.contiguous(),
            logits=logits,
            top_k=metadata.top_k,
        )

    def close(self) -> None:
        self._first_token_ids = None
        self._prefix1_offsets = None
        self._prefix1_values = None
        self._prefix1_pair_keys = None
        self._prefix2_value_offsets = None
        self._prefix2_values = None
        self._op = None
