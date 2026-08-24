# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU backend owning only six Device tensors, metadata, and one callable."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import cast

import torch

from vllm_gr.constraints.backend import (
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
    build_constraint_output,
    validate_select_inputs,
)

NPUOp = Callable[..., tuple[torch.Tensor, torch.Tensor]]
NPUTables = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
NPU_CONSTRAINT_MAX_TOP_K = 256

_NPU_TABLE_DTYPES = (
    ("first_token_ids", torch.int32),
    ("prefix1_offsets", torch.int32),
    ("prefix1_values", torch.int32),
    ("prefix1_pair_keys", torch.int64),
    ("prefix2_value_offsets", torch.int32),
    ("prefix2_values", torch.int32),
)


def _is_npu_device(device: torch.device) -> bool:
    return device.type == "npu"


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
    """Internal strict-ABI implementation; construct it through the Worker loader."""

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
        info: ConstraintBackendInfo,
        op: NPUOp,
    ) -> None:
        if not callable(op):
            raise TypeError("NPU constraint op must be callable")
        if not isinstance(info, ConstraintBackendInfo):
            raise TypeError("NPU constraint info must be ConstraintBackendInfo")
        if (
            isinstance(info.max_top_k, bool)
            or not isinstance(info.max_top_k, int)
            or not 1 <= info.max_top_k <= NPU_CONSTRAINT_MAX_TOP_K
        ):
            raise ValueError(
                "NPU rec_constrained_topk supports top_k in "
                f"[1, {NPU_CONSTRAINT_MAX_TOP_K}], got {info.max_top_k!r}"
            )
        named_tables = (
            ("first_token_ids", first_token_ids),
            ("prefix1_offsets", prefix1_offsets),
            ("prefix1_values", prefix1_values),
            ("prefix1_pair_keys", prefix1_pair_keys),
            ("prefix2_value_offsets", prefix2_value_offsets),
            ("prefix2_values", prefix2_values),
        )
        expected_dtypes = dict(_NPU_TABLE_DTYPES)
        for name, table in named_tables:
            if not isinstance(table, torch.Tensor):
                raise TypeError(f"NPU constraint table {name} must be a torch.Tensor")
            if table.dtype != expected_dtypes[name]:
                raise ValueError(
                    f"NPU constraint table {name} must have dtype "
                    f"{expected_dtypes[name]}, got {table.dtype}"
                )
            if table.dim() != 1:
                raise ValueError(
                    f"NPU constraint table {name} must be 1-D, got shape {tuple(table.shape)}"
                )
            if not table.is_contiguous():
                raise ValueError(f"NPU constraint table {name} must be contiguous")
        table_device = first_token_ids.device
        if not _is_npu_device(table_device):
            raise ValueError(f"NPU constraint tables must be on an NPU device, got {table_device}")
        for name, table in named_tables[1:]:
            if table.device != table_device:
                raise ValueError(
                    f"NPU constraint table {name} device {table.device} "
                    f"does not match {table_device}"
                )

        self._first_token_ids: torch.Tensor | None = first_token_ids
        self._prefix1_offsets: torch.Tensor | None = prefix1_offsets
        self._prefix1_values: torch.Tensor | None = prefix1_values
        self._prefix1_pair_keys: torch.Tensor | None = prefix1_pair_keys
        self._prefix2_value_offsets: torch.Tensor | None = prefix2_value_offsets
        self._prefix2_values: torch.Tensor | None = prefix2_values
        self._info = info
        self._op: NPUOp | None = op

    @property
    def info(self) -> ConstraintBackendInfo:
        return self._info

    def _runtime_state(self) -> tuple[NPUOp, NPUTables]:
        op = self._op
        tables = (
            self._first_token_ids,
            self._prefix1_offsets,
            self._prefix1_values,
            self._prefix1_pair_keys,
            self._prefix2_value_offsets,
            self._prefix2_values,
        )
        if op is None or any(table is None for table in tables):
            raise RuntimeError("NPU constraint backend is closed")
        return op, cast(NPUTables, tables)

    def select(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput:
        validate_select_inputs(logits, metadata, self._info)
        op, tables = self._runtime_state()
        if any(table.device != logits.device for table in tables):
            raise RuntimeError("all NPU constraint tables must share logits.device")
        if not logits.is_contiguous():
            raise ValueError("NPU constraint logits must be contiguous")

        prefixes = metadata.prefix_token_ids
        if prefixes.device != logits.device:
            raise ValueError(
                f"NPU prefix_token_ids device {prefixes.device} must match logits {logits.device}"
            )
        if prefixes.dtype != torch.int32:
            raise ValueError(f"NPU prefix_token_ids must be int32, got {prefixes.dtype}")
        if prefixes.shape[1] <= metadata.current_step:
            raise ValueError(
                f"NPU prefix_token_ids width {prefixes.shape[1]} must be greater than "
                f"current_step {metadata.current_step}"
            )
        if not prefixes.is_contiguous():
            raise ValueError("NPU prefix_token_ids must be contiguous")

        temperatures = metadata.temperatures
        if temperatures.device != logits.device:
            raise ValueError(
                f"NPU temperatures device {temperatures.device} must match logits {logits.device}"
            )
        if not temperatures.is_contiguous():
            raise ValueError("NPU temperatures must be contiguous")

        # The current ACS ABI accepts [B, W, max_steps]; a singleton W keeps
        # the public Backend contract flat at [rows, max_steps]. ``unsqueeze``
        # is a view because the input contract already requires contiguity.
        sequence_group = prefixes.unsqueeze(1)
        token_ids, logprobs = op(
            logits,
            sequence_group,
            *tables,
            temperatures,
            metadata.current_step,
            metadata.top_k,
            self._info.table.max_prefix1_degree,
            self._info.table.max_prefix2_degree,
        )
        return build_constraint_output(
            token_ids,
            logprobs,
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
