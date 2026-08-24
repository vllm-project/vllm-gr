# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable runtime contract owned by exactly one ModelRunner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from vllm_gr.constraints.table import (
    ConstraintTableInfo,
    validate_constraint_table_info,
)
from vllm_gr.ops.rec_constrained_topk import _INVALID_LOGIT

INVALID_LOGIT: float = _INVALID_LOGIT


def _validate_backend_integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"constraint backend {name} must be a {qualifier} integer")
    return value


@dataclass(frozen=True, slots=True)
class ConstraintBackendInfo:
    table: ConstraintTableInfo
    max_top_k: int
    device_memory_bytes: int

    def __post_init__(self) -> None:
        validate_constraint_table_info(self.table, require_artifact_digest=True)
        _validate_backend_integer(self.max_top_k, "max_top_k", minimum=1)
        _validate_backend_integer(
            self.device_memory_bytes,
            "device_memory_bytes",
            minimum=0,
        )


@dataclass(frozen=True, slots=True)
class ConstraintMetadata:
    prefix_token_ids: torch.Tensor  # [rows, max_steps]
    temperatures: torch.Tensor  # [1] or [rows]
    current_step: int
    top_k: int


@dataclass(frozen=True, slots=True)
class ConstraintOutput:
    token_ids: torch.Tensor
    logprobs: torch.Tensor
    valid_mask: torch.Tensor


@runtime_checkable
class ConstraintBackend(Protocol):
    @property
    def info(self) -> ConstraintBackendInfo: ...

    def select(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput: ...

    def close(self) -> None: ...


def validate_select_inputs(
    logits: torch.Tensor,
    metadata: ConstraintMetadata,
    info: ConstraintBackendInfo,
) -> None:
    if logits.dim() != 2:
        raise ValueError(f"logits must be [rows, vocab], got {tuple(logits.shape)}")
    rows, vocab_size = logits.shape
    if vocab_size != info.table.vocab_size:
        raise ValueError(
            f"logits vocab {vocab_size} must equal constraint vocab_size {info.table.vocab_size}"
        )
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"logits dtype must be fp16/bf16/fp32, got {logits.dtype}")

    prefixes = metadata.prefix_token_ids
    if prefixes.dim() != 2 or prefixes.shape[0] != rows:
        raise ValueError(f"prefix_token_ids must be [rows, max_steps], got {tuple(prefixes.shape)}")
    if prefixes.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"prefix_token_ids must be int32/int64, got {prefixes.dtype}")
    if metadata.current_step not in (0, 1, 2):
        raise ValueError(f"current_step must be 0, 1, or 2, got {metadata.current_step!r}")
    if prefixes.shape[1] < metadata.current_step:
        raise ValueError(
            f"prefix_token_ids width {prefixes.shape[1]} is smaller than step "
            f"{metadata.current_step}"
        )
    if (
        isinstance(metadata.top_k, bool)
        or not isinstance(metadata.top_k, int)
        or not 1 <= metadata.top_k <= info.max_top_k
    ):
        raise ValueError(f"top_k must be in [1, {info.max_top_k}], got {metadata.top_k!r}")
    temperatures = metadata.temperatures
    if temperatures.dim() != 1 or temperatures.numel() not in (1, rows):
        raise ValueError(
            f"temperatures must have shape [1] or [{rows}], got {tuple(temperatures.shape)}"
        )
    if temperatures.dtype != torch.float32:
        raise ValueError(f"temperatures must be float32, got {temperatures.dtype}")


def effective_temperatures(temperatures: torch.Tensor) -> torch.Tensor:
    """Map greedy temperature zero to an unscaled divisor without Host sync."""

    return torch.where(temperatures == 0, torch.ones_like(temperatures), temperatures)


def build_constraint_output(
    token_ids: torch.Tensor,
    logprobs: torch.Tensor,
    *,
    logits: torch.Tensor,
    top_k: int,
) -> ConstraintOutput:
    expected_shape = (logits.shape[0], top_k)
    if tuple(token_ids.shape) != expected_shape or tuple(logprobs.shape) != expected_shape:
        raise RuntimeError(
            f"constraint backend returned shape {tuple(token_ids.shape)} / "
            f"{tuple(logprobs.shape)}, expected {expected_shape}"
        )
    if token_ids.dtype != torch.int32:
        raise RuntimeError(f"constraint token_ids dtype must be int32, got {token_ids.dtype}")
    if logprobs.dtype != torch.float32:
        raise RuntimeError(f"constraint logprobs dtype must be float32, got {logprobs.dtype}")
    for name, tensor in (("token_ids", token_ids), ("logprobs", logprobs)):
        if tensor.device != logits.device:
            raise RuntimeError(
                f"constraint {name} device {tensor.device} must match logits {logits.device}"
            )
        if not tensor.is_contiguous():
            raise RuntimeError(f"constraint {name} must be contiguous")
    valid_mask = logprobs > INVALID_LOGIT / 2
    if not valid_mask.is_contiguous():
        raise RuntimeError("constraint valid_mask must be contiguous")
    return ConstraintOutput(
        token_ids=token_ids,
        logprobs=logprobs,
        valid_mask=valid_mask,
    )
