# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable runtime contract owned by exactly one ModelRunner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from vllm_gr.constraints.table import ConstraintTableInfo
from vllm_gr.ops.rec_constrained_topk import _INVALID_LOGIT as INVALID_LOGIT


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
    def info(self) -> ConstraintTableInfo: ...

    def select(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput: ...

    def close(self) -> None: ...


def validate_select_inputs(
    logits: torch.Tensor,
    metadata: ConstraintMetadata,
    info: ConstraintTableInfo,
) -> None:
    if logits.dim() != 2:
        raise ValueError(f"logits must be [rows, vocab], got {tuple(logits.shape)}")
    rows, vocab_size = logits.shape
    if vocab_size != info.vocab_size:
        raise ValueError(
            f"logits vocab {vocab_size} must equal constraint vocab_size {info.vocab_size}"
        )
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"logits dtype must be fp16/bf16/fp32, got {logits.dtype}")

    prefixes = metadata.prefix_token_ids
    if prefixes.dim() != 2 or prefixes.shape[0] != rows:
        raise ValueError(
            f"prefix_token_ids must be [rows, max_steps], got {tuple(prefixes.shape)}"
        )
    if prefixes.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"prefix_token_ids must be int32/int64, got {prefixes.dtype}")
    if metadata.current_step not in (0, 1, 2):
        raise ValueError(
            f"current_step must be 0, 1, or 2, got {metadata.current_step!r}"
        )
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
    if not bool(torch.isfinite(temperatures).all()) or bool((temperatures <= 0).any()):
        raise ValueError("temperatures must contain only finite positive values")


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
    token_ids = token_ids.to(dtype=torch.int32)
    logprobs = logprobs.to(dtype=torch.float32)
    for name, tensor in (("token_ids", token_ids), ("logprobs", logprobs)):
        if tensor.device != logits.device:
            raise RuntimeError(
                f"constraint {name} device {tensor.device} must match logits {logits.device}"
            )
        if not tensor.is_contiguous():
            raise RuntimeError(f"constraint {name} must be contiguous")
    valid_mask = logprobs > INVALID_LOGIT / 2
    return ConstraintOutput(
        token_ids=token_ids,
        logprobs=logprobs,
        valid_mask=valid_mask,
    )
