# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Device-agnostic runtime contract for canonical constraint tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from vllm_gr.config.models import ConstraintTableBackendEnum
from vllm_gr.constraints.table import ConstraintTableMetadata

INVALID_LOGIT = -1.0e20
_LOGIT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True, slots=True)
class ConstraintBackendInfo:
    """Immutable construction and ownership diagnostics for one Backend."""

    table_metadata: ConstraintTableMetadata
    backend: ConstraintTableBackendEnum
    max_top_k: int
    max_rows: int
    host_memory_bytes: int
    device_memory_bytes: int


@dataclass(frozen=True, slots=True)
class ConstraintMetadata:
    """Per-call constrained-selection metadata.

    ``temperatures`` is a caller-validated tensor of finite, non-negative
    values.  Zero has the established constrained-operator meaning of one:
    logits are left unscaled, rather than divided by zero.  Positive values
    scale logits before normalization over the complete legal candidate set.

    Value-domain validation intentionally remains at the request boundary.
    Inspecting a device tensor here would synchronize every NPU ``select``;
    this Backend boundary therefore validates only its structural ABI.
    """

    prefix_token_ids: torch.Tensor  # int32 [rows, 3]
    temperatures: torch.Tensor  # float32 [1] or [rows]
    current_step: int  # 0 / 1 / 2
    top_k: int


@dataclass(frozen=True, slots=True)
class ConstraintOutput:
    """Fixed-width constrained candidates on the logits device.

    Without a deterministic fused-kernel tie-break, an exact tie at the Top-K
    cutoff does not guarantee token-set or slot-order parity across Backends.
    Every returned valid token and its normalized logprob remain contractual.
    """

    token_ids: torch.Tensor  # int32 [rows, top_k]
    logprobs: torch.Tensor  # float32 [rows, top_k]
    valid_mask: torch.Tensor  # bool [rows, top_k]


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
    """Validate the common structural ABI without copying or device sync.

    In particular, the caller must have already established that temperature
    values are finite and non-negative as documented by
    :class:`ConstraintMetadata`.
    """

    if not isinstance(metadata, ConstraintMetadata):
        raise ValueError("metadata must be ConstraintMetadata")
    if logits.dim() != 2:
        raise ValueError(f"logits must be [rows, vocab], got {tuple(logits.shape)}")
    if not logits.is_contiguous():
        raise ValueError("logits must be contiguous")
    if logits.dtype not in _LOGIT_DTYPES:
        raise ValueError(f"logits dtype must be fp16/bf16/fp32, got {logits.dtype}")

    rows, vocab_size = logits.shape
    if not 1 <= rows <= info.max_rows:
        raise ValueError(f"logits rows must be in [1, {info.max_rows}], got {rows}")
    expected_vocab = info.table_metadata.vocab_size
    if vocab_size != expected_vocab:
        raise ValueError(
            f"logits vocab {vocab_size} must equal constraint vocab_size {expected_vocab}"
        )

    prefixes = metadata.prefix_token_ids
    if tuple(prefixes.shape) != (rows, 3):
        raise ValueError(
            f"prefix_token_ids must have shape [{rows}, 3], got {tuple(prefixes.shape)}"
        )
    if prefixes.dtype != torch.int32:
        raise ValueError(f"prefix_token_ids must be int32, got {prefixes.dtype}")

    temperatures = metadata.temperatures
    if temperatures.dim() != 1 or temperatures.numel() not in (1, rows):
        raise ValueError(
            f"temperatures must have shape [1] or [{rows}], got {tuple(temperatures.shape)}"
        )
    if temperatures.dtype != torch.float32:
        raise ValueError(f"temperatures must be float32, got {temperatures.dtype}")

    for name, tensor in (
        ("prefix_token_ids", prefixes),
        ("temperatures", temperatures),
    ):
        if tensor.device != logits.device:
            raise ValueError(
                f"{name}.device {tensor.device} must match logits.device {logits.device}"
            )

    current_step = metadata.current_step
    if isinstance(current_step, bool) or not isinstance(current_step, int):
        raise ValueError(f"current_step must be an integer, got {current_step!r}")
    if current_step not in (0, 1, 2):
        raise ValueError(f"current_step must be 0, 1, or 2, got {current_step}")

    top_k = metadata.top_k
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= info.max_top_k:
        raise ValueError(f"top_k must be in [1, {info.max_top_k}], got {top_k!r}")


def build_constraint_output(
    token_ids: torch.Tensor,
    logprobs: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    logits: torch.Tensor,
    top_k: int,
) -> ConstraintOutput:
    """Validate raw outputs and their independently derived validity mask.

    Deliberately do not cast or make outputs contiguous here. A real operator
    returning the wrong ABI must fail at this boundary rather than be repaired
    invisibly in Python.
    """

    expected_shape = (logits.shape[0], top_k)
    if (
        tuple(token_ids.shape) != expected_shape
        or tuple(logprobs.shape) != expected_shape
        or tuple(valid_mask.shape) != expected_shape
    ):
        raise RuntimeError(
            "constraint backend returned shape "
            f"{tuple(token_ids.shape)} / {tuple(logprobs.shape)} / "
            f"{tuple(valid_mask.shape)}, expected {expected_shape}"
        )
    if token_ids.dtype != torch.int32:
        raise RuntimeError(f"constraint token_ids must be int32, got {token_ids.dtype}")
    if logprobs.dtype != torch.float32:
        raise RuntimeError(f"constraint logprobs must be float32, got {logprobs.dtype}")
    if valid_mask.dtype != torch.bool:
        raise RuntimeError(f"constraint valid_mask must be bool, got {valid_mask.dtype}")
    for name, tensor in (
        ("token_ids", token_ids),
        ("logprobs", logprobs),
        ("valid_mask", valid_mask),
    ):
        if tensor.device != logits.device:
            raise RuntimeError(
                f"constraint {name}.device {tensor.device} must match logits.device {logits.device}"
            )
        if not tensor.is_contiguous():
            raise RuntimeError(f"constraint {name} must be contiguous")

    return ConstraintOutput(
        token_ids=token_ids,
        logprobs=logprobs,
        valid_mask=valid_mask,
    )


__all__ = [
    "INVALID_LOGIT",
    "ConstraintBackend",
    "ConstraintBackendInfo",
    "ConstraintMetadata",
    "ConstraintOutput",
]
