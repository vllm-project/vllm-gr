# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Device-agnostic runtime contract for canonical constraint tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from vllm_gr.config.models import ConstraintBackendKind
from vllm_gr.constraints.scoring import INVALID_LOGIT
from vllm_gr.constraints.table import ConstraintTableMetadata

_LOGIT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True, slots=True)
class ConstraintBackendInfo:
    """Immutable construction and ownership diagnostics for one Backend."""

    table_metadata: ConstraintTableMetadata
    backend: ConstraintBackendKind
    max_top_k: int
    max_rows: int
    host_memory_bytes: int
    device_memory_bytes: int


@dataclass(frozen=True, slots=True)
class ConstraintMetadata:
    """Per-call constrained-selection metadata.

    ``temperatures`` is a caller-validated tensor of finite, non-negative
    values.  Zero has the established constrained-operator meaning of one:
    logits are left unscaled, rather than divided by zero. Positive values
    scale logits before normalization. The CPU and CUDA Backends normalize
    over the complete shared logits window so candidate scores are
    comparable across parent rows.

    Temperatures are validated at the request boundary. Checked Backend calls
    also validate root ID ranges, which synchronizes when IDs are on a device.
    Worker admission validates IDs once; unchecked sampling skips this check.
    """

    prefix_token_ids: torch.Tensor  # int32 [rows, 3]
    temperatures: torch.Tensor  # float32 [1] or [rows]
    current_step: int  # 0 / 1 / 2
    top_k: int
    trie_ids: torch.Tensor | None = None  # int32 [rows], validated at admission


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
    *,
    expected_logits_vocab_size: int | None = None,
) -> None:
    """Validate the common ABI and explicit root ID ranges.

    In particular, the caller must have already established that temperature
    values are finite and non-negative as documented by
    :class:`ConstraintMetadata`. Root range validation may synchronize a device;
    the prevalidated Worker sampling path does not call this function.
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
    if metadata.trie_ids is not None:
        roots = metadata.trie_ids
        if (
            roots.shape != (rows,)
            or roots.dtype != torch.int32
            or roots.device != logits.device
            or not roots.is_contiguous()
        ):
            raise ValueError("trie_ids must be contiguous int32 [rows] on logits.device")
        if bool(((roots < 0) | (roots >= info.table_metadata.trie_count)).any()):
            raise ValueError("trie_id is outside the constraint table")
    elif info.table_metadata.trie_count > 1:
        raise ValueError("multi-trie selection requires trie_ids")
    if not 1 <= rows <= info.max_rows:
        raise ValueError(f"logits rows must be in [1, {info.max_rows}], got {rows}")
    expected_vocab = (
        info.table_metadata.vocab_size
        if expected_logits_vocab_size is None
        else expected_logits_vocab_size
    )
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
