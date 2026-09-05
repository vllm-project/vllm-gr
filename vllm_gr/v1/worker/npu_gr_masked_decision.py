# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mask-aware device Beam selection for persistent NPU GR.

The installed xllm decision operators do not accept structural candidate
validity.  This stage therefore implements the missing ordering contract with
ordinary device tensor operators and caller-owned fixed storage.  It does not
materialize Beam state on the Host and does not invent an xllm symbol.

Selection uses two stable sorts.  The first orders by cumulative score and
original flat candidate index.  The second makes structural validity the
primary key while preserving the first ordering.  Consequently a legal
``-inf`` candidate is selected ahead of invalid ``-inf`` padding without
substituting a finite score sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class NPUMaskedDecisionStep:
    device_step: int
    token_ids: torch.Tensor
    parent_ids: torch.Tensor
    scores: torch.Tensor
    sequence: torch.Tensor
    active_mask: torch.Tensor
    valid_mask: torch.Tensor
    local_scores: torch.Tensor
    parent_prefix: torch.Tensor | None
    status: torch.Tensor

    @property
    def reorder_parents(self) -> torch.Tensor | None:
        return self.parent_ids if self.parent_prefix is not None else None

    @property
    def reorder_prefix(self) -> torch.Tensor | None:
        return self.parent_prefix


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype is not dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


class NPUMaskedBeamDecision:
    """Fixed-capacity B=1 Beam selector using only device tensor operations."""

    def __init__(
        self,
        *,
        beam_width: int,
        max_steps: int,
        device: torch.device | str,
        cpu_test_backend: bool = False,
    ) -> None:
        width = _positive_int(beam_width, "beam_width")
        steps = _positive_int(max_steps, "max_steps")
        resolved = torch.device(device)
        if resolved.type != "npu" and not cpu_test_backend:
            raise RuntimeError("persistent NPU masked decision requires an NPU device")
        candidates = width * width
        self.beam_width = width
        self.max_steps = steps
        self.device = resolved
        self._cpu_test_backend = cpu_test_backend

        self._combined = torch.empty((width, width), dtype=torch.float32, device=resolved)
        self._effective_valid = torch.empty((width, width), dtype=torch.bool, device=resolved)
        self._numeric_scratch = torch.empty_like(self._effective_valid)
        self._score_values = torch.empty((candidates,), dtype=torch.float32, device=resolved)
        self._score_order = torch.empty((candidates,), dtype=torch.int64, device=resolved)
        self._valid_values = torch.empty((candidates,), dtype=torch.bool, device=resolved)
        self._valid_order = torch.empty((candidates,), dtype=torch.int64, device=resolved)
        self._ordered_valid = torch.empty((candidates,), dtype=torch.bool, device=resolved)
        self._winner_flat = torch.empty((width,), dtype=torch.int64, device=resolved)
        self._winner_valid = torch.empty((width,), dtype=torch.bool, device=resolved)
        self._winner_parents = torch.empty((width,), dtype=torch.int64, device=resolved)
        self._winner_columns = torch.empty((width,), dtype=torch.int64, device=resolved)
        self._winner_tokens = torch.empty((width,), dtype=torch.int32, device=resolved)
        self._winner_scores = torch.empty((width,), dtype=torch.float32, device=resolved)
        self._winner_local_scores = torch.empty((width,), dtype=torch.float32, device=resolved)
        self._safe_parents = torch.empty((width,), dtype=torch.int64, device=resolved)
        self._safe_tokens = torch.empty((width,), dtype=torch.int32, device=resolved)
        self._safe_scores = torch.empty((width,), dtype=torch.float32, device=resolved)
        self._safe_local_scores = torch.empty((width,), dtype=torch.float32, device=resolved)
        self._group_parents = torch.empty((width,), dtype=torch.int64, device=resolved)
        self._group_order = torch.empty((width,), dtype=torch.int64, device=resolved)
        self._selected_tokens = torch.empty((1, width), dtype=torch.int32, device=resolved)
        self._selected_parents = torch.empty((1, width), dtype=torch.int32, device=resolved)
        self._selected_scores = torch.empty((1, width), dtype=torch.float32, device=resolved)
        self._selected_local_scores = torch.empty((1, width), dtype=torch.float32, device=resolved)
        self._selected_active = torch.empty((1, width), dtype=torch.bool, device=resolved)
        self._selected_valid = torch.empty((1, width), dtype=torch.bool, device=resolved)
        self._parent_counts = torch.empty((1, width), dtype=torch.int32, device=resolved)
        self._parent_prefix = torch.empty((1, width), dtype=torch.int32, device=resolved)
        self._next_sequence = torch.empty((1, width, steps), dtype=torch.int32, device=resolved)
        self._status = torch.empty((1,), dtype=torch.int32, device=resolved)
        self._bool_status = torch.empty((1,), dtype=torch.bool, device=resolved)
        self._ones = torch.ones((1, width), dtype=torch.int32, device=resolved)
        self._zero_parents = torch.zeros((width,), dtype=torch.int64, device=resolved)
        self._zero_tokens = torch.zeros((width,), dtype=torch.int32, device=resolved)
        self._negative_infinity = torch.full(
            (width,), float("-inf"), dtype=torch.float32, device=resolved
        )
        # Bind to the actual allocation device (for example ``npu:1``).
        self.device = self._combined.device

    def _validate_session(self, session: Any, expected_device_step: int) -> tuple[Any, Any]:
        step = _positive_int(expected_device_step, "expected_device_step")
        if step >= self.max_steps:
            raise ValueError("persistent NPU masked decision exceeds max_steps")
        if getattr(session, "is_active", None) is not True:
            raise RuntimeError("persistent NPU masked decision requires an active session")
        if getattr(session, "beam_width", None) != self.beam_width:
            raise RuntimeError("persistent NPU masked decision Beam width changed")
        if getattr(session, "slot", None) != 0:
            raise RuntimeError("persistent NPU masked decision requires B=1 slot zero")
        if getattr(session, "decode_step", None) != step - 1:
            raise RuntimeError("persistent NPU masked decision session step mismatch")
        buffers = getattr(getattr(session, "context", None), "decision", None)
        if buffers is None or getattr(buffers, "worker_decode_steps", None) != step:
            raise RuntimeError("persistent NPU masked decision canonical step mismatch")
        sequence = _tensor(
            getattr(buffers, "sequence", None),
            name="canonical sequence",
            shape=(1, self.beam_width, self.max_steps),
            dtype=torch.int32,
            device=self.device,
        )
        scores = _tensor(
            getattr(buffers, "beam_scores", None),
            name="canonical scores",
            shape=(1, self.beam_width),
            dtype=torch.float32,
            device=self.device,
        )
        return buffers, (sequence, scores)

    def select(
        self,
        session: Any,
        candidate_token_ids: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        *,
        expected_device_step: int,
        is_last_step: bool,
        result_width: int,
    ) -> NPUMaskedDecisionStep:
        """Select one generation stage and commit canonical device state."""
        buffers, canonical = self._validate_session(session, expected_device_step)
        sequence, cumulative = canonical
        width = self.beam_width
        _tensor(
            candidate_token_ids,
            name="candidate_token_ids",
            shape=(width, width),
            dtype=torch.int32,
            device=self.device,
        )
        _tensor(
            candidate_scores,
            name="candidate_scores",
            shape=(width, width),
            dtype=torch.float32,
            device=self.device,
        )
        _tensor(
            candidate_valid_mask,
            name="candidate_valid_mask",
            shape=(width, width),
            dtype=torch.bool,
            device=self.device,
        )
        if not isinstance(is_last_step, bool):
            raise TypeError("is_last_step must be a boolean")
        result_width = _positive_int(result_width, "result_width")
        if result_width > width or (not is_last_step and result_width != width):
            raise ValueError("persistent NPU masked decision result_width is invalid")

        torch.add(cumulative.view(width, 1), candidate_scores, out=self._combined)
        # ``isnan``/``isposinf`` do not expose ``out=`` on all supported
        # PyTorch builds.  Equality gives the same classification while
        # retaining fixed scratch: NaN is the only value unequal to itself.
        torch.eq(self._combined, self._combined, out=self._numeric_scratch)
        self._numeric_scratch.logical_not_()
        torch.eq(self._combined, float("inf"), out=self._effective_valid)
        self._numeric_scratch.logical_or_(self._effective_valid)
        torch.logical_not(self._numeric_scratch, out=self._effective_valid)
        self._effective_valid.logical_and_(candidate_valid_mask)

        # Score order is stable by original flat candidate index.  The second
        # stable sort promotes every structurally valid entry ahead of padding
        # while preserving that score order, including legal -inf ties.
        torch.sort(
            self._combined.view(-1),
            descending=True,
            stable=True,
            out=(self._score_values, self._score_order),
        )
        torch.index_select(
            self._effective_valid.view(-1),
            0,
            self._score_order,
            out=self._ordered_valid,
        )
        torch.sort(
            self._ordered_valid,
            descending=True,
            stable=True,
            out=(self._valid_values, self._valid_order),
        )
        torch.index_select(
            self._score_order,
            0,
            self._valid_order[:width],
            out=self._winner_flat,
        )
        torch.index_select(
            self._effective_valid.view(-1),
            0,
            self._winner_flat,
            out=self._winner_valid,
        )
        torch.div(self._winner_flat, width, rounding_mode="floor", out=self._winner_parents)
        torch.remainder(self._winner_flat, width, out=self._winner_columns)
        torch.index_select(
            candidate_token_ids.view(-1),
            0,
            self._winner_flat,
            out=self._winner_tokens,
        )
        torch.index_select(
            self._combined.view(-1),
            0,
            self._winner_flat,
            out=self._winner_scores,
        )
        torch.index_select(
            candidate_scores.view(-1),
            0,
            self._winner_flat,
            out=self._winner_local_scores,
        )
        torch.where(
            self._winner_valid,
            self._winner_parents,
            self._zero_parents,
            out=self._safe_parents,
        )
        torch.where(
            self._winner_valid,
            self._winner_tokens,
            self._zero_tokens,
            out=self._safe_tokens,
        )
        torch.where(
            self._winner_valid,
            self._winner_scores,
            self._negative_infinity,
            out=self._safe_scores,
        )
        torch.where(
            self._winner_valid,
            self._winner_local_scores,
            self._negative_infinity,
            out=self._safe_local_scores,
        )

        if is_last_step:
            self._selected_tokens[0].copy_(self._safe_tokens)
            self._selected_parents[0].copy_(self._safe_parents)
            self._selected_scores[0].copy_(self._safe_scores)
            self._selected_local_scores[0].copy_(self._safe_local_scores)
            self._selected_valid[0].copy_(self._winner_valid)
            torch.gt(
                self._safe_scores,
                float("-inf"),
                out=self._selected_active[0],
            )
            self._selected_active[0].logical_and_(self._selected_valid[0])
            parent_prefix = None
            sequence_parents = self._safe_parents
        else:
            torch.sort(
                self._safe_parents,
                stable=True,
                out=(self._group_parents, self._group_order),
            )
            torch.index_select(
                self._safe_tokens,
                0,
                self._group_order,
                out=self._selected_tokens[0],
            )
            self._selected_parents[0].copy_(self._group_parents)
            torch.index_select(
                self._safe_scores,
                0,
                self._group_order,
                out=self._selected_scores[0],
            )
            torch.index_select(
                self._safe_local_scores,
                0,
                self._group_order,
                out=self._selected_local_scores[0],
            )
            torch.index_select(
                self._winner_valid,
                0,
                self._group_order,
                out=self._selected_valid[0],
            )
            torch.gt(
                self._selected_scores,
                float("-inf"),
                out=self._selected_active,
            )
            self._selected_active.logical_and_(self._selected_valid)
            self._parent_counts.zero_()
            self._parent_counts.scatter_add_(1, self._group_parents.view(1, width), self._ones)
            torch.cumsum(self._parent_counts, dim=1, out=self._parent_prefix)
            parent_prefix = self._parent_prefix
            sequence_parents = self._group_parents

        selected_width = result_width if is_last_step else width
        torch.index_select(
            sequence[0],
            0,
            sequence_parents[:selected_width],
            out=self._next_sequence[0, :selected_width],
        )
        self._next_sequence[0, :selected_width, expected_device_step].copy_(
            self._selected_tokens[0, :selected_width]
        )

        # Defer empty/invalid-numeric failure to the single final status copy.
        torch.logical_not(
            self._selected_valid[0, :selected_width],
            out=self._valid_values[:selected_width],
        )
        self._bool_status.copy_(torch.any(self._valid_values[:selected_width]))
        self._status.copy_(self._bool_status)

        # Canonical state changes only after input validation and selection.
        sequence[:, :selected_width].copy_(self._next_sequence[:, :selected_width])
        cumulative[:, :selected_width].copy_(self._selected_scores[:, :selected_width])
        buffers.worker_decode_steps = expected_device_step + 1
        return NPUMaskedDecisionStep(
            device_step=expected_device_step + 1,
            token_ids=self._selected_tokens[:, :selected_width],
            parent_ids=self._selected_parents[:, :selected_width],
            scores=cumulative[:, :selected_width],
            sequence=sequence[:, :selected_width],
            active_mask=self._selected_active[:, :selected_width],
            valid_mask=self._selected_valid[:, :selected_width],
            local_scores=self._selected_local_scores[:, :selected_width],
            parent_prefix=parent_prefix,
            status=self._status,
        )

    @property
    def device_memory_bytes(self) -> int:
        total = 0
        seen: set[int] = set()
        for value in vars(self).values():
            if not torch.is_tensor(value):
                continue
            storage = value.untyped_storage()
            pointer = int(storage.data_ptr())
            if pointer not in seen:
                seen.add(pointer)
                total += int(storage.nbytes())
        return total


__all__ = ["NPUMaskedBeamDecision", "NPUMaskedDecisionStep"]
