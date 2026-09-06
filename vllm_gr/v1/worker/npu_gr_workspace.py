# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU-backend-owned fixed storage for persistent GR Beam result lineage.

This module owns only device tensors.  It deliberately has no NPU stream,
Event, page-locked Host buffer, D2H transport, or ModelRunner hook; those live
in the composite NPU backend. Explicit CPU allocation exists solely for
contract tests, while the production adapter requires an NPU device.

The session's ``context.decision.sequence`` and ``beam_scores`` remain the
only canonical Beam state.  This workspace retains only raw parent lineage,
FP32 cumulative-score deltas, final-pack scratch, and two control-storage
slots.  The delta is the current GPU ABI semantics; it is not raw sampler
logprob and can differ because of FP32 rounding.

Validity comes from the constraint backend's structural candidate mask,
separately from Beam activity: legal padding and inactive ``-inf`` scores are
preserved. NaN/+inf and inconsistent masks fail closed. The final transport
rejects nonzero device status after its one final D2H. This helper alone does
not establish NPU hardware parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_gr.v1.worker.gr_model_runner_state import GR_FINAL_OUTPUT_FORMAT


@dataclass(frozen=True, slots=True)
class NPUWorkspaceLease:
    """One request generation's ownership token."""

    owner_id: str
    epoch: int
    generation_steps: int
    result_width: int


@dataclass(frozen=True, slots=True)
class NPUFinalPack:
    """Device-only compact result; no Host delivery is implied."""

    buffer_format: str
    staging: torch.Tensor
    payload: torch.Tensor
    payload_words: int
    shape: tuple[int, int]
    status: torch.Tensor


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


@dataclass(slots=True)
class NPUWorkspace:
    """Fixed-capacity, single-owner device workspace for persistent NPU GR."""

    beam_width: int
    max_steps: int
    device: torch.device
    parent_history: torch.Tensor
    cumulative_delta_history: torch.Tensor
    active_mask: torch.Tensor
    previous_scores: torch.Tensor
    parent_scores: torch.Tensor
    gather_indices: torch.Tensor
    backtrace_indices: torch.Tensor
    backtrace_parents: torch.Tensor
    backtrace_deltas: torch.Tensor
    status: torch.Tensor
    final_staging_i32: torch.Tensor
    control_storage_0: torch.Tensor
    control_storage_1: torch.Tensor
    _epoch: int = 0
    _lease: NPUWorkspaceLease | None = None
    _last_released: NPUWorkspaceLease | None = None
    _canonical_sequence: torch.Tensor | None = None
    _canonical_scores: torch.Tensor | None = None
    _recorded_steps: int = 0
    _snapshot_step: int | None = None
    _reset_done: bool = False
    _zero_scores: torch.Tensor = field(repr=False, default=None)  # type: ignore[assignment]

    @classmethod
    def allocate(
        cls,
        *,
        beam_width: int,
        max_steps: int,
        device: torch.device | str,
    ) -> "NPUWorkspace":
        width = _positive_int(beam_width, "beam_width")
        steps = _positive_int(max_steps, "max_steps")
        resolved = torch.device(device)
        beam = (1, width)
        history = (1, width, steps)
        final_words = 1 + 3 * width * steps + width
        final_i32 = torch.empty((final_words,), dtype=torch.int32, device=resolved)
        # Bind the actual allocation device. An index-free "npu" resolves
        # to "npu:N"; retaining the requested alias would reject our own
        # tensors later (similarly, "cpu:0" allocations report "cpu").
        resolved = final_i32.device
        return cls(
            beam_width=width,
            max_steps=steps,
            device=resolved,
            parent_history=torch.empty(history, dtype=torch.int32, device=resolved),
            cumulative_delta_history=torch.empty(history, dtype=torch.float32, device=resolved),
            active_mask=torch.empty(beam, dtype=torch.bool, device=resolved),
            previous_scores=torch.empty(beam, dtype=torch.float32, device=resolved),
            parent_scores=torch.empty(beam, dtype=torch.float32, device=resolved),
            gather_indices=torch.empty(beam, dtype=torch.int64, device=resolved),
            backtrace_indices=torch.empty((2, 1, width), dtype=torch.int64, device=resolved),
            backtrace_parents=torch.empty(beam, dtype=torch.int32, device=resolved),
            backtrace_deltas=torch.empty(beam, dtype=torch.float32, device=resolved),
            status=torch.empty((1,), dtype=torch.int32, device=resolved),
            final_staging_i32=final_i32,
            control_storage_0=torch.empty((1, 1), dtype=torch.int32, device=resolved),
            control_storage_1=torch.empty((1, 1), dtype=torch.int32, device=resolved),
            _zero_scores=torch.zeros(beam, dtype=torch.float32, device=resolved),
        )

    def acquire(
        self,
        owner_id: str,
        *,
        generation_steps: int,
        result_width: int,
        canonical_sequence: torch.Tensor,
        canonical_scores: torch.Tensor,
    ) -> NPUWorkspaceLease:
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("owner_id must be a non-empty string")
        if self._lease is not None:
            raise RuntimeError("persistent NPU workspace already has an owner")
        steps = _positive_int(generation_steps, "generation_steps")
        width = _positive_int(result_width, "result_width")
        if steps > self.max_steps:
            raise ValueError("generation_steps exceeds fixed workspace capacity")
        if width > self.beam_width:
            raise ValueError("result_width exceeds fixed workspace beam width")
        _tensor(
            canonical_sequence,
            name="canonical sequence",
            shape=(1, self.beam_width, self.max_steps),
            dtype=torch.int32,
            device=self.device,
        )
        _tensor(
            canonical_scores,
            name="canonical scores",
            shape=(1, self.beam_width),
            dtype=torch.float32,
            device=self.device,
        )
        self._epoch += 1
        lease = NPUWorkspaceLease(owner_id, self._epoch, steps, width)
        self._lease = lease
        self._canonical_sequence = canonical_sequence
        self._canonical_scores = canonical_scores
        self._recorded_steps = 0
        self._snapshot_step = None
        self._reset_done = False
        return lease

    def _require_lease(self, lease: NPUWorkspaceLease) -> None:
        if not isinstance(lease, NPUWorkspaceLease) or lease is not self._lease:
            raise RuntimeError("stale or foreign persistent NPU workspace owner")
        if not self._reset_done:
            raise RuntimeError("persistent NPU workspace must be reset before use")

    def reset(self, lease: NPUWorkspaceLease) -> None:
        if not isinstance(lease, NPUWorkspaceLease) or lease is not self._lease:
            raise RuntimeError("stale or foreign persistent NPU workspace owner")
        if self._reset_done:
            raise RuntimeError("persistent NPU workspace lease was already reset")
        self.parent_history.zero_()
        self.cumulative_delta_history.zero_()
        self.active_mask.zero_()
        self.previous_scores.zero_()
        self.parent_scores.zero_()
        self.gather_indices.zero_()
        self.backtrace_indices.zero_()
        self.backtrace_parents.zero_()
        self.backtrace_deltas.zero_()
        self.status.zero_()
        self.final_staging_i32.zero_()
        self.control_storage_0.zero_()
        self.control_storage_1.zero_()
        self._recorded_steps = 0
        self._snapshot_step = None
        self._reset_done = True

    def _require_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if scores is not self._canonical_scores:
            raise RuntimeError("persistent NPU workspace canonical scores identity changed")
        return _tensor(
            scores,
            name="canonical scores",
            shape=(1, self.beam_width),
            dtype=torch.float32,
            device=self.device,
        )

    def _require_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence is not self._canonical_sequence:
            raise RuntimeError("persistent NPU workspace canonical sequence identity changed")
        return _tensor(
            sequence,
            name="canonical sequence",
            shape=(1, self.beam_width, self.max_steps),
            dtype=torch.int32,
            device=self.device,
        )

    def _accumulate_invalid(self, invalid: torch.Tensor) -> None:
        invalid_i32 = invalid.to(dtype=torch.int32).amax(dim=1)
        torch.maximum(self.status, invalid_i32, out=self.status)

    def before_decision(
        self,
        lease: NPUWorkspaceLease,
        canonical_scores: torch.Tensor,
        *,
        expected_device_step: int,
    ) -> torch.Tensor:
        self._require_lease(lease)
        self._require_scores(canonical_scores)
        if (
            isinstance(expected_device_step, bool)
            or not isinstance(expected_device_step, int)
            or expected_device_step != self._recorded_steps
            or expected_device_step < 1
        ):
            raise RuntimeError("persistent NPU score snapshot/device step mismatch")
        if expected_device_step >= lease.generation_steps:
            raise RuntimeError("persistent NPU score snapshot is past the terminal step")
        if self._snapshot_step is not None:
            raise RuntimeError("persistent NPU score snapshot was not consumed")
        self.previous_scores.copy_(canonical_scores)
        self._snapshot_step = expected_device_step
        return self.previous_scores

    def record_bootstrap(
        self,
        lease: NPUWorkspaceLease,
        canonical_scores: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        candidate_valid_mask: torch.Tensor,
    ) -> bool:
        self._require_lease(lease)
        scores = self._require_scores(canonical_scores)
        if self._recorded_steps != 0 or self._snapshot_step is not None:
            raise RuntimeError("persistent NPU bootstrap lineage is out of sequence")
        mask = _tensor(
            active_mask,
            name="bootstrap active_mask",
            shape=(1, self.beam_width),
            dtype=torch.bool,
            device=self.device,
        )
        candidates = _tensor(
            candidate_valid_mask,
            name="bootstrap candidate_valid_mask",
            shape=(1, self.beam_width),
            dtype=torch.bool,
            device=self.device,
        )
        # Match the GPU Prefill count contract, not a magnitude heuristic or
        # an inference from active beams. A legal score can itself be -inf.
        self._accumulate_invalid(~candidates.any(dim=1, keepdim=True))
        if self.beam_width > 1:
            self._accumulate_invalid(candidates[:, 1:] & ~candidates[:, :-1])
        bad_numeric = torch.isnan(scores) | torch.isposinf(scores)
        expected_active = candidates & (scores > float("-inf"))
        invalid = bad_numeric | (mask != expected_active)
        invalid |= ~candidates & ~torch.isneginf(scores)
        self._accumulate_invalid(invalid)
        self.active_mask.copy_(expected_active & ~invalid)
        self.parent_history[:, :, 0].zero_()
        # Preserve padding -inf in the ABI; sanitize only failed numeric input.
        torch.where(bad_numeric, self._zero_scores, scores, out=self.backtrace_deltas)
        self.cumulative_delta_history[:, :, 0].copy_(self.backtrace_deltas)
        self._recorded_steps = 1
        return self._recorded_steps == lease.generation_steps

    def record_decision(
        self,
        lease: NPUWorkspaceLease,
        parent_ids: torch.Tensor,
        selected_cumulative_scores: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        expected_device_step: int,
        selected_valid_mask: torch.Tensor | None = None,
        selected_local_scores: torch.Tensor | None = None,
    ) -> bool:
        self._require_lease(lease)
        if isinstance(expected_device_step, bool) or not isinstance(expected_device_step, int):
            raise TypeError("expected_device_step must be an integer")
        if self._snapshot_step != expected_device_step:
            raise RuntimeError("persistent NPU decision has no matching score snapshot")
        if expected_device_step != self._recorded_steps:
            raise RuntimeError("persistent NPU lineage/device step mismatch")
        if expected_device_step >= lease.generation_steps:
            raise RuntimeError("persistent NPU lineage is past the terminal step")
        decision_width = (
            lease.result_width
            if expected_device_step + 1 == lease.generation_steps
            else self.beam_width
        )
        parents = _tensor(
            parent_ids,
            name="selected parent_ids",
            shape=(1, decision_width),
            dtype=torch.int32,
            device=self.device,
        )
        selected = self._require_scores(selected_cumulative_scores)[:, :decision_width]
        mask = _tensor(
            active_mask,
            name="selected active_mask",
            shape=(1, decision_width),
            dtype=torch.bool,
            device=self.device,
        )
        explicit_structural = selected_valid_mask is not None or selected_local_scores is not None
        if explicit_structural and (selected_valid_mask is None or selected_local_scores is None):
            raise ValueError(
                "selected_valid_mask and selected_local_scores must be provided together"
            )
        if explicit_structural:
            structural = _tensor(
                selected_valid_mask,
                name="selected valid_mask",
                shape=(1, decision_width),
                dtype=torch.bool,
                device=self.device,
            )
            local_scores = _tensor(
                selected_local_scores,
                name="selected local_scores",
                shape=(1, decision_width),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            # Compatibility for the isolated finite-only xllm contract.  The
            # production composite backend always supplies structural
            # validity and exact selected local scores.
            structural = mask
            local_scores = None

        parent_valid = torch.logical_and(parents >= 0, parents < self.beam_width)
        indices = self.gather_indices[:, :decision_width]
        indices.copy_(parents)
        indices.clamp_(0, self.beam_width - 1)
        torch.gather(
            self.previous_scores,
            1,
            indices,
            out=self.parent_scores[:, :decision_width],
        )
        parent_scores = self.parent_scores[:, :decision_width]
        selected_numeric = ~torch.isnan(selected) & ~torch.isposinf(selected)
        parent_numeric = ~torch.isnan(parent_scores) & ~torch.isposinf(parent_scores)
        finite_pair = torch.logical_and(torch.isfinite(selected), torch.isfinite(parent_scores))
        valid = structural & parent_valid & selected_numeric & parent_numeric
        if local_scores is not None:
            local_numeric = ~torch.isnan(local_scores) & ~torch.isposinf(local_scores)
            valid &= local_numeric
            expected_active = structural & (selected > float("-inf")) & valid
            self._accumulate_invalid(~valid | (mask != expected_active))
        else:
            valid &= finite_pair
            self._accumulate_invalid(torch.logical_not(valid))

        zero_scores = self._zero_scores[:, :decision_width]
        local_deltas = self.backtrace_deltas[:, :decision_width]
        safe_selected = torch.where(torch.isfinite(selected), selected, zero_scores)
        safe_parents = torch.where(torch.isfinite(parent_scores), parent_scores, zero_scores)
        torch.sub(safe_selected, safe_parents, out=local_deltas)
        delta_finite = torch.isfinite(local_deltas)
        if local_scores is None:
            self._accumulate_invalid(torch.logical_not(delta_finite))
            torch.where(delta_finite, local_deltas, zero_scores, out=local_deltas)
            valid = torch.logical_and(valid, delta_finite)
        else:
            # Finite cumulative values retain the CUDA ABI's FP32 delta.
            # When either cumulative value is legal -inf, subtraction is NaN;
            # the explicit selector supplies the exact local score instead.
            torch.where(finite_pair, local_deltas, local_scores, out=local_deltas)
        self.parent_history[:, :decision_width, expected_device_step].copy_(parents)
        self.cumulative_delta_history[:, :decision_width, expected_device_step].copy_(local_deltas)
        self.active_mask.zero_()
        self.active_mask[:, :decision_width].copy_(mask & valid)
        self._recorded_steps += 1
        self._snapshot_step = None
        return self._recorded_steps == lease.generation_steps

    def control_slot(self, lease: NPUWorkspaceLease, *, device_step: int) -> torch.Tensor:
        self._require_lease(lease)
        if (
            isinstance(device_step, bool)
            or not isinstance(device_step, int)
            or device_step < 1
            or device_step > self._recorded_steps
        ):
            raise RuntimeError("persistent NPU control slot/device step mismatch")
        return self.control_storage_1 if device_step % 2 else self.control_storage_0

    def pack_final(
        self,
        lease: NPUWorkspaceLease,
        canonical_sequence: torch.Tensor,
        canonical_scores: torch.Tensor,
    ) -> NPUFinalPack:
        self._require_lease(lease)
        if self._recorded_steps != lease.generation_steps:
            raise RuntimeError("persistent NPU final pack requested before terminal step")
        if self._snapshot_step is not None:
            raise RuntimeError("persistent NPU final pack has an unconsumed score snapshot")
        canonical_sequence = self._require_sequence(canonical_sequence)
        scores = self._require_scores(canonical_scores)
        width = lease.result_width
        steps = lease.generation_steps
        count = width * steps
        packed = self.final_staging_i32[1:]
        packed[:count].view(width, steps).copy_(canonical_sequence[0, :width, :steps])
        packed_parents = packed[count : 2 * count].view(width, steps)
        packed_deltas = packed[2 * count : 3 * count].view(torch.float32).view(width, steps)
        current = self.backtrace_indices[0, 0, :width]
        following = self.backtrace_indices[1, 0, :width]
        torch.arange(width, dtype=torch.int64, device=self.device, out=current)
        for generation_step in range(steps - 1, -1, -1):
            index_valid = torch.logical_and(current >= 0, current < self.beam_width)
            self._accumulate_invalid(torch.logical_not(index_valid).view(1, width))
            current.clamp_(0, self.beam_width - 1)
            torch.index_select(
                self.parent_history[0, :, generation_step],
                0,
                current,
                out=self.backtrace_parents[0, :width],
            )
            torch.index_select(
                self.cumulative_delta_history[0, :, generation_step],
                0,
                current,
                out=self.backtrace_deltas[0, :width],
            )
            packed_parents[:, generation_step].copy_(self.backtrace_parents[0, :width])
            packed_deltas[:, generation_step].copy_(self.backtrace_deltas[0, :width])
            following.copy_(self.backtrace_parents[0, :width])
            current, following = following, current
        packed[3 * count : 3 * count + width].copy_(scores[0, :width].view(torch.int32))
        self.final_staging_i32[:1].copy_(self.status)
        payload_words = 3 * count + width
        return NPUFinalPack(
            buffer_format=GR_FINAL_OUTPUT_FORMAT,
            staging=self.final_staging_i32[: 1 + payload_words],
            payload=packed[:payload_words],
            payload_words=payload_words,
            shape=(width, steps),
            status=self.final_staging_i32[:1],
        )

    def release(self, lease: NPUWorkspaceLease) -> bool:
        if not isinstance(lease, NPUWorkspaceLease):
            raise RuntimeError("stale or foreign persistent NPU workspace owner")
        if self._lease is not None and lease is self._lease:
            self._lease = None
            self._last_released = lease
            self._canonical_sequence = None
            self._canonical_scores = None
            self._snapshot_step = None
            self._reset_done = False
            return True
        if self._lease is None and lease is self._last_released:
            return False
        raise RuntimeError("stale or foreign persistent NPU workspace owner")

    @property
    def device_memory_bytes(self) -> int:
        tensors = (
            self.parent_history,
            self.cumulative_delta_history,
            self.active_mask,
            self.previous_scores,
            self.parent_scores,
            self.gather_indices,
            self.backtrace_indices,
            self.backtrace_parents,
            self.backtrace_deltas,
            self.status,
            self.final_staging_i32,
            self.control_storage_0,
            self.control_storage_1,
            self._zero_scores,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


__all__ = ["NPUFinalPack", "NPUWorkspace", "NPUWorkspaceLease"]
