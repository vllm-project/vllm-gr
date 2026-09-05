# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-only constrained sampling boundary for persistent NPU GR requests.

This helper is deliberately not wired into ``NPUModelRunner`` yet.  It owns
the request's fixed metadata and public sampler scratch, calls the canonical
NPU constraint Backend directly, and leaves both raw candidates and their
structural validity on device.  It never reconstructs Beam state on the Host.

The Backend currently allocates its raw ``ConstraintOutput`` tensors.  Those
tensors are retained by :class:`NPUConstrainedCandidates` without copying.
Only the public wire logprobs, top-1 sampled token, ranks, structural counts,
and deferred status use fixed request-owned buffers.  A result remains valid
until it is consumed and the next selection overwrites those fixed buffers.

This boundary detects empty rows, non-prefix-packed validity masks, and NaN
or positive-infinite valid logprobs without reading device values.  Legal
negative-infinite candidates remain structural candidates.  These diagnostics
are deferred device status, not a complete Beam-decision exhaustion check,
and structural validity is never inferred from logprob finiteness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from vllm_gr.config.models import ConstraintTableBackendEnum
from vllm_gr.constraints.backend import (
    ConstraintBackend,
    ConstraintMetadata,
    ConstraintOutput,
    build_constraint_output,
)

_MAX_CONSTRAINT_STEPS = 3

# Per-row deferred status.  Later decision code must reject any non-zero value
# before treating the safe padding token as a real candidate.
NPU_SAMPLER_STATUS_OK = 0
NPU_SAMPLER_STATUS_EMPTY = 1
NPU_SAMPLER_STATUS_VALIDITY_HOLE = 2
NPU_SAMPLER_STATUS_INVALID_NUMERIC = 3


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"persistent NPU sampler {name} is invalid")
    return value


def _storage_id(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


@dataclass(frozen=True, slots=True)
class _TensorBinding:
    tensor: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    data_ptr: int
    storage_ptr: int

    @classmethod
    def capture(cls, tensor: torch.Tensor) -> "_TensorBinding":
        return cls(
            tensor=tensor,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
            data_ptr=int(tensor.data_ptr()),
            storage_ptr=_storage_id(tensor),
        )

    def validate(self, tensor: Any) -> None:
        if (
            tensor is not self.tensor
            or not torch.is_tensor(tensor)
            or tuple(tensor.shape) != self.shape
            or tuple(tensor.stride()) != self.stride
            or tensor.dtype is not self.dtype
            or tensor.device != self.device
            or int(tensor.data_ptr()) != self.data_ptr
            or _storage_id(tensor) != self.storage_ptr
        ):
            raise RuntimeError("persistent NPU sampler fixed tensor binding changed")


@dataclass(frozen=True, slots=True)
class NPUConstrainedCandidates:
    """One device-resident constrained selection awaiting decision consume.

    ``constraint_output`` is the Backend's original output.  ``valid_counts``
    and ``status`` are fixed-buffer views with one element per logits row.
    ``sampler_output`` follows vLLM's candidate ABI but must not be delivered
    to ordinary output bookkeeping; the future NPU adapter must replace it
    with O(B) control output before construction of an EngineCore output.
    """

    request_id: str
    device_step: int
    constraint_output: ConstraintOutput
    valid_counts: torch.Tensor
    status: torch.Tensor
    sampler_output: Any


class NPUPersistentConstraintSampler:
    """Fixed B=1 constrained sampler owner for one persistent request."""

    def __init__(
        self,
        request_id: str,
        beam_width: int,
        generation_steps: int,
        temperature: float,
        device: torch.device,
        backend: ConstraintBackend,
        *,
        cpu_test_backend: bool = False,
    ) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("persistent NPU sampler request_id is invalid")
        width = _strict_int(beam_width, "Beam width", minimum=1)
        steps = _strict_int(generation_steps, "generation steps", minimum=1)
        if steps > _MAX_CONSTRAINT_STEPS:
            raise RuntimeError("persistent NPU sampler supports only constraint steps 0/1/2")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise RuntimeError("persistent NPU sampler temperature is invalid")
        temperature_value = float(temperature)
        if not math.isfinite(temperature_value) or temperature_value < 0:
            raise RuntimeError("persistent NPU sampler temperature must be finite and non-negative")
        if not isinstance(device, torch.device):
            raise RuntimeError("persistent NPU sampler device must be torch.device")
        if not isinstance(cpu_test_backend, bool):
            raise RuntimeError("cpu_test_backend must be a boolean")
        if cpu_test_backend:
            if device.type != "cpu":
                raise RuntimeError("explicit persistent NPU CPU test Backend requires CPU device")
        elif device.type != "npu":
            raise RuntimeError("persistent NPU sampler production tensors must be NPU-resident")

        select = getattr(backend, "select", None)
        info = getattr(backend, "info", None)
        if not callable(select) or info is None:
            raise RuntimeError("persistent NPU sampler constraint Backend is invalid")
        max_rows = _strict_int(getattr(info, "max_rows", None), "Backend max_rows", minimum=1)
        max_top_k = _strict_int(getattr(info, "max_top_k", None), "Backend max_top_k", minimum=1)
        if max_rows < width or max_top_k < width:
            raise RuntimeError("persistent NPU sampler Backend capacity is too small")
        table_metadata = getattr(info, "table_metadata", None)
        vocab_size = _strict_int(
            getattr(table_metadata, "vocab_size", None),
            "Backend vocab size",
            minimum=1,
        )
        if (
            not cpu_test_backend
            and getattr(info, "backend", None) is not ConstraintTableBackendEnum.NPU
        ):
            raise RuntimeError("persistent NPU sampler requires the NPU constraint Backend")

        self._request_id = request_id
        self._width = width
        self._steps = steps
        self._backend = backend
        self._backend_info = info
        self._select_backend = select
        self._vocab_size = vocab_size
        self._prefix = torch.zeros((width, 3), dtype=torch.int32, device=device)
        # An unindexed device resolves to the device that actually owns the
        # first allocation.  Bind every subsequent tensor and input check to
        # that concrete placement.
        self._device = self._prefix.device
        self._temperature = torch.full(
            (1,), temperature_value, dtype=torch.float32, device=self._device
        )
        self._wire_logprobs = torch.empty((width, width), dtype=torch.float32, device=self._device)
        self._sampled_token_ids = torch.empty((width, 1), dtype=torch.int32, device=self._device)
        self._safe_token_ids = torch.zeros((width, 1), dtype=torch.int32, device=self._device)
        self._selected_token_ranks = torch.zeros(width, dtype=torch.int64, device=self._device)
        self._valid_counts = torch.empty(width, dtype=torch.int32, device=self._device)
        self._status = torch.empty(width, dtype=torch.int32, device=self._device)
        self._column_ids = torch.arange(width, dtype=torch.int32, device=self._device)
        self._mask_scratch = torch.empty((width, width), dtype=torch.bool, device=self._device)
        self._row_scratch = torch.empty(width, dtype=torch.bool, device=self._device)
        workspace_tensors = (
            self._prefix,
            self._temperature,
            self._wire_logprobs,
            self._sampled_token_ids,
            self._safe_token_ids,
            self._selected_token_ranks,
            self._valid_counts,
            self._status,
            self._column_ids,
            self._mask_scratch,
            self._row_scratch,
        )
        self._workspace_bindings = tuple(
            _TensorBinding.capture(tensor) for tensor in workspace_tensors
        )
        self._canonical: _TensorBinding | None = None
        self._next_step = 0
        self._pending: NPUConstrainedCandidates | None = None
        self._poisoned = False
        self._poison_error: BaseException | None = None
        self._released = False

    def _validate_workspace(self) -> None:
        current_tensors = (
            self._prefix,
            self._temperature,
            self._wire_logprobs,
            self._sampled_token_ids,
            self._safe_token_ids,
            self._selected_token_ranks,
            self._valid_counts,
            self._status,
            self._column_ids,
            self._mask_scratch,
            self._row_scratch,
        )
        for binding, tensor in zip(self._workspace_bindings, current_tensors, strict=True):
            binding.validate(tensor)

    def _validate_canonical(self, sequence: Any) -> torch.Tensor:
        if not torch.is_tensor(sequence):
            raise RuntimeError("persistent NPU sampler canonical sequence must be a Tensor")
        if (
            tuple(sequence.shape[:2]) != (1, self._width)
            or sequence.ndim != 3
            or sequence.shape[2] < self._steps
            or sequence.dtype is not torch.int32
            or sequence.device != self._device
            or not sequence.is_contiguous()
        ):
            raise RuntimeError("persistent NPU sampler canonical sequence ABI mismatch")
        # Prefix preparation zeroes its scratch before reading the canonical
        # history. Reject aliasing even through a distinct tensor/view, so
        # sampler metadata can never erase the sole authoritative Beam state.
        storage_ptr = _storage_id(sequence)
        if any(binding.storage_ptr == storage_ptr for binding in self._workspace_bindings):
            raise RuntimeError("persistent NPU sampler canonical sequence aliases workspace")
        if self._canonical is None:
            self._canonical = _TensorBinding.capture(sequence)
        else:
            self._canonical.validate(sequence)
        return sequence

    def _validate_logits(self, logits: Any, step: int) -> torch.Tensor:
        if not torch.is_tensor(logits) or logits.ndim != 2:
            raise RuntimeError("persistent NPU sampler logits must be a 2-D Tensor")
        rows = 1 if step == 0 else self._width
        if (
            tuple(logits.shape) != (rows, self._vocab_size)
            or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or logits.device != self._device
            or not logits.is_contiguous()
        ):
            raise RuntimeError("persistent NPU sampler logits ABI mismatch")
        return logits

    def select(
        self,
        logits: torch.Tensor,
        canonical_sequence: torch.Tensor,
        *,
        expected_device_step: int,
    ) -> NPUConstrainedCandidates:
        """Select W candidates per row without materializing device values.

        Step zero consumes one Prefill logits row and an all-zero prefix.
        Steps one and two consume W Decode rows and the first ``step`` tokens
        from the canonical session sequence.  The prefix is copied D2D into
        fixed request-owned storage; no Host Beam suffix participates.
        """

        if self._released:
            raise RuntimeError("persistent NPU sampler is released")
        if self._poisoned:
            raise RuntimeError("persistent NPU sampler is poisoned") from self._poison_error
        if self._pending is not None:
            raise RuntimeError("persistent NPU sampler result is awaiting decision consume")
        step = _strict_int(expected_device_step, "expected device step")
        if step != self._next_step or step >= self._steps:
            raise RuntimeError("persistent NPU sampler device step is stale or out of range")
        self._validate_workspace()
        sequence = self._validate_canonical(canonical_sequence)
        logits = self._validate_logits(logits, step)
        if getattr(self._backend, "info", None) is not self._backend_info:
            raise RuntimeError("persistent NPU sampler Backend ownership changed")

        rows = 1 if step == 0 else self._width
        self._poisoned = True
        try:
            prefixes = self._prefix[:rows]
            prefixes.zero_()
            if step:
                prefixes[:, :step].copy_(sequence[0, :, :step])
            metadata = ConstraintMetadata(
                prefix_token_ids=prefixes,
                temperatures=self._temperature,
                current_step=step,
                top_k=self._width,
            )
            raw = self._select_backend(logits, metadata)
            if not isinstance(raw, ConstraintOutput):
                raise RuntimeError("persistent NPU sampler Backend returned wrong output type")
            # Validate with the common strict ABI but retain the Backend's
            # original dataclass and raw tensor identities.
            build_constraint_output(
                raw.token_ids,
                raw.logprobs,
                raw.valid_mask,
                logits=logits,
                top_k=self._width,
            )

            counts = self._valid_counts[:rows]
            status = self._status[:rows]
            mask = self._mask_scratch[:rows]
            row_mask = self._row_scratch[:rows]
            torch.sum(raw.valid_mask, dim=1, dtype=torch.int32, out=counts)
            status.zero_()
            torch.eq(counts, 0, out=row_mask)
            status.masked_fill_(row_mask, NPU_SAMPLER_STATUS_EMPTY)

            torch.lt(self._column_ids.unsqueeze(0), counts.unsqueeze(1), out=mask)
            mask.logical_xor_(raw.valid_mask)
            torch.any(mask, dim=1, out=row_mask)
            status.masked_fill_(row_mask, NPU_SAMPLER_STATUS_VALIDITY_HOLE)

            # NaN and +inf both fail x < +inf; legal -inf passes. Keep the
            # numeric check in fixed scratch instead of allocating two W×W
            # isfinite-classification tensors on every generation stage.
            torch.lt(raw.logprobs, float("inf"), out=mask)
            mask.logical_not_().logical_and_(raw.valid_mask)
            torch.any(mask, dim=1, out=row_mask)
            status.masked_fill_(row_mask, NPU_SAMPLER_STATUS_INVALID_NUMERIC)

            wire = self._wire_logprobs[:rows]
            wire.copy_(raw.logprobs)
            torch.logical_not(raw.valid_mask, out=mask)
            wire.masked_fill_(mask, float("-inf"))
            sampled = self._sampled_token_ids[:rows]
            torch.where(
                raw.valid_mask[:, :1],
                raw.token_ids[:, :1],
                self._safe_token_ids[:rows],
                out=sampled,
            )
            ranks = self._selected_token_ranks[:rows]

            from vllm.v1.outputs import LogprobsTensors, SamplerOutput

            sampler_output = SamplerOutput(
                sampled_token_ids=sampled,
                logprobs_tensors=LogprobsTensors(
                    logprob_token_ids=raw.token_ids,
                    logprobs=wire,
                    selected_token_ranks=ranks,
                ),
            )
            sampler_output._gr_constraint_token_ids = raw.token_ids
            sampler_output._gr_constraint_logprobs = raw.logprobs
            sampler_output._gr_constraint_valid_mask = raw.valid_mask
            sampler_output._gr_constraint_valid_counts = counts
            sampler_output._gr_npu_sampler_status = status
            result = NPUConstrainedCandidates(
                request_id=self._request_id,
                device_step=step,
                constraint_output=raw,
                valid_counts=counts,
                status=status,
                sampler_output=sampler_output,
            )
        except BaseException as error:
            self._poison_error = error
            raise
        self._poisoned = False
        self._pending = result
        self._next_step = step + 1
        return result

    def consume(self, result: NPUConstrainedCandidates) -> None:
        """Transfer the one pending result to the future decision boundary."""

        if (
            self._released
            or self._poisoned
            or not isinstance(result, NPUConstrainedCandidates)
            or self._pending is None
            or result is not self._pending
        ):
            raise RuntimeError("stale or foreign persistent NPU sampler result")
        self._pending = None

    def release(self, request_id: str) -> bool:
        if request_id != self._request_id:
            raise RuntimeError("stale or foreign persistent NPU sampler request")
        if self._released:
            return False
        self._released = True
        return True

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def prefix_token_ids(self) -> torch.Tensor:
        return self._prefix

    @property
    def temperatures(self) -> torch.Tensor:
        return self._temperature


__all__ = [
    "NPUConstrainedCandidates",
    "NPUPersistentConstraintSampler",
    "NPU_SAMPLER_STATUS_EMPTY",
    "NPU_SAMPLER_STATUS_INVALID_NUMERIC",
    "NPU_SAMPLER_STATUS_OK",
    "NPU_SAMPLER_STATUS_VALIDITY_HOLE",
]
