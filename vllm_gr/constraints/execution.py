# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Constrained Top-K execution protocol and platform adapters.

After ``ConstraintTableCompiler`` produces a
:class:`~vllm_gr.constraints.compiler.ConstraintTableResource` (NumPy tables +
digest), a :class:`ConstrainedTopKExecution` loads those tables onto a device
and runs ``rec_constrained_topk`` on raw logits, returning per-row top-k valid
token IDs and log-probabilities.

Design rules:

- ``run`` returns a fixed-shape ``[rows, top_k]`` result plus an explicit
  structural validity mask derived from the canonical table ranges. Padding
  validity is never inferred from a logprob value.
- ``load_tables`` copies the NumPy tables to device tensors (the resource
  arrays are read-only bytes-backed) and is idempotent by digest: same digest
  reuses, a different digest is rejected rather than silently replaced.
- The NPU adapter calls ``rec_constrained_topk_npu`` directly and fails
  explicitly when the custom op is unavailable; it never silently falls back
  to Torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

# Single source of truth for the operator's legacy padding payload. Structural
# validity is carried separately and never inferred from this numeric value.
from vllm_gr.config.models import MAX_CANDIDATE_WIDTH
from vllm_gr.constraints.compiler import validate_resource_tables
from vllm_gr.constraints.scoring import INVALID_LOGIT as _INVALID_LOGIT
from vllm_gr.constraints.validity import (
    constraint_candidate_counts,
    valid_mask_from_counts,
)

INVALID_LOGIT = _INVALID_LOGIT

if TYPE_CHECKING:
    from vllm_gr.constraints.compiler import ConstraintTableResource

    _DEVICE = torch.device


@dataclass(frozen=True)
class ConstrainedTopKTensorResult:
    """Per-row constrained top-k token IDs and log-probabilities.

    ``top_tokens`` and ``top_logprobs`` both have shape ``[rows, top_k]``.
    ``valid_mask`` is authoritative; logprobs may legally have any finite
    float32 value, including the historical padding sentinel value.
    """

    top_tokens: torch.Tensor
    top_logprobs: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        if not (self.top_tokens.shape == self.top_logprobs.shape == self.valid_mask.shape):
            raise ValueError(
                "top_tokens, top_logprobs, and valid_mask must have the same shape, got "
                f"{tuple(self.top_tokens.shape)} vs {tuple(self.top_logprobs.shape)} "
                f"vs {tuple(self.valid_mask.shape)}"
            )
        if self.top_tokens.dim() != 2:
            raise ValueError(f"expected a 2-D result, got {self.top_tokens.dim()} dims")
        if self.top_tokens.dtype != torch.int32:
            raise ValueError(
                f"top_tokens must be int32 (constraint operator ABI), got {self.top_tokens.dtype}"
            )
        if self.top_logprobs.dtype != torch.float32:
            raise ValueError(f"top_logprobs must be float32, got {self.top_logprobs.dtype}")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError(f"valid_mask must be bool, got {self.valid_mask.dtype}")
        if not self.valid_mask.is_contiguous():
            raise ValueError("valid_mask must be contiguous")
        if not (self.top_tokens.device == self.top_logprobs.device == self.valid_mask.device):
            raise ValueError("top_tokens, top_logprobs, and valid_mask must share one device")

    @property
    def rows(self) -> int:
        return int(self.top_tokens.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.top_tokens.shape[1])


def valid_candidate_mask(valid_counts: torch.Tensor, top_k: int) -> torch.Tensor:
    """Expand authoritative per-row counts into a fixed-width mask."""

    return valid_mask_from_counts(valid_counts, top_k)


class ConstrainedTopKExecution(Protocol):
    """Load constraint tables and run constrained top-k on raw logits."""

    def is_available(self) -> bool:
        """Return whether the platform can execute the constrained top-k op."""
        ...

    def load_tables(self, resource: "ConstraintTableResource") -> None:
        """Move tables to device and cache them idempotently by digest.

        Reuses already-loaded tables for the same digest and rejects a
        different digest (no silent replacement).
        """
        ...

    def run(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> ConstrainedTopKTensorResult:
        """Run constrained top-k and return per-row token ids + logprobs."""
        ...


def _tables_to_device(
    resource: "ConstraintTableResource",
    device: "_DEVICE",
) -> dict[str, torch.Tensor]:
    """Copy the resource's NumPy tables to device tensors.

    The resource arrays are read-only bytes-backed; ``torch.from_numpy`` on a
    read-only array produces a non-writable tensor with undefined behavior, so
    each array is copied to a writable NumPy array first.
    """
    arrays = {
        "first_token_ids": resource.first_token_ids,
        "prefix1_offsets": resource.prefix1_offsets,
        "prefix1_values": resource.prefix1_values,
        "prefix1_pair_keys": resource.prefix1_pair_keys,
        "prefix2_value_offsets": resource.prefix2_value_offsets,
        "prefix2_values": resource.prefix2_values,
    }
    return {
        # np.array(copy=True) produces a writable copy of the read-only
        # bytes-backed resource array so torch.from_numpy gets a writable buffer.
        name: torch.from_numpy(np.array(array, copy=True)).to(device=device)
        for name, array in arrays.items()
    }


class _ConstrainedTopKExecutionBase:
    """Shared table lifecycle for the Torch and NPU adapters.

    Owns the loaded-table state and the digest-keyed idempotent lifecycle:

    - ``load_tables`` reuses the same digest and rejects a different digest.
    - ``reset`` drops the loaded state so a fresh digest can be loaded again
      (pair with :meth:`ConstraintResourceManager.clear`).
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        self._device = device
        self._tables: dict[str, torch.Tensor] = {}
        self._digest: str | None = None
        self._vocab_size: int | None = None
        self._max_degrees: dict[str, int] = {}

    def load_tables(self, resource: "ConstraintTableResource") -> None:
        device = self._device or self._default_device()
        if self._digest == resource.digest:
            return  # idempotent reuse
        if self._digest is not None and self._digest != resource.digest:
            raise ValueError(
                f"attempted to replace loaded constraint tables digest "
                f"{self._digest} with {resource.digest}; reloading is not allowed "
                "(call reset() first)"
            )
        # Validate the full table contract before moving anything to device.  A
        # directly-constructed or pickle-tampered resource bypasses from_wire's
        # validation, and must not reach the device tables.
        validate_resource_tables(resource)
        self._tables = _tables_to_device(resource, device)
        self._digest = resource.digest
        self._vocab_size = resource.vocab_size
        self._max_degrees = {
            "max_prefix1_degree": resource.max_prefix1_degree,
            "max_prefix2_degree": resource.max_prefix2_degree,
        }

    def reset(self) -> None:
        """Drop loaded tables and digest so a new digest can be loaded again.

        This must be paired with
        :meth:`~vllm_gr.constraints.resource_manager.ConstraintResourceManager.clear`
        so the CPU-side manager and the device-side execution stay in sync.
        """
        self._tables = {}
        self._digest = None
        self._vocab_size = None
        self._max_degrees = {}

    def _default_device(self) -> torch.device:
        return _default_device()

    def _call_op(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _expected_logits_vocab_size(self, logits: torch.Tensor) -> int | None:
        """Return the logits width required by this execution adapter."""

        del logits
        return self._vocab_size

    def run(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> ConstrainedTopKTensorResult:
        if self._digest is None:
            raise RuntimeError("constraint tables not loaded; call load_tables first")
        if self._vocab_size is None:
            raise RuntimeError("loaded constraint tables are missing vocab_size")
        self._validate_runtime_inputs(logits, sequence_group, temperatures, current_step, top_k)

        top_tokens, top_logprobs = self._call_op(
            logits, sequence_group, temperatures, current_step, top_k
        )
        self._validate_op_output(logits, top_k, top_tokens, top_logprobs)
        valid_counts = constraint_candidate_counts(
            sequence_group,
            self._tables["first_token_ids"],
            self._tables["prefix1_offsets"],
            self._tables["prefix1_pair_keys"],
            self._tables["prefix2_value_offsets"],
            current_step=current_step,
            vocab_size=self._vocab_size,
        )
        # Normalize to the constraint operator ABI: top-k token ids are int32 (the Torch
        # operator returns int64); downstream consumers require int32.
        return ConstrainedTopKTensorResult(
            top_tokens=top_tokens.to(torch.int32),
            top_logprobs=top_logprobs,
            valid_mask=valid_mask_from_counts(valid_counts, top_k),
        )

    def _validate_op_output(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_tokens: torch.Tensor,
        top_logprobs: torch.Tensor,
    ) -> None:
        """Validate the operator's output against the fixed result contract.

        A misbehaving custom op could return the wrong row count, the wrong
        device, or a non-contiguous tensor; catching it here (instead of in a
        later remux) keeps the adapter boundary strict.
        """
        expected_shape = (logits.shape[0], top_k)
        if tuple(top_tokens.shape) != expected_shape or tuple(top_logprobs.shape) != expected_shape:
            raise ValueError(
                f"operator returned shape {tuple(top_tokens.shape)} / "
                f"{tuple(top_logprobs.shape)}, expected {expected_shape} "
                f"([logits.rows, top_k])"
            )
        for name, tensor in (("top_tokens", top_tokens), ("top_logprobs", top_logprobs)):
            if tensor.device != logits.device:
                raise ValueError(
                    f"{name}.device {tensor.device} must match logits.device {logits.device}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

    def _validate_runtime_inputs(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> None:
        _validate_runtime_inputs(
            logits=logits,
            sequence_group=sequence_group,
            temperatures=temperatures,
            current_step=current_step,
            top_k=top_k,
            vocab_size=self._expected_logits_vocab_size(logits),
            tables=self._tables,
        )


class TorchConstrainedTopKExecution(_ConstrainedTopKExecutionBase):
    """Pure-PyTorch constrained top-k used as the test oracle / dev backend."""

    def is_available(self) -> bool:
        return True

    def _call_op(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_gr.ops.rec_constrained_topk import rec_constrained_topk_torch

        return rec_constrained_topk_torch(
            logits,
            sequence_group,
            self._tables["first_token_ids"],
            self._tables["prefix1_offsets"],
            self._tables["prefix1_values"],
            self._tables["prefix1_pair_keys"],
            self._tables["prefix2_value_offsets"],
            self._tables["prefix2_values"],
            temperatures,
            current_step,
            top_k,
            self._max_degrees["max_prefix1_degree"],
            self._max_degrees["max_prefix2_degree"],
        )


class CUDAConstrainedTopKExecution(_ConstrainedTopKExecutionBase):
    """CUDA constrained Top-K backed by the Triton lookup/gather kernel."""

    def load_tables(self, resource: "ConstraintTableResource") -> None:
        from vllm_gr.ops.cuda.rec_constrained_topk import validate_cuda_sort_capacity

        validate_cuda_sort_capacity(
            int(resource.first_token_ids.size),
            resource.max_prefix1_degree,
            resource.max_prefix2_degree,
        )
        super().load_tables(resource)

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def _default_device(self) -> torch.device:
        return _default_device(prefer="cuda")

    def _vocab_window(self, logits: torch.Tensor) -> tuple[int, int]:
        """Resolve ``(offset, full_vocab_size)`` for full or sliced logits."""

        if self._vocab_size is None:
            raise RuntimeError("loaded constraint tables are missing vocab_size")
        if logits.shape[1] == self._vocab_size:
            return 0, self._vocab_size

        from vllm_gr.ops.sliced_logits import get_vocab_slice_config

        offset, slice_size, _ = get_vocab_slice_config()
        if offset is None or slice_size is None:
            raise ValueError(
                f"CUDA constraint logits vocab {logits.shape[1]} differs from table "
                f"vocab_size {self._vocab_size}, but vocab slicing is not configured"
            )
        if logits.shape[1] != slice_size:
            raise ValueError(
                f"CUDA constraint logits vocab {logits.shape[1]} must equal configured "
                f"vocab_slice_size {slice_size}"
            )
        if offset < 0 or offset + slice_size > self._vocab_size:
            raise ValueError(
                "configured CUDA vocabulary slice is outside the constraint table "
                f"vocabulary: offset={offset}, size={slice_size}, "
                f"table_vocab_size={self._vocab_size}"
            )
        return offset, self._vocab_size

    def _expected_logits_vocab_size(self, logits: torch.Tensor) -> int:
        self._vocab_window(logits)
        return int(logits.shape[1])

    def _call_op(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_gr.ops.cuda.rec_constrained_topk import rec_constrained_topk_cuda

        vocab_offset, table_vocab_size = self._vocab_window(logits)
        tokens, logprobs, _, _ = rec_constrained_topk_cuda(
            logits,
            sequence_group,
            self._tables["first_token_ids"],
            self._tables["prefix1_offsets"],
            self._tables["prefix1_values"],
            self._tables["prefix1_pair_keys"],
            self._tables["prefix2_value_offsets"],
            self._tables["prefix2_values"],
            temperatures,
            current_step,
            top_k,
            self._max_degrees["max_prefix1_degree"],
            self._max_degrees["max_prefix2_degree"],
            vocab_offset=vocab_offset,
            table_vocab_size=table_vocab_size,
        )
        return tokens, logprobs

    def run_out(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
        *,
        out_token_ids: torch.Tensor,
        out_logprobs: torch.Tensor,
        out_valid_mask: torch.Tensor,
        out_valid_counts: torch.Tensor,
        out_wire_logprobs: torch.Tensor | None = None,
    ) -> ConstrainedTopKTensorResult:
        """CUDA-only caller-owned output path used by Worker postprocessing."""

        return self._run_out_impl(
            logits,
            sequence_group,
            temperatures,
            current_step,
            top_k,
            out_token_ids=out_token_ids,
            out_logprobs=out_logprobs,
            out_valid_mask=out_valid_mask,
            out_valid_counts=out_valid_counts,
            out_wire_logprobs=out_wire_logprobs,
            validate=True,
        )

    def run_out_unchecked(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
        *,
        out_token_ids: torch.Tensor,
        out_logprobs: torch.Tensor,
        out_valid_mask: torch.Tensor,
        out_valid_counts: torch.Tensor,
        out_wire_logprobs: torch.Tensor | None = None,
        out_prefill_status: torch.Tensor | None = None,
        out_prefill_payload: torch.Tensor | None = None,
    ) -> ConstrainedTopKTensorResult:
        """Launch into a prewarmed Worker workspace without repeated validation."""

        return self._run_out_impl(
            logits,
            sequence_group,
            temperatures,
            current_step,
            top_k,
            out_token_ids=out_token_ids,
            out_logprobs=out_logprobs,
            out_valid_mask=out_valid_mask,
            out_valid_counts=out_valid_counts,
            out_wire_logprobs=out_wire_logprobs,
            out_prefill_status=out_prefill_status,
            out_prefill_payload=out_prefill_payload,
            validate=False,
        )

    def _run_out_impl(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
        *,
        out_token_ids: torch.Tensor,
        out_logprobs: torch.Tensor,
        out_valid_mask: torch.Tensor,
        out_valid_counts: torch.Tensor,
        out_wire_logprobs: torch.Tensor | None,
        out_prefill_status: torch.Tensor | None = None,
        out_prefill_payload: torch.Tensor | None = None,
        validate: bool,
    ) -> ConstrainedTopKTensorResult:
        """Shared checked/unchecked caller-owned launch implementation."""

        if self._digest is None:
            raise RuntimeError("constraint tables not loaded; call load_tables first")
        if validate:
            self._validate_runtime_inputs(
                logits,
                sequence_group,
                temperatures,
                current_step,
                top_k,
            )
        from vllm_gr.ops.cuda.rec_constrained_topk import (
            _rec_constrained_topk_cuda_out_unchecked,
            rec_constrained_topk_cuda_out,
        )

        vocab_offset, table_vocab_size = self._vocab_window(logits)
        launch = (
            rec_constrained_topk_cuda_out if validate else _rec_constrained_topk_cuda_out_unchecked
        )
        launch_kwargs = {
            "vocab_offset": vocab_offset,
            "table_vocab_size": table_vocab_size,
            "out_wire_logprobs": out_wire_logprobs,
        }
        if not validate:
            launch_kwargs["out_prefill_status"] = out_prefill_status
            launch_kwargs["out_prefill_payload"] = out_prefill_payload
        launch(
            logits,
            sequence_group,
            self._tables["first_token_ids"],
            self._tables["prefix1_offsets"],
            self._tables["prefix1_values"],
            self._tables["prefix1_pair_keys"],
            self._tables["prefix2_value_offsets"],
            self._tables["prefix2_values"],
            temperatures,
            current_step,
            top_k,
            self._max_degrees["max_prefix1_degree"],
            self._max_degrees["max_prefix2_degree"],
            out_token_ids,
            out_logprobs,
            out_valid_mask,
            out_valid_counts,
            **launch_kwargs,
        )
        return ConstrainedTopKTensorResult(
            top_tokens=out_token_ids,
            top_logprobs=out_logprobs,
            valid_mask=out_valid_mask,
        )


class NPUConstrainedTopKExecution(_ConstrainedTopKExecutionBase):
    """NPU constrained top-k backed by the fused ``custom_ops_lib`` operator.

    Calls ``rec_constrained_topk_npu`` directly and fails explicitly when the
    custom op is unavailable; it never silently falls back to Torch.
    """

    def is_available(self) -> bool:
        from vllm_gr.ops.npu.rec_constrained_topk import is_available

        return is_available()

    def _default_device(self) -> torch.device:
        return _default_device(prefer="npu")

    def _call_op(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_gr.ops.npu.rec_constrained_topk import rec_constrained_topk_npu

        return rec_constrained_topk_npu(
            logits,
            sequence_group,
            self._tables["first_token_ids"],
            self._tables["prefix1_offsets"],
            self._tables["prefix1_values"],
            self._tables["prefix1_pair_keys"],
            self._tables["prefix2_value_offsets"],
            self._tables["prefix2_values"],
            temperatures,
            current_step,
            top_k,
            self._max_degrees["max_prefix1_degree"],
            self._max_degrees["max_prefix2_degree"],
        )


def _validate_runtime_inputs(
    *,
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    temperatures: torch.Tensor,
    current_step: int,
    top_k: int,
    vocab_size: int | None,
    tables: dict[str, torch.Tensor],
) -> None:
    """Common run-time input validation shared by both adapters."""
    if not isinstance(current_step, int) or isinstance(current_step, bool):
        raise ValueError(f"current_step must be an int, got {type(current_step).__name__}")
    if current_step not in (0, 1, 2):
        raise ValueError(f"unsupported current_step={current_step}; expected 0, 1, or 2")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError(f"top_k must be an int, got {type(top_k).__name__}")
    if not (1 <= top_k <= MAX_CANDIDATE_WIDTH):
        raise ValueError(f"top_k must be in [1, {MAX_CANDIDATE_WIDTH}], got {top_k}")
    _validate_input_dtypes(logits, sequence_group, temperatures, tables)
    _validate_shapes(logits, sequence_group, current_step, vocab_size)
    _validate_same_device(logits, sequence_group, temperatures)
    _validate_table_device(logits, tables)
    _validate_temperature_shape(temperatures, logits.shape[0])


def _validate_input_dtypes(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    temperatures: torch.Tensor,
    tables: dict[str, torch.Tensor],
) -> None:
    """Enforce the operator input dtype contract.

    - ``logits``: fp16 / bf16 / fp32 (the operator supports all three).
    - ``sequence_group``: int32.
    - ``temperatures``: float32.
    - tables: offsets/values/first are int32; ``prefix1_pair_keys`` is int64.
    """
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"logits.dtype must be fp16/bf16/fp32, got {logits.dtype}")
    if sequence_group.dtype != torch.int32:
        raise ValueError(f"sequence_group.dtype must be int32, got {sequence_group.dtype}")
    if temperatures.dtype != torch.float32:
        raise ValueError(f"temperatures.dtype must be float32, got {temperatures.dtype}")
    expected = {
        "first_token_ids": torch.int32,
        "prefix1_offsets": torch.int32,
        "prefix1_values": torch.int32,
        "prefix1_pair_keys": torch.int64,
        "prefix2_value_offsets": torch.int32,
        "prefix2_values": torch.int32,
    }
    for name, want in expected.items():
        got = tables[name].dtype
        if got != want:
            raise ValueError(f"table {name}.dtype must be {want}, got {got}")


def _default_device(prefer: str | None = None) -> torch.device:
    """Pick a sensible default device, honouring an explicit preference.

    ``prefer`` names a device type (``"npu"``, ``"cuda"``, ``"cpu"``).  When a
    type is requested it is returned regardless of availability so an NPU
    adapter never accidentally lands tables on CPU while logits are on NPU
    (which would make the operator fail with a device mismatch).  Without a
    preference, fall back to CUDA then CPU.
    """
    if prefer == "npu":
        try:
            return torch.device("npu")
        except RuntimeError:
            # torch on non-Ascend builds does not recognise the "npu" device
            # type; fall through to the generic default.
            pass
    if prefer == "cuda":
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_shapes(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    current_step: int,
    vocab_size: int | None = None,
) -> None:
    """Enforce the operator input shape contracts.

    - ``logits``: ``[rows, vocab]``, 2-D; ``vocab`` must equal ``vocab_size``
      when the loaded tables know it.
    - prefill (step 0): ``sequence_group [B, 1, 1]`` with ``rows == B``.
    - decode (steps 1/2): ``sequence_group [B, W, max_step]`` with
      ``rows == B * W`` (one row per parent beam) and ``max_step > current_step``.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D, got {tuple(logits.shape)}")
    rows = logits.shape[0]
    if vocab_size is not None and logits.shape[1] != vocab_size:
        raise ValueError(f"logits vocab {logits.shape[1]} must equal table vocab_size {vocab_size}")
    if current_step == 0:
        if tuple(sequence_group.shape) != (rows, 1, 1):
            raise ValueError(
                f"prefill sequence_group must be [B, 1, 1], got {tuple(sequence_group.shape)}"
            )
    else:
        if sequence_group.dim() != 3:
            raise ValueError(
                f"decode sequence_group must be [B, W, max_step], got {tuple(sequence_group.shape)}"
            )
        batch, width = sequence_group.shape[0], sequence_group.shape[1]
        if rows != batch * width:
            raise ValueError(f"decode logits rows {rows} must equal batch*width {batch}*{width}")
        if sequence_group.shape[2] <= current_step:
            raise ValueError(
                f"decode sequence_group max_step {sequence_group.shape[2]} must be "
                f"> current_step {current_step}"
            )


def _validate_same_device(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    temperatures: torch.Tensor,
) -> None:
    """Require logits, sequence_group, and temperatures to share one device.

    The device check is unconditional (including empty tensors) because the
    operator receives all three tensors and a device mismatch is always a bug,
    even when a tensor has zero elements.
    """
    for name, tensor in (
        ("sequence_group", sequence_group),
        ("temperatures", temperatures),
    ):
        if tensor.device != logits.device:
            raise ValueError(
                f"{name}.device {tensor.device} must match logits.device {logits.device}"
            )


def _validate_table_device(
    logits: torch.Tensor,
    tables: dict[str, torch.Tensor],
) -> None:
    """Require every loaded table to sit on the same device as logits."""
    for name, table in tables.items():
        if table.device != logits.device:
            raise ValueError(
                f"table {name} on {table.device} must match logits.device {logits.device}"
            )


def _validate_temperature_shape(
    temperatures: torch.Tensor,
    rows: int,
) -> None:
    """temperatures must be empty, ``[1]``, or ``[rows]``."""
    if temperatures.dim() != 1 or temperatures.numel() not in (0, 1, rows):
        raise ValueError(
            f"temperatures must have shape [0]/[1]/[{rows}], got {tuple(temperatures.shape)}"
        )
