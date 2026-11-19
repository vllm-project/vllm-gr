# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Fixed-address CUDA Beam postprocess workspace and its prewarmed pool.

This module owns the memory layout contract only: shape/dtype identity, the
preallocated device/pinned-host buffers, and the lookup used by the decision
paths.  Test-only shape views live in
``tests/beam_postprocess_workspace_utils.py``.  It deliberately holds no beam
decision logic or cross-step request state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm_gr.v1.beam.decision_ops import workspace_candidate_width

if TYPE_CHECKING:
    from vllm_gr.ops.cuda.beam_state import BeamStateScratch


def _validate_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class BeamPostprocessWorkspaceKey:
    """Shape/dtype identity for one fixed-address postprocess workspace.

    The CUDA kernels have exact contiguous shape contracts.  In particular, a
    top-left ``[:width, :top_k]`` slice of a larger two-dimensional bucket is
    not contiguous, so a request may only use a workspace whose width/top-k
    exactly match its operator geometry.  ``logits_dtype`` is part of the key
    even though the operator outputs are int32/fp32; this prevents an fp16
    warmup from silently standing in for a bf16 deployment.
    """

    beam_width: int
    top_k: int
    logits_dtype: torch.dtype


@dataclass(slots=True)
class BeamPostprocessWorkspace:
    """Runner-owned fixed storage for one CUDA Beam postprocess geometry.

    This object is scratch/output storage only.  It deliberately owns no
    decode-step counter, request identity, or continuation state; the existing
    ``BeamDecisionBuffers`` remain the source of cumulative cross-step state.
    Keeping this distinction explicit prevents an allocation optimization from
    growing into a second Beam lifecycle protocol.

    The first dimension is always ``max_batch``.  Runtime code takes leading
    batch slices, while tests can construct ``max_batch=2`` even though the
    current Worker-level Beam context accepts one active session.
    """

    key: BeamPostprocessWorkspaceKey
    max_batch: int
    max_decode_steps: int
    device: torch.device

    # Constrained Top-K outputs: [Bmax, W, K] / [Bmax, W].
    topk_token_ids: torch.Tensor
    topk_scores: torch.Tensor
    topk_valid_mask: torch.Tensor
    topk_valid_counts: torch.Tensor
    wire_topk_scores: torch.Tensor

    # Constraint metadata staging.  Host tensors are pinned for one H2D copy.
    prefix_token_ids: torch.Tensor
    host_prefix_token_ids: torch.Tensor
    host_prefix_token_ids_numpy: Any
    temperatures: torch.Tensor
    host_temperatures: torch.Tensor

    # BeamSearchGroup outputs: [Bmax, W] and fixed sequence extent.
    selected_token_ids: torch.Tensor
    selected_parent_ids: torch.Tensor
    selected_scores: torch.Tensor
    selected_flat_indices: torch.Tensor
    next_sequence: torch.Tensor

    # Current SamplerOutput compatibility buffers.
    sampled_token_ids: torch.Tensor
    selected_token_ranks: torch.Tensor

    # Caller-owned hierarchical selection ping-pong buffers.
    selection_scratch: Any

    # Optional persistent-runner scratch, allocated once when the runner binds.
    # Ordinary postprocess callers retain their existing allocations and ABI.
    state_scratch: BeamStateScratch | None = None
    execution_owner: Any = None

    @classmethod
    def allocate(
        cls,
        *,
        max_batch: int,
        beam_width: int,
        top_k: int,
        max_decode_steps: int,
        device: torch.device | str,
        logits_dtype: torch.dtype,
        pin_host: bool | None = None,
    ) -> BeamPostprocessWorkspace:
        """Allocate every tensor once; callers retain the returned object."""

        _validate_positive_int("max_batch", max_batch)
        _validate_positive_int("beam_width", beam_width)
        _validate_positive_int("top_k", top_k)
        _validate_positive_int("max_decode_steps", max_decode_steps)

        resolved_device = torch.device(device)
        if pin_host is None:
            pin_host = resolved_device.type == "cuda"
        if pin_host and resolved_device.type != "cuda":
            raise ValueError("pinned postprocess staging is only supported for CUDA workspaces")

        from vllm_gr.ops.cuda.beam_selection import allocate_beam_selection_scratch

        key = BeamPostprocessWorkspaceKey(beam_width, top_k, logits_dtype)
        candidate_shape = (max_batch, beam_width, top_k)
        beam_shape = (max_batch, beam_width)
        sequence_shape = (max_batch, beam_width, max_decode_steps)

        def device_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=resolved_device)

        def host_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=bool(pin_host))

        # Group selects W winners from W*K candidates.  Its scratch is the
        # largest selection scratch needed by the worker-specific terminal
        # reorder as well, so no second final-step allocation is necessary.
        selection_scratch = allocate_beam_selection_scratch(
            max_batch,
            beam_width,
            top_k,
            beam_width,
            resolved_device,
        )
        host_prefix_token_ids = host_empty((max_batch, beam_width, 3), torch.int32)

        return cls(
            key=key,
            max_batch=max_batch,
            max_decode_steps=max_decode_steps,
            device=resolved_device,
            topk_token_ids=device_empty(candidate_shape, torch.int32),
            topk_scores=device_empty(candidate_shape, torch.float32),
            topk_valid_mask=device_empty(candidate_shape, torch.bool),
            topk_valid_counts=device_empty(beam_shape, torch.int32),
            wire_topk_scores=device_empty(candidate_shape, torch.float32),
            prefix_token_ids=device_empty((max_batch, beam_width, 3), torch.int32),
            host_prefix_token_ids=host_prefix_token_ids,
            host_prefix_token_ids_numpy=host_prefix_token_ids.numpy(),
            temperatures=device_empty(beam_shape, torch.float32),
            host_temperatures=host_empty(beam_shape, torch.float32),
            selected_token_ids=device_empty(beam_shape, torch.int32),
            selected_parent_ids=device_empty(beam_shape, torch.int32),
            selected_scores=device_empty(beam_shape, torch.float32),
            selected_flat_indices=device_empty(beam_shape, torch.int32),
            next_sequence=device_empty(sequence_shape, torch.int32),
            sampled_token_ids=device_empty((max_batch * beam_width, 1), torch.int32),
            selected_token_ranks=torch.zeros(
                (max_batch * beam_width,), dtype=torch.int64, device=resolved_device
            ),
            selection_scratch=selection_scratch,
        )

    def device_tensors(self) -> tuple[torch.Tensor, ...]:
        scratch = self.selection_scratch
        tensors = (
            self.topk_token_ids,
            self.topk_scores,
            self.topk_valid_mask,
            self.topk_valid_counts,
            self.wire_topk_scores,
            self.prefix_token_ids,
            self.temperatures,
            self.selected_token_ids,
            self.selected_parent_ids,
            self.selected_scores,
            self.selected_flat_indices,
            self.next_sequence,
            self.sampled_token_ids,
            self.selected_token_ranks,
            scratch.local_scores,
            scratch.local_flat_indices,
            scratch.merge_scores,
            scratch.merge_flat_indices,
        )
        if self.state_scratch is not None:
            tensors += self.state_scratch.device_tensors()
        return tensors

    @property
    def device_memory_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.device_tensors())

    @property
    def pinned_memory_bytes(self) -> int:
        tensors = (
            self.host_prefix_token_ids,
            self.host_temperatures,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


class BeamPostprocessWorkspacePool:
    """Prewarmed workspace buckets; request-time creation is intentionally absent."""

    def __init__(self) -> None:
        self._buckets: dict[BeamPostprocessWorkspaceKey, BeamPostprocessWorkspace] = {}

    def add(self, workspace: BeamPostprocessWorkspace) -> None:
        if workspace.key in self._buckets:
            raise ValueError(f"duplicate Beam postprocess workspace bucket {workspace.key}")
        self._buckets[workspace.key] = workspace

    def require(self, key: BeamPostprocessWorkspaceKey) -> BeamPostprocessWorkspace:
        try:
            return self._buckets[key]
        except KeyError as exc:
            raise KeyError(
                f"Beam postprocess workspace {key} was not prewarmed; "
                "the hot path is not allowed to allocate a bucket"
            ) from exc

    @property
    def keys(self) -> tuple[BeamPostprocessWorkspaceKey, ...]:
        return tuple(self._buckets)

    @property
    def device_memory_bytes(self) -> int:
        """Return the device storage owned by all prewarmed buckets."""

        return sum(workspace.device_memory_bytes for workspace in self._buckets.values())


def get_beam_postprocess_workspace(
    runner: Any,
    beam_width: int,
    candidate_width: int | None = None,
) -> BeamPostprocessWorkspace | None:
    """Return an exact prewarmed CUDA bucket, never allocating on a miss."""

    device = getattr(runner, "device", None)
    if str(getattr(device, "type", "")) != "cuda":
        return None
    pool = getattr(runner, "_beam_postprocess_workspaces", None)
    if not isinstance(pool, BeamPostprocessWorkspacePool):
        return None
    dtype = getattr(getattr(runner, "model_config", None), "dtype", None)
    if not isinstance(dtype, torch.dtype):
        return None
    if candidate_width is None:
        candidate_width = workspace_candidate_width(runner, beam_width)
    key = BeamPostprocessWorkspaceKey(beam_width, candidate_width, dtype)
    try:
        return pool.require(key)
    except KeyError:
        return None


__all__ = [
    "BeamPostprocessWorkspace",
    "BeamPostprocessWorkspaceKey",
    "BeamPostprocessWorkspacePool",
    "get_beam_postprocess_workspace",
]
