# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side beam decision for incremental beam decode (ACS-aligned).

After the accelerator model forward produces top-k candidates (from the sampler's
logprob tensors, or the Constraint Table ``rec_constrained_topk`` override),
the worker can run the beam search decision itself instead of round-tripping
the full ``W * K`` logprob tensor to the frontend for a NumPy ``argpartition``:

- non-final steps: ``beam_search_group`` selects the surviving ``W`` beams and
  rewrites the sequence history;
- final step: ``beam_search_rec_final_select`` returns ``result_width`` beams
  directly (the 32-row tiling constraint applies only to the NPU kernel).

Cumulative scores and sequence history live in the device-resident
:class:`~vllm_gr.v1.beam.beam_attention_context.BeamDecisionBuffers`
(``context.decision``), which the worker treats as the single source of truth,
mirroring the ACS ``BeamSearchContext.decode_select`` + ``init_from_prefill``
save flow.  The result dict (``out_token_ids`` / ``out_token_index`` /
``out_log_probs`` / ``out_sequence``) is attached to
``ModelRunnerOutput.beam_search_result`` and consumed by the serving frontend.

The public ``onerec_decode`` operators dispatch to the active accelerator:
CUDA uses the fused CUDA kernels, NPU uses the xllm operators, and CPU keeps a
PyTorch reference path for parity tests.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_gr.ops.cuda.beam_state import BeamStateScratch

# Kernel tiling constraint: beam_search_rec_final_select requires result_width
# to be a multiple of 32 (verified on device; see ops/npu/beam_ops.py ABI notes).
_FINAL_SELECT_ALIGNMENT = 32
_BEAM_RESULT_BUFFER_FORMAT = "beam_result_i32_f32_v1"
_CUDA_PREFILL_POSTPROCESS_WIDTHS = frozenset((64, 128, 256))
_CUDA_PREFILL_INVALID_TOKEN_ID = 0


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
    selected_parent_count_prefix: torch.Tensor
    selected_flat_indices: torch.Tensor
    next_sequence: torch.Tensor

    # Current SamplerOutput compatibility buffers (never transported as the
    # canonical Beam result payload).
    sampled_token_ids: torch.Tensor
    selected_token_ranks: torch.Tensor

    # Terminal-step outputs.
    final_token_ids: torch.Tensor
    final_parent_ids: torch.Tensor
    final_scores: torch.Tensor
    final_sequence: torch.Tensor

    # Caller-owned hierarchical selection ping-pong buffers.
    selection_scratch: Any

    # Compact result transport.  Scores occupy int32 slots by bit view.
    packed_result: torch.Tensor
    host_result: torch.Tensor
    host_result_numpy: Any
    copy_event: Any | None

    # CUDA prefill-only status + compact Beam result transport.  Keep this
    # separate from ``packed_result`` so the D->D result path and its offsets
    # remain unchanged. Layout is [status Bmax][token|parent|score_bits].
    prefill_staging: torch.Tensor
    host_prefill_staging: torch.Tensor
    host_prefill_staging_numpy: Any
    prefill_copy_event: Any | None

    # Optional persistent-runner scratch, allocated once when the runner binds.
    # Ordinary postprocess callers retain their existing allocations and ABI.
    state_scratch: BeamStateScratch | None = None

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
    ) -> "BeamPostprocessWorkspace":
        """Allocate every tensor once; callers retain the returned object."""

        if isinstance(max_batch, bool) or not isinstance(max_batch, int) or max_batch < 1:
            raise ValueError("max_batch must be a positive integer")
        if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
            raise ValueError("beam_width must be a positive integer")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if (
            isinstance(max_decode_steps, bool)
            or not isinstance(max_decode_steps, int)
            or max_decode_steps < 1
        ):
            raise ValueError("max_decode_steps must be a positive integer")

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
        result_capacity = 3 * max_batch * beam_width
        host_prefix_token_ids = host_empty((max_batch, beam_width, 3), torch.int32)
        host_result = host_empty((result_capacity,), torch.int32)
        prefill_capacity = max_batch + result_capacity
        host_prefill_staging = host_empty((prefill_capacity,), torch.int32)
        copy_event = torch.Event(enable_timing=False) if resolved_device.type == "cuda" else None
        prefill_copy_event = (
            torch.Event(enable_timing=False) if resolved_device.type == "cuda" else None
        )

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
            selected_parent_count_prefix=device_empty(beam_shape, torch.int32),
            selected_flat_indices=device_empty(beam_shape, torch.int32),
            next_sequence=device_empty(sequence_shape, torch.int32),
            sampled_token_ids=device_empty((max_batch * beam_width, 1), torch.int32),
            selected_token_ranks=torch.zeros(
                (max_batch * beam_width,), dtype=torch.int64, device=resolved_device
            ),
            final_token_ids=device_empty(beam_shape, torch.int32),
            final_parent_ids=device_empty(beam_shape, torch.int32),
            final_scores=device_empty(beam_shape, torch.float32),
            final_sequence=device_empty(sequence_shape, torch.int32),
            selection_scratch=selection_scratch,
            packed_result=device_empty((result_capacity,), torch.int32),
            host_result=host_result,
            host_result_numpy=host_result.numpy(),
            copy_event=copy_event,
            prefill_staging=device_empty((prefill_capacity,), torch.int32),
            host_prefill_staging=host_prefill_staging,
            host_prefill_staging_numpy=host_prefill_staging.numpy(),
            prefill_copy_event=prefill_copy_event,
        )

    def _validate_batch_size(self, batch_size: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an int")
        if not 1 <= batch_size <= self.max_batch:
            raise ValueError(f"batch_size must be in [1, {self.max_batch}], got {batch_size}")

    def candidate_rows(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        *,
        compact: bool = False,
    ) -> torch.Tensor:
        """Return an exact contiguous ``[rows, K]`` view without allocating.

        Decode uses ``rows=B*W``.  The compact bootstrap uses the first ``B``
        rows of the same fixed storage, matching the CUDA Beam step-0 ABI.
        """

        self._validate_batch_size(batch_size)
        rows = batch_size if compact else batch_size * self.key.beam_width
        return tensor.view(self.max_batch * self.key.beam_width, self.key.top_k)[:rows]

    def active_beams(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Return the leading contiguous ``[B, W, ...]`` workspace slice."""

        self._validate_batch_size(batch_size)
        return tensor[:batch_size]

    def result_count(self, batch_size: int, result_width: int | None = None) -> int:
        self._validate_batch_size(batch_size)
        width = self.key.beam_width if result_width is None else result_width
        if isinstance(width, bool) or not isinstance(width, int):
            raise TypeError("result_width must be an int")
        if not 1 <= width <= self.key.beam_width:
            raise ValueError(f"result_width must be in [1, {self.key.beam_width}], got {width}")
        return batch_size * width

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
            self.selected_parent_count_prefix,
            self.selected_flat_indices,
            self.next_sequence,
            self.sampled_token_ids,
            self.selected_token_ranks,
            self.final_token_ids,
            self.final_parent_ids,
            self.final_scores,
            self.final_sequence,
            scratch.local_scores,
            scratch.local_flat_indices,
            scratch.merge_scores,
            scratch.merge_flat_indices,
            self.packed_result,
            self.prefill_staging,
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
            self.host_result,
            self.host_prefill_staging,
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


@dataclass(slots=True)
class BeamNpuTransportWorkspace:
    """Minimal pinned-host transport for one NPU beam-decision geometry.

    The CUDA :class:`BeamPostprocessWorkspace` carries ~25 CUDA-kernel scratch
    buffers that the NPU path never uses; rather than grow that class with an
    NPU branch, this object holds *only* what the compact result encoding
    needs: a pinned int32 host staging buffer, its cached NumPy view, and one
    NPU copy event.

    Layout of the staging buffer (int32 words, packed back-to-back):
    ``[ids | parents | score_bits | sequence?]`` — the first three segments are
    the equal-width compact wire arrays (``beam_result_i32_f32_v1``); the
    optional trailing segment carries the terminal step's ``out_sequence``
    (``[B, W, steps]`` int32) so the FINAL step also moves its whole payload in
    one D2H.  The nested ``out_sequence`` / ``out_log_probs`` lists the serving
    terminal short-circuit (``_worker_final_sequences``) consumes verbatim are
    then derived from the host view — pure host-memory reads, no second sync.

    The encoding reuses the CUDA ``beam_result_i32_f32_v1`` wire format for the
    compact arrays, so the frontend decodes it through the device-agnostic
    ``unpack_beam_result_buffer`` path with no consumer-side change.
    """

    key: BeamPostprocessWorkspaceKey
    max_batch: int
    max_decode_steps: int
    device: torch.device
    host_result: torch.Tensor
    host_result_numpy: Any
    copy_event: Any | None
    # In-flight guard for the deferred terminal D2H (see ``_LazyBeamPayload``):
    # the pinned ``host_result`` is a single shared staging buffer across every
    # session, so the next terminal encode must not rewrite it while the async
    # output thread is still materializing the previous terminal's copy.  Set
    # on the busy-loop thread right after ``copy_event.record()``; cleared when
    # the lazy payload materializes (event-grained sync) or, defensively, at
    # the top of the next terminal fast path before any staging write.  Mirrors
    # the ``_h2d_pending`` flag in ``beam_attn_metadata._pack_int32_to_dev``.
    copy_pending: bool = False

    @property
    def capacity(self) -> int:
        """Staging words: 3 compact arrays + the terminal sequence segment."""

        return (3 + self.max_decode_steps) * self.max_batch * self.key.beam_width

    @classmethod
    def allocate(
        cls,
        *,
        max_batch: int,
        beam_width: int,
        max_decode_steps: int,
        device: torch.device | str,
        logits_dtype: torch.dtype,
    ) -> "BeamNpuTransportWorkspace":
        """Allocate the pinned host staging pair once; callers retain it."""

        if isinstance(max_batch, bool) or not isinstance(max_batch, int) or max_batch < 1:
            raise ValueError("max_batch must be a positive integer")
        if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
            raise ValueError("beam_width must be a positive integer")
        if (
            isinstance(max_decode_steps, bool)
            or not isinstance(max_decode_steps, int)
            or max_decode_steps < 1
        ):
            raise ValueError("max_decode_steps must be a positive integer")
        resolved_device = torch.device(device)
        if resolved_device.type != "npu":
            raise ValueError("BeamNpuTransportWorkspace is only supported for NPU workspaces")

        capacity = (3 + max_decode_steps) * max_batch * beam_width
        host_result = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        copy_event = None
        try:
            copy_event = torch.npu.Event(enable_timing=False)
        except Exception:
            # torch.npu.Event unavailable (older CANN / import-time): degrade to
            # a global ``torch.npu.synchronize()`` in the encoder.  Still one
            # synchronization per step rather than one per tensor.
            copy_event = None
        return cls(
            key=BeamPostprocessWorkspaceKey(beam_width, beam_width, logits_dtype),
            max_batch=max_batch,
            max_decode_steps=max_decode_steps,
            device=resolved_device,
            host_result=host_result,
            host_result_numpy=host_result.numpy(),
            copy_event=copy_event,
        )

    @property
    def pinned_memory_bytes(self) -> int:
        return self.host_result.numel() * self.host_result.element_size()


def get_beam_postprocess_workspace(
    runner: Any,
    beam_width: int,
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
    key = BeamPostprocessWorkspaceKey(beam_width, beam_width, dtype)
    try:
        return pool.require(key)
    except KeyError:
        return None


def get_beam_npu_transport_workspace(
    runner: Any,
    beam_width: int,
) -> BeamNpuTransportWorkspace | None:
    """Return the prewarmed NPU transport workspace, never allocating on a miss."""

    device = getattr(runner, "device", None)
    if str(getattr(device, "type", "")) != "npu":
        return None
    workspace = getattr(runner, "_beam_npu_transport_workspace", None)
    if not isinstance(workspace, BeamNpuTransportWorkspace):
        return None
    if workspace.key.beam_width != beam_width:
        return None
    return workspace


def _encode_cuda_beam_result(
    decision: dict[str, Any],
    *,
    is_last_step: bool,
    workspace: BeamPostprocessWorkspace | None = None,
) -> tuple[dict[str, Any], list[int]] | None:
    """Encode CUDA winners with one device-to-host synchronization.

    The canonical result channel is an untyped ``dict[str, Any]``.  Sending
    tensors through it loses msgpack auxiliary frames, while independently
    calling ``tensor.to('cpu').tolist()`` for ids, parents and scores forces
    multiple synchronizations and creates thousands of Python objects.

    Store the three equal-width arrays in one int32 transport buffer (float32
    scores are bit-reinterpreted, not converted), perform one D2H copy, and
    send the resulting bytes in ``beam_search_result``.  A supplied CUDA
    workspace reuses its fixed device buffer, pinned host buffer, cached NumPy
    view, and copy event; the allocating branch remains as the compatibility
    fallback.  This is intentionally distinct from the removed FlatLogprobs
    packed-wire path: token ids and parents remain separate arrays and no
    sentinel or token-bit packing is involved.
    """
    token_ids = decision.get("out_token_ids")
    parents = decision.get("out_token_index")
    scores = decision.get("out_log_probs")
    if not all(torch.is_tensor(value) for value in (token_ids, parents, scores)):
        return None
    if token_ids.ndim != 2 or parents.shape != token_ids.shape or scores.shape != token_ids.shape:
        return None

    count = token_ids.numel()
    if (
        workspace is not None
        and token_ids.device == workspace.packed_result.device
        and parents.device == workspace.packed_result.device
        and scores.device == workspace.packed_result.device
        and workspace.device.type == "cuda"
        and token_ids.dtype is torch.int32
        and parents.dtype is torch.int32
        and scores.dtype is torch.float32
        and token_ids.is_contiguous()
        and parents.is_contiguous()
        and scores.is_contiguous()
        and count <= workspace.max_batch * workspace.key.beam_width
        and workspace.copy_event is not None
        and workspace.host_result.is_pinned()
    ):
        packed = workspace.packed_result
        packed[:count].copy_(token_ids.view(-1))
        packed[count : 2 * count].copy_(parents.view(-1))
        packed[2 * count : 3 * count].copy_(scores.view(-1).view(torch.int32))
        host = workspace.host_result[: 3 * count]
        host.copy_(packed[: 3 * count], non_blocking=True)
        workspace.copy_event.record()
        workspace.copy_event.synchronize()
        host_numpy = workspace.host_result_numpy[: 3 * count]
    else:
        token_ids_i32 = token_ids if token_ids.dtype is torch.int32 else token_ids.to(torch.int32)
        parents_i32 = parents if parents.dtype is torch.int32 else parents.to(torch.int32)
        scores_f32 = scores if scores.dtype is torch.float32 else scores.to(torch.float32)
        pieces = (
            token_ids_i32.contiguous().reshape(-1),
            parents_i32.contiguous().reshape(-1),
            scores_f32.contiguous().reshape(-1).view(torch.int32),
        )
        host = torch.cat(pieces).to(device="cpu")
        host_numpy = host.numpy()
    payload = {
        "buffer_format": _BEAM_RESULT_BUFFER_FORMAT,
        "shape": [int(token_ids.shape[0]), int(token_ids.shape[1])],
        "buffer": host_numpy.tobytes(),
        "is_last_step": bool(is_last_step),
    }
    return payload, [int(parent) for parent in host_numpy[count : 2 * count].tolist()]


class _LazyBeamPayload(dict):
    """A terminal beam-decision payload that defers its D2H sync + materialization.

    The terminal step's ``copy_event.synchronize()`` + ``tobytes()``/``tolist()``
    is the heaviest host-side work in a beam session (the W*steps ``tolist`` of
    ``out_sequence`` alone).  Running it synchronously inside ``sample_tokens``
    blocks the worker busy-loop thread, which under async scheduling serializes
    the *next* session's ``execute_model``/``sample_tokens`` issuance even though
    the terminal step has nothing left to schedule for its own session.

    This dict subclass carries the cheap scalar fields (``buffer_format``,
    ``shape``, ``is_last_step``, and the caller-added ``stage``/``session_id``)
    immediately so EngineCore's ``patched_update_from_output`` can route it
    without touching the device, while the three heavy fields (``buffer``,
    ``out_log_probs``, ``out_sequence``) are materialized lazily at the first
    pickle.  The worker→EngineCore IPC goes through ``MessageQueue.enqueue`` →
    ``pickle.dumps`` (shm_broadcast.py), so ``__reduce__`` fires exactly there —
    on the async output thread when async scheduling is on, i.e. off the
    busy-loop critical path; on the busy-loop thread otherwise (identical to
    the previous inline behavior, no regression).

    The pinned ``host_result`` staging is a single shared buffer across
    sessions; the lazy payload keeps the ``copy_event`` reference and the
    ``copy_pending`` guard on the workspace ensures the next terminal encode
    waits on this event before rewriting staging (cross-thread staging-rewrite
    race, same class as ``_pack_int32_to_dev``).
    """

    __slots__ = ("_materialize_fn", "_materialized", "_materialized_dict")

    def __init__(self, scalars: dict, materialize_fn):
        super().__init__(scalars)
        self._materialize_fn = materialize_fn
        self._materialized = False
        self._materialized_dict: dict | None = None

    def get(self, key, default=None):
        # In uniprocess deployments there is no pickle boundary to trigger
        # __reduce__. Materialize only when a consumer asks for a heavy field.
        if not self._materialized and key not in self:
            return self._materialize().get(key, default)
        return super().get(key, default)

    def __reduce__(self):
        return (dict, (self._materialize(),))

    def _materialize(self) -> dict:
        if self._materialized:
            assert self._materialized_dict is not None
            return self._materialized_dict
        heavy = self._materialize_fn()
        # Keep normal dict semantics after the first lazy read: later ``get``
        # and ``__getitem__`` calls must see the same materialized fields.
        super().update(heavy)
        result = dict(self)
        self._materialized = True
        self._materialized_dict = result
        # Drop the closure reference once materialized so the pinned buffer /
        # workspace are not retained past the pickle boundary.
        self._materialize_fn = None  # type: ignore[assignment]
        return result


def _encode_npu_beam_result(
    decision: dict[str, Any],
    *,
    is_last_step: bool,
    workspace: BeamNpuTransportWorkspace | None = None,
) -> tuple[dict[str, Any], list[int]] | None:
    """Encode NPU winners with one device-to-host synchronization.

    Mirrors ``_encode_cuda_beam_result``'s fast path: pack the three
    equal-width arrays (int32 token ids, int32 parents, fp32 scores
    bit-reinterpreted as int32) into the pinned host staging buffer, issue a
    single ``non_blocking`` D2H copy, and synchronize once.  This replaces the
    legacy per-key ``.to("cpu").tolist()`` loop, which cost ~30ms per tensor
    on NPU (profiling bw128_1k_t3) — 3 tensors per step stalled ~90ms.

    The wire format is identical to the CUDA compact buffer
    (``beam_result_i32_f32_v1``), so the frontend decodes it through the same
    device-agnostic ``unpack_beam_result_buffer`` path.

    Terminal step (``is_last_step``): the decision additionally carries
    ``out_sequence`` ``[B, W_eff, steps]`` (cumulative decode state).  It is
    appended to the SAME pinned transfer (staging layout
    ``[ids | parents | score_bits | sequence]``).  When a ``copy_event`` is
    available the terminal step is **deferred**: the D2H + ``record()`` is
    issued here, but the ``synchronize()`` + ``tobytes``/``tolist``
    materialization is postponed to the first pickle of the returned
    ``_LazyBeamPayload``.  Under async scheduling that pickle happens on the
    output thread (off the worker busy-loop), so the worker can immediately
    issue the next session's compute while the terminal tail (final select
    D2H + W*steps tolist + pickle) overlaps — the terminal step has nothing
    left to schedule for its own session.  Without an event, or for
    non-terminal steps, the sync stays inline (one sync per session).

    Returns ``(payload, parent_host)`` — ``parent_host`` is the request-0 parent
    mapping derived from the same D2H copy, so the KV-reorder record needs no
    second device→host sync (cf. the ``_finalize_decision`` comment).  For the
    deferred terminal path ``parent_host`` is an empty list (the terminal step
    has no successor, so the KV-reorder record is never consumed).  Returns
    ``None`` when the payload shape/dtype does not satisfy the fast path; the
    caller keeps the legacy per-key list fallback for correctness.
    """
    token_ids = decision.get("out_token_ids")
    parents = decision.get("out_token_index")
    scores = decision.get("out_log_probs")
    sequence = decision.get("out_sequence")
    if not all(torch.is_tensor(value) for value in (token_ids, parents, scores)):
        return None
    if token_ids.ndim != 2 or parents.shape != token_ids.shape or scores.shape != token_ids.shape:
        return None
    # Terminal step requires the sequence tensor for the same-transfer packing;
    # a missing/odd-shaped sequence on the terminal step falls back to the
    # legacy path (correctness first).
    if is_last_step and (
        not torch.is_tensor(sequence)
        or sequence.ndim != 3
        or sequence.shape[0] != token_ids.shape[0]
        or sequence.shape[1] != token_ids.shape[1]
    ):
        return None

    count = token_ids.numel()
    seq_words = sequence.numel() if (is_last_step and torch.is_tensor(sequence)) else 0
    fast_path = (
        workspace is not None
        and token_ids.device == workspace.device
        and parents.device == workspace.device
        and scores.device == workspace.device
        and token_ids.dtype is torch.int32
        and parents.dtype is torch.int32
        and scores.dtype is torch.float32
        and token_ids.is_contiguous()
        and parents.is_contiguous()
        and scores.is_contiguous()
        and count <= workspace.max_batch * workspace.key.beam_width
        and workspace.host_result.is_pinned()
    )
    if is_last_step and fast_path:
        # The sequence segment must fit the prewarmed staging tail; its extent
        # is bounded by beam_max_decode_steps at allocation time.
        fast_path = (
            torch.is_tensor(sequence)
            and sequence.device == workspace.device
            and sequence.dtype is torch.int32
            and sequence.is_contiguous()
            and seq_words
            <= workspace.max_batch
            * workspace.key.beam_width
            * workspace.max_decode_steps
        )
    if fast_path:
        # Staging-rewrite guard: the pinned ``host_result`` is a single shared
        # buffer across sessions.  A previous terminal step may have issued a
        # DEFERRED D2H (``copy_event.record()`` without an inline sync) whose
        # materialization is still pending on the async output thread.  Wait on
        # that recorded event — event-grained, not a stream sync — before
        # rewriting staging so the two transfers never overlap.  No-op for the
        # common case where the previous step materialized inline (non-terminal
        # or fallback path clears the flag on sync).
        if workspace.copy_pending:
            if workspace.copy_event is not None:
                workspace.copy_event.synchronize()
            else:
                torch.npu.synchronize()
            workspace.copy_pending = False
        # One device-side concat → one D2H copy → one event sync.  The concat
        # keeps the copy a single contiguous transfer; the int32 bit-view of
        # the fp32 scores avoids any float↔int conversion on the host side.
        # Terminal step: the int32 sequence rides in the same transfer.
        pieces = (
            token_ids.view(-1),
            parents.view(-1),
            scores.view(-1).view(torch.int32),
        )
        if is_last_step:
            pieces = pieces + (sequence.view(-1),)
        packed = torch.cat(pieces)
        total_words = 3 * count + seq_words
        host = workspace.host_result[:total_words]
        host.copy_(packed, non_blocking=True)

        # Deferred terminal path: when an event is available, issue the D2H
        # ``record()`` and return WITHOUT syncing or materializing.  The sync
        # ``tobytes``/``tolist`` happens lazily at the first pickle (see
        # ``_LazyBeamPayload``), which under async scheduling lands on the
        # output thread — off the busy-loop critical path — so the worker can
        # immediately dequeue/issue the next session's execute_model.  The
        # terminal step has no subsequent scheduling of its own, so deferring
        # its wait only ever overlaps it with useful next-batch work.
        if is_last_step and workspace.copy_event is not None:
            workspace.copy_event.record()
            workspace.copy_pending = True
            batch, width = int(token_ids.shape[0]), int(token_ids.shape[1])
            steps = seq_words // (batch * width)
            # ``parent_host`` is consumed only for non-terminal steps (caller
            # builds ``state.kv_reorder`` from it); the terminal step has no
            # successor, so an empty list is the correct deferred return.
            scalars = {
                "buffer_format": _BEAM_RESULT_BUFFER_FORMAT,
                "shape": [batch, width],
                "is_last_step": True,
            }
            ws_host = workspace.host_result
            ws_numpy = workspace.host_result_numpy
            copy_event = workspace.copy_event

            def _materialize(
                _copy_event=copy_event,
                _ws=workspace,
                _host=ws_host,
                _numpy=ws_numpy,
                _count=count,
                _seq_words=seq_words,
                _batch=batch,
                _width=width,
                _steps=steps,
            ) -> dict[str, Any]:
                _copy_event.synchronize()
                _ws.copy_pending = False
                return {
                    "buffer": _numpy[: 3 * _count].tobytes(),
                    "out_log_probs": _host[2 * _count : 3 * _count]
                    .view(torch.float32)
                    .reshape(_batch, _width)
                    .tolist(),
                    "out_sequence": _host[3 * _count : 3 * _count + _seq_words]
                    .reshape(_batch, _width, _steps)
                    .tolist(),
                }

            return _LazyBeamPayload(scalars, _materialize), []

        # Inline path: non-terminal steps, or terminal without a usable event
        # (fallback to the previous behavior — one sync per terminal session).
        if workspace.copy_event is not None:
            workspace.copy_event.record()
            workspace.copy_event.synchronize()
        else:
            # Event API unavailable: a global device sync is coarser (all NPU
            # work, not just this copy) but still collapses the per-tensor
            # syncs into one.
            torch.npu.synchronize()
        host_numpy = workspace.host_result_numpy[:total_words]
    else:
        token_ids_i32 = token_ids if token_ids.dtype is torch.int32 else token_ids.to(torch.int32)
        parents_i32 = parents if parents.dtype is torch.int32 else parents.to(torch.int32)
        scores_f32 = scores if scores.dtype is torch.float32 else scores.to(torch.float32)
        pieces = (
            token_ids_i32.contiguous().reshape(-1),
            parents_i32.contiguous().reshape(-1),
            scores_f32.contiguous().reshape(-1).view(torch.int32),
        )
        if is_last_step and torch.is_tensor(sequence):
            seq_i32 = sequence if sequence.dtype is torch.int32 else sequence.to(torch.int32)
            pieces = pieces + (seq_i32.contiguous().reshape(-1),)
        host = torch.cat(pieces).to(device="cpu")
        host_numpy = host.numpy()
    payload = {
        "buffer_format": _BEAM_RESULT_BUFFER_FORMAT,
        "shape": [int(token_ids.shape[0]), int(token_ids.shape[1])],
        "buffer": host_numpy[: 3 * count].tobytes(),
        "is_last_step": bool(is_last_step),
    }
    if is_last_step:
        # Derive the plain nested lists the serving terminal short-circuit
        # reads (``_worker_final_sequences``) from the already-synced host
        # tensor — host-memory reads only, no further D2H.  The fp32 score
        # re-interpretation uses a torch dtype bit-view (int32 and float32
        # share the 4-byte itemsize); both branches keep ``host`` as a
        # contiguous CPU int32 tensor.
        batch, width = int(token_ids.shape[0]), int(token_ids.shape[1])
        steps = seq_words // (batch * width)
        payload["out_log_probs"] = (
            host[2 * count : 3 * count].view(torch.float32).reshape(batch, width).tolist()
        )
        payload["out_sequence"] = (
            host[3 * count : 3 * count + seq_words].reshape(batch, width, steps).tolist()
        )
    return payload, [int(parent) for parent in host_numpy[count : 2 * count].tolist()]


def _encode_cuda_deferred_beam_result(
    decision: dict[str, Any],
    workspace: BeamPostprocessWorkspace,
    *,
    is_last_step: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[int]] | None:
    """Finish a Worker-packed decode result with one pinned D2H copy."""

    if not bool(decision.get("_gr_deferred_cuda_status", False)):
        return None
    token_ids = decision.get("out_token_ids")
    parents = decision.get("out_token_index")
    scores = decision.get("out_log_probs")
    if not all(torch.is_tensor(value) for value in (token_ids, parents, scores)):
        return None
    batch = decision.get("_gr_result_batch")
    width = decision.get("_gr_result_width")
    if (
        not isinstance(batch, int)
        or not isinstance(width, int)
        or tuple(token_ids.shape) != (batch, width)
        or parents.shape != token_ids.shape
        or scores.shape != token_ids.shape
        or batch < 1
        or batch > workspace.max_batch
        or width < 1
        or width > workspace.key.beam_width
        or workspace.prefill_copy_event is None
        or not workspace.host_prefill_staging.is_pinned()
    ):
        return None

    count = batch * width
    copy_words = workspace.max_batch + 3 * count
    stream = torch.cuda.current_stream(workspace.device)
    with torch.autograd.profiler.record_function("decode_result_d2h"):
        workspace.host_prefill_staging[:copy_words].copy_(
            workspace.prefill_staging[:copy_words],
            non_blocking=True,
        )
        workspace.prefill_copy_event.record(stream)
        workspace.prefill_copy_event.synchronize()

    # The workspace is reused by the next decode step, while an async vLLM
    # output may retain these compatibility tensors until get_output().  Give
    # every published output its own O(W) host storage after the D2H has
    # completed; a CUDA event alone cannot protect the host buffer lifetime.
    host_snapshot = workspace.host_prefill_staging[:copy_words].clone()
    host_numpy = host_snapshot.numpy()
    bad_rows = [row for row in range(batch) if int(host_numpy[row]) != 0]
    if bad_rows:
        raise RuntimeError(
            "constraint Backend returned no legal candidate for batch rows "
            f"{bad_rows}; refusing to sample a padding token"
        )
    payload_begin = workspace.max_batch
    payload_numpy = host_numpy[payload_begin : payload_begin + 3 * count]
    payload = {
        "buffer_format": _BEAM_RESULT_BUFFER_FORMAT,
        "shape": [batch, width],
        "buffer": payload_numpy.tobytes(),
        "is_last_step": bool(is_last_step),
    }
    compatibility = dict(decision)
    compatibility.update(
        {
            "_host_token_ids": host_snapshot[payload_begin : payload_begin + count].view(
                batch, width
            ),
            "_host_log_probs": host_snapshot[payload_begin + 2 * count : payload_begin + 3 * count]
            .view(torch.float32)
            .view(batch, width),
            "_host_selected_token_ranks": host_snapshot[:batch],
        }
    )
    parent_host = [int(parent) for parent in payload_numpy[count : 2 * count].tolist()]
    return payload, compatibility, parent_host


def unpack_beam_result_buffer(
    decision: Any,
) -> tuple[list[int], list[float], list[int]] | None:
    """Decode the compact ``beam_search_result`` bytes on EngineCore/frontend."""
    if not isinstance(decision, dict):
        return None
    if decision.get("buffer_format") != _BEAM_RESULT_BUFFER_FORMAT:
        return None
    shape = decision.get("shape")
    data = decision.get("buffer")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in shape)
        or not isinstance(data, (bytes, bytearray, memoryview))
    ):
        return None
    count = int(shape[0]) * int(shape[1])
    if len(data) != count * 3 * 4:
        return None
    view = memoryview(data)
    token_ids = list(struct.unpack_from(f"<{count}i", view, 0))
    parents = list(struct.unpack_from(f"<{count}i", view, count * 4))
    scores = list(struct.unpack_from(f"<{count}f", view, count * 8))
    return token_ids, scores, parents


def flatten_decision_value(value: Any) -> list[Any] | None:
    """Flatten one worker beam-decision payload value to a flat Python list.

    The return-channel payload crosses the EngineCore IPC boundary inside
    ``kv_transfer_params`` (``dict[str, Any]``): msgpack only round-trips
    plain containers through an untyped mapping, so the worker ships plain
    nested lists (see ``_maybe_run_worker_beam_decision``).  In-process
    consumers may still hand over the original host tensors, and a payload
    that skipped normalization arrives as the raw encoder triple
    ``('int64', shape, data)`` — the ``data`` slot is either an inline
    memoryview or an aux-buffer INDEX whose backing frame never reaches this
    side, i.e. unrecoverable — so return ``None`` and let the caller fall
    back to its legacy path.
    """
    if torch.is_tensor(value):
        return value.detach().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        # Serialized-tensor triple: ('<dtype>', shape, memoryview | index).
        if len(value) == 3 and isinstance(value[0], str) and isinstance(value[1], (list, tuple)):
            return None
        flat: list[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                nested = flatten_decision_value(item)
                if nested is None:
                    return None
                flat.extend(nested)
            else:
                flat.append(item)
        return flat
    return None


def _decision_buffers(runner: Any):
    """Return the active beam session's BeamDecisionBuffers (or None)."""
    beam_info = getattr(runner, "_current_beam_info", None)
    if beam_info is None:
        return None
    session = beam_info.get("beam_session")
    if session is None or not getattr(session, "is_active", False):
        return None
    context = getattr(session, "context", None)
    if context is None or not hasattr(context, "decision"):
        return None
    return context.decision


def _is_beam_decode_step(runner: Any) -> bool:
    """True when the current run is an incremental beam decode step."""
    beam_info = getattr(runner, "_current_beam_info", None)
    if beam_info is None:
        return False
    for mdata in (beam_info.get("beam_data") or {}).values():
        if mdata.get("is_beam_decode", False) and "decode_step" in mdata:
            return True
    return False


def _cuda_worker_decision_width_supported(beam_width: object) -> bool:
    """Whether the fused CUDA decision kernels support ``beam_width``."""
    if isinstance(beam_width, bool) or not isinstance(beam_width, int):
        return False
    from vllm_gr.config.models import MAX_BEAM_WIDTH

    return 1 <= beam_width <= MAX_BEAM_WIDTH


def cuda_prefill_postprocess_enabled(
    runner: Any,
    beam_width: object,
    batch_size: object,
) -> bool:
    """Return whether the private CUDA P->D fast path may defer validation."""

    if getattr(runner, "_enable_cuda_prefill_postprocess_fast_path", True) is False:
        return False
    device = getattr(runner, "device", None)
    if str(getattr(device, "type", "")) != "cuda":
        return False
    if (
        isinstance(beam_width, bool)
        or not isinstance(beam_width, int)
        or beam_width not in _CUDA_PREFILL_POSTPROCESS_WIDTHS
        or batch_size != 1
    ):
        return False
    from vllm_gr.config import ConstraintBackend
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    gr_cfg = _get_gr_config(runner)
    if not bool(gr_cfg.beam_worker_decision):
        return False
    if gr_cfg.constraint_backend != ConstraintBackend.CONSTRAINT_TABLE:
        return False
    workspace = get_beam_postprocess_workspace(runner, int(beam_width))
    return workspace is not None and workspace.max_batch >= int(batch_size)


def cuda_decode_postprocess_enabled(
    runner: Any,
    beam_width: object,
    rows: object,
) -> bool:
    """Whether CUDA Decode may defer candidate status to BeamSearchGroup."""

    if getattr(runner, "_enable_cuda_decode_postprocess_fast_path", True) is False:
        return False
    if not worker_decision_enabled(runner):
        return False
    if (
        isinstance(beam_width, bool)
        or not isinstance(beam_width, int)
        or beam_width not in _CUDA_PREFILL_POSTPROCESS_WIDTHS
        or isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows < 1
        or rows % beam_width != 0
    ):
        return False
    workspace = get_beam_postprocess_workspace(runner, beam_width)
    return workspace is not None and rows // beam_width <= workspace.max_batch


def _xllm_decode_mode_active() -> bool:
    """True iff the NPU decode execution mode resolves to the fused xllm path.

    The worker-side beam decision is the post-processing half of the xllm
    pipeline: its same-step ``select_unshared_kv`` reorder assumes the fused
    pool layout, and running it under ``cascade`` mode would mix the two
    execution paths.  Resolved lazily through the same per-process cache the
    attention backend uses (``_get_npu_decode_execution``; unset config
    defaults to ``cascade``, NPU-with-library defaults to ``xllm``).
    """
    try:
        from vllm_gr.v1.attention.backends.beam_attn import _get_npu_decode_execution

        return _get_npu_decode_execution() == "xllm"
    except Exception:
        return False


def worker_decision_enabled(runner: Any) -> bool:
    """Return whether worker-side beam decision is active for this run.

    Requires the GRConfig opt-in, an active BeamSearchSession, an incremental
    beam-decode step, a CUDA or NPU device, and worker-side candidate
    filtering. NPU additionally requires the fused xllm decode mode; CUDA
    dispatches through the canonical CUDA operators. Trie filtering remains
    frontend-side and therefore keeps the frontend NumPy decision path.
    """
    from vllm_gr.config import ConstraintBackend
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    gr_cfg = _get_gr_config(runner)
    if not bool(gr_cfg.beam_worker_decision):
        return False
    if _is_beam_decode_step(runner) is False:
        return False
    if _decision_buffers(runner) is None:
        return False
    device = getattr(runner, "device", None)
    device_type = str(getattr(device, "type", ""))
    if device_type not in ("cuda", "npu"):
        return False
    if device_type == "cuda":
        beam_info = getattr(runner, "_current_beam_info", None)
        session = beam_info.get("beam_session") if beam_info else None
        if not _cuda_worker_decision_width_supported(getattr(session, "beam_width", None)):
            return False
    # NPU same-step KV selection is defined only for fused xllm mode. CUDA
    # uses the generic session dispatcher and has no xllm dependency.
    if device_type == "npu" and not _xllm_decode_mode_active():
        return False
    # Trie-catalog filtering runs frontend-side; Constraint Table candidates
    # are already filtered by the worker's rec_constrained_topk.  Only decide
    # worker-side when no frontend-side catalog pruning applies.
    if (
        gr_cfg.catalog_path is not None
        and gr_cfg.constraint_backend != ConstraintBackend.CONSTRAINT_TABLE
    ):
        return False
    return True


def _resident_beam_decode_step(beam_info: Any) -> bool:
    """True when the current step is a Stage 1 resident beam-decode step.

    The ``resident`` metadata flag is set by ``build_beam_request_metadata``
    from ``Request.is_beam_resident`` and only ever originates from the
    ``beam_async`` entry point's session template, so it is exactly the
    Stage 1 async-scheduling form — no extra config lookup is needed.
    """
    if not beam_info:
        return False
    for mdata in (beam_info.get("beam_data") or {}).values():
        if mdata.get("is_beam_decode", False) and mdata.get("resident", False):
            return True
    return False


def enqueue_cuda_prefill_workspace_result(
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    candidate_counts: torch.Tensor,
    out_status: torch.Tensor,
    out_payload: torch.Tensor,
    device_copy: torch.Tensor,
    host_copy: torch.Tensor,
    copy_event: Any,
    *,
    stream: Any,
) -> None:
    """Unchecked fixed-workspace ResultOut + pinned-copy launch sequence."""

    from vllm_gr.ops.cuda.prefill_beam_result import _prefill_beam_result_out_unchecked

    _prefill_beam_result_out_unchecked(
        token_ids,
        scores,
        candidate_counts,
        out_status,
        out_payload,
        _CUDA_PREFILL_INVALID_TOKEN_ID,
    )
    host_copy.copy_(device_copy, non_blocking=True)
    copy_event.record(stream)


def enqueue_cuda_prefill_postprocess(runner: Any, sampler_output: Any) -> bool:
    """Enqueue result packing and pinned D2H without waiting for the host."""

    if bool(getattr(sampler_output, "_gr_prefill_result_enqueued", False)):
        return True
    if not bool(getattr(sampler_output, "_gr_deferred_prefill_status", False)):
        return False
    beam_info = getattr(runner, "_current_beam_info", None)
    beam_data = (beam_info.get("beam_data") or {}) if beam_info else {}
    prefills = [mdata for mdata in beam_data.values() if mdata.get("is_constraint_prefill", False)]
    if len(prefills) != 1:
        return False
    width = prefills[0].get("beam_width")
    if not cuda_prefill_postprocess_enabled(runner, width, len(prefills)):
        return False

    workspace = getattr(sampler_output, "_gr_beam_postprocess_workspace", None)
    expected_workspace = get_beam_postprocess_workspace(runner, int(width))
    if workspace is None or workspace is not expected_workspace:
        return False
    token_ids = getattr(sampler_output, "_gr_constraint_token_ids", None)
    scores = getattr(sampler_output, "_gr_constraint_logprobs", None)
    candidate_counts = getattr(sampler_output, "_gr_constraint_valid_counts", None)
    if (
        not torch.is_tensor(token_ids)
        or not torch.is_tensor(scores)
        or not torch.is_tensor(candidate_counts)
        or tuple(token_ids.shape) != (1, int(width))
        or tuple(scores.shape) != tuple(token_ids.shape)
        or candidate_counts.numel() < 1
        or workspace.prefill_copy_event is None
        or not workspace.host_prefill_staging.is_pinned()
    ):
        return False

    stream = torch.cuda.current_stream(workspace.device)
    with torch.autograd.profiler.record_function("prefill_candidate_finalize"):
        enqueue_cuda_prefill_workspace_result(
            token_ids,
            scores,
            candidate_counts[:1],
            workspace.prefill_staging[:1],
            workspace.prefill_staging[workspace.max_batch :],
            workspace.prefill_staging[: workspace.max_batch + 3 * int(width)],
            workspace.host_prefill_staging[: workspace.max_batch + 3 * int(width)],
            workspace.prefill_copy_event,
            stream=stream,
        )
    copy_words = workspace.max_batch + 3 * int(width)
    sampler_output._gr_prefill_result_enqueued = True
    sampler_output._gr_prefill_result_width = int(width)
    sampler_output._gr_prefill_result_copy_words = copy_words
    return True


def _run_cuda_prefill_postprocess(
    runner: Any,
    sampler_output: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]] | None:
    """Finalize an enqueued CUDA prefill result at the sampling boundary."""

    if not enqueue_cuda_prefill_postprocess(runner, sampler_output):
        return None
    beam_info = getattr(runner, "_current_beam_info", None)
    beam_data = (beam_info.get("beam_data") or {}) if beam_info else {}
    prefills = [mdata for mdata in beam_data.values() if mdata.get("is_constraint_prefill", False)]
    if len(prefills) != 1:
        return None
    width = prefills[0].get("beam_width")
    if not cuda_prefill_postprocess_enabled(runner, width, len(prefills)):
        return None

    workspace = getattr(sampler_output, "_gr_beam_postprocess_workspace", None)
    expected_workspace = get_beam_postprocess_workspace(runner, int(width))
    if workspace is None or workspace is not expected_workspace:
        return None
    token_ids = getattr(sampler_output, "_gr_constraint_token_ids", None)
    scores = getattr(sampler_output, "_gr_constraint_logprobs", None)
    candidate_counts = getattr(sampler_output, "_gr_constraint_valid_counts", None)
    if (
        not torch.is_tensor(token_ids)
        or not torch.is_tensor(scores)
        or not torch.is_tensor(candidate_counts)
        or tuple(token_ids.shape) != (1, int(width))
        or tuple(scores.shape) != tuple(token_ids.shape)
        or candidate_counts.numel() < 1
        or workspace.prefill_copy_event is None
        or not workspace.host_prefill_staging.is_pinned()
    ):
        return None

    from vllm_gr.ops.cuda.prefill_beam_result import STATUS_OK

    batch_size = 1
    active_payload_words = 3 * batch_size * int(width)
    with torch.autograd.profiler.record_function("prefill_result_wait"):
        workspace.prefill_copy_event.synchronize()

    copy_words = workspace.max_batch + active_payload_words
    # See _encode_cuda_deferred_beam_result: compatibility tensors may outlive
    # this step under async scheduling, so they must not alias the reusable
    # pinned workspace.
    host_snapshot = workspace.host_prefill_staging[:copy_words].clone()
    host_numpy = host_snapshot.numpy()
    bad_rows = [row for row in range(batch_size) if int(host_numpy[row]) != STATUS_OK]
    if bad_rows:
        raise RuntimeError(f"No valid constrained candidates for batch rows {bad_rows}")
    payload_numpy = host_numpy[workspace.max_batch : workspace.max_batch + active_payload_words]
    with torch.autograd.profiler.record_function("prefill_beam_bootstrap"):
        payload = {
            "buffer_format": _BEAM_RESULT_BUFFER_FORMAT,
            "shape": [batch_size, int(width)],
            "buffer": payload_numpy.tobytes(),
            "is_last_step": False,
        }

    lp_tensors = getattr(sampler_output, "logprobs_tensors", None)
    wire_scores = getattr(lp_tensors, "logprobs", None)
    if not torch.is_tensor(wire_scores) or tuple(wire_scores.shape) != tuple(token_ids.shape):
        wire_scores = scores
    compatibility_views = {
        "out_token_ids": token_ids,
        "out_log_probs": wire_scores,
        "_host_token_ids": host_snapshot[
            workspace.max_batch : workspace.max_batch + batch_size * int(width)
        ].view(batch_size, int(width)),
        "_host_log_probs": host_snapshot[
            workspace.max_batch + 2 * batch_size * int(width) : workspace.max_batch
            + 3 * batch_size * int(width)
        ]
        .view(torch.float32)
        .view(batch_size, int(width)),
        "_host_selected_token_ranks": host_snapshot[:batch_size],
    }
    return payload, compatibility_views


def _run_cuda_prefill_decision(runner: Any, sampler_output: Any) -> dict[str, torch.Tensor] | None:
    """Publish constrained prefill winners from the CUDA worker.

    Constraint prefill has one candidate row and no ``BeamSearchSession`` yet.
    The backend row is already deterministically ordered, so its W entries are
    the first W beam survivors (all with parent 0).  Publishing them through
    the canonical result channel avoids the remaining frontend NumPy select.

    The decode session initializes independently from its first per-parent
    candidate matrix; prefill scores must not seed its persistent buffers.
    """
    from vllm_gr.config import ConstraintBackend
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    if bool(getattr(sampler_output, "_gr_deferred_prefill_status", False)):
        counts = getattr(sampler_output, "_gr_constraint_valid_counts", None)
        if not torch.is_tensor(counts):
            raise RuntimeError("deferred CUDA prefill status is missing candidate counts")
        bad_rows = (counts <= 0).nonzero().reshape(-1).to(device="cpu").tolist()
        if bad_rows:
            raise RuntimeError(f"No valid constrained candidates for batch rows {bad_rows}")

    device = getattr(runner, "device", None)
    if str(getattr(device, "type", "")) != "cuda":
        return None
    gr_cfg = _get_gr_config(runner)
    if not bool(gr_cfg.beam_worker_decision):
        return None
    if gr_cfg.constraint_backend != ConstraintBackend.CONSTRAINT_TABLE:
        return None

    beam_info = getattr(runner, "_current_beam_info", None)
    beam_data = (beam_info.get("beam_data") or {}) if beam_info else {}
    prefills = [
        (req_id, mdata)
        for req_id, mdata in beam_data.items()
        if mdata.get("is_constraint_prefill", False)
    ]
    # The result ABI is one logical request. Multi-request prefill keeps the
    # established frontend path until the result channel becomes batched.
    if len(prefills) != 1:
        return None
    prefill_req_id, mdata = prefills[0]
    width = mdata.get("beam_width")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        return None
    if not _cuda_worker_decision_width_supported(width):
        return None

    lp_tensors = getattr(sampler_output, "logprobs_tensors", None)
    token_ids = getattr(lp_tensors, "logprob_token_ids", None)
    scores = getattr(lp_tensors, "logprobs", None)
    if (
        not torch.is_tensor(token_ids)
        or not torch.is_tensor(scores)
        or token_ids.ndim != 2
        or scores.shape != token_ids.shape
        or token_ids.shape[0] != 1
        or token_ids.shape[1] < width
    ):
        return None

    workspace = get_beam_postprocess_workspace(runner, width)
    if workspace is not None:
        out_ids = workspace.selected_token_ids[:1]
        out_scores = workspace.selected_scores[:1]
        out_parents = workspace.selected_parent_ids[:1]
        out_ids.copy_(token_ids[:, :width])
        out_scores.copy_(scores[:, :width])
        out_parents.zero_()
    else:
        out_ids = token_ids[:, :width].to(dtype=torch.int32).contiguous()
        out_scores = scores[:, :width].to(dtype=torch.float32).contiguous()
        out_parents = torch.zeros_like(out_ids, dtype=torch.int32)
    return {
        "out_token_ids": out_ids,
        "out_token_index": out_parents,
        "out_log_probs": out_scores,
        # Ownership markers (review P5/P8): EngineCore's engine-driven prefill
        # branch keys on stage="prefill" and validates session_id against the
        # output it was attached to, so a mixed batch (A decode + B prefill)
        # can never chain B's decision onto A's session.
        "stage": "prefill",
        "session_id": prefill_req_id,
    }


def _run_unconstrained_prefill_decision(
    runner: Any,
    sampler_output: Any,
) -> dict[str, torch.Tensor] | None:
    """Publish the top-W prefill winners for an UNCONSTRAINED beam session.

    The Stage 1 resident form (``GRLLM.async_beam_search``) chains step 0
    from a prefill worker decision on every accelerator.  Constraint Table
    prefills publish theirs through ``_run_cuda_prefill_decision`` (the
    constrained top-k operator's fixed-width row); an unconstrained session
    has no such operator, but the ordinary sampler already computed the
    flat top-K of the prefill logits (``logprobs == beam_width`` on the
    prefill's sampling params), so the first W candidates ARE the step-0
    beam survivors — all with parent 0, mirroring the CUDA contract
    (``_run_cuda_prefill_decision``'s ownership markers included).

    The decode session seeds its own persistent buffers later from the
    step-0 candidate matrix (``init_beam_decision_state``); prefill scores
    never touch them (same isolation rule as the CUDA path).
    """
    from vllm_gr.config import ConstraintBackend
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    device = getattr(runner, "device", None)
    if str(getattr(device, "type", "")) not in ("npu", "cuda"):
        return None
    gr_cfg = _get_gr_config(runner)
    if not bool(gr_cfg.beam_worker_decision):
        return None
    if gr_cfg.constraint_backend == ConstraintBackend.CONSTRAINT_TABLE:
        # Constraint Table prefills publish through the dedicated CUDA path
        # (constrained top-k operator + fixed-width rows).
        return None

    beam_info = getattr(runner, "_current_beam_info", None)
    beam_data = (beam_info.get("beam_data") or {}) if beam_info else {}
    prefills = [
        (req_id, mdata)
        for req_id, mdata in beam_data.items()
        if mdata.get("is_beam_prefill", False)
    ]
    if len(prefills) > 1:
        # Async resident B>1: seed every request row directly into its
        # registry slot. The prefill decision never leaves the worker.
        if not all(
            mdata.get("is_beam_single_request", False)
            or mdata.get("single_request", False)
            for _, mdata in prefills
        ):
            return None
        widths = {mdata.get("beam_width") for _, mdata in prefills}
        if len(widths) != 1:
            return None
        width = next(iter(widths))
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            return None

        lp_tensors = getattr(sampler_output, "logprobs_tensors", None)
        token_ids = getattr(lp_tensors, "logprob_token_ids", None)
        scores = getattr(lp_tensors, "logprobs", None)
        if (
            not torch.is_tensor(token_ids)
            or not torch.is_tensor(scores)
            or token_ids.ndim != 2
            or scores.shape != token_ids.shape
            or token_ids.shape[0] != len(prefills)
        ):
            return None
        n_cols = min(token_ids.shape[1], scores.shape[1])
        if n_cols > width:
            out_ids = token_ids[:, 1 : width + 1].to(dtype=torch.int32).contiguous()
            out_scores = scores[:, 1 : width + 1].to(dtype=torch.float32).contiguous()
        elif n_cols == width:
            out_ids = token_ids[:, :width].to(dtype=torch.int32).contiguous()
            out_scores = scores[:, :width].to(dtype=torch.float32).contiguous()
        else:
            return None

        prefill_map = dict(prefills)
        scheduled_order = [
            req_id
            for req_id in (beam_info.get("req_ids") or ())
            if req_id in prefill_map
        ]
        if len(scheduled_order) != len(prefills):
            scheduled_order = [req_id for req_id, _ in prefills]

        from vllm_gr.v1.worker.model_runner_common import (
            _create_single_request_beam_session,
        )

        sessions = []
        for row, req_id in enumerate(scheduled_order):
            mdata = prefill_map[req_id]
            session = _create_single_request_beam_session(
                runner,
                req_id=req_id,
                beam_width=width,
                prefix_len=int(mdata.get("prefix_len", 0)),
                max_decode_steps=int(mdata.get("max_decode_steps", 0)),
            )
            if session is None:
                raise RuntimeError(
                    f"unable to allocate resident beam session for {req_id!r}"
                )
            bufs = getattr(session.context, "decision", None)
            if bufs is None:
                raise RuntimeError("resident beam context has no decision buffers")
            slot = int(session.slot)
            winners = out_ids[row, :width]
            if winners.dtype is not bufs.sequence.dtype:
                winners = winners.to(bufs.sequence.dtype)
            win_scores = out_scores[row, :width]
            if win_scores.dtype is not bufs.beam_scores.dtype:
                win_scores = win_scores.to(bufs.beam_scores.dtype)
            bufs.sequence[slot, :width, 0].copy_(winners)
            bufs.beam_scores[slot, :width].copy_(win_scores)
            bufs.worker_decode_steps = 1
            sessions.append(session)

        if beam_info is not None:
            beam_info["beam_session"] = sessions[0]
        return {"stage": "prefill", "session_id": scheduled_order[0]}

    # The result ABI is one logical request (mirrors the CUDA gate).
    if len(prefills) != 1:
        return None
    prefill_req_id, mdata = prefills[0]
    width = mdata.get("beam_width")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        return None

    lp_tensors = getattr(sampler_output, "logprobs_tensors", None)
    token_ids = getattr(lp_tensors, "logprob_token_ids", None)
    scores = getattr(lp_tensors, "logprobs", None)
    if (
        not torch.is_tensor(token_ids)
        or not torch.is_tensor(scores)
        or token_ids.ndim != 2
        or scores.shape != token_ids.shape
        or token_ids.shape[0] != 1
    ):
        return None
    # Flat layout is [sampled, top1, ..., topK]: drop the sampled-token
    # column (dedup semantics, cf. extract_and_dedup_flat_logprobs) before
    # taking the W beam candidates.  A row already exactly W wide is the
    # constrained-override shape and never reaches this unconstrained gate,
    # but keep the min() guard from _select_candidates so slicing cannot
    # overflow.
    n_cols = min(token_ids.shape[1], scores.shape[1])
    if n_cols > width:
        cand_ids = token_ids[:, 1 : width + 1]
        cand_scores = scores[:, 1 : width + 1]
    elif n_cols == width:
        cand_ids = token_ids[:, :width]
        cand_scores = scores[:, :width]
    else:
        return None

    out_ids = cand_ids.to(dtype=torch.int32).contiguous()
    out_scores = cand_scores.to(dtype=torch.float32).contiguous()
    out_parents = torch.zeros_like(out_ids, dtype=torch.int32)

    # Single-Request form: the prefill IS the session's first step, so seed
    # the device-resident decision buffers NOW (column 0 = prefill winners,
    # ``worker_decode_steps = 1``) — before any decode step runs.  This makes
    # the first decode's input overwrite read the real winners instead of the
    # reset-zeros a freshly-``begin_session``'d buffer would hold, and lets the
    # subsequent ``_get_or_create_beam_session`` find an already-seeded session
    # (no second ``begin_session`` reset).  The engine-driven two-Request path
    # keeps the old contract (buffers seeded later by ``init_beam_decision_state``
    # during the step-0 decode decision).
    if mdata.get("is_beam_single_request", False) or mdata.get("single_request", False):
        from vllm_gr.v1.worker.model_runner_common import (
            _create_single_request_beam_session,
        )

        session = _create_single_request_beam_session(
            runner,
            req_id=prefill_req_id,
            beam_width=width,
            prefix_len=int(mdata.get("prefix_len", 0)),
            max_decode_steps=int(mdata.get("max_decode_steps", 0)),
        )
        if session is not None:
            bufs = getattr(session.context, "decision", None)
            if bufs is not None:
                n = min(width, bufs.sequence.shape[1])
                winners = out_ids[0, :n]
                if winners.dtype is not bufs.sequence.dtype:
                    winners = winners.to(bufs.sequence.dtype)
                win_scores = out_scores[0, :n]
                if win_scores.dtype is not bufs.beam_scores.dtype:
                    win_scores = win_scores.to(bufs.beam_scores.dtype)
                # Device-side seed (same stream as the prefill forward / the
                # next decode's overwrite — ordered before both).
                bufs.sequence[session.slot, :n, 0].copy_(winners)
                bufs.beam_scores[session.slot, :n].copy_(win_scores)
                bufs.worker_decode_steps = 1
            # Publish the session so the prefill iteration's worker-decision
            # bookkeeping and the next decode's execute_model see it.
            if beam_info is not None:
                beam_info["beam_session"] = session
        # The prefill decision has no host-side consumer in the Single-Request
        # form (no engine-driven prefill→step0 chain, the frontend drain loop
        # drops non-terminal outputs), so return a tensor-free stub instead of
        # the full candidate payload.  ``maybe_run_worker_beam_decision``'s
        # compact-encode path sees no tensors and publishes nothing — avoiding
        # the per-prefill pinned D2H + ``copy_event.synchronize()`` that the
        # two-Request form pays to ship the candidates to the frontend.  The
        # ownership markers are kept so any downstream ``_beam_advance_from_*``
        # hook still recognises the (suppressed) prefill stage.
        return {
            "stage": "prefill",
            "session_id": prefill_req_id,
        }

    return {
        "out_token_ids": out_ids,
        "out_token_index": out_parents,
        "out_log_probs": out_scores,
        # Ownership markers: same contract as _run_cuda_prefill_decision.
        "stage": "prefill",
        "session_id": prefill_req_id,
    }


def init_beam_decision_state(
    session: Any,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
) -> None:
    """Seed the persistent decision buffers from the first candidate round.

    Mirrors ACS ``BeamSearchContext.init_from_prefill``: at the first
    decision step each beam (one row of ``[B*W, W]`` candidates) contributes
    its top candidate token and score to ``sequence[..., 0]`` and
    ``beam_scores``, so the subsequent ``beam_search_group`` steps read
    cumulative state entirely from device-resident buffers.  Runs once per
    session before any ``decode_select``-style step.

    Args:
        session: the active BeamSearchSession (its ``context.decision`` buffers).
        top_tokens / top_logprobs: ``[B * W, W]`` candidate token ids / probs.
    """
    bufs = getattr(session.context, "decision", None)
    if bufs is None:
        return
    width = session.beam_width
    n_rows = top_tokens.shape[0]
    if width == 0 or n_rows == 0 or n_rows % width != 0:
        return
    batch = n_rows // width
    # [B, W, W]: the last dim holds the top-candidate ordering per beam.
    # Column-0 views are already contiguous per-row, and the dtype guards
    # keep the constrained path (int32 ids / float32 probs straight from
    # rec_constrained_topk) copy-free — the profiled ~12ms-per-copy host
    # stalls (aclnnInplaceCopyGetWorkspaceSize) came from casting first and
    # copying the cast result; matching dtypes lets copy_ take the
    # contiguous fast path directly.
    best_token = top_tokens.reshape(batch, width, width)[:, :, 0]
    best_score = top_logprobs.reshape(batch, width, width)[:, :, 0]
    if best_token.dtype is not bufs.sequence.dtype:
        best_token = best_token.to(bufs.sequence.dtype)
    if best_score.dtype is not bufs.beam_scores.dtype:
        best_score = best_score.to(bufs.beam_scores.dtype)
    n = min(width, bufs.sequence.shape[1])
    bufs.sequence[:batch, :n, 0].copy_(best_token[:, :n])
    bufs.beam_scores[:batch, :n].copy_(best_score[:, :n])
    # Worker-side token-count for the decision: one committed token per beam.
    bufs.worker_decode_steps = 1


def init_beam_decision_from_parents(
    session: Any,
    parent_sequences: Sequence[Sequence[int]],
    parent_cumulative_logprobs: Sequence[float],
) -> None:
    """Restore persistent decision state from already-selected parents.

    The first grouped decode evaluates candidates after the prefill winner
    (for OneRec, ``s_b`` after ``s_a``).  Seeding from the current candidate
    matrix would lose the prefill score and count the current local score
    twice, changing beam rank order.
    """
    import math

    width = session.beam_width
    if len(parent_sequences) != width or len(parent_cumulative_logprobs) != width:
        raise RuntimeError(
            f"worker beam decision parent state must contain exactly beam_width={width} rows"
        )
    lengths = {len(sequence) for sequence in parent_sequences}
    if len(lengths) != 1:
        raise RuntimeError("worker beam decision parent sequences must have equal lengths")
    steps = next(iter(lengths), 0)
    bufs = session.context.decision
    if steps < 1 or steps > bufs.sequence.shape[2]:
        raise RuntimeError(
            f"worker beam decision parent sequence length {steps} is outside "
            f"[1, {bufs.sequence.shape[2]}]"
        )
    scores = [float(score) for score in parent_cumulative_logprobs]
    # -inf is a legitimate parent-score sentinel at the EOS-mask protocol
    # boundary. Only NaN/+inf are protocol errors.
    if any(math.isnan(score) or score == math.inf for score in scores):
        raise RuntimeError(
            "worker beam decision parent scores must not be NaN or +inf; "
            "-inf marks an EOS-masked beam and is a valid score"
        )
    device = bufs.sequence.device
    sequence_tensor = torch.tensor(parent_sequences, dtype=torch.int32, device=device)
    score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
    row = int(getattr(session, "slot", 0))
    if row < 0 or row >= bufs.sequence.shape[0]:
        raise RuntimeError(
            f"worker beam decision slot {row} exceeds decision capacity "
            f"{bufs.sequence.shape[0]}"
        )
    bufs.sequence[row, :width, :steps].copy_(sequence_tensor)
    bufs.beam_scores[row, :width].copy_(score_tensor)
    if bufs.worker_decode_steps not in (0, steps):
        raise RuntimeError("synchronized beam sessions have inconsistent parent sequence lengths")
    bufs.worker_decode_steps = steps


def _select_candidates(
    sampler_output: Any,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extract ``[B*W, W]`` (token_ids int32, probs fp32) from sampler output.

    The sampler logprob tensors carry ``[rows, max_logprobs + 1]`` with the
    sampled token in column 0; the constrained override already replaced the
    constrained rows with the fixed-width ``W`` operator output.  For
    unconstrained rows we drop the sampled-token column to get exactly ``W``
    candidates per beam, matching ACS ``model_runner_v1``.

    dtype: the ``.to(...)`` casts are skipped when the source tensor already
    carries the target dtype (``torch.Tensor.to`` short-circuits, but the
    check avoids even the dispatch — profiling bw128_1k_t3 attributed a
    ~30ms host stall to these casts when they ran per candidate row).  The
    constrained paths already emit int32 token ids / float32 probs, so in
    the worker-decision flow the casts are no-ops; the unconstrained
    sampler path (int64 ids from upstream topk) still converts once per
    step here instead of per row.
    """
    lp_tensors = getattr(sampler_output, "logprobs_tensors", None)
    if lp_tensors is None:
        return None
    token_ids = getattr(lp_tensors, "logprob_token_ids", None)
    logprobs = getattr(lp_tensors, "logprobs", None)
    if token_ids is None or logprobs is None:
        return None
    if token_ids.dim() != 2 or logprobs.dim() != 2:
        return None
    if token_ids.shape[0] != logprobs.shape[0]:
        return None
    # Constrained rows already have width == W; unconstrained rows have
    # width == max_logprobs + 1.  Use min() so slicing can't overflow.
    n_cols = min(token_ids.shape[1], logprobs.shape[1])
    if n_cols > width:
        top_tokens = token_ids[:, 1 : width + 1]
        top_probs = logprobs[:, 1 : width + 1]
    elif n_cols == width:
        top_tokens = token_ids[:, :width]
        top_probs = logprobs[:, :width]
    else:
        return None
    if top_tokens.dtype is not torch.int32:
        top_tokens = top_tokens.to(torch.int32)
    if top_probs.dtype is not torch.float32:
        top_probs = top_probs.to(torch.float32)
    if not top_tokens.is_contiguous():
        top_tokens = top_tokens.contiguous()
    if not top_probs.is_contiguous():
        top_probs = top_probs.contiguous()
    return top_tokens, top_probs


def _select_candidates_into(
    sampler_output: Any,
    width: int,
    workspace: BeamPostprocessWorkspace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Normalize sampler candidates into fixed CUDA workspace buffers."""

    lp_tensors = getattr(sampler_output, "logprobs_tensors", None)
    if lp_tensors is None:
        return None
    token_ids = getattr(sampler_output, "_gr_constraint_token_ids", None)
    logprobs = getattr(sampler_output, "_gr_constraint_logprobs", None)
    valid_mask = getattr(sampler_output, "_gr_constraint_valid_mask", None)
    if not torch.is_tensor(token_ids):
        token_ids = getattr(lp_tensors, "logprob_token_ids", None)
    if not torch.is_tensor(logprobs):
        logprobs = getattr(lp_tensors, "logprobs", None)
    if not torch.is_tensor(token_ids) or not torch.is_tensor(logprobs):
        return None
    if token_ids.ndim != 2 or logprobs.ndim != 2 or token_ids.shape[0] != logprobs.shape[0]:
        return None
    n_cols = min(token_ids.shape[1], logprobs.shape[1])
    if n_cols > width:
        source_tokens = token_ids[:, 1 : width + 1]
        source_scores = logprobs[:, 1 : width + 1]
        source_mask = None
    elif n_cols == width:
        source_tokens = token_ids[:, :width]
        source_scores = logprobs[:, :width]
        source_mask = valid_mask if torch.is_tensor(valid_mask) else None
    else:
        return None

    rows = source_tokens.shape[0]
    capacity_rows = workspace.max_batch * width
    if rows > capacity_rows:
        return None
    out_tokens = workspace.topk_token_ids.view(capacity_rows, width)[:rows]
    out_scores = workspace.topk_scores.view(capacity_rows, width)[:rows]
    if source_tokens.data_ptr() != out_tokens.data_ptr() or source_tokens.dtype != torch.int32:
        out_tokens.copy_(source_tokens)
    if source_scores.data_ptr() != out_scores.data_ptr() or source_scores.dtype != torch.float32:
        out_scores.copy_(source_scores)

    out_mask: torch.Tensor | None = None
    if source_mask is not None:
        if tuple(source_mask.shape) != (rows, width):
            return None
        out_mask = workspace.topk_valid_mask.view(capacity_rows, width)[:rows]
        if source_mask.data_ptr() != out_mask.data_ptr():
            out_mask.copy_(source_mask)
    return out_tokens, out_scores, out_mask


def _reorder_kv_same_step(
    session: Any,
    beam_index: torch.Tensor,
    group_token_num: torch.Tensor,
) -> None:
    """Apply the beam decision's KV reorder inside the SAME step (ACS timing).

    ACS ``decode_select`` runs ``select_unshared_kv`` immediately after
    ``beam_search_group`` (context.py:471-480) — the next step's attention
    then reads an already-reordered pool with no frontend round-trip in the
    critical path.  ``beam_search_group`` output is grouped by source beam
    ascending (the torch reference's stable ``argsort(parent)`` locks the
    contract), so at full pool width the fused kernel's prefix-sum mapping
    applies directly — bypassing ``reorder_unshared_kv``'s per-step D2H
    sortedness check + bincount round-trip.  Sub-width pools, CPU hosts and
    unavailable ops fall back to the standard dispatcher.
    """
    context = session.context
    pools = context.pools
    state = session._state
    step = state.decode_step  # 0-based; the reorder covers [0, step]
    pool_beam_width = pools[0].unshared_key.shape[1]

    # The fused operator only executes on NPU tensors; CPU hosts (unit tests,
    # GPU-debug parity runs) go straight to the dispatcher fallback instead of
    # raising through the ACL layer first.
    if beam_index.numel() == pool_beam_width and str(beam_index.device.type) == "npu":
        try:
            from vllm_gr.ops.npu import beam_ops

            if beam_ops.is_available():
                # Reuse the persistent block-table buffer filled by this
                # step's execute_model (fixed address, slot already current).
                block_table = getattr(context, "xllm_block_table_buf", None)
                if block_table is None:
                    block_table = torch.tensor(
                        [session.slot], dtype=torch.int32, device=beam_index.device
                    )
                else:
                    block_table = block_table[session.slot : session.slot + 1]
                beam_ops.select_unshared_kv(
                    beam_index.contiguous(),
                    [p.unshared_key for p in pools],
                    [p.unshared_value for p in pools],
                    block_table,
                    group_token_num.contiguous(),
                    step + 1,
                    pool_beam_width,
                    len(pools),
                )
                state.decode_step += 1
                return
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("vllm-gr fused same-step KV reorder failed: %s; fallback", exc)

    # Sub-width pools / CPU hosts / op-unavailable: the dispatcher picks the
    # cascade kernel (arbitrary widths) and step_complete advances decode_step.
    # Runner-owned pools, aux buffers, and device pointer tables were created
    # and validated together with this persistent context.  Avoid rescanning
    # every K/V layer between BeamSearchGroup and SelectUnsharedKV.
    session.step_complete(beam_index, layout_prevalidated=True)


def _finalize_decision(
    session: Any,
    bufs: Any,
    out_ids: torch.Tensor,
    out_idx: torch.Tensor,
    out_lp: torch.Tensor,
    out_seq: torch.Tensor,
    prefix: torch.Tensor | None,
    batch: int,
    beam_width: int,
    result_width: int,
) -> dict[str, torch.Tensor]:
    """Persist cumulative state + run the ACS decode_select post-processing.

    Non-final steps (``prefix`` is not None): convert the operator's GLOBAL
    ``request * beam_width`` offsets to request-local form for both the
    parent mapping and the prefix sums (ACS context.py:458-468), persist
    ``group_token_num`` for the fused ``select_unshared_kv`` kernel, and
    reorder the KV pool in the SAME step.  Final steps (``prefix is None``):
    no reorder (ACS skips select_unshared_kv on the last step).
    """
    # Persist cumulative state back into the fixed-address buffers.
    write_w = min(out_seq.shape[1], bufs.sequence.shape[1])
    bufs.sequence[:batch, :write_w, : out_seq.shape[2]].copy_(out_seq[:, :write_w])
    bufs.beam_scores[:batch, :result_width].copy_(out_lp[:, :result_width])

    decision = {
        "out_token_ids": out_ids,
        "out_token_index": out_idx,
        "out_log_probs": out_lp,
        "out_sequence": out_seq,
    }
    if prefix is None:
        return decision

    request_offset = (
        torch.arange(batch, dtype=torch.int32, device=prefix.device).unsqueeze(1) * beam_width
    )
    local_idx = out_idx[:batch, :beam_width].to(torch.int32) - request_offset
    local_prefix = prefix[:batch, :beam_width].to(torch.int32) - request_offset
    bufs.group_token_num[:batch, :beam_width].copy_(local_prefix)

    state = session._state
    if not state.is_last_step:
        # Each session owns one W-wide pool row.  Apply each request-local
        # mapping to its own row; flattening all B rows would incorrectly feed
        # a [B*W] mapping into a [W]-wide session pool.
        sessions = getattr(session, "_batch_sessions", (session,))
        for row, item in enumerate(sessions):
            _reorder_kv_same_step(item, local_idx[row], local_prefix[row])
        # The applied mapping is NOT materialized here: reading
        # ``local_idx[0].tolist()`` is a device→host sync that stalls the
        # pipeline every step (~30ms on NPU, profiling bw128_1k_t3).  The
        # return channel (``_maybe_run_worker_beam_decision``) already copies
        # ``out_token_index`` to host for the frontend, and its request-0 row
        # equals this request-local mapping (global offset 0), so the record
        # is derived there from that copy — no second D2H.
    return decision


def _run_npu_decision(
    session: Any,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    is_last_step: bool,
    result_width: int,
) -> dict[str, torch.Tensor] | None:
    """Run the fused xllm decision operators (ACS decode_select equivalent)."""
    from vllm_gr.ops.npu import beam_ops


    bufs = session.context.decision
    width = bufs.worker_decode_steps
    beam_width = session.beam_width
    batch = top_tokens.shape[0] // beam_width
    step = width  # worker-side token count == next sequence slot to write

    log_probs = bufs.beam_scores[:batch, :beam_width].reshape(batch * beam_width, 1)
    seq = bufs.sequence[:batch, :beam_width, :].contiguous()

    prefix: torch.Tensor | None = None
    if is_last_step and result_width % _FINAL_SELECT_ALIGNMENT == 0:
        out_ids, out_idx, out_lp, out_seq = beam_ops.beam_search_rec_final_select(
            log_probs,
            top_tokens,
            top_probs,
            seq,
            step,
            result_width,
        )
    else:
        out_ids, out_idx, out_lp, prefix, out_seq = beam_ops.beam_search_group(
            log_probs,
            top_tokens,
            top_probs,
            seq,
            step,
            beam_width,
        )
        result_width = beam_width

    out_ids = out_ids.reshape(batch, result_width)
    out_idx = out_idx.reshape(batch, result_width)
    out_lp = out_lp.reshape(batch, result_width)
    bufs.worker_decode_steps = step + 1
    return _finalize_decision(
        session, bufs, out_ids, out_idx, out_lp, out_seq, prefix, batch, beam_width, result_width
    )


def _run_cuda_decision_out(
    session: Any,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None,
    candidate_counts: torch.Tensor | None,
    workspace: BeamPostprocessWorkspace,
    is_last_step: bool,
    result_width: int,
) -> dict[str, torch.Tensor] | None:
    """Run the fixed-address CUDA Group/KV/Final pipeline for one step."""

    from vllm_gr.ops.cuda.beam_selection import (
        _beam_search_group_cuda_worker_out_unchecked,
        beam_result_pack_cuda_worker_out,
        beam_search_group_cuda_out,
        beam_search_rec_final_select_grouped_cuda_out,
    )

    bufs = session.context.decision
    beam_width = session.beam_width
    if beam_width != workspace.key.beam_width or top_tokens.shape[1] != workspace.key.top_k:
        return None
    if top_tokens.shape[0] % beam_width != 0:
        return None
    batch = top_tokens.shape[0] // beam_width
    if not 1 <= batch <= workspace.max_batch:
        return None
    step = bufs.worker_decode_steps

    log_probs = bufs.beam_scores[:batch, :beam_width]
    sequence = bufs.sequence[:batch, :beam_width, :]
    if not log_probs.is_contiguous() or not sequence.is_contiguous():
        return None
    log_probs_rows = log_probs.view(batch * beam_width, 1)

    selected_ids = workspace.selected_token_ids[:batch]
    selected_parents = workspace.selected_parent_ids[:batch]
    selected_scores = workspace.selected_scores[:batch]
    selected_prefix = workspace.selected_parent_count_prefix[:batch]
    selected_sequence = workspace.next_sequence[:batch]
    selected_flat = workspace.selected_flat_indices[:batch]
    scratch = workspace.selection_scratch
    deferred_status = bool(candidate_counts is not None)
    status = workspace.prefill_staging[:batch]
    payload = workspace.prefill_staging[workspace.max_batch :]
    if deferred_status:
        # Constraint TopK rows are score-sorted, enabling the physical W-way
        # merge + finalize implementation (two launches total).
        _beam_search_group_cuda_worker_out_unchecked(
            log_probs_rows,
            top_tokens,
            top_probs,
            sequence,
            step,
            selected_ids,
            selected_parents,
            selected_scores,
            selected_sequence,
            selected_flat,
            scratch.local_scores,
            scratch.local_flat_indices,
            candidate_valid_mask,
            candidate_counts,
            status,
            payload,
        )
    else:
        # Non-constraint and externally supplied sampler rows do not promise
        # row ordering.  Preserve the checked hierarchical implementation.
        beam_search_group_cuda_out(
            log_probs_rows,
            top_tokens,
            top_probs,
            sequence,
            step,
            beam_width,
            selected_ids,
            selected_parents,
            selected_scores,
            selected_prefix,
            selected_sequence,
            scratch.local_scores,
            scratch.local_flat_indices,
            scratch.merge_scores,
            scratch.merge_flat_indices,
            candidate_valid_mask,
            selected_flat,
        )
    bufs.worker_decode_steps = step + 1

    if is_last_step:
        if not 1 <= result_width <= beam_width:
            return None
        result_rows = batch * result_width
        final_ids = workspace.final_token_ids.view(-1)[:result_rows].view(batch, result_width)
        final_parents = workspace.final_parent_ids.view(-1)[:result_rows].view(batch, result_width)
        final_scores = workspace.final_scores.view(-1)[:result_rows].view(batch, result_width)
        final_sequence = workspace.final_sequence.view(
            workspace.max_batch * beam_width,
            workspace.max_decode_steps,
        )[:result_rows].view(batch, result_width, workspace.max_decode_steps)
        final_positions = workspace.selected_flat_indices.view(-1)[:result_rows].view(
            batch, result_width
        )
        beam_search_rec_final_select_grouped_cuda_out(
            selected_ids,
            selected_parents,
            selected_scores,
            selected_sequence,
            result_width,
            final_ids,
            final_parents,
            final_scores,
            final_sequence,
            final_positions,
        )
        if deferred_status:
            beam_result_pack_cuda_worker_out(
                final_ids,
                final_parents,
                final_scores,
                status,
                payload,
                beam_width=beam_width,
            )
        bufs.sequence[:batch, :result_width].copy_(final_sequence)
        bufs.beam_scores[:batch, :result_width].copy_(final_scores)
        return {
            "out_token_ids": final_ids,
            "out_token_index": final_parents,
            "out_log_probs": final_scores,
            "out_sequence": final_sequence,
            "_gr_deferred_cuda_status": deferred_status,
            "_gr_result_batch": batch,
            "_gr_result_width": result_width,
        }

    bufs.sequence[:batch, :beam_width].copy_(selected_sequence)
    bufs.beam_scores[:batch, :beam_width].copy_(selected_scores)
    # The current CUDA Beam context owns one active pool slot, so request row
    # zero's global parent ids are already request-local.  Keep them on device
    # and feed the existing duplicate-safe gather/scatter reorder directly.
    _reorder_kv_same_step(session, selected_parents[0], selected_prefix[0])
    return {
        "out_token_ids": selected_ids,
        "out_token_index": selected_parents,
        "out_log_probs": selected_scores,
        "out_sequence": selected_sequence,
        "_gr_deferred_cuda_status": deferred_status,
        "_gr_result_batch": batch,
        "_gr_result_width": beam_width,
    }


def _run_torch_decision(
    session: Any,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    is_last_step: bool,
    result_width: int,
) -> dict[str, torch.Tensor] | None:
    """PyTorch reference decision (GPU/host parity oracle for the NPU path)."""
    from vllm_gr.ops.onerec_decode import (
        beam_search_group,
        beam_search_rec_final_select,
    )

    bufs = session.context.decision
    width = bufs.worker_decode_steps
    beam_width = session.beam_width
    batch = top_tokens.shape[0] // beam_width
    step = width

    log_probs = bufs.beam_scores[:batch, :beam_width].reshape(batch * beam_width, 1)
    seq = bufs.sequence[:batch, :beam_width, :].contiguous()

    prefix: torch.Tensor | None = None
    if is_last_step and (
        top_tokens.device.type == "cuda" or result_width % _FINAL_SELECT_ALIGNMENT == 0
    ):
        out_ids, out_idx, out_lp, out_seq = beam_search_rec_final_select(
            log_probs, top_tokens, top_probs, seq, step, result_width
        )
    else:
        out_ids, out_idx, out_lp, prefix, out_seq = beam_search_group(
            log_probs, top_tokens, top_probs, seq, step, beam_width
        )
        result_width = beam_width

    out_ids = out_ids.reshape(batch, result_width)
    out_idx = out_idx.reshape(batch, result_width)
    out_lp = out_lp.reshape(batch, result_width)
    bufs.worker_decode_steps = step + 1
    return _finalize_decision(
        session, bufs, out_ids, out_idx, out_lp, out_seq, prefix, batch, beam_width, result_width
    )


def run_beam_decision(
    runner: Any,
    session: Any,
    sampler_output: Any,
    is_last_step: bool,
    result_width: int | None = None,
) -> dict[str, torch.Tensor] | None:
    """Run the worker-side beam decision for one decode step.

    Args:
        runner: the model runner (must expose ``_current_beam_info``).
        session: the active BeamSearchSession.
        sampler_output: the sampler output whose logprob tensors carry the
            per-beam top-k candidates (post constraint override).
        is_last_step: whether this is the final decode step.
        result_width: output width for the final step; defaults to beam width.

    Returns a dict with ``out_token_ids`` / ``out_token_index`` /
    ``out_log_probs`` / ``out_sequence`` (device tensors, ``[B, W_eff]`` /
    ``[B, W_eff]`` / ``[B, W_eff]`` / ``[B, W_eff, steps]``), plus a copy of
    the cumulative ``beam_scores`` (``probs`` key), or ``None`` when the step
    cannot be decided worker-side (caller keeps the frontend path).
    """
    width = session.beam_width
    run_device = getattr(runner, "device", None)
    device_type = str(run_device.type) if run_device is not None else ""
    if device_type == "cuda" and not _cuda_worker_decision_width_supported(width):
        return None
    workspace = get_beam_postprocess_workspace(runner, width) if device_type == "cuda" else None
    candidates = (
        _select_candidates_into(sampler_output, width, workspace)
        if workspace is not None
        else _select_candidates(sampler_output, width)
    )
    if candidates is None:
        return None

    top_tokens, top_probs = candidates[:2]
    candidate_valid_mask = candidates[2] if len(candidates) == 3 else None
    candidate_counts = getattr(sampler_output, "_gr_constraint_valid_counts", None)
    if not bool(getattr(sampler_output, "_gr_deferred_decode_status", False)):
        candidate_counts = None
    n_rows = top_tokens.shape[0]
    if n_rows == 0 or n_rows % width != 0:
        return None
    # Keep the fused mapping well-defined: require one survivor per beam.
    beam_info = getattr(runner, "_current_beam_info", None)
    for mdata in (beam_info.get("beam_data") or {}).values():
        if mdata.get("is_beam_decode", False):
            parent = mdata.get("parent_beam_ids")
            if parent is not None and len(parent) != width:
                return None

    result_width = result_width or width
    if result_width <= 0 or result_width < width:
        result_width = width

    bufs = session.context.decision
    if bufs.worker_decode_steps == 0:
        # A decode candidate matrix cannot reconstruct the scores of its
        # already-selected parents. Fall back to frontend selection instead
        # of silently dropping prefill scores and counting this local score
        # twice. Normal engine paths seed from request metadata at creation.
        return None

    if device_type == "cuda" and workspace is not None:
        return _run_cuda_decision_out(
            session,
            top_tokens,
            top_probs,
            candidate_valid_mask,
            candidate_counts,
            workspace,
            is_last_step,
            result_width,
        )

    try:
        from vllm_gr.ops.npu import beam_ops

        # The fused xllm operators execute on NPU tensors; on CPU/CUDA hosts
        # (unit-test / GPU-debug environments) always take the PyTorch
        # reference so the decision logic stays verifiable off-device.
        if beam_ops.is_available() and device_type.startswith("npu"):
            decision = _run_npu_decision(session, top_tokens, top_probs, is_last_step, result_width)
            if decision is not None:
                decision["probs"] = session.context.decision.beam_scores[:1, :width].clone()
                return decision
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("vllm-gr worker beam decision (NPU) failed: %s; fallback", exc)

    decision = _run_torch_decision(session, top_tokens, top_probs, is_last_step, result_width)
    if decision is not None:
        decision["probs"] = session.context.decision.beam_scores[:1, :width].clone()
    return decision


def _run_batch_worker_decision(runner: Any, batch: Any, sampler_output: Any) -> None:
    """Run one device decision and publish request-local cohort results."""
    first = batch.sessions[0]
    first._batch_sessions = batch.sessions
    beam_data = runner._current_beam_info["beam_data"]
    result_widths = {
        int(beam_data[session.session_id].get("result_width") or session.beam_width)
        for session in batch.sessions
    }
    if len(result_widths) != 1:
        del first._batch_sessions
        raise RuntimeError("multi-request beam decision requires one common result width")
    try:
        with torch.autograd.profiler.record_function("beam_decision_device"):
            decision = run_beam_decision(
                runner,
                first,
                sampler_output,
                is_last_step=first._state.is_last_step,
                result_width=next(iter(result_widths)),
            )
    finally:
        del first._batch_sessions
    if decision is None:
        raise RuntimeError("batched worker decision unavailable; refusing partial output")

    request_ids = [session.session_id for session in batch.sessions]
    resident_batch = all(
        beam_data[request_id].get("resident", False)
        and beam_data[request_id].get("single_request", False)
        for request_id in request_ids
    )
    runner._beam_decision_tensors = {
        key: value
        for key, value in decision.items()
        if key in ("out_token_ids", "out_log_probs", "out_token_index")
        and torch.is_tensor(value)
    }

    if resident_batch and not first._state.is_last_step:
        # The next inputs and KV mapping are already device resident. Publish
        # only per-request routing stubs and perform no D2H on intermediate steps.
        results = {}
        for session in batch.sessions:
            session._state.kv_reorder = []
            results[session.session_id] = {
                "stage": "decode",
                "session_id": session.session_id,
                "is_last_step": False,
            }
        runner._beam_decision_result = {"by_request": results}
        return

    from vllm_gr.v1.beam.execution_batch import split_batch_decision

    device_type = str(getattr(getattr(runner, "device", None), "type", ""))
    if resident_batch and first._state.is_last_step and device_type == "npu":
        compact = _encode_npu_beam_result(
            decision,
            is_last_step=True,
            workspace=get_beam_npu_transport_workspace(runner, first.beam_width),
        )
        if compact is not None:
            global_payload, _ = compact

            def _materialize_batch(
                _payload=global_payload,
                _request_ids=tuple(request_ids),
                _width=first.beam_width,
            ) -> dict[str, Any]:
                full = (
                    _payload._materialize()
                    if isinstance(_payload, _LazyBeamPayload)
                    else dict(_payload)
                )
                unpacked = unpack_beam_result_buffer(full)
                if unpacked is None:
                    raise RuntimeError("invalid compact batched beam result")
                token_ids, scores, parents = unpacked
                rows = len(_request_ids)
                host = {
                    "out_token_ids": [
                        token_ids[row * _width : (row + 1) * _width]
                        for row in range(rows)
                    ],
                    "out_token_index": [
                        parents[row * _width : (row + 1) * _width]
                        for row in range(rows)
                    ],
                    "out_log_probs": [
                        scores[row * _width : (row + 1) * _width]
                        for row in range(rows)
                    ],
                    "out_sequence": full["out_sequence"],
                }
                return {
                    "by_request": split_batch_decision(
                        _request_ids, _width, host, True
                    )
                }

            runner._beam_decision_result = _LazyBeamPayload(
                {"session_id": None}, _materialize_batch
            )
            return

    keys = ("out_token_ids", "out_token_index", "out_log_probs", "out_sequence")
    host = {
        key: decision[key].detach().to("cpu").tolist()
        for key in keys
        if torch.is_tensor(decision.get(key))
    }
    results = split_batch_decision(
        request_ids,
        first.beam_width,
        host,
        first._state.is_last_step,
    )
    for session in batch.sessions:
        if not session._state.is_last_step:
            session._state.kv_reorder = results[session.session_id]["out_token_index"][0]
    runner._beam_decision_result = {"by_request": results}


def maybe_run_worker_beam_decision(runner: Any, sampler_output: Any) -> None:
    """Run and publish the active accelerator's worker-side beam decision."""
    fast_prefill = _run_cuda_prefill_postprocess(runner, sampler_output)
    if fast_prefill is not None:
        payload, compatibility_views = fast_prefill
        runner._beam_decision_tensors = compatibility_views
        runner._beam_decision_result = payload
        runner._beam_result_host_ready = True
        return

    prefill_decision = _run_cuda_prefill_decision(runner, sampler_output)
    if prefill_decision is None:
        # Unconstrained Stage 1 beam prefill (NPU or CUDA): the ordinary
        # sampler's flat top-K carries the step-0 beam candidates.
        prefill_decision = _run_unconstrained_prefill_decision(runner, sampler_output)
    if prefill_decision is not None:
        # Single-Request prefill stub: the device buffers were seeded inside
        # ``_run_unconstrained_prefill_decision`` and the tensor-free payload
        # has no host-side consumer (no engine-driven chain, frontend drops
        # non-terminal outputs).  Skip the compact-encode + D2H sync entirely
        # — publish nothing so ``_attach_beam_search_result`` attaches None.
        if not torch.is_tensor(prefill_decision.get("out_token_ids")):
            runner._beam_decision_tensors = None
            runner._beam_decision_result = None
            return
        runner._beam_decision_tensors = prefill_decision
        width = int(prefill_decision["out_token_ids"].shape[1])
        device_type = str(getattr(getattr(runner, "device", None), "type", ""))
        # Compact transport: one D2H sync instead of three per-tensor
        # .tolist() calls.  The CUDA workspace fast path only exists on
        # CUDA; on NPU (or without a prewarmed workspace) the allocating
        # branch of _encode_cuda_beam_result packs the same
        # ``beam_result_i32_f32_v1`` wire format on any device.
        compact = _encode_cuda_beam_result(
            prefill_decision,
            is_last_step=False,
            workspace=get_beam_postprocess_workspace(runner, width)
            if device_type == "cuda"
            else None,
        )
        runner._beam_decision_result = compact[0] if compact is not None else None
        if runner._beam_decision_result is not None:
            # _encode_cuda_beam_result keeps only the packed arrays; re-attach
            # the ownership markers so downstream consumers stay keyed.
            runner._beam_decision_result["stage"] = "prefill"
            runner._beam_decision_result["session_id"] = prefill_decision["session_id"]
        return

    if not worker_decision_enabled(runner):
        runner._beam_decision_result = None
        return
    beam_info = getattr(runner, "_current_beam_info", None)
    session = beam_info.get("beam_session") if beam_info else None
    if session is None:
        runner._beam_decision_result = None
        return

    if hasattr(session, "sessions"):
        _run_batch_worker_decision(runner, session, sampler_output)
        return

    state = session._state
    result_width = None
    for mdata in (beam_info.get("beam_data") or {}).values():
        if mdata.get("is_beam_decode", False):
            result_width = mdata.get("result_width")
            break

    decision = run_beam_decision(
        runner,
        session,
        sampler_output,
        is_last_step=state.is_last_step,
        result_width=result_width,
    )
    if decision is not None:
        # Keep the device winners alive until bookkeeping has replaced the
        # redundant W-by-W sampler payload with its O(W) compatibility row.
        device_type = str(getattr(getattr(runner, "device", None), "type", ""))
        if (
            not state.is_last_step
            and _resident_beam_decode_step(beam_info)
            and not bool(decision.get("_gr_deferred_cuda_status", False))
        ):
            # Stage 1 resident intermediate step: NOTHING consumes the
            # decision payload on the host.  The EngineCore resident early
            # return only reads ``is_last_step`` (False) to set the chained
            # marker, the frontend drain loop drops non-terminal outputs,
            # the next step's inputs come from the device-resident sequence
            # buffer, and the KV reorder was already applied on-device by
            # ``_finalize_decision`` (``parent_beam_ids`` is None for
            # resident rows, so the reorder record's VALUE is never
            # compared — only its non-None-ness gates the next step's
            # ``decode_step`` advance in ``_get_or_create_beam_session``).
            # Publishing a host-only stub therefore preserves every
            # consumer contract while removing the per-step pinned D2H
            # ``copy_event.synchronize()`` that stalls the async-scheduling
            # pipeline every decode step.  The terminal step keeps the
            # full compact encode (one sync per session, not per step).
            # CUDA deferred-status results already staged a host payload
            # without a worker-side sync; they fall through to the normal
            # encode, which only packs the already-staged host buffer.
            runner._beam_decision_tensors = decision
            # Non-empty sentinel (never compared, only truth-tested as a
            # ``is not None`` record) so the next prepare consumes it and
            # skips the duplicate decode_step advance.
            state.kv_reorder = []
            runner._beam_decision_result = {
                "stage": "decode",
                "is_last_step": False,
            }
            decode_req_id_stub = next(
                (
                    req_id
                    for req_id, mdata in (beam_info.get("beam_data") or {}).items()
                    if mdata.get("is_beam_decode", False)
                ),
                None,
            )
            if decode_req_id_stub is not None:
                runner._beam_decision_result["session_id"] = decode_req_id_stub
            return
        workspace = (
            get_beam_postprocess_workspace(runner, session.beam_width)
            if device_type == "cuda"
            else None
        )
        npu_transport = (
            get_beam_npu_transport_workspace(runner, session.beam_width)
            if device_type == "npu"
            else None
        )
        deferred = (
            _encode_cuda_deferred_beam_result(
                decision,
                workspace,
                is_last_step=state.is_last_step,
            )
            if workspace is not None
            else None
        )
        if deferred is not None:
            payload, compatibility, parent_host = deferred
            runner._beam_decision_tensors = compatibility
            runner._beam_result_host_ready = True
            if not state.is_last_step:
                state.kv_reorder = parent_host[: session.beam_width]
            runner._beam_decision_result = payload
            return

        runner._beam_decision_tensors = decision
        # NPU: collapse the per-tensor D2H syncs into one pinned copy + event
        # sync via the compact buffer (the CUDA transport optimization, now
        # extended to NPU).  Both step kinds take it: non-terminal steps pack
        # ids/parents/scores; the terminal step additionally packs
        # ``out_sequence`` into the same transfer and derives the nested
        # ``out_sequence`` / ``out_log_probs`` lists the serving terminal
        # short-circuit reads from the synced host view.
        compact = (
            _encode_cuda_beam_result(
                decision,
                is_last_step=state.is_last_step,
                workspace=workspace,
            )
            if device_type == "cuda"
            else (
                _encode_npu_beam_result(
                    decision,
                    is_last_step=state.is_last_step,
                    workspace=npu_transport,
                )
                if device_type == "npu"
                else None
            )
        )
        if compact is not None:
            payload, parent_host = compact
            if not state.is_last_step:
                state.kv_reorder = parent_host[: session.beam_width]
            # Terminal NPU compact path returns a ``_LazyBeamPayload`` with
            # ``parent_host == []``: the terminal step has no successor, so the
            # KV-reorder record is unused, and deferring its materialization
            # keeps the sync off the busy-loop thread (see _LazyBeamPayload).
            decision = payload
        else:
            # Legacy plain-container ABI: NPU without a prewarmed transport
            # workspace, or a payload whose shape/dtype bypassed the compact
            # fast path.  One per-key ``.to("cpu").tolist()`` sync remains —
            # correct, but ~30ms per tensor on NPU.
            keys = (
                ("out_token_ids", "out_token_index", "out_log_probs")
                if not state.is_last_step
                else ("out_token_ids", "out_token_index", "out_log_probs", "out_sequence")
            )
            payload = {}
            for key in keys:
                value = decision.get(key)
                if value is None:
                    continue
                from torch.autograd.profiler import record_function
                with record_function("## to cpu ##"):
                    payload[key] = (
                        value.detach().to("cpu").tolist() if torch.is_tensor(value) else value
                    )
            payload["is_last_step"] = bool(state.is_last_step)
            if not state.is_last_step:
                out_idx_host = payload.get("out_token_index")
                if out_idx_host:
                    state.kv_reorder = [int(parent) for parent in out_idx_host[0]]
            decision = payload
    if decision is not None:
        # Ownership markers (review P5): mixed batches must never let session
        # B's decision chain onto session A's step update downstream.
        decision["stage"] = "decode"
        decode_req_id = next(
            (
                req_id
                for req_id, mdata in (beam_info.get("beam_data") or {}).items()
                if mdata.get("is_beam_decode", False)
            ),
            None,
        )
        if decode_req_id is not None:
            decision["session_id"] = decode_req_id
    runner._beam_decision_result = decision


__all__ = [
    "BeamNpuTransportWorkspace",
    "BeamPostprocessWorkspace",
    "BeamPostprocessWorkspaceKey",
    "BeamPostprocessWorkspacePool",
    "enqueue_cuda_prefill_postprocess",
    "enqueue_cuda_prefill_workspace_result",
    "flatten_decision_value",
    "get_beam_npu_transport_workspace",
    "get_beam_postprocess_workspace",
    "init_beam_decision_from_parents",
    "init_beam_decision_state",
    "maybe_run_worker_beam_decision",
    "run_beam_decision",
    "unpack_beam_result_buffer",
    "worker_decision_enabled",
]
