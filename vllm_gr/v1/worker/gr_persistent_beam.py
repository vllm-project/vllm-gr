# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inactive GPU ModelRunner V1 persistent-Beam preparation backend.

The backend is intentionally not registered in ``execute_model``.  Tests and
the future A3 adapter call it directly.  Beam data remains in Worker-owned
device tensors; acknowledgements carry control state only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import torch

from vllm_gr.constraints.backend import ConstraintBackend, ConstraintMetadata
from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.beam_decision import (
    BeamPostprocessWorkspace,
    BeamPostprocessWorkspaceKey,
    BeamPostprocessWorkspacePool,
)
from vllm_gr.v1.beam.persistent_contracts import (
    GRBeamSessionSpec,
    GRStageAck,
    GRStageKind,
    GRWorkerStageSpec,
)
from vllm_gr.v1.worker.gr_model_runner_state import (
    GRModelRunnerState,
    GRModelRunnerStateRegistry,
    GRModelRunnerStatus,
)


@dataclass(frozen=True, slots=True)
class GRPrefillDeviceInputs:
    """Constrained Prefill candidates already resident on the Worker device."""

    candidate_token_ids: torch.Tensor
    candidate_logprobs: torch.Tensor
    valid_mask: torch.Tensor
    prefix_len: int


@dataclass(frozen=True, slots=True)
class GRDecodeDeviceInputs:
    """Fixed ModelRunner input buffer populated from persistent Beam state."""

    input_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class GRFinalDeviceView:
    """Worker-local terminal views; this object must never cross IPC."""

    sampled_token_ids: torch.Tensor
    cumulative_logprobs: torch.Tensor
    parent_beam_indices: torch.Tensor
    generated_token_ids: torch.Tensor
    valid_mask: torch.Tensor

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("GRFinalDeviceView is Worker-local and cannot be serialized")

    # Architecture-document compatibility aliases.  New code should use the
    # vLLM-aligned field names above.
    @property
    def token_ids(self) -> torch.Tensor:
        return self.sampled_token_ids

    @property
    def scores(self) -> torch.Tensor:
        return self.cumulative_logprobs

    @property
    def parent_ids(self) -> torch.Tensor:
        return self.parent_beam_indices

    @property
    def sequences(self) -> torch.Tensor:
        return self.generated_token_ids


@runtime_checkable
class GRPersistentBeamBackend(Protocol):
    def create_session(self, spec: GRBeamSessionSpec) -> None: ...

    def bootstrap_from_prefill(
        self,
        stage: GRWorkerStageSpec,
        inputs: GRPrefillDeviceInputs,
    ) -> GRStageAck: ...

    def prepare_decode_inputs(
        self,
        stage: GRWorkerStageSpec,
        inputs: GRDecodeDeviceInputs,
    ) -> None: ...

    def advance_beam(
        self,
        stage: GRWorkerStageSpec,
        logits: torch.Tensor,
    ) -> GRStageAck: ...

    def get_final_device_view(self, session_id: str) -> GRFinalDeviceView: ...

    def abort_session(self, session_id: str) -> None: ...

    def release_session(self, session_id: str) -> None: ...


@dataclass(slots=True)
class GRPersistentBeamMetrics:
    state_created: int = 0
    state_released: int = 0
    beam_decisions: int = 0
    kv_reorders: int = 0
    persistent_state_data_ptr_changed: int = 0


class GPUModelRunnerPersistentBeamBackend:
    """Direct-call implementation of the persistent Beam data path."""

    def __init__(
        self,
        *,
        context: BeamAttentionContext,
        constraint_backends: Mapping[str, ConstraintBackend],
        workspace_pool: BeamPostprocessWorkspacePool,
    ) -> None:
        if context.device.type not in ("cpu", "cuda"):
            raise ValueError(
                "GPU persistent Beam backend only supports CPU test harnesses and CUDA"
            )
        self.context = context
        self.constraint_backends = dict(constraint_backends)
        self.workspace_pool = workspace_pool
        self.registry = GRModelRunnerStateRegistry(context)
        self.metrics = GRPersistentBeamMetrics()

    def _workspace(self, spec: GRBeamSessionSpec) -> BeamPostprocessWorkspace:
        key = BeamPostprocessWorkspaceKey(
            beam_width=spec.beam_width,
            top_k=spec.beam_width,
            logits_dtype=self.context.dtype,
        )
        workspace = self.workspace_pool.require(key)
        if workspace.max_batch < self.context.max_batch:
            raise ValueError("Beam postprocess workspace max_batch must cover every state slot")
        if workspace.max_decode_steps < spec.max_tokens:
            raise ValueError(
                f"workspace token capacity {workspace.max_decode_steps} is smaller than "
                f"max_tokens={spec.max_tokens}"
            )
        if workspace.device != self.context.device:
            raise ValueError("Beam context and postprocess workspace must share one device")
        if self.context.device.type == "cuda" and spec.beam_width != self.context.max_beams:
            raise ValueError(
                "CUDA persistent Beam requires the fixed session beam_width to equal "
                "the prewarmed Beam context width; dynamic Beam width is out of scope"
            )
        return workspace

    def create_session(self, spec: GRBeamSessionSpec) -> None:
        workspace = self._workspace(spec)
        state = self.registry.create(spec, workspace=workspace)
        workspace.temperatures[state.slot_id, : spec.beam_width].fill_(float(spec.temperature))
        self.metrics.state_created += 1

    @staticmethod
    def _validate_tensor(
        name: str,
        tensor: torch.Tensor,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must use {dtype}, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def _fail(self, state: GRModelRunnerState) -> None:
        if state.status not in (
            GRModelRunnerStatus.FINAL_READY,
            GRModelRunnerStatus.RELEASED,
            GRModelRunnerStatus.FAILED,
            GRModelRunnerStatus.ABORTED,
        ):
            state.mark_failed()
        self.registry.record_terminal(state)

    def _validate_stable_addresses(self, state: GRModelRunnerState) -> None:
        state_changed = (
            state.buffers.data_ptrs() + state.scratch.data_ptrs() != state.initial_data_ptrs
        )
        workspace_changed = (
            tuple(tensor.data_ptr() for tensor in state.workspace.device_tensors())
            != state.initial_workspace_data_ptrs
        )
        if state_changed or workspace_changed:
            self.metrics.persistent_state_data_ptr_changed += 1
            raise RuntimeError(
                f"persistent Beam state address changed for session {state.session_id!r}"
            )

    def bootstrap_from_prefill(
        self,
        stage: GRWorkerStageSpec,
        inputs: GRPrefillDeviceInputs,
    ) -> GRStageAck:
        state = self.registry.get(stage.session_id)
        try:
            state.validate_stage(stage)
            self._validate_stable_addresses(state)
            if stage.stage_kind is not GRStageKind.PREFILL:
                raise ValueError("bootstrap_from_prefill requires a Prefill stage")
            if state.status is not GRModelRunnerStatus.NEW:
                raise RuntimeError(
                    f"session {state.session_id!r} cannot bootstrap while {state.status.value}"
                )
            state.status = GRModelRunnerStatus.PREFILL_RUNNING
            width = state.spec.beam_width
            shape = (1, width)
            self._validate_tensor(
                "candidate_token_ids",
                inputs.candidate_token_ids,
                shape=shape,
                dtype=torch.int32,
                device=self.context.device,
            )
            self._validate_tensor(
                "candidate_logprobs",
                inputs.candidate_logprobs,
                shape=shape,
                dtype=torch.float32,
                device=self.context.device,
            )
            self._validate_tensor(
                "valid_mask",
                inputs.valid_mask,
                shape=shape,
                dtype=torch.bool,
                device=self.context.device,
            )
            if isinstance(inputs.prefix_len, bool) or not isinstance(inputs.prefix_len, int):
                raise TypeError("prefix_len must be an int")
            if inputs.prefix_len < 0:
                raise ValueError("prefix_len must be non-negative")

            buffers = state.buffers
            buffers.sampled_token_ids.copy_(inputs.candidate_token_ids)
            buffers.generated_token_ids[:, :, 0].copy_(inputs.candidate_token_ids)
            buffers.cumulative_logprobs.copy_(inputs.candidate_logprobs)
            buffers.cumulative_logprobs.masked_fill_(~inputs.valid_mask, float("-inf"))
            buffers.valid_mask.copy_(inputs.valid_mask)
            buffers.parent_beam_indices[:, :, 0].zero_()
            torch.arange(
                1,
                width + 1,
                dtype=torch.int32,
                device=self.context.device,
                out=buffers.parent_count_prefix.reshape(-1),
            )
            state.beam_session._state.prefix_len = inputs.prefix_len
            state.next_expected_stage = 1
            state.state_version += 1
            terminal = stage.is_last_stage
            state.status = (
                GRModelRunnerStatus.FINAL_READY
                if terminal
                else GRModelRunnerStatus.PREFILL_BOOTSTRAPPED
            )
            if terminal:
                self.registry.record_terminal(state)
            return GRStageAck(state.session_id, stage.stage_index, True, terminal)
        except Exception:
            self._fail(state)
            raise

    def prepare_decode_inputs(
        self,
        stage: GRWorkerStageSpec,
        inputs: GRDecodeDeviceInputs,
    ) -> None:
        state = self.registry.get(stage.session_id)
        try:
            state.validate_stage(stage)
            self._validate_stable_addresses(state)
            if stage.stage_kind is not GRStageKind.DECODE:
                raise ValueError("prepare_decode_inputs requires a Decode stage")
            if state.status not in (
                GRModelRunnerStatus.PREFILL_BOOTSTRAPPED,
                GRModelRunnerStatus.DECODE_RUNNING,
            ):
                raise RuntimeError(
                    f"session {state.session_id!r} cannot prepare Decode while {state.status.value}"
                )
            if state.pending_stage_index is not None:
                raise RuntimeError(
                    f"session {state.session_id!r} already prepared stage "
                    f"{state.pending_stage_index}"
                )
            width = state.spec.beam_width
            if tuple(inputs.input_ids.shape) not in ((width,), (width, 1)):
                raise ValueError(
                    f"input_ids must have shape [{width}] or [{width}, 1], got "
                    f"{tuple(inputs.input_ids.shape)}"
                )
            if inputs.input_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must use int32 or int64")
            if inputs.input_ids.device != self.context.device:
                raise ValueError("input_ids must share the persistent state device")
            if not inputs.input_ids.is_contiguous():
                raise ValueError("input_ids must be contiguous")
            input_data_ptr = inputs.input_ids.data_ptr()
            if state.decode_input_data_ptr is None:
                state.decode_input_data_ptr = input_data_ptr
            elif input_data_ptr != state.decode_input_data_ptr:
                raise RuntimeError(
                    "persistent Beam decode input buffer address changed during the session"
                )
            inputs.input_ids.reshape(-1).copy_(state.buffers.sampled_token_ids.reshape(-1))
            self.context.decode_step_buf.fill_(state.beam_session.decode_step)

            workspace = state.workspace
            prefix = workspace.prefix_token_ids[state.slot_id, :width]
            prefix.zero_()
            prefix_steps = min(stage.stage_index, prefix.shape[1])
            prefix[:, :prefix_steps].copy_(state.buffers.generated_token_ids[0, :, :prefix_steps])
            state.pending_stage_index = stage.stage_index
            state.status = GRModelRunnerStatus.DECODE_RUNNING
            state.state_version += 1
        except Exception:
            self._fail(state)
            raise

    def _constraint_backend(self, state: GRModelRunnerState) -> ConstraintBackend:
        handle = state.spec.constraint_handle
        if handle is None:
            raise RuntimeError("persistent Beam Decode requires a constraint_handle")
        try:
            return self.constraint_backends[handle]
        except KeyError as exc:
            raise KeyError(f"unknown constraint_handle {handle!r}") from exc

    def _select_candidates(
        self,
        state: GRModelRunnerState,
        stage: GRWorkerStageSpec,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = state.spec.beam_width
        workspace = state.workspace
        slot = state.slot_id
        metadata = ConstraintMetadata(
            prefix_token_ids=workspace.prefix_token_ids[slot, :width],
            temperatures=workspace.temperatures[slot, :width],
            current_step=stage.stage_index,
            top_k=width,
        )
        backend = self._constraint_backend(state)
        token_ids = workspace.topk_token_ids[slot].view(width, width)
        logprobs = workspace.topk_scores[slot].view(width, width)
        valid_mask = workspace.topk_valid_mask[slot].view(width, width)
        valid_counts = workspace.topk_valid_counts[slot, :width]
        select_out = getattr(backend, "select_out", None)
        if callable(select_out):
            select_out(
                logits,
                metadata,
                out_token_ids=token_ids,
                out_logprobs=logprobs,
                out_valid_mask=valid_mask,
                out_valid_counts=valid_counts,
                out_wire_logprobs=None,
            )
        else:
            output = backend.select(logits, metadata)
            token_ids.copy_(output.token_ids)
            logprobs.copy_(output.logprobs)
            valid_mask.copy_(output.valid_mask)
        # The score accumulator is seeded with the Prefill (first SID token)
        # scores by bootstrap_from_prefill and must be preserved here. The
        # legacy frontend accumulates cum_logprob as 0 + logprob(s_a) +
        # logprob(s_b) + ... ; overwriting it with the first Decode candidate
        # column would drop the Prefill score and change the final beam rank
        # order. BeamSearchGroup adds each candidate score to that accumulator.
        # Invalid candidates are masked on device; validity remains separate.
        logprobs.masked_fill_(~valid_mask, float("-inf"))
        return token_ids, logprobs, valid_mask

    def _advance_cuda(
        self,
        state: GRModelRunnerState,
        stage: GRWorkerStageSpec,
        top_tokens: torch.Tensor,
        top_logprobs: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm_gr.ops.cuda.beam_selection import (
            _beam_search_group_cuda_worker_out_unchecked,
            beam_search_rec_final_select_grouped_cuda_out,
        )

        workspace = state.workspace
        slot = state.slot_id
        width = state.spec.beam_width
        sequence = state.buffers.generated_token_ids
        cumulative = state.buffers.cumulative_logprobs.reshape(width, 1)
        selected_ids = workspace.selected_token_ids[slot : slot + 1, :width]
        selected_parents = workspace.selected_parent_ids[slot : slot + 1, :width]
        selected_scores = workspace.selected_scores[slot : slot + 1, :width]
        selected_flat = workspace.selected_flat_indices[slot : slot + 1, :width]
        next_sequence = workspace.next_sequence[slot : slot + 1, :width, :]
        scratch = workspace.selection_scratch
        # Constraint candidates are score-sorted per parent. Use the same
        # fixed-width W-way merge as the legacy worker-decision path so both
        # paths share identical score ordering and tie-breaking.
        _beam_search_group_cuda_worker_out_unchecked(
            cumulative,
            top_tokens,
            top_logprobs,
            sequence,
            stage.stage_index,
            selected_ids,
            selected_parents,
            selected_scores,
            next_sequence,
            selected_flat,
            scratch.local_scores[slot : slot + 1],
            scratch.local_flat_indices[slot : slot + 1],
            candidate_valid_mask,
            None,
            None,
            None,
        )

        selected_valid = state.scratch.selected_valid_mask.reshape(-1)
        torch.index_select(
            candidate_valid_mask.reshape(-1),
            0,
            selected_flat.reshape(-1),
            out=selected_valid,
        )
        if not stage.is_last_stage:
            state.buffers.valid_mask.copy_(selected_valid.view(1, width))
            return selected_ids, selected_parents, selected_scores, next_sequence

        final_ids = workspace.final_token_ids[slot : slot + 1, :width]
        final_parents = workspace.final_parent_ids[slot : slot + 1, :width]
        final_scores = workspace.final_scores[slot : slot + 1, :width]
        final_sequence = workspace.final_sequence[slot : slot + 1, :width, :]
        final_positions = workspace.selected_flat_indices[slot : slot + 1, :width]
        beam_search_rec_final_select_grouped_cuda_out(
            selected_ids,
            selected_parents,
            selected_scores,
            next_sequence,
            width,
            final_ids,
            final_parents,
            final_scores,
            final_sequence,
            final_positions,
        )
        torch.index_select(
            selected_valid,
            0,
            final_positions.reshape(-1),
            out=state.buffers.valid_mask.reshape(-1),
        )
        return final_ids, final_parents, final_scores, final_sequence

    def _advance_reference(
        self,
        state: GRModelRunnerState,
        stage: GRWorkerStageSpec,
        top_tokens: torch.Tensor,
        top_logprobs: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm_gr.ops.onerec_decode import (
            beam_search_group,
            beam_search_rec_final_select,
        )

        width = state.spec.beam_width
        cumulative = state.buffers.cumulative_logprobs.reshape(width, 1)
        sequence = state.buffers.generated_token_ids
        if stage.is_last_stage:
            out_ids, out_parents, out_scores, out_sequence = beam_search_rec_final_select(
                cumulative,
                top_tokens,
                top_logprobs,
                sequence,
                stage.stage_index,
                width,
            )
            scores = (
                cumulative.reshape(1, width, 1) + top_logprobs.reshape(1, width, width)
            ).reshape(1, -1)
            flat = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :width]
            selected_valid = torch.gather(candidate_valid_mask.reshape(1, -1), 1, flat)
            state.buffers.valid_mask.copy_(selected_valid)
            return (
                out_ids.reshape(1, width),
                out_parents.reshape(1, width),
                out_scores.reshape(1, width),
                out_sequence,
            )

        out_ids, out_parents, out_scores, prefix, out_sequence = beam_search_group(
            cumulative,
            top_tokens,
            top_logprobs,
            sequence,
            stage.stage_index,
            width,
        )
        scores = (cumulative.reshape(1, width, 1) + top_logprobs.reshape(1, width, width)).reshape(
            1, -1
        )
        flat = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :width]
        parent = torch.div(flat, width, rounding_mode="floor")
        order = torch.argsort(parent, dim=1, stable=True)
        grouped_flat = torch.gather(flat, 1, order)
        state.buffers.valid_mask.copy_(
            torch.gather(candidate_valid_mask.reshape(1, -1), 1, grouped_flat)
        )
        state.buffers.parent_count_prefix.copy_(prefix)
        return out_ids, out_parents, out_scores, out_sequence

    def advance_beam(
        self,
        stage: GRWorkerStageSpec,
        logits: torch.Tensor,
    ) -> GRStageAck:
        state = self.registry.get(stage.session_id)
        try:
            state.validate_stage(stage)
            self._validate_stable_addresses(state)
            if stage.stage_kind is not GRStageKind.DECODE:
                raise ValueError("advance_beam requires a Decode stage")
            if state.pending_stage_index != stage.stage_index:
                raise RuntimeError(
                    f"session {state.session_id!r} stage {stage.stage_index} was not prepared"
                )
            if state.status is not GRModelRunnerStatus.DECODE_RUNNING:
                raise RuntimeError(
                    f"session {state.session_id!r} cannot advance while {state.status.value}"
                )
            if logits.dim() != 2 or logits.shape[0] != state.spec.beam_width:
                raise ValueError(f"logits must be [beam_width, vocab], got {tuple(logits.shape)}")
            if logits.device != self.context.device:
                raise ValueError("logits must share the persistent state device")
            if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError("logits must use fp16, bf16, or fp32")
            # RecConstrainedTopK reads explicit row/vocabulary strides. A
            # sliced LM head has row padding; copying it is unnecessary.
            if logits.shape[1] == 0 or logits.stride(0) <= 0 or logits.stride(1) <= 0:
                raise ValueError("logits must have a nonempty vocabulary and positive strides")

            top_tokens, top_logprobs, valid_mask = self._select_candidates(state, stage, logits)
            if self.context.device.type == "cuda":
                out_ids, out_parents, out_scores, out_sequence = self._advance_cuda(
                    state,
                    stage,
                    top_tokens,
                    top_logprobs,
                    valid_mask,
                )
            else:
                out_ids, out_parents, out_scores, out_sequence = self._advance_reference(
                    state,
                    stage,
                    top_tokens,
                    top_logprobs,
                    valid_mask,
                )

            buffers = state.buffers
            buffers.generated_token_ids.copy_(out_sequence)
            buffers.cumulative_logprobs.copy_(out_scores)
            buffers.sampled_token_ids.copy_(out_ids)
            next_parent_beam_indices = state.scratch.next_parent_beam_indices
            torch.index_select(
                buffers.parent_beam_indices[0],
                0,
                out_parents.reshape(-1),
                out=next_parent_beam_indices[0],
            )
            next_parent_beam_indices[:, :, stage.stage_index].copy_(out_parents)
            buffers.parent_beam_indices.copy_(next_parent_beam_indices)
            if not stage.is_last_stage:
                state.beam_session.step_complete(
                    out_parents.reshape(-1),
                    layout_prevalidated=self.context.device.type == "cuda",
                )
                self.metrics.kv_reorders += 1

            state.pending_stage_index = None
            state.next_expected_stage += 1
            state.decode_step += 1
            state.state_version += 1
            state.status = (
                GRModelRunnerStatus.FINAL_READY
                if stage.is_last_stage
                else GRModelRunnerStatus.DECODE_RUNNING
            )
            self.metrics.beam_decisions += 1
            if stage.is_last_stage:
                self.registry.record_terminal(state)
            return GRStageAck(
                state.session_id,
                stage.stage_index,
                True,
                stage.is_last_stage,
            )
        except Exception:
            self._fail(state)
            raise

    def get_final_device_view(self, session_id: str) -> GRFinalDeviceView:
        state = self.registry.get(session_id)
        self._validate_stable_addresses(state)
        if state.status is not GRModelRunnerStatus.FINAL_READY:
            raise RuntimeError(
                f"session {session_id!r} final view is unavailable while {state.status.value}"
            )
        buffers = state.buffers
        return GRFinalDeviceView(
            sampled_token_ids=buffers.sampled_token_ids,
            cumulative_logprobs=buffers.cumulative_logprobs,
            parent_beam_indices=buffers.parent_beam_indices[:, :, : state.spec.max_tokens],
            generated_token_ids=buffers.generated_token_ids[:, :, : state.spec.max_tokens],
            valid_mask=buffers.valid_mask,
        )

    def abort_session(self, session_id: str) -> None:
        self.registry.abort(session_id)

    def release_session(self, session_id: str) -> None:
        was_released = self.registry.is_released(session_id)
        self.registry.release(session_id)
        if not was_released:
            self.metrics.state_released += 1


class GPUModelRunnerPersistentBeamAdapter:
    """Inactive ModelRunner V1 seam for direct-call tests and future A3 wiring."""

    def __init__(self, backend: GRPersistentBeamBackend) -> None:
        self.backend = backend

    def create_session(self, spec: GRBeamSessionSpec) -> None:
        self.backend.create_session(spec)

    def on_prefill_output(
        self,
        stage: GRWorkerStageSpec,
        inputs: GRPrefillDeviceInputs,
    ) -> GRStageAck:
        return self.backend.bootstrap_from_prefill(stage, inputs)

    def prepare_decode_inputs(
        self,
        stage: GRWorkerStageSpec,
        inputs: GRDecodeDeviceInputs,
    ) -> None:
        self.backend.prepare_decode_inputs(stage, inputs)

    def on_decode_output(
        self,
        stage: GRWorkerStageSpec,
        logits: torch.Tensor,
    ) -> GRStageAck:
        return self.backend.advance_beam(stage, logits)

    def final_device_view(self, session_id: str) -> GRFinalDeviceView:
        return self.backend.get_final_device_view(session_id)

    def abort(self, session_id: str) -> None:
        self.backend.abort_session(session_id)

    def release(self, session_id: str) -> None:
        self.backend.release_session(session_id)


__all__ = [
    "GRDecodeDeviceInputs",
    "GRFinalDeviceView",
    "GRPersistentBeamBackend",
    "GRPersistentBeamMetrics",
    "GRPrefillDeviceInputs",
    "GPUModelRunnerPersistentBeamAdapter",
    "GPUModelRunnerPersistentBeamBackend",
]
