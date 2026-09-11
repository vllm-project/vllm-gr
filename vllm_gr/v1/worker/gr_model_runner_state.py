# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-owned lifecycle and stable device views for persistent Beam V1."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.persistent_contracts import GRBeamSessionSpec, GRWorkerStageSpec


class GRModelRunnerStatus(str, Enum):
    NEW = "new"
    PREFILL_RUNNING = "prefill_running"
    PREFILL_BOOTSTRAPPED = "prefill_bootstrapped"
    DECODE_RUNNING = "decode_running"
    FINAL_READY = "final_ready"
    RELEASED = "released"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class GRBeamStateBuffers:
    """Slot-local views into the existing runner-owned Beam buffers."""

    generated_token_ids: torch.Tensor
    cumulative_logprobs: torch.Tensor
    parent_beam_indices: torch.Tensor
    sampled_token_ids: torch.Tensor
    valid_mask: torch.Tensor
    parent_count_prefix: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.generated_token_ids,
            self.cumulative_logprobs,
            self.parent_beam_indices,
            self.sampled_token_ids,
            self.valid_mask,
            self.parent_count_prefix,
        )

    def data_ptrs(self) -> tuple[int, ...]:
        return tuple(tensor.data_ptr() for tensor in self.tensors())


@dataclass(frozen=True, slots=True)
class GRBeamScratchBuffers:
    """A4-only fixed scratch that carries no authoritative Beam history."""

    selected_valid_mask: torch.Tensor
    next_parent_beam_indices: torch.Tensor

    def data_ptrs(self) -> tuple[int, ...]:
        return (
            self.selected_valid_mask.data_ptr(),
            self.next_parent_beam_indices.data_ptr(),
        )


@dataclass(slots=True)
class GRModelRunnerState:
    """Single source of truth for one persistent Worker Beam session."""

    spec: GRBeamSessionSpec
    slot_id: int
    beam_session: BeamSearchSession
    buffers: GRBeamStateBuffers
    scratch: GRBeamScratchBuffers
    workspace: Any | None = None
    next_expected_stage: int = 0
    decode_step: int = 0
    status: GRModelRunnerStatus = GRModelRunnerStatus.NEW
    state_version: int = 0
    pending_stage_index: int | None = None
    terminal_event: Any | None = None
    terminal_event_recorded: bool = False
    initial_data_ptrs: tuple[int, ...] = ()
    initial_workspace_data_ptrs: tuple[int, ...] = ()
    decode_input_data_ptr: int | None = None

    @property
    def session_id(self) -> str:
        return self.spec.session_id

    @property
    def request_id(self) -> str:
        return self.spec.request_id

    def validate_stage(self, stage: GRWorkerStageSpec) -> None:
        if stage.session_id != self.session_id:
            raise ValueError(
                f"stage session_id {stage.session_id!r} does not match {self.session_id!r}"
            )
        if self.status in (
            GRModelRunnerStatus.RELEASED,
            GRModelRunnerStatus.FAILED,
            GRModelRunnerStatus.ABORTED,
        ):
            raise RuntimeError(
                f"session {self.session_id!r} cannot run a stage while {self.status.value}"
            )
        if stage.stage_index != self.next_expected_stage:
            raise RuntimeError(
                f"session {self.session_id!r} expected stage {self.next_expected_stage}, "
                f"got {stage.stage_index}"
            )
        expected_last = stage.stage_index == self.spec.max_tokens - 1
        if stage.is_last_stage is not expected_last:
            raise ValueError(
                f"stage {stage.stage_index} is_last_stage must be {expected_last} "
                f"for max_tokens={self.spec.max_tokens}"
            )

    def mark_failed(self) -> None:
        self.pending_stage_index = None
        self.status = GRModelRunnerStatus.FAILED
        self.state_version += 1


class GRModelRunnerStateRegistry:
    """Slot-indexed state registry layered over ``BeamSessionRegistry``.

    Recently released IDs remain in a bounded tombstone set for idempotence.
    A CUDA slot is returned only after its terminal event reports completion;
    no synchronize is performed by the registry.
    """

    def __init__(self, context: BeamAttentionContext) -> None:
        self.context = context
        self._states: dict[str, GRModelRunnerState] = {}
        self._pending_release: set[str] = set()
        self._released_session_ids: set[str] = set()
        self._released_session_order: deque[str] = deque()
        self._max_released_session_tombstones = max(64, context.max_batch * 8)
        # GPU-A4-only state with no equivalent in the legacy/NPU context.
        # Sequence and cumulative scores are deliberately absent: their sole
        # authority remains ``context.decision``.
        lineage_shape = (
            context.max_batch,
            context.max_beams,
            context.max_decode_steps,
        )
        self._parent_beam_indices = torch.zeros(
            lineage_shape, dtype=torch.int32, device=context.device
        )
        self._sampled_token_ids = torch.zeros(
            (context.max_batch, context.max_beams),
            dtype=torch.int32,
            device=context.device,
        )
        self._valid_mask = torch.zeros(
            (context.max_batch, context.max_beams),
            dtype=torch.bool,
            device=context.device,
        )
        self._selected_valid_mask = torch.zeros(
            (context.max_batch, context.max_beams),
            dtype=torch.bool,
            device=context.device,
        )
        self._next_parent_beam_indices = torch.zeros(
            lineage_shape, dtype=torch.int32, device=context.device
        )

    def __contains__(self, session_id: str) -> bool:
        state = self._states.get(session_id)
        return state is not None and state.status is not GRModelRunnerStatus.RELEASED

    def create(
        self, spec: GRBeamSessionSpec, *, workspace: Any | None = None
    ) -> GRModelRunnerState:
        self.reap_completed()
        if (
            spec.session_id in self._states
            or spec.session_id in self._released_session_ids
        ):
            raise ValueError(f"session {spec.session_id!r} was already created")
        if spec.beam_width > self.context.max_beams:
            raise ValueError(
                f"beam_width={spec.beam_width} exceeds persistent capacity {self.context.max_beams}"
            )
        if spec.max_tokens > self.context.max_decode_steps:
            raise ValueError(
                f"max_tokens={spec.max_tokens} exceeds persistent token capacity "
                f"{self.context.max_decode_steps}"
            )

        slot = self.context.registry.allocate(
            spec.session_id,
            beam_width=spec.beam_width,
            prefix_len=0,
        )
        session = BeamSearchSession(spec.session_id, slot, self.context)
        decision = self.context.decision
        width = spec.beam_width
        decision.sequence[slot].zero_()
        decision.beam_scores[slot].zero_()
        decision.group_token_num[slot].zero_()
        self._parent_beam_indices[slot].zero_()
        self._sampled_token_ids[slot].zero_()
        self._valid_mask[slot].zero_()
        self._selected_valid_mask[slot].zero_()
        self._next_parent_beam_indices[slot].zero_()
        buffers = GRBeamStateBuffers(
            generated_token_ids=decision.sequence[slot : slot + 1, :width, :],
            cumulative_logprobs=decision.beam_scores[slot : slot + 1, :width],
            parent_beam_indices=self._parent_beam_indices[slot : slot + 1, :width, :],
            sampled_token_ids=self._sampled_token_ids[slot : slot + 1, :width],
            valid_mask=self._valid_mask[slot : slot + 1, :width],
            parent_count_prefix=decision.group_token_num[slot : slot + 1, :width],
        )
        scratch = GRBeamScratchBuffers(
            selected_valid_mask=self._selected_valid_mask[slot : slot + 1, :width],
            next_parent_beam_indices=self._next_parent_beam_indices[slot : slot + 1, :width, :],
        )
        event = (
            torch.cuda.Event(enable_timing=False) if self.context.device.type == "cuda" else None
        )
        state = GRModelRunnerState(
            spec=spec,
            slot_id=slot,
            beam_session=session,
            buffers=buffers,
            scratch=scratch,
            workspace=workspace,
            terminal_event=event,
            initial_data_ptrs=buffers.data_ptrs() + scratch.data_ptrs(),
            initial_workspace_data_ptrs=(
                tuple(tensor.data_ptr() for tensor in workspace.device_tensors())
                if workspace is not None
                else ()
            ),
        )
        self._states[spec.session_id] = state
        return state

    def get(self, session_id: str) -> GRModelRunnerState:
        try:
            state = self._states[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown persistent Beam session {session_id!r}") from exc
        if state.status is GRModelRunnerStatus.RELEASED:
            raise KeyError(f"persistent Beam session {session_id!r} is released")
        return state

    def record_terminal(self, state: GRModelRunnerState) -> None:
        if state.terminal_event is not None and not state.terminal_event_recorded:
            state.terminal_event.record(torch.cuda.current_stream(self.context.device))
            state.terminal_event_recorded = True

    def abort(self, session_id: str) -> None:
        state = self.get(session_id)
        if state.status in (
            GRModelRunnerStatus.ABORTED,
            GRModelRunnerStatus.FAILED,
            GRModelRunnerStatus.FINAL_READY,
        ):
            return
        state.pending_stage_index = None
        state.status = GRModelRunnerStatus.ABORTED
        state.state_version += 1
        self.record_terminal(state)

    def release(self, session_id: str) -> None:
        state = self._states.get(session_id)
        if state is None:
            if session_id in self._released_session_ids:
                return
            raise KeyError(f"unknown persistent Beam session {session_id!r}")
        if state.status is GRModelRunnerStatus.RELEASED:
            self.reap_completed()
            return
        state.pending_stage_index = None
        state.status = GRModelRunnerStatus.RELEASED
        state.state_version += 1
        self.record_terminal(state)
        self._pending_release.add(session_id)
        self.reap_completed()

    def is_released(self, session_id: str) -> bool:
        state = self._states.get(session_id)
        return (
            state is not None and state.status is GRModelRunnerStatus.RELEASED
        ) or session_id in self._released_session_ids

    def _remember_released(self, session_id: str) -> None:
        if session_id in self._released_session_ids:
            return
        while (
            len(self._released_session_order)
            >= self._max_released_session_tombstones
        ):
            expired = self._released_session_order.popleft()
            self._released_session_ids.discard(expired)
        self._released_session_order.append(session_id)
        self._released_session_ids.add(session_id)

    def reap_completed(self) -> None:
        for session_id in tuple(self._pending_release):
            state = self._states[session_id]
            event = state.terminal_event
            if event is not None and state.terminal_event_recorded and not event.query():
                continue
            if self.context.registry.has_session(session_id):
                self.context.registry.release(session_id)
            self._pending_release.remove(session_id)
            self._states.pop(session_id, None)
            self._remember_released(session_id)

    @property
    def active_session_count(self) -> int:
        return sum(
            state.status is not GRModelRunnerStatus.RELEASED for state in self._states.values()
        )


__all__ = [
    "GRBeamScratchBuffers",
    "GRBeamStateBuffers",
    "GRModelRunnerState",
    "GRModelRunnerStateRegistry",
    "GRModelRunnerStatus",
]
