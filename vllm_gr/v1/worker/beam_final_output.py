# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compact final output for the persistent GPU Beam runner."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

import torch
import triton

from vllm_gr.ops.cuda import beam_state as beam_state_ops
from vllm_gr.ops.cuda.beam_selection import (
    _hierarchical_topk_out,
    allocate_beam_selection_scratch,
)
from vllm_gr.ops.cuda.beam_state import FINISH_STOP
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchOutput,
    BeamBatchParams,
    BeamBatchResult,
    BeamOutputOptions,
    BeamOutputSequence,
    BeamRequestOutput,
)

_CONTROL_WORDS = 3

if TYPE_CHECKING:
    from vllm_gr.v1.worker.gpu_beam_stage_runner import (
        BeamRequestBinding,
        GPUBeamStageRunner,
    )


@dataclass(frozen=True, slots=True)
class _PayloadLayout:
    """Fixed compact wire layout: count followed by one row per returned Beam."""

    width: int
    steps: int
    row_stride: int
    size: int

    @classmethod
    def create(cls, width: int, steps: int) -> _PayloadLayout:
        row_stride = 2 * steps + 4
        return cls(width, steps, row_stride, _CONTROL_WORDS + 1 + width * row_stride)


def _build_result(
    payload: Any,
    layout: _PayloadLayout,
    *,
    session_id: str,
    request_id: str,
    item_index: int,
    options: BeamOutputOptions,
) -> BeamBatchResult:
    """Wrap the GPU-ranked compact payload in the existing transport ABI."""

    values = payload.numpy() if isinstance(payload, torch.Tensor) else payload
    count = int(values[_CONTROL_WORDS])
    sequences = []
    steps = layout.steps
    stride = layout.row_stride
    rows = values[_CONTROL_WORDS + 1 :].reshape(layout.width, stride)
    include_score = options.include_cumulative_logprob
    include_parents = options.include_parent_beam_indices
    include_stop = options.include_stop_token
    # The wire row uses native int32/float32 words without alignment padding.
    # Unpack directly to Python tuples, avoiding two tolist conversions and
    # per-field scalar casts. The small CPU copy also detaches reused storage.
    for row in struct.iter_unpack(f"={2 * steps}ifiii", rows[:count].tobytes()):
        length, reason, stop_token = row[-3:]
        stopped = reason == FINISH_STOP
        token_count = length - (stopped and not include_stop)
        sequences.append(
            BeamOutputSequence(
                row[:token_count],
                row[-4] if include_score else None,
                row[steps : steps + length] if include_parents else (),
                "stop" if stopped else "length",
                stop_token if stopped else None,
            )
        )

    return BeamBatchResult(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=session_id,
        output=(
            BeamBatchOutput((BeamRequestOutput(item_index, request_id, tuple(sequences)),))
            if sequences
            else None
        ),
        failure=(
            None
            if sequences
            else BeamBatchFailure("no_valid_beams", "No valid final Beam candidates", item_index)
        ),
    )


class GPUBeamFinalOutput:
    """Rank and materialize final paths on GPU, then transfer one compact payload."""

    def __init__(
        self,
        owner: GPUBeamStageRunner,
        binding: BeamRequestBinding,
        params: BeamBatchParams,
        options: BeamOutputOptions,
    ) -> None:
        state = owner.beam.state
        self._storage_key = (
            state.beam_width,
            state.max_new_tokens,
            state.completed_capacity,
            state.device,
        )
        self.layout = _PayloadLayout.create(state.beam_width, state.max_new_tokens)
        with torch.inference_mode(False):
            self.device = torch.empty(self.layout.size, dtype=torch.int32, device=state.device)
            self.host = torch.empty(self.layout.size, dtype=torch.int32, pin_memory=True)
            self._zero_score = torch.zeros((1, 1), dtype=torch.float32, device=state.device)
            self._ranking_scores = torch.empty(
                (1, state.completed_capacity), dtype=torch.float32, device=state.device
            )
            self._selection = allocate_beam_selection_scratch(
                1,
                1,
                state.completed_capacity,
                state.beam_width,
                state.device,
            )
        self.host_numpy = self.host.numpy()
        self.pack_ready = torch.Event()
        self.ready = torch.Event()
        self.done = torch.Event()
        self._lock = Lock()
        self.reset(owner, binding, params, options)
        self.final_d2h_count = 0
        self.final_d2h_bytes = 0

    def reset(
        self,
        owner: GPUBeamStageRunner,
        binding: BeamRequestBinding,
        params: BeamBatchParams,
        options: BeamOutputOptions,
    ) -> None:
        with self._lock:
            self.binding = binding
            self.options = options
            self.length_penalty = params.length_penalty
            self.prompt_length = owner.beam.session._state.prefix_len
            self._result: BeamBatchResult | None = None
            self._enqueued = False
            self._delivered = False
            self._aborted = False
            self._done_recorded = False
            self._control_embedded = False

    def is_compatible(self, owner: GPUBeamStageRunner) -> bool:
        state = owner.beam.state
        return self._storage_key == (
            state.beam_width,
            state.max_new_tokens,
            state.completed_capacity,
            state.device,
        )

    def _record_done(self, owner: GPUBeamStageRunner) -> None:
        if self._done_recorded:
            return
        stream = torch.cuda.current_stream(owner.beam.state.device)
        stream.wait_event(owner.last_work)
        if self._enqueued:
            stream.wait_event(self.ready)
        self.done.record(stream)
        self._done_recorded = True

    def _enqueue(
        self,
        owner: GPUBeamStageRunner,
        control: torch.Tensor | None = None,
    ) -> None:
        if self._enqueued:
            return
        state = owner.beam.state
        row = self.binding.state_row
        result_width = self.options.num_return_sequences
        produced = torch.cuda.current_stream(state.device)
        produced.wait_event(owner.last_work)
        ranking_scores = state.completed_scores[row : row + 1]
        if control is not None:
            self.device[:_CONTROL_WORDS].copy_(control)
            self._control_embedded = True
        # Ignored EOS completes every survivor at the same effective budget;
        # the common length penalty cannot change their ranking.
        if self.length_penalty != 0 and not state.ignore_stop_tokens_config:
            beam_state_ops._rank_final_beams_kernel[(triton.cdiv(state.completed_capacity, 256),)](
                ranking_scores,
                state.completed_lengths[row],
                state.completed_finish_reasons[row],
                self._ranking_scores,
                CAPACITY=state.completed_capacity,
                PROMPT_LENGTH=self.prompt_length,
                MAX_TOKENS=state.max_new_tokens,
                LENGTH_PENALTY=self.length_penalty,
                BLOCK=256,
            )
            ranking_scores = self._ranking_scores
        _, selected_indices, _ = _hierarchical_topk_out(
            self._zero_score,
            ranking_scores,
            None,
            1,
            1,
            state.completed_capacity,
            result_width,
            self._selection,
        )
        beam_state_ops._pack_final_beams_kernel[(result_width,)](
            selected_indices,
            state.num_completed[row : row + 1],
            state.completed_token_ids[row],
            state.completed_scores[row],
            state.completed_parent_ids[row],
            state.completed_lengths[row],
            state.completed_finish_reasons[row],
            state.history_token_ids[row],
            state.history_parent_ids[row],
            self.device[_CONTROL_WORDS:],
            WIDTH=state.beam_width,
            STEPS=state.max_new_tokens,
            RESULT_WIDTH=result_width,
            ROW_STRIDE=self.layout.row_stride,
            ROW_BLOCK=triton.next_power_of_2(2 * state.max_new_tokens),
        )
        self.pack_ready.record(produced)
        copy_words = _CONTROL_WORDS + 1 + result_width * self.layout.row_stride
        with torch.cuda.stream(owner.copy_stream):
            owner.copy_stream.wait_event(self.pack_ready)
            self.host[:copy_words].copy_(self.device[:copy_words], non_blocking=True)
            self.ready.record(owner.copy_stream)
        self.final_d2h_count += 1
        self.final_d2h_bytes += copy_words * self.device.element_size()
        self._enqueued = True

    def enqueue(self, owner: GPUBeamStageRunner, control: torch.Tensor) -> bool:
        """Submit final selection and D2H while the producing sample is still active."""

        with self._lock:
            if self._delivered or self._aborted:
                return False
            self._enqueue(owner, control)
            return self._enqueued and self._control_embedded

    def wait_embedded_control(self) -> list[int]:
        """Wait once for the terminal payload and return its control header."""

        with self._lock:
            if not self._enqueued or not self._control_embedded:
                raise RuntimeError("terminal Beam payload has no embedded control header")
            self.ready.synchronize()
            return self.host_numpy[:_CONTROL_WORDS].tolist()

    def abort(self, owner: GPUBeamStageRunner) -> None:
        with self._lock:
            if not self._delivered:
                self._aborted = True
            self._record_done(owner)

    def failure(self, owner: GPUBeamStageRunner, result: BeamBatchResult) -> BeamBatchResult | None:
        with self._lock:
            if self._delivered or self._aborted:
                return None
            self._result = result
            self._delivered = True
            self._record_done(owner)
            return result

    def consume(
        self,
        owner: GPUBeamStageRunner,
        *,
        ready: bool = False,
    ) -> BeamBatchResult | None:
        with self._lock:
            if self._delivered or self._aborted:
                return None
            # Early-stop termination is not known to the Host at sample time;
            # it uses this fallback. The fixed-length path is already enqueued.
            self._enqueue(owner)
            if not ready:
                self.ready.synchronize()
            self._result = _build_result(
                self.host_numpy,
                self.layout,
                session_id=self.binding.session_id,
                request_id=self.binding.native_request_id,
                item_index=self.binding.item_index,
                options=self.options,
            )
            self._delivered = True
            self._record_done(owner)
            return self._result

    def consumer_done(self, owner: GPUBeamStageRunner) -> torch.Event:
        with self._lock:
            self._record_done(owner)
            return self.done


__all__ = ["GPUBeamFinalOutput"]
