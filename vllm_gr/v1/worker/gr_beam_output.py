# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-local final-only output and resource retirement for persistent Beam.

Intermediate stages keep returning A4 enqueue acknowledgements. Only
``enqueue_final`` transfers Beam data to Host. A4 integration calls this class;
EngineCore exception/shutdown wiring belongs to A6. Calls are Worker-serialized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchOutput,
    BeamBatchResult,
    BeamOutputOptions,
    BeamOutputSequence,
    BeamRequestOutput,
    _validate_output_options,
    validate_beam_batch_result,
)
from vllm_gr.v1.beam.persistent_contracts import GRBeamSessionSpec
from vllm_gr.v1.worker.gr_persistent_beam import GPUModelRunnerPersistentBeamBackend


@dataclass(slots=True)
class _FinalOutputState:
    spec: GRBeamSessionSpec
    options: BeamOutputOptions
    slot: int
    event: torch.cuda.Event | None
    enqueued: bool = False
    retiring: bool = False
    result: BeamBatchResult | None = None


class GRBeamFinalOutput:
    """Own fixed final staging slots and the unified Worker cleanup boundary.

    Use ``create_session`` instead of creating sessions directly on the backend.
    After the terminal A4 ack, call ``enqueue_final`` then ``get_result``. Nonblocking
    polling returns None until the final D2H is complete; wait=True is allowed only
    at this final boundary. The caller owns the returned immutable Host result.
    Call ``cleanup_session`` after delivery, or with abort=True on cancellation.
    Repeated output reads/enqueues are idempotent until cleanup. Cleanup itself is
    idempotent. Retired slots are reaped without synchronizing the Host.

    State and Beam KV/workspaces are returned to their existing pools, not freed
    and reallocated per request. Scheduler-owned paged KV is never freed here.
    """

    def __init__(self, backend: GPUModelRunnerPersistentBeamBackend) -> None:
        self.backend = backend
        context = backend.context
        self._steps = context.max_decode_steps
        # int32 tokens / parent lineage / score bits / validity, one payload.
        shape = (context.max_batch, context.max_beams, 2 * self._steps + 2)
        self._device = torch.empty(shape, dtype=torch.int32, device=context.device)
        self._host = torch.empty(
            shape, dtype=torch.int32, device="cpu", pin_memory=context.device.type == "cuda"
        )
        self._sessions: dict[str, _FinalOutputState] = {}
        self.final_d2h_count = 0
        self.final_d2h_bytes = 0

    def create_session(self, spec: GRBeamSessionSpec, options: BeamOutputOptions) -> None:
        _validate_output_options(options, beam_width=spec.beam_width)
        self.reap_completed()
        # Allocate the Event before acquiring backend resources.
        event = torch.cuda.Event() if self.backend.context.device.type == "cuda" else None
        self.backend.create_session(spec)
        state = self.backend.registry.get(spec.session_id)
        self._sessions[spec.session_id] = _FinalOutputState(spec, options, state.slot_id, event)

    def _get(self, session_id: str) -> _FinalOutputState:
        try:
            state = self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown final-output session {session_id!r}") from exc
        if state.retiring:
            raise RuntimeError(f"final-output session {session_id!r} is retiring")
        return state

    def enqueue_final(self, session_id: str) -> None:
        state = self._get(session_id)
        if state.enqueued:
            return
        view = self.backend.get_final_device_view(session_id)
        runner_state = self.backend.registry.get(session_id)
        event = state.event
        if event is not None:
            stream = torch.cuda.current_stream(self.backend.context.device)
            if runner_state.terminal_event_recorded:
                stream.wait_event(runner_state.terminal_event)
        try:
            width, steps = state.spec.beam_width, state.spec.max_tokens
            payload = self._device[state.slot, :width]
            payload.zero_()
            payload[:, :steps].copy_(view.generated_token_ids[0])
            payload[:, self._steps : self._steps + steps].copy_(view.parent_beam_indices[0])
            # Bitwise score packing: no float conversion, rounding or re-ranking.
            payload[:, -2].view(torch.float32).copy_(view.cumulative_logprobs[0])
            payload[:, -1].copy_(view.valid_mask[0])
            self._host[state.slot, :width].copy_(payload, non_blocking=event is not None)
            if event is not None:
                self.final_d2h_count += 1
                self.final_d2h_bytes += payload.numel() * payload.element_size()
            state.enqueued = True
        except Exception:
            state.retiring = True
            raise
        finally:
            # Replace A4's earlier computation fence, including partially
            # submitted packing/copy work on failure. Never recycle before it.
            if event is not None:
                event.record(torch.cuda.current_stream(self.backend.context.device))
                runner_state.terminal_event = event
                runner_state.terminal_event_recorded = True
            if state.retiring:
                self.backend.release_session(session_id)

    def get_result(self, session_id: str, *, wait: bool = False) -> BeamBatchResult | None:
        state = self._get(session_id)
        if not state.enqueued:
            raise RuntimeError("final output has not been enqueued")
        if state.result is not None:
            return state.result
        if state.event is not None:
            if wait:
                state.event.synchronize()
            elif not state.event.query():
                return None
        try:
            state.result = self._build_result(state)
        except Exception:
            self.cleanup_session(session_id, abort=True)
            raise
        return state.result

    def _build_result(self, state: _FinalOutputState) -> BeamBatchResult:
        # Only CPU staging is materialized, after the completion fence.
        rows = self._host[state.slot, : state.spec.beam_width]
        values = rows.tolist()
        scores = rows[:, -2].view(torch.float32).tolist()
        sequences = []
        steps = state.spec.max_tokens
        for row, score in zip(values, scores, strict=True):
            if not row[-1] or score == -math.inf:
                continue
            if not math.isfinite(score):
                raise ValueError("final Beam score must be finite or a dead-beam -inf")
            sequences.append(
                BeamOutputSequence(
                    token_ids=tuple(row[:steps]),
                    cumulative_logprob=score if state.options.include_cumulative_logprob else None,
                    parent_beam_indices=(
                        tuple(row[self._steps : self._steps + steps])
                        if state.options.include_parent_beam_indices
                        else ()
                    ),
                    finish_reason="length",
                    stop_token_id=None,
                )
            )
            if len(sequences) == state.options.num_return_sequences:
                break
        result = BeamBatchResult(
            protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
            batch_id=state.spec.session_id,
            output=(
                BeamBatchOutput(
                    (
                        BeamRequestOutput(
                            item_index=0,
                            request_id=state.spec.request_id,
                            sequences=tuple(sequences),
                        ),
                    )
                )
                if sequences
                else None
            ),
            failure=(
                None
                if sequences
                else BeamBatchFailure(
                    code="no_valid_beams", message="No valid final Beam candidates", item_index=0
                )
            ),
        )
        validate_beam_batch_result(result)
        return result

    def cleanup_session(self, session_id: str, *, abort: bool = False) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            if self.backend.registry.is_released(session_id):
                return
            raise KeyError(f"unknown final-output session {session_id!r}")
        if not state.retiring:
            if abort:
                self.backend.abort_session(session_id)
            self.backend.release_session(session_id)
            state.retiring = True
        self.reap_completed()

    def reap_completed(self) -> None:
        self.backend.registry.reap_completed()
        for session_id, state in tuple(self._sessions.items()):
            if state.retiring and not self.backend.context.registry.has_session(session_id):
                del self._sessions[session_id]
