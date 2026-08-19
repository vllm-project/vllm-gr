# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Engine-layer CPU controller for the legacy vLLM-GR beam algorithm."""

from __future__ import annotations

import enum
import time
from typing import Any

from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs

from vllm_gr.beam_search_decision_utils import (
    complete_eos_candidates,
    extract_and_dedup_segment,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.constraints.candidate_filter import filter_candidates_by_token_ids
from vllm_gr.constraints.models import ParentCandidateBuilder
from vllm_gr.constraints.trie_resource import TrieConstraintResource
from vllm_gr.logprobs import project_flat_logprobs, reconstruct_beam_logprobs
from vllm_gr.beam_output import BeamOutputSpec
from vllm_gr.v1.beam.controller_types import (
    BeamControllerDecision,
    BeamControllerFinalBeam,
    BeamControllerFinalBeamLite,
    BeamControllerFinalResult,
    BeamControllerFinalResultLite,
    BeamControllerMetrics,
    BeamControllerRequest,
    BeamControllerStepOutput,
)


class BeamControllerState(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    FINALIZED = "finalized"


def _flat_to_wire(flat: FlatLogprobs) -> dict[str, list[Any]]:
    return {
        "start_indices": list(flat.start_indices),
        "end_indices": list(flat.end_indices),
        "token_ids": list(flat.token_ids),
        "logprobs": list(flat.logprobs),
        "ranks": list(flat.ranks),
        "decoded_tokens": list(flat.decoded_tokens),
    }


class BeamSearchController:
    """Session-local beam state and legacy CPU/Numpy decision logic."""

    def __init__(
        self,
        request: BeamControllerRequest,
        constraint_resource: TrieConstraintResource | None = None,
    ) -> None:
        if request.beam_width <= 0:
            raise ValueError("beam_width must be greater than zero")
        if request.catalog_strategy not in ("fill", "null"):
            raise ValueError("catalog_strategy must be 'fill' or 'null'")
        if request.beam_control_mode not in (
            "engine_per_step_request",
            "engine_persistent_request",
        ):
            raise ValueError(
                "EngineCore BeamSearchController requires an EngineCore beam_control_mode"
            )
        if request.constraint_backend == "trie":
            if request.constraint_resource_id is not None and constraint_resource is None:
                raise ValueError(
                    f"Trie resource {request.constraint_resource_id!r} was not provided"
                )
            if (
                constraint_resource is not None
                and request.constraint_resource_id != constraint_resource.digest
            ):
                raise ValueError("controller constraint resource digest mismatch")
        elif request.constraint_backend == "constraint_table":
            raise NotImplementedError(
                "EngineCore beam control does not yet support constraint_table"
            )
        else:
            raise ValueError(f"unsupported constraint backend {request.constraint_backend!r}")

        if request.prompt_length != len(request.prompt_token_ids):
            raise ValueError("prompt_length must match prompt_token_ids")
        expected_initial_tokens = list(request.prompt_token_ids)
        if request.begin_token_id is not None:
            expected_initial_tokens.append(request.begin_token_id)
        if request.initial_tokens != expected_initial_tokens:
            raise ValueError(
                "initial_tokens must contain prompt_token_ids and the optional begin token"
            )

        pre_calc = int(request.begin_token_id is not None) + int(request.end_token_id is not None)
        if request.max_tokens < pre_calc:
            raise ValueError(
                f"max tokens should be not lower than {pre_calc} to accommodate "
                "reserved begin/end tokens"
            )

        self.request = request
        self.beam_session_id = request.beam_session_id
        self.beam_width = request.beam_width
        self.max_tokens = request.max_tokens
        self.current_step = 0
        self.total_execution_steps = request.max_tokens - pre_calc
        self.initial_tokens = list(request.initial_tokens)
        self.constraint_resource = constraint_resource
        self.completed_beams: list[BeamSearchSequence] = []
        self.previous_internal_ids: list[str] = [request.beam_session_id]
        self.fork_info: list[tuple[int, int]] | None = None
        self.num_generation_tokens = 0
        self.generation_time = 0.0
        self._step_dispatched_at: float | None = None
        self._started_at = time.perf_counter()
        self._prefill_finished_at: float | None = None
        self._terminal_at: float | None = None
        self._sort_time = 0.0
        self._error: str | None = None
        self._final_result: BeamControllerFinalResult | BeamControllerFinalResultLite | None = None
        self.state = BeamControllerState.RUNNING

        initial_logprobs = FlatLogprobs()
        if request.begin_token_id is not None:
            initial_logprobs.append_fast([request.begin_token_id], [0.0], [None], [None])
        self.initial_logprobs = initial_logprobs

        root_beam = BeamSearchSequence(
            orig_prompt=tokens_input(self.initial_tokens),
            tokens=list(self.initial_tokens),
            cum_logprob=0.0,
            logprobs=initial_logprobs,
            lora_request=None,
        )
        root_beam._lp_parent = None
        root_beam._lp_step_data = None
        self.active_beams: list[BeamSearchSequence] = [root_beam]
        self.active_constraint_states = [self._initial_constraint_state()]

        if self.total_execution_steps == 0:
            self._complete("length")

    @classmethod
    def from_request(
        cls,
        request: BeamControllerRequest,
        constraint_resource: TrieConstraintResource | None = None,
    ) -> BeamSearchController:
        return cls(request, constraint_resource)

    @property
    def requires_step(self) -> bool:
        return self.state is BeamControllerState.RUNNING and bool(self.active_beams)

    @property
    def active_suffix_tokens(self) -> list[list[int]]:
        prefix_len = len(self.initial_tokens)
        return [list(beam.tokens[prefix_len:]) for beam in self.active_beams]

    def mark_step_dispatched(self, timestamp: float | None = None) -> None:
        if self.state is not BeamControllerState.RUNNING:
            raise RuntimeError("terminal controller cannot dispatch another step")
        dispatched_at = timestamp if timestamp is not None else time.perf_counter()
        # The legacy Serving loop classifies the first execution plus its CPU
        # decision as prefill, then starts decode timing immediately before
        # dispatching step 1.
        if self.current_step == 1 and self._prefill_finished_at is None:
            self._prefill_finished_at = dispatched_at
        self._step_dispatched_at = dispatched_at

    def consume_output(self, output: BeamControllerStepOutput) -> BeamControllerDecision:
        if self.state is not BeamControllerState.RUNNING:
            raise RuntimeError(
                f"controller {self.beam_session_id!r} is terminal ({self.state.value})"
            )
        completed_step = self.current_step
        now = time.perf_counter()
        if completed_step > 0 and self._step_dispatched_at is not None:
            self.generation_time += now - self._step_dispatched_at
            self.num_generation_tokens += len(self.active_beams)

        if output.error is not None:
            self._fail(output.error)
            return self._terminal_decision(completed_step)

        builder = ParentCandidateBuilder()
        parent_count = len(self.active_beams)
        if parent_count == 0:
            self._complete("constraint")
            return self._terminal_decision(completed_step)

        total_candidates = len(output.token_ids)
        if len(output.logprobs) != total_candidates:
            self._fail("candidate token/logprob length mismatch")
            return self._terminal_decision(completed_step)
        if total_candidates % parent_count != 0:
            self._fail(
                f"candidate row of length {total_candidates} cannot be divided "
                f"across {parent_count} active beams"
            )
            return self._terminal_decision(completed_step)

        stride = total_candidates // parent_count
        for parent_index, parent in enumerate(self.active_beams):
            start = parent_index * stride
            end = start + stride
            token_ids, local_logprobs, ranks, decoded = extract_and_dedup_segment(
                output.token_ids,
                output.logprobs,
                output.ranks,
                output.decoded_tokens,
                start,
                end,
                self.beam_width,
            )
            if ranks is None:
                ranks = [None] * len(token_ids)
            if decoded is None:
                decoded = [None] * len(token_ids)
            if self.constraint_resource is not None:
                state = self.active_constraint_states[parent_index]
                valid = (
                    set() if state is None else self.constraint_resource.valid_next_token_ids(state)
                )
                token_ids, local_logprobs, ranks, decoded = filter_candidates_by_token_ids(
                    token_ids,
                    local_logprobs,
                    ranks,
                    decoded,
                    valid,
                    self.request.catalog_strategy,
                )
            builder.add_parent(
                parent_index=parent_index,
                parent_cumulative_logprob=parent.cum_logprob,
                token_ids=token_ids,
                local_logprobs=local_logprobs,
                ranks=ranks,
                decoded_tokens=decoded,
            )

        candidate_batch = builder.build()
        scores = candidate_batch.cumulative_logprobs
        if not self.request.ignore_eos:
            stop_ids = {self.request.eos_token_id}
            if self.request.end_token_id is not None:
                stop_ids.add(self.request.end_token_id)
            eos_beams, scores = complete_eos_candidates(
                parents=self.active_beams,
                token_ids=candidate_batch.token_ids,
                scores=scores,
                beam_idx=candidate_batch.parent_indices,
                beam_flat_cache=candidate_batch.parent_cache,
                stop_token_ids=stop_ids,
                initial_logprobs=self.initial_logprobs,
                include_eos_token=self.request.include_stop_token,
                inherit_lora_request=False,
            )
            self.completed_beams.extend(eos_beams)

        selected = select_top_indices(
            scores,
            self.beam_width,
            self.request.catalog_strategy,
        )
        new_beams, fork_info = materialize_selected_beams(
            parents=self.active_beams,
            selected_indices=selected,
            token_ids=candidate_batch.token_ids,
            scores=scores,
            beam_idx=candidate_batch.parent_indices,
            beam_flat_cache=candidate_batch.parent_cache,
            initial_logprobs=self.initial_logprobs,
        )
        new_constraint_states: list[int | None] = []
        if self.constraint_resource is not None:
            for parent_index, token_id in fork_info:
                parent_state = self.active_constraint_states[parent_index]
                new_constraint_states.append(
                    None
                    if parent_state is None
                    else self.constraint_resource.advance(parent_state, token_id)
                )
        else:
            new_constraint_states = [None] * len(new_beams)

        self.active_beams = new_beams
        self.active_constraint_states = new_constraint_states
        self.fork_info = fork_info
        self.current_step += 1
        self.previous_internal_ids = [
            f"{self.beam_session_id}-step-{self.current_step}-beam-{index}"
            for index in range(len(new_beams))
        ]
        if not new_beams:
            self._complete("constraint")
            return self._terminal_decision(completed_step)
        if self.current_step >= self.total_execution_steps:
            self._complete("length")
            return self._terminal_decision(completed_step)

        return BeamControllerDecision(
            completed_step=completed_step,
            next_tokens=[token_id for _, token_id in fork_info],
            parent_beam_indices=[parent_index for parent_index, _ in fork_info],
            active_width=len(new_beams),
            finished=False,
        )

    def cancel(self) -> None:
        if self.state in (
            BeamControllerState.CANCELLED,
            BeamControllerState.FINALIZED,
        ):
            return
        if self.state in (BeamControllerState.COMPLETED, BeamControllerState.ERROR):
            return
        self.state = BeamControllerState.CANCELLED
        self._terminal_at = time.perf_counter()

    def finalize(self) -> BeamControllerFinalResult | BeamControllerFinalResultLite:
        if self._final_result is not None:
            return self._final_result
        if self.state is BeamControllerState.RUNNING:
            raise RuntimeError("cannot finalize a running beam controller")
        if self.state is BeamControllerState.CANCELLED:
            result_type = (
                BeamControllerFinalResultLite
                if self.request.output_spec is not None
                and self.request.output_spec.logprobs_mode == "none"
                else BeamControllerFinalResult
            )
            result = result_type(
                beam_session_id=self.beam_session_id,
                external_request_id=self.request.external_request_id,
                beams=[],
                metrics=None,
                finish_reason="cancelled",
                error="beam search cancelled",
            )
            self._final_result = result
            self.state = BeamControllerState.FINALIZED
            return result
        if self.state is BeamControllerState.ERROR:
            result_type = (
                BeamControllerFinalResultLite
                if self.request.output_spec is not None
                and self.request.output_spec.logprobs_mode == "none"
                else BeamControllerFinalResult
            )
            result = result_type(
                beam_session_id=self.beam_session_id,
                external_request_id=self.request.external_request_id,
                beams=[],
                metrics=None,
                finish_reason="error",
                error=self._error or "beam controller error",
            )
            self._final_result = result
            self.state = BeamControllerState.FINALIZED
            return result

        sort_start = time.perf_counter()
        sorted_completed = sorted(
            self.completed_beams,
            key=lambda beam: beam.cum_logprob,
            reverse=True,
        )
        output_spec = self.request.output_spec
        legacy_full_output = output_spec is None
        if output_spec is None:
            output_spec = BeamOutputSpec(
                result_width=self.beam_width,
                logprobs_mode="top_k",
                top_logprobs=self.beam_width,
            )
        output_spec.validate(self.beam_width)
        best_beams = sorted_completed[: output_spec.result_width]
        self._sort_time = time.perf_counter() - sort_start
        lite_beams: list[BeamControllerFinalBeamLite] = []
        final_beams: list[BeamControllerFinalBeam] = []
        for beam in best_beams:
            completion_tokens = list(beam.tokens[self.request.prompt_length :])
            if output_spec.logprobs_mode == "none":
                lite_beams.append(
                    BeamControllerFinalBeamLite(
                        token_ids=completion_tokens,
                        cumulative_logprob=float(beam.cum_logprob),
                        finish_reason=beam.finish_reason or "length",
                        stop_reason=beam.stop_reason,
                    )
                )
                continue
            reconstruct_beam_logprobs(
                beam,
                self.initial_logprobs,
                self.request.end_token_id,
            )
            output_logprobs = beam.logprobs
            if not legacy_full_output:
                output_logprobs = project_flat_logprobs(
                    output_logprobs,
                    completion_tokens,
                    output_spec.logprobs_mode,
                    output_spec.top_logprobs,
                )
            decode_tokens = completion_tokens
            if (
                completion_tokens
                and completion_tokens[-1] == self.request.eos_token_id
                and not self.request.ignore_eos
            ):
                decode_tokens = completion_tokens[:-1]
            final_beams.append(
                BeamControllerFinalBeam(
                    token_ids=completion_tokens,
                    decode_token_ids=decode_tokens,
                    cumulative_logprob=float(beam.cum_logprob),
                    logprobs=_flat_to_wire(output_logprobs),
                    finish_reason=beam.finish_reason or "length",
                    stop_reason=beam.stop_reason,
                )
            )

        terminal_at = time.perf_counter()
        self._terminal_at = terminal_at
        prefill_at = self._prefill_finished_at or self._started_at
        prefill_time = max(0.0, prefill_at - self._started_at)
        decode_time = max(0.0, terminal_at - prefill_at)
        metrics = BeamControllerMetrics(
            beam_search_overhead=max(0.0, decode_time - self.generation_time),
            beam_search_prefill_time=prefill_time,
            beam_search_decode_time=decode_time,
            beam_search_sort_time=self._sort_time,
            num_generation_tokens=self.num_generation_tokens,
            num_beams=self.beam_width,
        )
        if output_spec.logprobs_mode == "none":
            result = BeamControllerFinalResultLite(
                beam_session_id=self.beam_session_id,
                external_request_id=self.request.external_request_id,
                beams=lite_beams,
                metrics=metrics,
                finish_reason="completed",
            )
        else:
            result = BeamControllerFinalResult(
                beam_session_id=self.beam_session_id,
                external_request_id=self.request.external_request_id,
                beams=final_beams,
                metrics=metrics,
                finish_reason="completed",
            )
        self._final_result = result
        self.state = BeamControllerState.FINALIZED
        return result

    def _initial_constraint_state(self) -> int | None:
        if self.constraint_resource is None:
            return None
        state: int | None = self.constraint_resource.root_state()
        for token_id in self.initial_tokens[self.request.prompt_length :]:
            if state is None:
                break
            state = self.constraint_resource.advance(state, token_id)
        return state

    def _complete(self, reason: str) -> None:
        if self.state is not BeamControllerState.RUNNING:
            return
        if self.request.end_token_id is not None:
            for beam in self.active_beams:
                beam.tokens.append(self.request.end_token_id)
        self.completed_beams.extend(self.active_beams)
        self.active_beams = []
        self.active_constraint_states = []
        self.state = BeamControllerState.COMPLETED
        self._terminal_at = time.perf_counter()
        self._completion_reason = reason

    def _fail(self, error: str) -> None:
        self._error = error
        self.state = BeamControllerState.ERROR
        self._terminal_at = time.perf_counter()

    def _terminal_decision(self, completed_step: int) -> BeamControllerDecision:
        result = self.finalize()
        return BeamControllerDecision(
            completed_step=completed_step,
            next_tokens=[],
            parent_beam_indices=[],
            active_width=0,
            finished=True,
            finish_reason=result.finish_reason,
            final_beams=result.beams,
            error=result.error,
        )
