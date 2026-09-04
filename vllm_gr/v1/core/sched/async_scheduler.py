# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async scheduler support for a persistent GR request."""

from __future__ import annotations

from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_gr.config import MAX_BEAM_WIDTH, GRConfig
from vllm_gr.v1.beam.beam_decision import unpack_beam_result_buffer
from vllm_gr.v1.engine.types import GR_REQUEST_METADATA_KEY, GRRequestMetadata
from vllm_gr.v1.metrics.stats import GRRequestCounters


class GRAsyncScheduler(AsyncScheduler):
    """vLLM AsyncScheduler with one persistent logical GR request.

    Ordinary requests use the upstream implementation unchanged. Persistent
    runner-state GR requests use the same one-step output-placeholder lookahead
    as vLLM's native async scheduler: the scheduler advances one logical token,
    while the worker obtains the W physical Beam inputs from device state.
    """

    def add_request(self, request: Request) -> None:
        params = request.sampling_params
        extra_args = params.extra_args if params is not None else None
        raw = extra_args.get(GR_REQUEST_METADATA_KEY) if extra_args else None
        if raw is not None:
            if not isinstance(raw, dict):
                raise ValueError("sampling_params.extra_args['vllm_gr'] must be a mapping")
            metadata = GRRequestMetadata(**raw)
            if metadata.request_id != request.request_id:
                raise ValueError(
                    "persistent GR metadata request_id does not match the native request: "
                    f"{metadata.request_id!r} != {request.request_id!r}"
                )
            if metadata.prefix_len != request.num_prompt_tokens:
                raise ValueError(
                    "persistent GR prefix_len does not match the prompt length: "
                    f"{metadata.prefix_len} != {request.num_prompt_tokens}"
                )
            if not 1 <= metadata.beam_width <= MAX_BEAM_WIDTH:
                raise ValueError(f"persistent GR beam_width must be in [1, {MAX_BEAM_WIDTH}]")
            if metadata.max_tokens < 1:
                raise ValueError("persistent GR max_tokens must be at least 1")
            gr_config = GRConfig.from_vllm_config(
                getattr(self, "vllm_config", None)
            )
            configured_digest = (
                gr_config.constraint_table.artifact_digest
                if gr_config.constraint_table is not None
                else None
            )
            if (
                configured_digest is not None
                and metadata.constraint_digest != configured_digest
            ):
                raise ValueError(
                    "persistent GR constraint_digest does not match the configured "
                    "Constraint Table artifact"
                )
            # The request/config digest is the host-side immutable binding.
            # Resolving it to an installed worker resource is a backend concern
            # and is deliberately deferred to the Step-4 backend boundary.
            if any(self._is_gr_request(existing) for existing in self.requests.values()):
                raise ValueError("persistent GR requests currently require B=1")

            request.is_gr_request = True
            request.gr_metadata = metadata
            # ``gr_step`` remains the number of outputs reconciled on the host
            # for compatibility with the Phase-A counters and frontend.  Async
            # lookahead needs a second cursor because schedule() can issue the
            # next GR stage before update_from_output() advances ``gr_step``.
            request.gr_step = 0
            request.gr_issued_steps = 0
            request.gr_completed_steps = 0
            request.gr_persistent_runner_state = gr_config.persistent_runner_state
            request.gr_parent_ids = [0] * metadata.beam_width
            request.gr_beam_suffix = []
            request.gr_counters = GRRequestCounters(
                native_request_created_per_session=1,
                scheduler_add_request_per_session=1,
            )

        super().add_request(request)

    @staticmethod
    def _is_gr_request(request: Request) -> bool:
        return bool(getattr(request, "is_gr_request", False))

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        if not any(
            self._is_gr_request(self.requests[req_id])
            for req_id in scheduler_output.num_scheduled_tokens
        ):
            # Preserve upstream AsyncScheduler bookkeeping byte-for-byte for
            # an ordinary batch; the custom placeholder rules below exist
            # only for batches that actually contain a GR request.
            return super()._update_after_schedule(scheduler_output)

        # Keep vLLM's normal request accounting, then reproduce the upstream
        # AsyncScheduler placeholder logic.  A persistent runner-state GR
        # request is safe to schedule one stage ahead because its next Beam
        # tokens and parents never have to return to the scheduler.  The legacy
        # host-state compatibility path still waits for its Beam result.
        Scheduler._update_after_schedule(self, scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue
            if self._is_gr_request(request):
                if not bool(getattr(request, "gr_persistent_runner_state", False)):
                    continue
                issued_steps = int(getattr(request, "gr_issued_steps", 0))
                if issued_steps >= request.gr_metadata.max_tokens:
                    continue
                # GR does not use speculative tokens. One placeholder represents
                # the control token produced by this output-bearing stage.
                request.num_output_placeholders += 1
                continue
            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            request.spec_token_ids = self._spec_token_placeholders

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if not self._is_gr_request(request):
            return super()._update_request_with_output(request, new_token_ids)

        if request.async_tokens_to_discard > 0:
            request.async_tokens_to_discard -= 1
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = Scheduler._update_request_with_output(self, request, new_token_ids)
        if bool(getattr(request, "gr_persistent_runner_state", False)):
            # Match AsyncScheduler's accounting for the lookahead control
            # token. Without consuming the placeholder, vLLM's max-token guard
            # sees an ever-growing speculative tail and schedules one GR stage
            # beyond the device state's terminal step.
            request.num_output_placeholders -= len(new_token_ids)
            if request.num_output_placeholders < 0:
                raise RuntimeError("persistent GR output placeholder underflow")
        if status_before_update == RequestStatus.RUNNING:
            # Native control tokens may own allocated blocks, but only the
            # prompt prefix is valid reusable KV for a GR request.
            metadata: GRRequestMetadata = request.gr_metadata
            cache_tokens = min(request.num_computed_tokens, metadata.prefix_len)
            self.kv_cache_manager.cache_blocks(request, cache_tokens)
        return new_token_ids, stopped

    def _decorate_gr_output(self, output: SchedulerOutput) -> None:
        """Attach Worker metadata and advance the issue cursor for one batch."""
        existing_beam_data = getattr(output, "beam_data", None)
        beam_data: dict[str, dict[str, Any]] = (
            existing_beam_data if existing_beam_data is not None else {}
        )
        scheduled_gr = False
        for req_id, logical_tokens in tuple(output.num_scheduled_tokens.items()):
            request = self.requests.get(req_id)
            if request is None or not self._is_gr_request(request):
                continue
            scheduled_gr = True

            metadata: GRRequestMetadata = request.gr_metadata
            # Use the issue cursor, not the host completion cursor. With a
            # BatchQueue of depth two these intentionally differ by one.
            step = int(getattr(request, "gr_issued_steps", request.gr_step))
            persistent_runner_state = bool(getattr(request, "gr_persistent_runner_state", False))
            if step == 0:
                is_last_prefill = request.num_computed_tokens >= metadata.prefix_len
                beam_data[req_id] = {
                    "is_beam_decode": False,
                    "is_constraint_prefill": True,
                    "is_constrained": True,
                    "constraint_step": 0,
                    "beam_width": metadata.beam_width,
                    "beam_suffix": [],
                    "prefix_len": metadata.prefix_len,
                    "cache_len": request.num_computed_tokens - logical_tokens,
                    "decode_step": 0,
                    "max_decode_steps": metadata.max_tokens,
                    "persistent_request": True,
                    "persistent_runner_state": persistent_runner_state,
                    "continue_token_id": metadata.continue_token_id,
                    "finish_token_id": metadata.finish_token_id,
                    "is_last_prefill": is_last_prefill,
                    # Intermediate prefill chunks do not produce a GR token and
                    # therefore do not advance either lifecycle cursor.
                    "generation_index": 0 if is_last_prefill else None,
                }
                if is_last_prefill:
                    request.gr_issued_steps = 1
            else:
                if step >= metadata.max_tokens:
                    raise RuntimeError(
                        "persistent GR scheduled beyond max_tokens: "
                        f"issued={step}, max_tokens={metadata.max_tokens}"
                    )
                beam_data[req_id] = {
                    "is_beam_decode": True,
                    "is_constrained": True,
                    "constraint_step": step,
                    "beam_width": metadata.beam_width,
                    "beam_decode_steps": step,
                    "beam_suffix": ([] if persistent_runner_state else request.gr_beam_suffix),
                    "prefix_len": metadata.prefix_len,
                    "cache_len": metadata.prefix_len,
                    "decode_step": step - 1,
                    "parent_beam_ids": (None if persistent_runner_state else request.gr_parent_ids),
                    "max_decode_steps": metadata.max_tokens,
                    "result_width": metadata.num_return_sequences,
                    "persistent_request": True,
                    "persistent_runner_state": persistent_runner_state,
                    "continue_token_id": metadata.continue_token_id,
                    "finish_token_id": metadata.finish_token_id,
                    "generation_index": step,
                    "is_last_step": step + 1 >= metadata.max_tokens,
                }
                if not persistent_runner_state:
                    beam_data[req_id]["next_token_ids"] = [
                        suffix[-1] for suffix in request.gr_beam_suffix
                    ]
                # The scheduler advances one logical control token while the
                # worker executes W physical Beam rows. Runner-state requests
                # source their token ids directly from device state.
                output.num_scheduled_tokens[req_id] = metadata.beam_width
                output.total_num_scheduled_tokens += metadata.beam_width - logical_tokens
                request.gr_issued_steps = step + 1

            request.gr_counters.decode_scheduler_calls += int(step > 0)

        # An ordinary request must observe the upstream SchedulerOutput
        # unchanged; only GR scheduling introduces this sidecar.
        if scheduled_gr:
            output.beam_data = beam_data

    def schedule(self) -> SchedulerOutput:
        """Issue exactly one upstream SchedulerOutput and one GR stage."""
        output = super().schedule()
        self._decorate_gr_output(output)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        return self._update_single_output(scheduler_output, model_runner_output)

    def _update_single_output(self, scheduler_output, model_runner_output):
        beam_data = getattr(scheduler_output, "beam_data", {})
        logical_counts: dict[str, int] = {}
        for req_id, physical_tokens in tuple(scheduler_output.num_scheduled_tokens.items()):
            request = self.requests.get(req_id)
            if request is None or not self._is_gr_request(request):
                continue
            if bool(beam_data.get(req_id, {}).get("is_beam_decode", False)):
                logical_counts[req_id] = 1
                scheduler_output.num_scheduled_tokens[req_id] = 1
                scheduler_output.total_num_scheduled_tokens -= physical_tokens - 1

        decision = getattr(model_runner_output, "beam_search_result", None)
        scheduled_gr = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if self._is_gr_request(self.requests.get(req_id))
            and beam_data.get(req_id, {}).get("generation_index") is not None
        ]
        runner_state_enabled = any(
            bool(getattr(self.requests.get(req_id), "gr_persistent_runner_state", False))
            for req_id in scheduled_gr
        )
        if scheduled_gr and decision is None and not runner_state_enabled:
            raise RuntimeError("persistent GR worker output is missing beam_search_result")
        if decision is not None:
            for req_id in scheduler_output.num_scheduled_tokens:
                request = self.requests.get(req_id)
                if request is None or not self._is_gr_request(request):
                    continue
                if getattr(request, "gr_persistent_runner_state", False):
                    continue
                generation_index = int(beam_data[req_id]["generation_index"])
                completed_steps = int(getattr(request, "gr_completed_steps", request.gr_step))
                if generation_index != completed_steps:
                    raise RuntimeError(
                        "persistent GR output order mismatch: "
                        f"generation_index={generation_index}, completed={completed_steps}"
                    )
                unpacked = unpack_beam_result_buffer(decision)
                if unpacked is None:
                    raise RuntimeError("persistent GR requires a compact Beam decision")
                token_ids, _, parent_ids = unpacked
                width = request.gr_metadata.beam_width
                if len(token_ids) != width or len(parent_ids) != width:
                    raise RuntimeError(
                        "persistent GR Beam decision width does not match request metadata"
                    )
                old_suffix = request.gr_beam_suffix or [[]]
                request.gr_beam_suffix = [
                    list(old_suffix[parent]) + [int(token)]
                    for token, parent in zip(token_ids, parent_ids)
                ]
                request.gr_parent_ids = [int(parent) for parent in parent_ids]
                request.gr_completed_steps = generation_index + 1
                request.gr_step = request.gr_completed_steps

                req_index = model_runner_output.req_id_to_index[req_id]
                is_final = bool(decision.get("is_last_step"))
                counters: GRRequestCounters = request.gr_counters
                counters.decode_worker_dispatches += int(generation_index > 0)
                payload = decision.get("buffer")
                payload_bytes = (
                    len(payload) if isinstance(payload, (bytes, bytearray, memoryview)) else 0
                )
                if is_final:
                    counters.final_result_d2h_count += 1
                    counters.final_result_d2h_bytes += payload_bytes
                else:
                    counters.intermediate_beam_state_d2h_bytes += payload_bytes
                decision["counters"] = counters.to_dict()
                token_id = (
                    request.gr_metadata.finish_token_id
                    if is_final
                    else request.gr_metadata.continue_token_id
                )
                model_runner_output.sampled_token_ids[req_index] = [token_id]
                break

        if runner_state_enabled:
            for req_id in scheduled_gr:
                request = self.requests[req_id]
                generation_index = int(beam_data[req_id]["generation_index"])
                completed_steps = int(getattr(request, "gr_completed_steps", request.gr_step))
                if generation_index != completed_steps:
                    raise RuntimeError(
                        "persistent GR output order mismatch: "
                        f"generation_index={generation_index}, completed={completed_steps}"
                    )
                is_final_dispatch = generation_index + 1 >= request.gr_metadata.max_tokens
                if is_final_dispatch and decision is None:
                    raise RuntimeError(
                        "persistent GR final worker output is missing the final result"
                    )
                request.gr_completed_steps = generation_index + 1
                request.gr_step = request.gr_completed_steps
                is_final = bool(decision and decision.get("is_last_step"))
                if is_final != is_final_dispatch:
                    raise RuntimeError(
                        "persistent GR final marker does not match the scheduled generation"
                    )
                req_index = model_runner_output.req_id_to_index[req_id]
                model_runner_output.sampled_token_ids[req_index] = [
                    request.gr_metadata.finish_token_id
                    if is_final
                    else request.gr_metadata.continue_token_id
                ]
                counters: GRRequestCounters = request.gr_counters
                counters.decode_worker_dispatches += int(generation_index > 0)
                if is_final:
                    payload = decision.get("buffer")
                    counters.gr_state_created_per_session = int(decision.get("gr_state_created", 0))
                    counters.gr_state_freed_per_session = int(decision.get("gr_state_freed", 0))
                    counters.final_result_d2h_count = 1
                    counters.final_result_d2h_bytes = (
                        len(payload) if isinstance(payload, (bytes, bytearray, memoryview)) else 0
                    )
                    decision["counters"] = counters.to_dict()
                break

        result = super().update_from_output(scheduler_output, model_runner_output)

        # Preserve the worker decision for the existing frontend. Runner-state
        # requests publish this payload only on their final step.
        if decision is not None:
            # ``Scheduler.update_from_output`` removes a request that reaches
            # max_tokens. Keep the IDs captured before that call so a one-token
            # GR request, which finishes directly from prefill, still receives
            # its final worker result.
            beam_ids = set(scheduled_gr)
            for outputs in result.values():
                for output in outputs.outputs:
                    if output.request_id in beam_ids:
                        params = dict(output.kv_transfer_params or {})
                        params["beam_search_result"] = decision
                        output.kv_transfer_params = params
                        return result
        return result
