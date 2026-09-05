# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composite persistent GR backend for the NPU ModelRunner V1 boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm_gr.v1.worker.gr_model_runner_state import (
    GR_FINAL_OUTPUT_FORMAT,
    GRBackendStep,
)
from vllm_gr.v1.worker.npu_gr_masked_decision import (
    NPUMaskedBeamDecision,
    NPUMaskedDecisionStep,
)
from vllm_gr.v1.worker.npu_gr_sampler import (
    NPUConstrainedCandidates,
    NPUPersistentConstraintSampler,
)
from vllm_gr.v1.worker.npu_gr_workspace import (
    NPUFinalPack,
    NPUWorkspace,
    NPUWorkspaceLease,
)


@dataclass(slots=True)
class _NPURequestState:
    workspace: NPUWorkspace
    lease: NPUWorkspaceLease
    sampler: NPUPersistentConstraintSampler
    selector: NPUMaskedBeamDecision
    status_scratch: torch.Tensor
    host_final: torch.Tensor | None
    final_event: Any
    prefix_binding: Any = None
    kv: Any = None
    pending_candidates: NPUConstrainedCandidates | None = None
    pending_decision: NPUMaskedDecisionStep | None = None
    released: bool = False

    @property
    def device_memory_bytes(self) -> int:
        return self.workspace.device_memory_bytes + self.selector.device_memory_bytes


class NPUGRDeviceBackend:
    """Join NPU sampling, mask-aware decision, KV, control and final output."""

    def __init__(
        self,
        *,
        cpu_test_backend: bool = False,
        constraint_backend: Any = None,
        prefix_binder: Callable[[Any, Any], Any] | None = None,
        kv_factory: Callable[[Any], Any] | None = None,
        final_copy: Callable[[NPUFinalPack, _NPURequestState], bytes] | None = None,
        drain: Callable[[torch.device], None] | None = None,
    ) -> None:
        self._cpu_test_backend = cpu_test_backend
        self._constraint_backend = constraint_backend
        self._prefix_binder = prefix_binder
        self._kv_factory = kv_factory
        self._final_copy = final_copy
        self._drain = drain

    @staticmethod
    def _temperature(runner: Any, request_id: str) -> float:
        requests = getattr(runner, "requests", None)
        request = requests.get(request_id) if isinstance(requests, dict) else None
        params = getattr(request, "sampling_params", None)
        value = getattr(params, "temperature", None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("persistent NPU GR request temperature is unavailable")
        return float(value)

    @staticmethod
    def _result_width(metadata: dict[str, Any], beam_width: int) -> int:
        value = metadata.get("result_width")
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= beam_width:
            raise RuntimeError("persistent NPU GR result_width is invalid")
        return value

    def _new_request_state(
        self,
        runner: Any,
        state: Any,
        metadata: dict[str, Any],
    ) -> _NPURequestState:
        device = torch.device(getattr(runner, "device", "cpu"))
        if device.type != "npu" and not self._cpu_test_backend:
            raise RuntimeError(f"persistent NPU GR requires an NPU runner, got {device}")
        decision = getattr(getattr(state.session, "context", None), "decision", None)
        sequence = getattr(decision, "sequence", None)
        scores = getattr(decision, "beam_scores", None)
        if not torch.is_tensor(sequence) or not torch.is_tensor(scores):
            raise RuntimeError("persistent NPU GR canonical buffers are missing")
        width = state.beam_width
        if tuple(sequence.shape[:2]) != (1, width) or tuple(scores.shape) != (1, width):
            raise RuntimeError(
                "persistent NPU GR currently requires request width to equal fixed pool width"
            )
        capacity = int(sequence.shape[2])
        if capacity < state.max_tokens:
            raise RuntimeError("persistent NPU GR request exceeds fixed generation capacity")
        backend = self._constraint_backend or getattr(runner, "constraint_backend", None)
        if backend is None:
            raise RuntimeError("persistent NPU GR constraint Backend is unavailable")
        workspace = NPUWorkspace.allocate(
            beam_width=width,
            max_steps=capacity,
            device=device,
        )
        lease = workspace.acquire(
            state.request_id,
            generation_steps=state.max_tokens,
            result_width=self._result_width(metadata, width),
            canonical_sequence=sequence,
            canonical_scores=scores,
        )
        workspace.reset(lease)
        sampler = NPUPersistentConstraintSampler(
            state.request_id,
            width,
            state.max_tokens,
            self._temperature(runner, state.request_id),
            workspace.device,
            backend,
            cpu_test_backend=self._cpu_test_backend,
        )
        selector = NPUMaskedBeamDecision(
            beam_width=width,
            max_steps=capacity,
            device=workspace.device,
            cpu_test_backend=self._cpu_test_backend,
        )
        status_scratch = torch.empty((1,), dtype=torch.int32, device=workspace.device)
        if self._cpu_test_backend:
            host_final = None
            final_event = None
        else:
            host_final = torch.empty(
                workspace.final_staging_i32.shape,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            npu = getattr(torch, "npu", None)
            if npu is None:
                raise RuntimeError("persistent NPU GR torch.npu runtime is unavailable")
            final_event = npu.Event()
        result = _NPURequestState(
            workspace=workspace,
            lease=lease,
            sampler=sampler,
            selector=selector,
            status_scratch=status_scratch,
            host_final=host_final,
            final_event=final_event,
        )
        state.backend_state = result
        runner._gr_state_peak_bytes = max(
            int(getattr(runner, "_gr_state_peak_bytes", 0)),
            result.device_memory_bytes,
        )
        return result

    def sample(
        self,
        runner: Any,
        state: Any,
        logits: torch.Tensor,
        metadata: dict[str, Any],
    ) -> Any:
        backend_state = state.backend_state
        if not isinstance(backend_state, _NPURequestState) or backend_state.released:
            backend_state = self._new_request_state(runner, state, metadata)
        if backend_state.pending_candidates is not None:
            raise RuntimeError("persistent NPU GR candidates are already pending")
        update = getattr(
            getattr(runner, "input_batch", None),
            "update_async_output_token_ids",
            None,
        )
        if not callable(update):
            raise TypeError("persistent NPU GR input batch cannot update async tokens")
        update()
        result = backend_state.sampler.select(
            logits,
            state.session.context.decision.sequence,
            expected_device_step=state.device_step,
        )
        backend_state.pending_candidates = result
        return result.sampler_output

    @staticmethod
    def _take_candidates(
        backend_state: _NPURequestState,
        sampler_output: Any,
        expected_step: int,
    ) -> NPUConstrainedCandidates:
        candidates = backend_state.pending_candidates
        if (
            candidates is None
            or candidates.sampler_output is not sampler_output
            or candidates.device_step != expected_step
        ):
            raise RuntimeError("persistent NPU GR sampler/decision ownership mismatch")
        return candidates

    @staticmethod
    def _accumulate_sampler_status(
        backend_state: _NPURequestState,
        candidates: NPUConstrainedCandidates,
    ) -> None:
        torch.amax(
            candidates.status,
            dim=0,
            keepdim=True,
            out=backend_state.status_scratch,
        )
        torch.maximum(
            backend_state.workspace.status,
            backend_state.status_scratch,
            out=backend_state.workspace.status,
        )

    def bootstrap_from_prefill(
        self,
        runner: Any,
        state: Any,
        sampler_output: Any,
        metadata: dict[str, Any],
    ) -> GRBackendStep:
        backend_state = state.backend_state
        if not isinstance(backend_state, _NPURequestState) or backend_state.released:
            raise RuntimeError("persistent NPU GR backend state is not active")
        candidates = self._take_candidates(backend_state, sampler_output, 0)
        raw = candidates.constraint_output
        decision = state.session.context.decision
        sequence = decision.sequence
        scores = decision.beam_scores
        invalid = backend_state.workspace.active_mask
        torch.logical_not(raw.valid_mask, out=invalid)
        sequence[:, :, 0].copy_(raw.token_ids)
        sequence[:, :, 0].masked_fill_(invalid, 0)
        scores.copy_(raw.logprobs)
        scores.masked_fill_(invalid, float("-inf"))
        torch.gt(scores, float("-inf"), out=backend_state.workspace.active_mask)
        backend_state.workspace.active_mask.logical_and_(raw.valid_mask)
        decision.group_token_num.zero_()
        decision.worker_decode_steps = 1
        self._accumulate_sampler_status(backend_state, candidates)
        backend_state.workspace.record_bootstrap(
            backend_state.lease,
            scores,
            backend_state.workspace.active_mask,
            candidate_valid_mask=raw.valid_mask,
        )
        binder = self._prefix_binder
        if binder is None:
            from vllm_gr.v1.worker.npu_gr_metadata import bind_prefill_prefix

            binder = bind_prefill_prefix
        backend_state.prefix_binding = binder(runner, state)
        state.prefix_binding = backend_state.prefix_binding
        state.session.context._gr_persistent_prefix_binding = backend_state.prefix_binding
        if state.max_tokens > 1:
            if self._kv_factory is not None:
                backend_state.kv = self._kv_factory(state.session)
            else:
                from vllm_gr.v1.attention.backends.beam_attn import (
                    _get_npu_decode_execution,
                )
                from vllm_gr.v1.worker.npu_gr_kv import NPUKVReorder

                backend_state.kv = NPUKVReorder(
                    state.session,
                    mode=_get_npu_decode_execution(),
                )
        backend_state.sampler.consume(candidates)
        backend_state.pending_candidates = None
        control = backend_state.workspace.control_slot(backend_state.lease, device_step=1)
        control.fill_(state.finish_token_id if state.max_tokens == 1 else state.continue_token_id)
        return GRBackendStep(
            device_step=1,
            next_token_ids=sequence[:, :, 0],
            beam_scores=scores,
            active_mask=backend_state.workspace.active_mask,
            control_output=control,
        )

    def prepare_decode_inputs(
        self,
        runner: Any,
        state: Any,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
        metadata: dict[str, Any],
    ) -> None:
        del runner, scheduler_output, metadata
        backend_state = state.backend_state
        if not isinstance(backend_state, _NPURequestState) or backend_state.released:
            raise RuntimeError("persistent NPU GR backend state is not active")
        if len(num_scheduled_tokens) != 1 or int(num_scheduled_tokens[0]) != state.beam_width:
            raise RuntimeError("persistent NPU GR physical input width is invalid")

    def advance_beam(
        self,
        runner: Any,
        state: Any,
        sampler_output: Any,
        metadata: dict[str, Any],
        *,
        is_last_step: bool,
    ) -> GRBackendStep:
        del runner
        backend_state = state.backend_state
        if not isinstance(backend_state, _NPURequestState) or backend_state.released:
            raise RuntimeError("persistent NPU GR backend state is not active")
        candidates = self._take_candidates(backend_state, sampler_output, state.device_step)
        workspace = backend_state.workspace
        decision_buffers = state.session.context.decision
        workspace.before_decision(
            backend_state.lease,
            decision_buffers.beam_scores,
            expected_device_step=state.device_step,
        )
        decision = backend_state.selector.select(
            state.session,
            candidates.constraint_output.token_ids,
            candidates.constraint_output.logprobs,
            candidates.constraint_output.valid_mask,
            expected_device_step=state.device_step,
            is_last_step=is_last_step,
            result_width=(
                self._result_width(metadata, state.beam_width) if is_last_step else state.beam_width
            ),
        )
        if decision.parent_prefix is not None:
            decision_buffers.group_token_num.copy_(decision.parent_prefix)
        self._accumulate_sampler_status(backend_state, candidates)
        torch.maximum(workspace.status, decision.status, out=workspace.status)
        workspace.record_decision(
            backend_state.lease,
            decision.parent_ids,
            decision_buffers.beam_scores,
            decision.active_mask,
            expected_device_step=state.device_step,
            selected_valid_mask=decision.valid_mask,
            selected_local_scores=decision.local_scores,
        )
        backend_state.sampler.consume(candidates)
        backend_state.pending_candidates = None
        backend_state.pending_decision = decision
        control = workspace.control_slot(
            backend_state.lease,
            device_step=decision.device_step,
        )
        control.fill_(state.finish_token_id if is_last_step else state.continue_token_id)
        return GRBackendStep(
            device_step=decision.device_step,
            next_token_ids=decision.token_ids,
            beam_scores=decision.scores,
            active_mask=decision.active_mask,
            control_output=control,
        )

    def reorder_kv(
        self,
        runner: Any,
        state: Any,
        step_result: GRBackendStep,
        *,
        is_last_step: bool,
    ) -> None:
        del runner, step_result
        backend_state = state.backend_state
        if not isinstance(backend_state, _NPURequestState) or backend_state.released:
            raise RuntimeError("persistent NPU GR backend state is not active")
        decision = backend_state.pending_decision
        if decision is None or backend_state.kv is None:
            raise RuntimeError("persistent NPU GR KV decision is missing")
        backend_state.kv.reorder(
            state.session,
            decision,
            expected_device_step=state.device_step,
            is_final=is_last_step,
        )
        backend_state.pending_decision = None

    @staticmethod
    def _copy_final_production(
        pack: NPUFinalPack,
        backend_state: _NPURequestState,
    ) -> bytes:
        host = backend_state.host_final
        event = backend_state.final_event
        if host is None or event is None or not host.is_pinned():
            raise RuntimeError("persistent NPU GR final staging is not pinned/event-backed")
        words = 1 + pack.payload_words
        target = host[:words]
        target.copy_(pack.staging, non_blocking=True)
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise RuntimeError("persistent NPU GR torch.npu runtime is unavailable")
        event.record(npu.current_stream(pack.staging.device))
        event.synchronize()
        array = target.numpy()
        if int(array[0]) != 0:
            raise RuntimeError("persistent NPU GR constraint/decision status is nonzero")
        return array[1:].tobytes()

    def finalize_output(self, runner: Any, state: Any) -> dict[str, Any]:
        del runner
        backend_state = state.backend_state
        if not isinstance(backend_state, _NPURequestState) or backend_state.released:
            raise RuntimeError("persistent NPU GR backend state is not active")
        decision = state.session.context.decision
        pack = backend_state.workspace.pack_final(
            backend_state.lease,
            decision.sequence,
            decision.beam_scores,
        )
        copier = self._final_copy or self._copy_final_production
        payload = copier(pack, backend_state)
        if not isinstance(payload, bytes) or len(payload) != pack.payload_words * 4:
            raise RuntimeError("persistent NPU GR final copy returned an invalid payload")
        return {
            "request_id": state.request_id,
            "buffer_format": GR_FINAL_OUTPUT_FORMAT,
            "shape": list(pack.shape),
            "buffer": payload,
            "is_last_step": True,
            "step": state.step,
            "device_step": state.device_step,
        }

    def release(self, state: Any) -> None:
        backend_state = state.backend_state
        if not isinstance(backend_state, _NPURequestState):
            return
        if backend_state.released:
            return
        if self._drain is not None:
            self._drain(backend_state.workspace.device)
        elif not self._cpu_test_backend:
            npu = getattr(torch, "npu", None)
            if npu is None:
                raise RuntimeError("persistent NPU GR torch.npu runtime is unavailable")
            npu.synchronize(backend_state.workspace.device)
        if backend_state.kv is not None:
            backend_state.kv.release(state.session)
        backend_state.sampler.release(state.request_id)
        backend_state.workspace.release(backend_state.lease)
        backend_state.pending_candidates = None
        backend_state.pending_decision = None
        backend_state.prefix_binding = None
        state.prefix_binding = None
        if hasattr(state.session.context, "_gr_persistent_prefix_binding"):
            state.session.context._gr_persistent_prefix_binding = None
        backend_state.released = True


__all__ = ["NPUGRDeviceBackend"]
