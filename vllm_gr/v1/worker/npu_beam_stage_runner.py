# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initial eager NPU runtime for a fixed Beam Batch Session.

Part 1 intentionally stops after the first Decode forward. It establishes the
real Prefill -> device winner handoff -> B*W Decode execution path. Beam state
advance, KV reorder, terminal materialization, and retirement belong to the
following device-state PR.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from typing import Any

import numpy as np
import torch

from vllm_gr.constraints.backend import ConstraintMetadata
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.npu_execution_batch import NPUBeamExecutionBatch
from vllm_gr.v1.engine.beam_search_v1 import GRWorkerResult
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
)
from vllm_gr.v1.worker.npu_beam_stage_adapter import (
    NPUEagerFirstDecodeHandoff,
    NPUEagerStagePlan,
    validate_npu_eager_capabilities,
)


class NPUEagerBeamStageRunner:
    """Own the initial NPU Prefill and first-Decode execution contract."""

    def __init__(self, worker: Any) -> None:
        validate_npu_eager_capabilities(worker)
        self.worker = worker
        self.previous = None
        self.pending_plan: NPUEagerStagePlan | None = None
        self.execution_error: Exception | None = None
        self.logical_scheduler_output: Any = None
        self.handoff: NPUEagerFirstDecodeHandoff | None = None
        self.decode_candidates: Any = None
        self.logits: torch.Tensor | None = None
        self.attention_batch: NPUBeamExecutionBatch | None = None
        self.closed = False

    def _metadata(self, scheduler_output: Any) -> Any:
        mapping = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR)
        return next(iter(mapping.values()))

    def _validate_sequence(self, plan: NPUEagerStagePlan) -> None:
        current = plan.execution
        if self.previous is None:
            if current.kind != "prefill" or current.dispatch_id != 0:
                raise ValueError("NPU Batch Session must begin with dispatch 0 Prefill")
            return
        self.previous.assert_same_logical_batch(current)
        if current.dispatch_id != self.previous.dispatch_id + 1:
            raise ValueError("NPU Batch Session dispatch IDs must be contiguous")
        if current.kind != "decode" or current.decode_step != 0:
            raise NotImplementedError(
                "NPU Part 1 supports Prefill and Decode step zero only; "
                "Beam state advance is implemented in Part 2"
            )
        if self.previous.kind != "prefill" or self.handoff is None:
            raise RuntimeError("Decode step zero requires the Prefill winner handoff")

    def _ensure_attention_batch(self, plan: NPUEagerStagePlan) -> None:
        if self.attention_batch is not None:
            self.attention_batch.execution.assert_same_logical_batch(plan.execution)
            return
        from vllm_gr.v1.worker.model_runner_common import (
            _create_beam_context,
            _get_gr_config,
        )

        execution = plan.execution
        context = _create_beam_context(
            self.worker,
            _get_gr_config(self.worker),
            max_batch=execution.batch_size,
        )
        sessions = []
        for request_id, prompt_len in zip(
            execution.request_ids,
            execution.num_prompt_tokens_by_request,
        ):
            slot = context.registry.allocate(
                request_id,
                beam_width=execution.beam_width,
                prefix_len=prompt_len,
            )
            session = BeamSearchSession(request_id, slot, context)
            session.begin_session(execution.beam_width, prompt_len)
            sessions.append(session)
        self.attention_batch = NPUBeamExecutionBatch(execution, sessions)
        self.worker._beam_context = context
        self.worker._beam_sessions = {session.session_id: session for session in sessions}

    def _beam_data(self, plan: NPUEagerStagePlan, metadata: Any) -> dict[str, dict[str, Any]]:
        execution = plan.execution
        assert execution.decode_step == 0
        return {
            request_id: {
                "prefix_len": prompt_len,
                "cache_len": prompt_len,
                "max_decode_steps": metadata.beam_params.max_tokens,
                "is_beam_decode": True,
                "beam_width": execution.beam_width,
                "beam_decode_steps": 1,
                "parent_beam_ids": [0] * execution.beam_width,
                "decode_step": 0,
            }
            for request_id, prompt_len in zip(
                execution.request_ids,
                execution.num_prompt_tokens_by_request,
            )
        }

    @staticmethod
    def _runtime_scheduler_output(
        scheduler_output: Any,
        plan: NPUEagerStagePlan,
    ) -> Any:
        """Create physical forward counts; the NPU state hook retains logical counts."""
        if plan.execution.kind != "decode":
            return scheduler_output
        runtime = copy(scheduler_output)
        runtime.num_scheduled_tokens = {
            request_id: plan.execution.beam_width for request_id in plan.execution.request_ids
        }
        runtime.total_num_scheduled_tokens = plan.model_token_count
        return runtime

    def execute(
        self,
        scheduler_output: Any,
        original: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.execution_error is not None:
            raise RuntimeError(
                "NPU GR execution owner failed; retirement is required"
            ) from self.execution_error
        if self.closed or self.pending_plan is not None:
            raise RuntimeError("NPU GR sampling must consume the previous dispatch")
        plan = NPUEagerStagePlan.from_scheduler_output(scheduler_output)
        self._validate_sequence(plan)
        metadata = self._metadata(scheduler_output)
        if metadata.is_last_stage:
            raise NotImplementedError(
                "NPU Part 1 does not publish terminal Beam results; use a non-terminal "
                "Prefill/Decode-step-zero validation and add Part 2 before serving"
            )

        self.pending_plan = plan
        self.logical_scheduler_output = scheduler_output
        self.worker._gr_npu_stage_runner = self
        runtime_output = self._runtime_scheduler_output(scheduler_output, plan)
        self.worker._current_scheduler_output = runtime_output
        self.worker._gr_npu_stage_plan = plan
        beam_info = None
        token = None
        if plan.execution.kind == "decode":
            self._ensure_attention_batch(plan)
            assert self.attention_batch is not None
            beam_data = self._beam_data(plan, metadata)
            beam_info = {
                "beam_data": beam_data,
                "req_ids": [],
                "beam_session": self.attention_batch,
            }
            self.worker._current_beam_data = beam_data
            self.worker._current_beam_info = beam_info
            context = self.attention_batch.context
            context.decode_step_buf.fill_(0)
            context.decode_step_1based_buf.fill_(1)
            context.xllm_block_table_buf.copy_(
                torch.arange(
                    plan.execution.batch_size,
                    dtype=torch.int32,
                    device=self.worker.device,
                )
            )
            from vllm_gr.v1.beam.context import BEAM_DATA_VAR

            token = BEAM_DATA_VAR.set(beam_info)
        else:
            self.worker._current_beam_data = {}
            self.worker._current_beam_info = {"beam_data": {}, "req_ids": []}

        try:
            result = original(self.worker, runtime_output, *args, **kwargs)
            state = getattr(self.worker, "execute_model_state", None)
            logits = getattr(state, "logits", None)
            if logits is None:
                logits = getattr(result, "logits", None)
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("NPU GR stage did not publish model logits")
            self.logits = logits
            return result
        except Exception as error:
            self.execution_error = error
            self.pending_plan = None
            self.logits = None
            raise
        finally:
            if token is not None:
                from vllm_gr.v1.beam.context import BEAM_DATA_VAR

                BEAM_DATA_VAR.reset(token)

    def prepare_inputs(
        self,
        original: Callable[..., Any],
        scheduler_output: Any,
        num_scheduled_tokens: Any,
    ) -> tuple[Any, Any, int, Any]:
        plan = self.pending_plan
        if plan is None or scheduler_output is not self.worker._current_scheduler_output:
            return original(self.worker, scheduler_output, num_scheduled_tokens)

        execution = plan.execution
        prepared_output = scheduler_output
        prepared_counts = num_scheduled_tokens
        if execution.kind == "decode":
            physical = {request_id: execution.beam_width for request_id in execution.request_ids}
            prepared_output.num_scheduled_tokens = physical
            prepared_output.total_num_scheduled_tokens = plan.model_token_count
            if (
                not isinstance(num_scheduled_tokens, np.ndarray)
                or num_scheduled_tokens.shape != (execution.batch_size,)
                or num_scheduled_tokens.dtype != np.int32
            ):
                raise TypeError("NPU Decode counts must be an int32 ndarray of logical batch size")
            prepared_counts = np.full_like(num_scheduled_tokens, execution.beam_width)

        if self.worker._current_beam_info is not None:
            self.worker._current_beam_info["req_ids"] = list(execution.request_ids)
        out = original(self.worker, prepared_output, prepared_counts)
        logits_indices, spec_metadata, total, compressed = out
        if execution.kind == "decode" and total != plan.model_token_count:
            raise RuntimeError(
                "NPU Decode input preparation did not preserve the physical "
                f"B*W token count: {total} != {plan.model_token_count}"
            )

        from vllm_gr.v1.worker.model_runner_common import (
            InputBatchProxy,
            _get_positions_buffers,
            _materialize_logits_indices,
        )

        if execution.kind == "decode":
            if self.handoff is None:
                raise RuntimeError("Decode preparation has no Prefill winner handoff")
            input_ids = self.worker.input_ids.gpu
            positions, _ = _get_positions_buffers(self.worker)
            if positions is None:
                raise RuntimeError("NPU runner has no device position buffer")
            self.handoff.bind_first_decode(plan, input_ids, positions)
            original_batch = self.worker.input_batch
            self.worker.input_batch = InputBatchProxy(
                original_batch,
                list(execution.row_request_ids),
                list(execution.row_request_indices),
                list(plan.sampling_row_indices),
                req_ids_key=execution.request_ids,
            )
            total = plan.model_token_count
        logits_indices = _materialize_logits_indices(self.worker, list(plan.logits_indices))
        return logits_indices, spec_metadata, total, compressed

    def sample(self, grammar_output: Any = None) -> Any:
        if self.execution_error is not None:
            raise RuntimeError("NPU GR execute_model failed") from self.execution_error
        plan = self.pending_plan
        if grammar_output is not None or plan is None or self.logits is None:
            raise RuntimeError("NPU GR sample requires one pending stage and no grammar")
        metadata = ConstraintMetadata(
            prefix_token_ids=self._constraint_prefixes(plan),
            temperatures=torch.full(
                (plan.execution.num_execution_rows,),
                float(
                    self._metadata(self.worker._current_scheduler_output).beam_params.temperature
                ),
                dtype=torch.float32,
                device=self.logits.device,
            ),
            current_step=0 if plan.execution.kind == "prefill" else 1,
            top_k=plan.execution.beam_width,
        )
        backend = getattr(self.worker, "constraint_backend", None)
        if backend is None or not callable(getattr(backend, "select", None)):
            raise RuntimeError("NPU GR requires a loaded canonical constraint backend")
        candidates = backend.select(self.logits.contiguous(), metadata)
        expected = (
            plan.execution.num_execution_rows,
            plan.execution.beam_width,
        )
        if tuple(candidates.token_ids.shape) != expected:
            raise RuntimeError("NPU constraint backend returned an invalid candidate shape")
        if plan.execution.kind == "prefill":
            self.handoff = NPUEagerFirstDecodeHandoff.from_prefill(plan, candidates.token_ids)
        else:
            self.decode_candidates = candidates

        from vllm.v1.outputs import ModelRunnerOutput

        execution = plan.execution
        request_ids = list(execution.request_ids)
        output = ModelRunnerOutput(
            req_ids=request_ids,
            req_id_to_index={request_id: index for index, request_id in enumerate(request_ids)},
            sampled_token_ids=[[0] for _ in request_ids],
        )
        worker_result = GRWorkerResult(
            session_id=execution.session_id,
            dispatch_id=execution.dispatch_id,
            stage_index=execution.stage_index,
            produced_token_count=1,
            device_state=self._metadata(self.worker._current_scheduler_output).output_device_state,
            finished=False,
            owner_generation=execution.owner_generation,
        )
        setattr(output, GR_WORKER_RESULT_ATTR, {execution.session_id: worker_result})
        self.previous = execution
        self.pending_plan = None
        self.logical_scheduler_output = None
        self.logits = None
        self.worker.execute_model_state = None
        self.worker.kv_connector_output = None
        return output

    def _constraint_prefixes(self, plan: NPUEagerStagePlan) -> torch.Tensor:
        rows = plan.execution.num_execution_rows
        prefixes = torch.zeros((rows, 3), dtype=torch.int32, device=self.logits.device)
        if plan.execution.kind == "decode":
            if self.handoff is None:
                raise RuntimeError("Decode constraint selection has no winner handoff")
            prefixes[:, 0].copy_(self.handoff.winner_token_ids.reshape(-1))
        return prefixes


def get_npu_stage_runner(worker: Any, scheduler_output: Any) -> NPUEagerBeamStageRunner:
    """Return the exclusive local NPU Part 1 execution owner."""

    from vllm_gr.config import BeamExecutionMode
    from vllm_gr.config.execution import require_beam_execution_mode

    require_beam_execution_mode(worker.vllm_config, BeamExecutionMode.V1)
    existing = getattr(worker, "_gr_npu_stage_runner", None)
    if existing is not None and not existing.closed:
        return existing
    runner = NPUEagerBeamStageRunner(worker)
    worker._gr_npu_stage_runner = runner
    return runner


__all__ = ["NPUEagerBeamStageRunner", "get_npu_stage_runner"]
