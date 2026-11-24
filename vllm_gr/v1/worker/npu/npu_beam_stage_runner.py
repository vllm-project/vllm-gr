# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Persistent eager NPU runtime for a fixed Beam Batch Session.

Prefill stays logical B; Decode executes physical B*W rows. Candidate state,
request-local KV reorder, terminal materialization, and retirement stay owned
by this runner for the complete session.
"""

# Native eager-runner internals remain dynamically typed; the NPU worker mypy
# group still checks the module together with its input/bookkeeping callers.
# mypy: disable-error-code="index,misc,no-any-return,union-attr,var-annotated"

from __future__ import annotations

import math
from collections.abc import Callable
from copy import copy
from typing import Any

import numpy as np
import torch

from vllm_gr.config.models import ConstraintMode, configured_constraint_mode
from vllm_gr.constraints.backend import ConstraintMetadata, ConstraintOutput
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchOutput,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputSequence,
    BeamRequestOutput,
    BeamSequenceColumns,
)
from vllm_gr.v1.beam.dispatch import (
    GR_RETIRE_OUTPUT_ATTR,
    GR_RETIRE_RESULT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    GRRetireMetadata,
    GRRetireResult,
    GRStageMetadata,
    GRWorkerResult,
)
from vllm_gr.v1.beam.execution_batch import (
    BeamBatchExecutionView,
    BeamExecutionContractError,
)
from vllm_gr.v1.beam.session import BeamSearchSession
from vllm_gr.v1.worker.npu.beam_decision import run_beam_decision_candidates
from vllm_gr.v1.worker.npu.beam_execution_batch import NPUBeamExecutionBatch
from vllm_gr.v1.worker.npu.npu_beam_stage_adapter import (
    NPUEagerStagePlan,
    validate_npu_eager_capabilities,
)


class NPUEagerBeamStageRunner:
    """Own persistent eager NPU Beam execution for one fixed batch."""

    def __init__(self, worker: Any) -> None:
        validate_npu_eager_capabilities(worker)
        self.worker = worker
        self.previous: BeamBatchExecutionView | None = None
        self.pending_plan: NPUEagerStagePlan | None = None
        self.execution_error: Exception | None = None
        self.logical_scheduler_output: Any = None
        self.prefill_candidates: Any = None
        self.latest_token_ids: torch.Tensor | None = None
        self.parent_history: torch.Tensor | None = None
        self.session_request: BeamBatchRequest | None = None
        self._session_trie_ids: tuple[int, ...] | None = None
        self._trie_ids_by_width: dict[int, tuple[int, ...]] = {}
        self.decode_candidates: Any = None
        self.logits: torch.Tensor | None = None
        self.attention_batch: NPUBeamExecutionBatch | None = None
        self.dynamic_state: Any = None
        self.closed = False

    def _metadata(self, scheduler_output: Any) -> Any:
        mapping = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR)
        return next(iter(mapping.values()))

    def _uses_dynamic_state(self, plan: NPUEagerStagePlan) -> bool:
        metadata = self._metadata(self.worker._current_scheduler_output)
        output_options = getattr(metadata, "output_options", None)
        capacity = plan.execution.beam_width
        widths = plan.execution.dynamic_beam_widths
        # Preserve native fixed-width execution and its tie ordering. The
        # fused multi-step final selector requires a multiple of 32 results.
        native_schedule = widths is None or (
            all(width == capacity for width in widths) and (len(widths) == 1 or capacity % 32 == 0)
        )
        return (
            getattr(plan.execution, "per_parent_beam_widths", None) is not None
            or not native_schedule
            or getattr(output_options, "num_return_sequences", capacity) != capacity
        )

    def _needs_parent_history(self) -> bool:
        """Keep optional output lineage separate from required Beam/KV parent indices."""
        request = getattr(self, "session_request", None)
        return request is None or bool(request.output_options.include_parent_beam_indices)

    def _validate_sequence(self, plan: NPUEagerStagePlan) -> None:
        current = plan.execution
        if self.previous is None:
            if current.kind != "prefill" or current.dispatch_id != 0:
                raise ValueError("NPU Batch Session must begin with dispatch 0 Prefill")
            return
        self.previous.assert_same_logical_batch(current)
        if current.dispatch_id != self.previous.dispatch_id + 1:
            raise ValueError("NPU Batch Session dispatch IDs must be contiguous")
        if current.kind != "decode":
            raise ValueError("NPU Batch Session may Prefill only once")
        expected_step = 0 if self.previous.kind == "prefill" else int(self.previous.decode_step) + 1
        if current.decode_step != expected_step:
            raise ValueError("NPU Batch Session Decode steps must be contiguous")
        if self.latest_token_ids is None:
            raise RuntimeError("Decode requires a device-resident previous winner")

    def _ensure_attention_batch(self, plan: NPUEagerStagePlan) -> None:
        if self.attention_batch is not None:
            self.attention_batch.execution.assert_same_logical_batch(plan.execution)
            if plan.execution.dynamic_beam_widths is not None:
                self.attention_batch = NPUBeamExecutionBatch(
                    plan.execution, self.attention_batch.sessions
                )
            return
        from vllm_gr.v1.worker.gr_context import (
            _create_beam_context,
            _get_gr_config,
        )

        execution = plan.execution
        if self.prefill_candidates is None:
            raise RuntimeError("Decode state requires Prefill candidates")
        config = _get_gr_config(self.worker)
        required_steps = self._metadata(
            self.worker._current_scheduler_output
        ).beam_params.max_tokens
        shape = (execution.batch_size, execution.input_width)
        captured_shapes = getattr(
            self.worker,
            "_gr_npu_beam_graph_shapes",
            set(),
        )
        graph_context = getattr(self.worker, "_gr_npu_graph_context", None)
        use_graph_context = shape in captured_shapes or (
            execution.dynamic_beam_widths is not None
            and graph_context is not None
            and graph_context.max_batch == execution.batch_size
        )
        if use_graph_context:
            if (
                graph_context is None
                or graph_context.max_batch != execution.batch_size
                or graph_context.max_beams != execution.beam_width
            ):
                raise RuntimeError("captured NPU Beam graph has no matching fixed-address context")
            context = graph_context
        else:
            context = getattr(self.worker, "_gr_npu_eager_context", None)
            if (
                context is None
                or context.max_batch != execution.batch_size
                or context.max_beams != execution.beam_width
            ):
                if context is not None and context.registry.num_active_sessions:
                    raise RuntimeError("cannot replace an active NPU Beam eager fallback context")
                context = _create_beam_context(
                    self.worker,
                    config,
                    max_batch=execution.batch_size,
                )
                if getattr(
                    config, "beam_graph_widths", None
                ) is not None or self._uses_dynamic_state(plan):
                    from vllm_gr.v1.beam.dynamic_width import (
                        initialize_packed_stage_pools,
                    )

                    initialize_packed_stage_pools(context)
                self.worker._gr_npu_eager_context = context
        if required_steps > context.decision.sequence.shape[2]:
            raise ValueError("NPU Beam history exceeds the configured Decode capacity")
        if context.registry.num_active_sessions:
            # begin_session resets the shared decision buffers, so an NPU
            # batch must own the complete context before initializing slots.
            raise RuntimeError("cannot initialize an NPU Beam batch in an active attention context")
        if self._uses_dynamic_state(plan):
            from vllm_gr.v1.beam.dynamic_width import prune_stage_pools

            widths = execution.dynamic_beam_widths or (execution.beam_width,) * required_steps
            keep_widths = set(widths[:-1])
            if context is graph_context:
                keep_widths.update(
                    width for batch, width in captured_shapes if batch == execution.batch_size
                )
            prune_stage_pools(context, keep_widths)
        sessions = []
        allocated_request_ids = []
        dynamic_state = None
        parent_history = None
        try:
            for request_id, prompt_len in zip(
                execution.request_ids,
                execution.num_prompt_tokens_by_request,
            ):
                slot = context.registry.allocate(
                    request_id,
                    beam_width=execution.beam_width,
                    prefix_len=prompt_len,
                )
                # The slot already belongs to this attempt even if constructing
                # or beginning the session raises before it enters sessions.
                allocated_request_ids.append(request_id)
                session = BeamSearchSession(request_id, slot, context)
                session.begin_session(execution.beam_width, prompt_len)
                sessions.append(session)
            attention_batch = NPUBeamExecutionBatch(execution, sessions)
            if self._uses_dynamic_state(plan):
                from vllm_gr.v1.beam.dynamic_width import DynamicBeamState

                if self.previous is None or self.previous.kind != "prefill":
                    raise RuntimeError("dynamic NPU Beam state requires the preceding Prefill")
                dynamic_state = DynamicBeamState(context, execution.batch_size, required_steps)
                dynamic_state.initialize(self.prefill_candidates, self.previous.output_width)
            else:
                width = execution.beam_width
                candidates = self.prefill_candidates
                initial_scores = candidates.logprobs.masked_fill(
                    ~candidates.valid_mask, float("-inf")
                )
                # Candidates carry K columns while the decision state is W-wide, and the
                # two are independent.  The candidates are score-ordered, so the first
                # min(W, K) are the survivors; beams beyond that stay inactive with
                # -inf scores (0 token id, since Prefill parents are zero by definition).
                candidate_width = candidates.token_ids.shape[1]
                live = min(width, candidate_width)
                sequence = context.decision.sequence[: execution.batch_size, :width, 0]
                scores = context.decision.beam_scores[: execution.batch_size, :width]
                sequence[:, :live].copy_(candidates.token_ids[:, :live])
                scores[:, :live].copy_(initial_scores[:, :live])
                if live < width:
                    sequence[:, live:].zero_()
                    scores[:, live:].fill_(float("-inf"))
                context.decision.worker_decode_steps = 1
                parent_history = (
                    torch.zeros_like(context.decision.sequence)
                    if self._needs_parent_history()
                    else None
                )
            beam_sessions = {session.session_id: session for session in sessions}
        except Exception:
            # Registry release is host-only and also handles a begin_session
            # that failed before its facade could be published.
            for request_id in allocated_request_ids:
                context.registry.release(request_id)
            raise
        self.attention_batch = attention_batch
        self.dynamic_state = dynamic_state
        self.parent_history = parent_history
        self.worker._beam_context = context
        self.worker._beam_sessions = beam_sessions

    def _beam_data(self, plan: NPUEagerStagePlan, metadata: Any) -> dict[str, dict[str, Any]]:
        execution = plan.execution
        step = execution.decode_step
        assert step is not None
        return {
            request_id: {
                "prefix_len": prompt_len,
                "cache_len": prompt_len,
                "max_decode_steps": metadata.beam_params.max_tokens,
                "is_beam_decode": True,
                "beam_width": execution.input_width,
                "beam_decode_steps": step + 1,
                "parent_beam_ids": [0] * execution.input_width,
                "decode_step": step,
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
            request_id: plan.execution.input_width for request_id in plan.execution.request_ids
        }
        runtime.total_num_scheduled_tokens = plan.model_token_count
        return runtime

    @torch.inference_mode()
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
        session_init = getattr(metadata, "session_init", None)
        if session_init is not None:
            if plan.execution.kind != "prefill" or self.session_request is not None:
                raise ValueError("NPU session_init must appear once on Prefill")
        if plan.execution.kind == "prefill":
            trie_ids = self._validate_session_trie_ids(plan, session_init)
            self._session_trie_ids = trie_ids
            self._trie_ids_by_width = {}
        if session_init is not None:
            self.session_request = session_init

        self.pending_plan = plan
        self.logical_scheduler_output = scheduler_output
        self.worker._gr_npu_stage_runner = self
        token = None
        try:
            runtime_output = self._runtime_scheduler_output(scheduler_output, plan)
            self.worker._current_scheduler_output = runtime_output
            self.worker._gr_npu_stage_plan = plan
            if plan.execution.kind == "decode":
                self._ensure_attention_batch(plan)
                assert self.attention_batch is not None
                for session in self.attention_batch.sessions:
                    session._state.is_last_step = bool(metadata.is_last_stage)
                    if self._uses_dynamic_state(plan):
                        session._state.decode_step = plan.execution.decode_step
                beam_data = self._beam_data(plan, metadata)
                beam_info = {
                    "beam_data": beam_data,
                    "req_ids": [],
                    "beam_session": self.attention_batch,
                }
                self.worker._current_beam_data = beam_data
                self.worker._current_beam_info = beam_info
                context = self.attention_batch.context
                step = plan.execution.decode_step
                assert step is not None
                context.decode_step_buf.fill_(step)
                context.decode_step_1based_buf.fill_(step + 1)
                context.xllm_block_table_buf.copy_(
                    torch.arange(
                        plan.execution.batch_size,
                        dtype=torch.int32,
                        device=self.worker.device,
                    )
                )
                from vllm_gr.v1.beam import BEAM_DATA_VAR

                token = BEAM_DATA_VAR.set(beam_info)
            else:
                self.worker._current_beam_data = {}
                self.worker._current_beam_info = {"beam_data": {}, "req_ids": []}

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
                from vllm_gr.v1.beam import BEAM_DATA_VAR

                BEAM_DATA_VAR.reset(token)

    def prepare_inputs(
        self,
        original: Callable[..., Any],
        scheduler_output: Any,
        num_scheduled_tokens: Any,
    ) -> tuple[Any, Any, int]:
        plan = self.pending_plan
        if plan is None or scheduler_output is not self.worker._current_scheduler_output:
            return original(self.worker, scheduler_output, num_scheduled_tokens)

        execution = plan.execution
        prepared_output = scheduler_output
        prepared_counts = num_scheduled_tokens
        if execution.kind == "decode":
            physical = {request_id: execution.input_width for request_id in execution.request_ids}
            prepared_output.num_scheduled_tokens = physical
            prepared_output.total_num_scheduled_tokens = plan.model_token_count
            if (
                not isinstance(num_scheduled_tokens, np.ndarray)
                or num_scheduled_tokens.shape != (execution.batch_size,)
                or num_scheduled_tokens.dtype != np.int32
            ):
                raise TypeError("NPU Decode counts must be an int32 ndarray of logical batch size")
            prepared_counts = np.full_like(num_scheduled_tokens, execution.input_width)

        if self.worker._current_beam_info is not None:
            self.worker._current_beam_info["req_ids"] = list(execution.request_ids)
        out = original(self.worker, prepared_output, prepared_counts)
        logits_indices, spec_metadata, total = out
        if execution.kind == "decode" and total != plan.model_token_count:
            raise RuntimeError(
                "NPU Decode input preparation did not preserve the physical "
                f"B*W token count: {total} != {plan.model_token_count}"
            )

        from vllm_gr.v1.worker.npu.npu_model_runner_inputs import (
            InputBatchProxy,
            _get_positions_buffers,
            _materialize_logits_indices,
        )

        if execution.kind == "decode":
            if self.latest_token_ids is None:
                raise RuntimeError("Decode preparation has no previous Beam winners")
            input_ids = self.worker.input_ids.gpu
            positions, _ = _get_positions_buffers(self.worker)
            if positions is None:
                raise RuntimeError("NPU runner has no device position buffer")
            plan.prepare_decode_inputs(input_ids, positions, self.latest_token_ids)
            original_batch = self.worker.input_batch
            self.worker.input_batch = InputBatchProxy(
                original_batch,
                list(execution.row_request_ids),
                list(execution.row_request_indices),
                list(plan.sampling_row_indices),
                device=positions.device,
                req_ids_key=execution.request_ids,
            )
            total = plan.model_token_count
        logits_indices = _materialize_logits_indices(self.worker, list(plan.logits_indices))
        return logits_indices, spec_metadata, total

    @torch.inference_mode()
    def sample(self, grammar_output: Any = None) -> Any:
        if self.execution_error is not None:
            raise RuntimeError("NPU GR execute_model failed") from self.execution_error
        plan = self.pending_plan
        if grammar_output is not None or plan is None or self.logits is None:
            raise RuntimeError("NPU GR sample requires one pending stage and no grammar")
        from vllm_gr.v1.worker.gr_context import (
            _get_gr_config,
        )

        stage_metadata = self._metadata(self.worker._current_scheduler_output)
        if configured_constraint_mode(_get_gr_config(self.worker)) is ConstraintMode.NONE:
            candidates = self._select_unconstrained_candidates(
                self.logits.contiguous(), stage_metadata
            )
        else:
            metadata = ConstraintMetadata(
                prefix_token_ids=self._constraint_prefixes(plan),
                temperatures=self._constraint_temperatures(),
                current_step=plan.execution.stage_index,
                top_k=plan.execution.candidate_top_k,
            )
            backend = getattr(self.worker, "constraint_backend", None)
            if backend is None or not callable(getattr(backend, "select", None)):
                raise RuntimeError("NPU GR requires a loaded canonical constraint backend")
            candidates = self._select_constraint_candidates(
                backend,
                self.logits.contiguous(),
                metadata,
                trie_ids=self._constraint_trie_ids(plan),
            )
        expected = (
            plan.execution.num_execution_rows,
            plan.execution.candidate_top_k,
        )
        if tuple(candidates.token_ids.shape) != expected:
            raise RuntimeError("NPU candidate selection returned an invalid candidate shape")
        is_last = bool(stage_metadata.is_last_stage)
        terminal_result = None
        execution = plan.execution
        if execution.kind == "prefill":
            self.prefill_candidates = candidates
            expected = (execution.batch_size, execution.output_width)
            if tuple(candidates.token_ids.shape) != expected:
                raise BeamExecutionContractError(
                    f"Prefill winner_token_ids must have shape {expected}"
                )
            self.latest_token_ids = candidates.token_ids
            if is_last:
                # Filter and rank all finite candidates before the return limit.
                parents = None
                if self._needs_parent_history():
                    parents = torch.zeros(
                        (execution.batch_size, execution.output_width, 1),
                        dtype=torch.int32,
                        device=candidates.token_ids.device,
                    )
                terminal_result = self._build_terminal_result(
                    candidates.token_ids.unsqueeze(2),
                    candidates.logprobs.masked_fill(~candidates.valid_mask, float("-inf")),
                    parents,
                )
        else:
            self.decode_candidates = candidates
            if self.attention_batch is None:
                raise RuntimeError("Decode decision has no persistent attention batch")
            if self._uses_dynamic_state(plan):
                if self.dynamic_state is None:
                    raise RuntimeError("dynamic NPU Beam decision has no state")
                decision = self.dynamic_state.select(
                    candidates,
                    execution.input_width,
                    execution.output_width,
                    execution.stage_index,
                    is_last=is_last,
                    result_width=stage_metadata.output_options.num_return_sequences
                    if is_last
                    else execution.output_width,
                )
                if self._needs_parent_history():
                    self.parent_history = decision["out_parent_history"]
            else:
                decision = run_beam_decision_candidates(
                    self.worker,
                    self.attention_batch,
                    candidates.token_ids,
                    candidates.logprobs.masked_fill(~candidates.valid_mask, float("-inf")),
                    is_last_step=is_last,
                    result_width=execution.beam_width,
                )
            self.latest_token_ids = decision["out_token_ids"]
            if not self._uses_dynamic_state(plan) and self._needs_parent_history():
                self._advance_parent_history(decision, execution.stage_index)
            if is_last:
                parents = None
                if self._needs_parent_history():
                    if self.parent_history is None:
                        raise RuntimeError("NPU Beam parent history is not initialized")
                    parents = self.parent_history[:, : decision["out_sequence"].shape[1]]
                terminal_result = self._build_terminal_result(
                    decision["out_sequence"],
                    decision["out_log_probs"],
                    parents,
                )
        if is_last:
            if terminal_result is None:
                raise RuntimeError("terminal NPU Beam stage produced no result")
            self._release()

        from vllm.v1.outputs import ModelRunnerOutput

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
            finished=is_last,
            terminal_result=terminal_result,
            resources_released=is_last,
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

    def _select_unconstrained_candidates(
        self,
        logits: torch.Tensor,
        metadata: GRStageMetadata,
    ) -> ConstraintOutput:
        """Select full-vocabulary candidates without catalog or prefix state."""
        from vllm_gr.ops.npu.unconstrained_topk import select_unconstrained_candidates

        plan = self.pending_plan
        if plan is None:
            raise RuntimeError("NPU candidate selection requires a pending stage")
        temperatures = torch.full(
            (plan.execution.num_execution_rows,),
            float(metadata.beam_params.temperature),
            dtype=torch.float32,
            device=logits.device,
        )
        from vllm_gr.v1.worker.gr_context import (
            _get_gr_config,
        )

        if getattr(_get_gr_config(self.worker), "beam_sid_stage_vocab", False):
            from vllm_gr.ops.npu.sid_stage_vocab import select_sid_stage_candidates

            stage_vocab = getattr(self.worker, "_gr_npu_sid_stage_vocab", None)
            if stage_vocab is None:
                raise RuntimeError("NPU SID stage selection requires loaded tokenizer stage maps")
            compact = logits.shape[-1] != stage_vocab.full_vocab_size
            if compact:
                projection = getattr(self.worker, "_gr_npu_stage_projection", None)
                if (
                    projection is None
                    or projection.closed
                    or logits.shape[-1] != stage_vocab.vocab_size
                ):
                    raise RuntimeError(
                        "NPU SID stage selection received unsupported compact logits"
                    )
            return select_sid_stage_candidates(
                logits,
                temperatures,
                plan.execution.candidate_top_k,
                plan.execution.stage_index,
                stage_vocab,
                compact_logits=compact,
            )
        return select_unconstrained_candidates(logits, temperatures, plan.execution.candidate_top_k)

    def _select_constraint_candidates(
        self,
        backend: Any,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
        *,
        trie_ids: tuple[int, ...] | None = None,
    ) -> ConstraintOutput:
        """Select candidates within one Backend call's row capacity.

        The canonical NPU Backend admits the supported B*W Decode rows in one
        call. Backends with a smaller capacity still use contiguous chunks
        and restore the request-major result with concatenation.
        """
        max_rows = int(backend.info.max_rows)
        if max_rows < 1:
            raise RuntimeError(f"NPU constraint backend has invalid max_rows={max_rows}")
        if trie_ids is not None and len(trie_ids) != logits.shape[0]:
            raise RuntimeError("NPU Trie selectors must match physical candidate rows")

        select_for_rows: (
            Callable[[torch.Tensor, ConstraintMetadata, tuple[int, ...]], ConstraintOutput] | None
        ) = None
        if trie_ids is not None:
            host_selector = getattr(backend, "select_for_rows", None)
            if not callable(host_selector):
                raise RuntimeError("NPU multi-trie selection requires host-row routing")
            select_for_rows = host_selector
        if logits.shape[0] <= max_rows:
            if trie_ids is None:
                return backend.select(logits, metadata)
            assert select_for_rows is not None
            return select_for_rows(logits, metadata, trie_ids)

        outputs: list[ConstraintOutput] = []
        for start in range(0, logits.shape[0], max_rows):
            stop = min(start + max_rows, logits.shape[0])
            temperatures = metadata.temperatures
            if temperatures.numel() != 1:
                temperatures = temperatures[start:stop]
            chunk_metadata = ConstraintMetadata(
                prefix_token_ids=metadata.prefix_token_ids[start:stop],
                temperatures=temperatures,
                current_step=metadata.current_step,
                top_k=metadata.top_k,
                trie_ids=None if metadata.trie_ids is None else metadata.trie_ids[start:stop],
            )
            chunk_logits = logits[start:stop].contiguous()
            if trie_ids is None:
                outputs.append(backend.select(chunk_logits, chunk_metadata))
            else:
                assert select_for_rows is not None
                outputs.append(select_for_rows(chunk_logits, chunk_metadata, trie_ids[start:stop]))

        return ConstraintOutput(
            token_ids=torch.cat([output.token_ids for output in outputs], dim=0),
            logprobs=torch.cat([output.logprobs for output in outputs], dim=0),
            valid_mask=torch.cat([output.valid_mask for output in outputs], dim=0),
        )

    def _advance_parent_history(self, decision: dict[str, torch.Tensor], stage_index: int) -> None:
        if self.parent_history is None or self.attention_batch is None:
            raise RuntimeError("NPU Beam parent history is not initialized")
        width = self.attention_batch.beam_width
        batch = self.attention_batch.batch_size
        parents = decision["out_token_index"][:, :width].to(torch.int64)
        offsets = torch.arange(batch, dtype=torch.int64, device=parents.device).unsqueeze(1) * width
        local_parents = parents - offsets
        gather_index = local_parents.unsqueeze(2).expand_as(self.parent_history)
        history = self.parent_history.gather(1, gather_index)
        history[:, :, stage_index].copy_(local_parents.to(torch.int32))
        self.parent_history = history

    def _build_terminal_result(
        self,
        sequences: torch.Tensor,
        scores: torch.Tensor,
        parents: torch.Tensor | None,
    ) -> BeamBatchResult:
        request = self.session_request
        if request is None:
            raise RuntimeError("terminal NPU Beam output is missing session_init")
        if tuple(item.request.request_id for item in request.items) != tuple(
            self.pending_plan.execution.request_ids
        ):
            raise RuntimeError("terminal NPU Beam request order changed")

        options = request.output_options
        include_parents = options.include_parent_beam_indices
        if include_parents and parents is None:
            raise RuntimeError("terminal NPU Beam output is missing requested parent history")
        batch, width, steps = sequences.shape
        if tuple(scores.shape) != (batch, width) or (
            include_parents
            and parents is not None
            and (
                parents.ndim != 3
                or tuple(parents.shape[:2]) != (batch, width)
                or parents.shape[2] < steps
            )
        ):
            raise RuntimeError("terminal NPU Beam tensors have inconsistent shapes")
        history_words = batch * width * steps
        score_words = batch * width
        pieces = [sequences.to(torch.int32).contiguous().reshape(-1)]
        if include_parents and parents is not None:
            pieces.append(parents[:, :, :steps].to(torch.int32).contiguous().reshape(-1))
        pieces.append(scores.to(torch.float32).contiguous().reshape(-1).view(torch.int32))
        packed = torch.cat(pieces).to("cpu")
        params = request.beam_params
        stop_ids = set(params.stop_token_ids) if not params.ignore_stop_tokens else set()
        # Reuse the existing wire protocol after the existing packed D2H. Small
        # results and stop-aware histories keep the row representation. Public
        # clients still materialize every sequence before returning to callers.
        use_columns = (
            min(width, options.num_return_sequences) >= 64
            and not stop_ids
            and steps > 0
            and steps == getattr(params, "max_tokens", None)
        )
        sequence_view = packed[:history_words].view(batch, width, steps)
        sequence_rows = None if use_columns else sequence_view.tolist()
        parent_rows = None
        parent_view = None
        score_offset = history_words
        if include_parents:
            parent_view = packed[history_words : 2 * history_words].view(batch, width, steps)
            parent_rows = None if use_columns else parent_view.tolist()
            score_offset += history_words
        score_view = (
            packed[score_offset : score_offset + score_words].view(torch.float32).view(batch, width)
        )
        score_rows = score_view.tolist()
        outputs = []
        for row, item in enumerate(request.items):
            ranked = []
            prompt_length = (
                len(item.request.prompt_token_ids)
                if item.request.prompt_token_ids is not None
                else 0
            )
            normalization = None
            normalizations = {}
            for beam, score in enumerate(score_rows[row]):
                if not math.isfinite(score):
                    continue
                if stop_ids:
                    token_ids = sequence_rows[row][beam]
                    stop_index = next(
                        (index for index, token in enumerate(token_ids) if token in stop_ids),
                        None,
                    )
                    length = len(token_ids) if stop_index is None else stop_index + 1
                    normalization = normalizations.get(length)
                    if normalization is None:
                        normalization = (
                            float(prompt_length + length) ** params.length_penalty
                            if params.length_penalty != 0
                            else 1.0
                        )
                        normalizations[length] = normalization
                else:
                    stop_index = None
                    length = steps
                    # Evaluate lazily: an all-invalid row must still report
                    # no_valid_beams even if this exponent would overflow.
                    if normalization is None:
                        normalization = (
                            float(prompt_length + length) ** params.length_penalty
                            if params.length_penalty != 0
                            else 1.0
                        )
                ranked.append(
                    (
                        score / normalization,
                        score,
                        beam,
                        length,
                        stop_index,
                    )
                )
            ranked.sort(key=lambda value: (value[0], value[1], -value[2]), reverse=True)
            if use_columns and ranked:
                selected_indices = [value[2] for value in ranked[: options.num_return_sequences]]
                outputs.append(
                    BeamRequestOutput(
                        item.item_index,
                        item.request.request_id,
                        BeamSequenceColumns(
                            count=len(selected_indices),
                            steps=steps,
                            token_ids=sequence_view.numpy()[row, selected_indices]
                            .astype("<i4", copy=False)
                            .tobytes(),
                            cumulative_logprobs=score_view.numpy()[row, selected_indices]
                            .astype("<f4", copy=False)
                            .tobytes()
                            if options.include_cumulative_logprob
                            else None,
                            parent_beam_indices=parent_view.numpy()[row, selected_indices]
                            .astype("<i4", copy=False)
                            .tobytes()
                            if parent_view is not None
                            else None,
                        ),
                    )
                )
                continue
            selected = []
            for _, score, beam, length, stop_index in ranked[: options.num_return_sequences]:
                token_ids = sequence_rows[row][beam]
                finish_reason = "length" if stop_index is None else "stop"
                stop_token_id = None if stop_index is None else token_ids[stop_index]
                token_count = length
                if finish_reason == "stop" and not options.include_stop_token:
                    token_count -= 1
                selected.append(
                    BeamOutputSequence(
                        tuple(token_ids[:token_count]),
                        score if options.include_cumulative_logprob else None,
                        tuple(parent_rows[row][beam][:token_count])
                        if options.include_parent_beam_indices
                        else (),
                        finish_reason,
                        stop_token_id,
                    )
                )
            if not selected:
                return BeamBatchResult(
                    BEAM_BATCH_PROTOCOL_VERSION,
                    request.batch_id,
                    None,
                    BeamBatchFailure(
                        "no_valid_beams",
                        "No valid final Beam candidates",
                        item.item_index,
                    ),
                )
            outputs.append(
                BeamRequestOutput(
                    item.item_index,
                    item.request.request_id,
                    tuple(selected),
                )
            )
        return BeamBatchResult(
            BEAM_BATCH_PROTOCOL_VERSION,
            request.batch_id,
            BeamBatchOutput(tuple(outputs)),
            None,
        )

    def _release(self) -> None:
        if self.attention_batch is not None:
            request_ids = tuple(session.session_id for session in self.attention_batch.sessions)
            self.attention_batch.end_session()
            sessions = getattr(self.worker, "_beam_sessions", {})
            for request_id in request_ids:
                sessions.pop(request_id, None)
        self.closed = True
        self.parent_history = None
        self._session_trie_ids = None
        self._trie_ids_by_width = {}
        self.logical_scheduler_output = None
        self.execution_error = None

    def _validate_session_trie_ids(
        self, plan: NPUEagerStagePlan, request: BeamBatchRequest | None
    ) -> tuple[int, ...] | None:
        """Validate immutable request selectors before retaining session state."""
        backend = getattr(self.worker, "constraint_backend", None)
        trie_count = backend.info.table_metadata.trie_count if backend is not None else 1
        if type(trie_count) is not int or trie_count < 1:
            raise RuntimeError("NPU constraint backend has an invalid live Trie count")
        if request is None:
            if trie_count > 1:
                raise ValueError("NPU multi-trie execution requires the initial batch request")
            return None
        execution = plan.execution
        execution.validate_request_order(tuple(item.request.request_id for item in request.items))
        roots = tuple(item.trie_id for item in request.items)
        if any(type(root) is not int or not 0 <= root < trie_count for root in roots):
            raise ValueError("trie_id is outside the loaded constraint table")
        # A single Trie needs neither per-row selectors nor hot-path dispatch.
        return roots if trie_count > 1 else None

    def _constraint_trie_ids(self, plan: NPUEagerStagePlan) -> tuple[int, ...] | None:
        roots = getattr(self, "_session_trie_ids", None)
        if roots is None:
            return None
        width = plan.execution.input_width
        expanded = self._trie_ids_by_width.get(width)
        if expanded is None:
            expanded = plan.execution.expand_per_request(roots, name="trie_ids")
            self._trie_ids_by_width[width] = expanded
        return expanded

    def _constraint_prefixes(self, plan: NPUEagerStagePlan) -> torch.Tensor:
        rows = plan.execution.num_execution_rows
        # Each Worker owns one exclusive persistent NPU runner. Separate the
        # immutable Prefill zeros from Decode scratch so repeated requests do
        # not reallocate or clear inactive prefix columns at each stage.
        cache_name = (
            "_gr_npu_constraint_prefill_prefixes"
            if plan.execution.kind == "prefill"
            else "_gr_npu_constraint_decode_prefixes"
        )
        cached = getattr(self.worker, cache_name, None)
        if cached is None or cached.device != self.logits.device or cached.shape[0] < rows:
            cached = torch.zeros((rows, 3), dtype=torch.int32, device=self.logits.device)
            setattr(self.worker, cache_name, cached)
        prefixes = cached[:rows]
        if plan.execution.kind == "decode":
            if self.attention_batch is None:
                raise RuntimeError("Decode constraint selection has no Beam state")
            steps = plan.execution.stage_index
            dynamic_state = getattr(self, "dynamic_state", None)
            if dynamic_state is not None:
                sequence = dynamic_state.sequence[:, : plan.execution.input_width, :steps]
            else:
                sequence = self.attention_batch.context.decision.sequence[
                    : plan.execution.batch_size, : plan.execution.input_width, :steps
                ]
            prefixes[:, :steps].copy_(sequence.reshape(rows, steps))
        return prefixes

    def _constraint_temperatures(self) -> torch.Tensor:
        """Reuse one immutable scalar for this session's shared temperature."""
        temperature = float(
            self._metadata(self.worker._current_scheduler_output).beam_params.temperature
        )
        cached = getattr(self.worker, "_gr_npu_constraint_temperature", None)
        if cached is None or cached[0] != temperature or cached[1].device != self.logits.device:
            tensor = torch.full((1,), temperature, dtype=torch.float32, device=self.logits.device)
            cached = (temperature, tensor)
            self.worker._gr_npu_constraint_temperature = cached
        return cached[1]


def get_npu_stage_runner(worker: Any, scheduler_output: Any) -> NPUEagerBeamStageRunner:
    """Return the exclusive local NPU persistent execution owner."""

    from vllm_gr.config.execution import require_beam_config

    require_beam_config(worker.vllm_config)
    existing = getattr(worker, "_gr_npu_stage_runner", None)
    if existing is not None and not existing.closed:
        return existing
    metadata = next(iter(getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR).values()))
    candidate_widths = getattr(metadata.beam_params, "per_parent_beam_widths", None)
    if candidate_widths is not None:
        backend = getattr(worker, "constraint_backend", None)
        info = getattr(backend, "info", None)
        if backend is None or info is None:
            raise ValueError("NPU per-parent widths require a loaded constraint backend")
        if max(candidate_widths) > getattr(info, "max_top_k", 0):
            raise ValueError(
                "NPU constraint backend capacity is smaller than requested per-parent widths"
            )
    runner = NPUEagerBeamStageRunner(worker)
    worker._gr_npu_stage_runner = runner
    return runner


def execute_npu_gr_retirement(worker: Any, scheduler_output: Any) -> Any:
    """Release a drained eager NPU owner without another model invocation."""
    metadata = getattr(scheduler_output, GR_RETIRE_OUTPUT_ATTR)
    if (
        not isinstance(metadata, GRRetireMetadata)
        or scheduler_output.total_num_scheduled_tokens
        or scheduler_output.num_scheduled_tokens
        or getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None)
        or metadata.session_id not in scheduler_output.finished_req_ids
    ):
        raise ValueError("invalid Asynchronous scheduling retirement control frame")
    previous_metadata = getattr(worker, "_gr_npu_retirement_metadata", None)
    previous_output = getattr(worker, "_gr_npu_retirement_output", None)
    if previous_metadata == metadata and previous_output is not None:
        return previous_output
    owner = getattr(worker, "_gr_npu_stage_runner", None)
    if (
        owner is None
        or owner.closed
        or owner.pending_plan is not None
        or owner.previous is None
        or owner.previous.session_id != metadata.session_id
        or owner.previous.owner_generation != metadata.owner_generation
    ):
        raise RuntimeError("Asynchronous scheduling retirement has no matching live NPU owner")
    owner._release()
    event = worker.prepare_inputs_event
    if event is not None:
        event.synchronize()
    worker._update_states(scheduler_output)

    from vllm.v1.outputs import ModelRunnerOutput

    output = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
    setattr(
        output,
        GR_RETIRE_RESULT_ATTR,
        GRRetireResult(
            metadata.session_id,
            metadata.owner_generation,
            metadata.retire_id,
            True,
        ),
    )
    worker._gr_npu_retirement_metadata = metadata
    worker._gr_npu_retirement_output = output
    return output


__all__ = [
    "NPUEagerBeamStageRunner",
    "execute_npu_gr_retirement",
    "get_npu_stage_runner",
]
