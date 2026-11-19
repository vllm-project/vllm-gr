# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""NPU ModelRunner wrappers for grouped Beam execution."""

# Native NPU patch internals remain dynamically typed. Keep the helper
# signatures and caller-side keyword checks covered by the NPU worker group.
# mypy: disable-error-code="no-any-return,no-untyped-call,no-untyped-def,type-arg,union-attr,var-annotated"

from __future__ import annotations

from functools import wraps

import torch
from vllm.logger import init_logger

from vllm_gr.v1.worker.constraint_hook import (
    apply_constrained_sample_override,
    try_sample_with_constraint_backend,
)
from vllm_gr.v1.worker.model_runner_contracts import (
    _WORKER_BOOKKEEPING_PATCH_MARKER,
    _WORKER_PREPARE_PATCH_MARKER,
    _WORKER_SAMPLE_PATCH_MARKER,
    _WORKER_SAMPLE_TOKENS_PATCH_MARKER,
)
from vllm_gr.v1.worker.npu_model_runner_bookkeeping import (
    _attach_beam_search_result,
    _beam_collapse_bookkeeping,
)
from vllm_gr.v1.worker.npu_model_runner_capture import (  # noqa: F401
    capture_beam_decode_graph,
    make_patched_determine_batch_execution_and_padding_npu,
)
from vllm_gr.v1.worker.npu_model_runner_inputs import (
    InputBatchProxy,
    _beam_remap_inputs,
    _materialize_logits_indices,
)
from vllm_gr.v1.worker.npu_model_runner_prefill import (  # noqa: F401
    capture_prefill_graphs,
    make_patched_model_forward_npu,
)
from vllm_gr.v1.worker.npu_model_runner_sid import (  # noqa: F401
    make_patched_load_model,
    make_patched_reload_weights,
    make_patched_shutdown,
)

logger = init_logger(__name__)


def make_patched_update_states_npu(_orig_update_states):
    """Keep native request progression logical while forward counts are physical."""
    marker = "_vllm_gr_npu_update_states_patch"
    if getattr(_orig_update_states, marker, False):
        return _orig_update_states

    @wraps(_orig_update_states)
    def patched_update_states(self, scheduler_output, *args, **kwargs):
        stage = getattr(self, "_gr_npu_stage_runner", None)
        if (
            stage is not None
            and stage.pending_plan is not None
            and stage.pending_plan.execution.kind == "decode"
            and scheduler_output is self._current_scheduler_output
        ):
            scheduler_output = stage.logical_scheduler_output
        return _orig_update_states(self, scheduler_output, *args, **kwargs)

    setattr(patched_update_states, marker, True)
    return patched_update_states


def make_patched_prepare_inputs_npu(_orig_prepare_inputs):
    """Wrap the NPU four-item ``_prepare_inputs`` contract."""
    if getattr(_orig_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False):
        return _orig_prepare_inputs

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        # Publish the scheduler output so the _model_forward prefill-graph
        # interception can classify the batch and extract token counts.
        self._current_scheduler_output = scheduler_output

        # NPU sampling can leave a proxy behind when bookkeeping is skipped.
        if isinstance(self.input_batch, InputBatchProxy):
            self.input_batch = object.__getattribute__(self.input_batch, "_obj")

        stage_runner = getattr(self, "_gr_npu_stage_runner", None)
        if stage_runner is not None and stage_runner.pending_plan is not None:
            return stage_runner.prepare_inputs(
                _orig_prepare_inputs,
                scheduler_output,
                num_scheduled_tokens,
            )

        # NPU builds attention metadata inside the original _prepare_inputs.
        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = self.input_batch.req_ids

        out = _orig_prepare_inputs(self, scheduler_output, num_scheduled_tokens)

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data:
            return out

        (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
        ) = out

        new_logits_indices = _beam_remap_inputs(self, scheduler_output, num_scheduled_tokens)
        logits_indices = _materialize_logits_indices(self, new_logits_indices)

        return (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
        )

    setattr(patched_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, True)
    return patched_prepare_inputs


def make_patched_sample_npu(_orig_sample):
    """Wrap the NPU ``_sample`` to run Constraint Tables on raw logits.

    Installed only when the deployed vllm-ascend runner actually exposes
    ``_sample`` (``apply_worker_patches`` guards with ``hasattr``); builds
    without the hook fail at constraint-table injection instead of silently
    sampling unconstrained.  Signature-agnostic pass-through so it tolerates
    an Ascend ``_sample`` with a different metadata argument.

    After the original sampling (and the Constraint Table raw-logits
    override), the worker may run the beam decision itself (``beam_search_group``
    / ``beam_search_rec_final_select``) when ``GRConfig.beam_worker_decision``
    is enabled — the result is stashed on the runner as
    ``_beam_decision_result`` and attached to ``ModelRunnerOutput.beam_search_result``
    by ``make_patched_execute_model`` (asynchronous result flow).  When disabled
    or inapplicable the decision stays in the frontend and nothing changes.

    Sampler short-circuit (Asynchronous scheduling ``model_runner_v1`` constrained decode path):
    when the worker beam decision is active and every sampling row of the
    step is constrained, the original sampler run is dead work — its
    log_softmax + topk + gather_logprobs output is entirely overwritten by
    the constrained operator (profiling bw128_1k_t3: ~3.4ms of wasted device
    work plus the host-side gather chain per step).  The sampler output is
    built directly from the operator result instead (``logprobs`` width
    ``W``, top-1 candidate in column 0 — the exact shape the override path
    and ``_select_candidates`` produce).
    """
    if getattr(_orig_sample, _WORKER_SAMPLE_PATCH_MARKER, False):
        return _orig_sample

    @wraps(_orig_sample)
    def patched_sample(self, *args, **kwargs):
        logits = args[0] if args else kwargs.get("logits")
        spec_decode_metadata = args[1] if len(args) > 1 else kwargs.get("spec_decode_metadata")
        sampler_output = try_sample_with_constraint_backend(
            self,
            logits,
            spec_decode_metadata,
        )
        if sampler_output is None:
            sampler_output = _try_constrained_sampler_shortcut(self, args, kwargs)
        if sampler_output is None:
            sampler_output = _orig_sample(self, *args, **kwargs)
            apply_constrained_sample_override(self, logits, sampler_output)
        _maybe_run_worker_beam_decision(self, sampler_output)
        return sampler_output

    setattr(patched_sample, _WORKER_SAMPLE_PATCH_MARKER, True)
    return patched_sample


def _try_constrained_sampler_shortcut(runner, args, kwargs):
    """Build the sampler output from the constrained operator, or None.

    Returns a fully populated ``SamplerOutput`` when the original sampler
    can be skipped, ``None`` when the caller must fall back to the original
    (then override) path.

    Conditions (all must hold — any miss keeps the untouched legacy path):

    - the worker beam decision is active (``worker_decision_enabled``):
      the short-circuit is the post-processing half of the xllm pipeline,
      and the frontend decision path still consumes a full-width sampler
      output (P1-1 scope);
    - raw logits are available as the first positional argument;
    - no speculative decoding (beam decode never carries spec metadata,
      but the check keeps the shortcut off that path defensively);
    - the step has constrained rows and they are homogeneous (validated by
      ``collect_constrained_override_rows``, which mirrors
      ``apply_constrained_sample_override``'s guards).

    Row-count conservation: ``_validate_homogeneous_step`` guarantees every
    W_logits > 0 row belongs to a constrained request, so the operator
    output covers exactly the logits rows the sampler would have produced.
    """
    from vllm_gr.v1.beam.beam_decision import worker_decision_enabled
    from vllm_gr.v1.worker.constraint_hook import (
        build_constrained_sampler_output,
        collect_constrained_override_rows,
    )

    logits = args[0] if args else None
    if logits is None or not torch.is_tensor(logits):
        return None
    # Spec-decode steps carry bonus/target logits slices; the beam decode
    # shortcut is only defined for plain steps.
    if len(args) > 1 and args[1] is not None:
        return None
    if kwargs.get("spec_decode_metadata") is not None:
        return None
    if not worker_decision_enabled(runner):
        return None
    rows = collect_constrained_override_rows(runner)
    if not rows:
        return None
    try:
        return build_constrained_sampler_output(runner, logits, rows)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "vllm-gr constrained sampler shortcut failed: %s; falling back to the original sampler",
            exc,
        )
        return None


def make_patched_sample_tokens_npu(_orig_sample_tokens):
    """Wrap the NPU ``sample_tokens`` to attach the worker beam decision.

    Under async scheduling the runner's ``execute_model`` returns an
    ``ExecuteModelState`` and sampling happens later in ``sample_tokens``;
    the worker-side beam decision runs inside the patched ``_sample`` during
    that call.  The ModelRunnerOutput / AsyncModelRunnerOutput is only built at
    the end of ``sample_tokens``, so the decision must be attached to it here
    (after the original sampling) rather than in ``execute_model`` where the
    decision does not exist yet.  ``_attach_beam_search_result`` handles both
    the plain ModelRunnerOutput and the AsyncModelRunnerOutput inner object.
    """
    if getattr(_orig_sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, False):
        return _orig_sample_tokens

    @wraps(_orig_sample_tokens)
    def patched_sample_tokens(self, grammar_output, *args, **kwargs):
        stage_runner = getattr(self, "_gr_npu_stage_runner", None)
        if stage_runner is not None and (
            stage_runner.pending_plan is not None or stage_runner.execution_error is not None
        ):
            return stage_runner.sample(grammar_output)
        out = _orig_sample_tokens(self, grammar_output, *args, **kwargs)
        _attach_beam_search_result(self, out)
        return out

    setattr(patched_sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, True)
    return patched_sample_tokens


def _maybe_run_worker_beam_decision(runner, sampler_output) -> None:
    """Compatibility alias for the shared accelerator decision helper."""
    from vllm_gr.v1.beam.beam_decision import maybe_run_worker_beam_decision

    maybe_run_worker_beam_decision(runner, sampler_output)


def make_patched_bookkeeping_sync_npu(_orig_bookkeeping_sync):
    """Wrap the NPU six-item ``_bookkeeping_sync`` contract."""
    if getattr(_orig_bookkeeping_sync, _WORKER_BOOKKEEPING_PATCH_MARKER, False):
        return _orig_bookkeeping_sync

    @wraps(_orig_bookkeeping_sync)
    def patched_bookkeeping_sync(self, scheduler_output, *args, **kwargs):
        if (
            hasattr(self, "_beam_logprobs_cache")
            and self._beam_logprobs_cache
            and hasattr(scheduler_output, "finished_req_ids")
        ):
            for req_id in scheduler_output.finished_req_ids:
                self._beam_logprobs_cache.pop(req_id, None)

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data or not hasattr(scheduler_output, "_gr_request_row_configs"):
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
            sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
            return res

        request_row_configs = scheduler_output._gr_request_row_configs
        orig_input_batch = object.__getattribute__(self.input_batch, "_obj")
        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)

        # vLLM sizes its synchronous pinned sampled-token buffer by
        # max_num_seqs (logical requests), while Beam decode initially has one
        # sampler row per parent. Collapse the worker-selected payload before
        # upstream bookkeeping copies it to the fixed-size host buffer.
        if (
            sampler_output is not None
            and isinstance(getattr(sampler_output, "sampled_token_ids", None), torch.Tensor)
            and sampler_output.sampled_token_ids.shape[0] > len(request_row_configs)
        ):
            # The frontend-decision compatibility path has no worker result to
            # reduce, but sampled_token_ids is still ordinary bookkeeping: keep
            # one row per logical request before the fixed-size D2H copy. Beam
            # candidates remain intact in logprobs_tensors.
            physical_row = 0
            logical_sampled_rows = []
            for req_id, _, logits_rows, *_ in request_row_configs:
                if req_id is not None:
                    logical_sampled_rows.append(
                        sampler_output.sampled_token_ids[physical_row : physical_row + 1]
                    )
                physical_row += logits_rows
            sampler_output.sampled_token_ids = torch.cat(logical_sampled_rows, dim=0)

        try:
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
        finally:
            self.input_batch = orig_input_batch

        (
            logprobs_lists,
            _valid_sampled_token_ids,
            prompt_logprobs_dict,
            _req_ids_output_copy,
            _req_id_to_index_output_copy,
            invalid_req_indices,
        ) = res

        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(
            self, request_row_configs, beam_data, sampler_output
        )

        return (
            [[] for _ in range(n)] if logprobs_lists is not None else None,
            [[] for _ in range(n)],
            prompt_logprobs_dict,
            collapsed_req_ids,
            collapsed_idx,
            invalid_req_indices,
        )

    setattr(patched_bookkeeping_sync, _WORKER_BOOKKEEPING_PATCH_MARKER, True)
    return patched_bookkeeping_sync


# ---------------------------------------------------------------------------
# Beam decode ACL Graph startup capture (FULL mode)
# ---------------------------------------------------------------------------


def discover_worker_runners() -> tuple[tuple[str, type], ...]:
    """Discover NPUModelRunner through vLLM-Ascend's supported entry point."""
    try:
        # worker initializes vllm_ascend.ops before importing model_runner_v1.
        from vllm_ascend.worker.worker import NPUModelRunner
    except ImportError as exc:
        logger.warning(
            "Failed to import NPUModelRunner through "
            "vllm_ascend.worker.worker; skipping "
            "vllm-gr NPU beam-search runner patches for this attempt. "
            "The vLLM-Ascend NPU runtime itself is not disabled. "
            "ImportError: %s",
            exc,
        )
        logger.debug("NPUModelRunner import traceback", exc_info=True)
        return ()
    return (("NPUModelRunner", NPUModelRunner),)


make_patched_prepare_inputs = make_patched_prepare_inputs_npu
make_patched_update_states = make_patched_update_states_npu
make_patched_bookkeeping_sync = make_patched_bookkeeping_sync_npu
make_patched_sample = make_patched_sample_npu
# The dispatcher (model_runner_patch.py) looks up this exact name; without the
# alias the sample_tokens hook is silently skipped and the worker beam decision
# never reaches the ModelRunnerOutput (async scheduling keeps the decision path
# broken end-to-end: engine-driven chaining no-ops and the frontend keeps its RPC loop).
make_patched_sample_tokens = make_patched_sample_tokens_npu
make_patched_determine_batch_execution_and_padding = (
    make_patched_determine_batch_execution_and_padding_npu
)


#  Read actual packed row order; do not modify latest stage/decode hooks.
make_patched_model_forward = make_patched_model_forward_npu
