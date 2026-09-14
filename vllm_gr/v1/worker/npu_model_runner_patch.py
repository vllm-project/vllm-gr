# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU ModelRunner wrappers for grouped Beam execution."""

from __future__ import annotations

from functools import wraps

import torch
from vllm.logger import init_logger

from vllm_gr.v1.worker.constraint_hook import (
    apply_constrained_sample_override,
    try_sample_with_constraint_backend,
)
from vllm_gr.v1.worker.model_runner_common import (
    _WORKER_BOOKKEEPING_PATCH_MARKER,
    _WORKER_DETERMINE_PATCH_MARKER,
    _WORKER_MODEL_FORWARD_PATCH_MARKER,
    _WORKER_PREPARE_PATCH_MARKER,
    _WORKER_SAMPLE_PATCH_MARKER,
    _WORKER_SAMPLE_TOKENS_PATCH_MARKER,
    InputBatchProxy,
    _beam_collapse_bookkeeping,
    _beam_full_graph_ready,
    _beam_graph_enabled_for_run,
    _beam_remap_inputs,
    _create_beam_context,
    _materialize_logits_indices,
    _reduce_sampler_output_for_worker_decision,
)
from vllm_gr.v1.worker.prefill_graph_runner import (
    SUPPORTED_NUM_REQS,
    get_prefill_graph_runner,
    is_prefill_graph_enabled,
    is_prefill_phase_batch,
)

logger = init_logger(__name__)


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
            num_scheduled_tokens_compressed_list,
        ) = out

        new_logits_indices = _beam_remap_inputs(self, scheduler_output, num_scheduled_tokens)
        logits_indices = _materialize_logits_indices(self, new_logits_indices)

        return (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
            num_scheduled_tokens_compressed_list,
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
    by ``make_patched_execute_model`` (ACS-aligned save flow).  When disabled
    or inapplicable the decision stays in the frontend and nothing changes.

    Sampler short-circuit (ACS ``model_runner_v1`` constrained decode path):
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
        from vllm_gr.v1.worker.model_runner_common import _attach_beam_search_result

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
        if hasattr(self, "_beam_logprobs_cache") and self._beam_logprobs_cache:
            if hasattr(scheduler_output, "finished_req_ids"):
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
        _reduce_sampler_output_for_worker_decision(self, sampler_output)
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
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = res

        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(
            self, orig_input_batch, request_row_configs, beam_data, sampler_output
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


def _npu_xllm_decode_mode_active() -> bool:
    """True iff the NPU decode execution mode resolves to the fused xllm path.

    Only the xllm fused path is graph-safe: the cascade NPU path slices the
    beam pool with host-int decode_step values (beam_attn_npu.py), which a
    captured graph would freeze.  Resolved lazily through the same
    per-process cache the attention backend uses (``_get_npu_decode_execution``).
    """
    try:
        from vllm_gr.v1.attention.backends.beam_attn import _get_npu_decode_execution

        return _get_npu_decode_execution() == "xllm"
    except Exception:
        return False


def _downgrade_to_none(result):
    """Return result with mode NONE and an unpadded descriptor (GPU parity)."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    if result[0] == CUDAGraphMode.FULL:
        return (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=result[1].num_tokens)) + result[2:]
    return result


def make_patched_determine_batch_execution_and_padding_npu(_orig_determine):
    """Override cudagraph_mode to FULL for beam decode steps (NPU/ACL graph).

    Mirrors the GPU-side override (``make_patched_determine_batch_execution_
    and_padding_gpu``): beam decode batches never satisfy the native
    ``_is_uniform_decode`` predicate (beam_width tokens != query_len 1), so
    without this hook dispatch can only ever select PIECEWISE — 29 subgraph
    replays per forward with every beam attention call eager.  For a PURE
    beam-decode batch with an active BeamSearchSession, an exactly-matching
    startup-captured FULL graph and the graph-safe xllm execution mode, the
    dispatch result is rewritten to ``(FULL, BatchDescriptor(num_tokens=W,
    uniform=True))`` so the ACLGraphWrapper replays one whole graph per
    decode step.

    Every other beam-bearing batch keeps the native dispatch result, except
    that a native FULL grant is downgraded to NONE: native FULL graphs
    contain no beam kernels, and replaying one for beam tokens would compute
    garbage silently.
    """
    if getattr(_orig_determine, _WORKER_DETERMINE_PATCH_MARKER, False):
        return _orig_determine

    @wraps(_orig_determine)
    def patched_determine(self, *args, **kwargs):
        result = _orig_determine(self, *args, **kwargs)

        beam_info = getattr(self, "_current_beam_info", None)
        if beam_info is None:
            return result

        beam_data = beam_info.get("beam_data", {})
        beam_keys = {
            req_id
            for req_id, mdata in beam_data.items()
            if mdata.get("is_beam_decode", False) and "decode_step" in mdata
        }
        if not beam_keys:
            return result

        # Beam decode FULL requires an active BeamSearchSession — the xllm
        # execution path reads the pool slot and context buffers from it.
        session = beam_info.get("beam_session")
        if session is None or not getattr(session, "is_active", False):
            return _downgrade_to_none(result)

        # FULL is granted only for a PURE beam-decode batch: the captured
        # beam graph contains no standard-attention path.  ``req_ids``
        # (filled by the patched _prepare_inputs) enumerates the whole batch.
        batch_req_ids = set(beam_info.get("req_ids") or [])
        if batch_req_ids != beam_keys:
            return _downgrade_to_none(result)

        # The startup beam graph is captured only for the default no-LoRA
        # descriptor and ``_beam_full_graph_ready`` is keyed by width alone,
        # so a beam batch with active LoRA adapters must not be granted FULL:
        # the rewritten descriptor would drop has_lora/num_active_loras and
        # silently route the batch to the LoRA-free graph.  Run eager until
        # LoRA-aware beam graphs are captured per full descriptor.
        if getattr(result[1], "has_lora", False):
            return _downgrade_to_none(result)

        # The captured graph bakes the pool row pointer (slot 0); a session
        # on any other slot must not replay it.
        if getattr(session, "slot", 0) != 0:
            return _downgrade_to_none(result)

        # Beam graphs are opt-in via GRConfig (beam_max_width) and can be
        # disabled for A/B comparison (beam_graph_enabled / --no-beam-graph).
        if not _beam_graph_enabled_for_run(self):
            return _downgrade_to_none(result)

        # Only the fused xllm path is graph-safe on NPU (the cascade path
        # bakes host-int pool slices; see _npu_xllm_decode_mode_active).
        if not _npu_xllm_decode_mode_active():
            return _downgrade_to_none(result)

        # Exact-width gate: compare the UNPADDED scheduled token count
        # against the captured graph's width (the beam width for a pure
        # batch).  num_tokens is argument 0 on the NPU runner contract.
        num_tokens_arg = kwargs.get("num_tokens")
        if num_tokens_arg is None and args:
            num_tokens_arg = args[0]
        real_num_tokens = (
            int(num_tokens_arg) if num_tokens_arg is not None else result[1].num_tokens
        )

        # FULL iff a beam graph for this exact width exists in the captured
        # set; otherwise eager (and never a native FULL graph for beam rows).
        if not _beam_full_graph_ready(self, real_num_tokens):
            return _downgrade_to_none(result)

        from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

        new_descriptor = BatchDescriptor(num_tokens=real_num_tokens, uniform=True)
        logger.debug(
            "Beam decode (NPU): cudagraph_mode %s -> FULL, batch_descriptor %s -> %s",
            result[0],
            result[1],
            new_descriptor,
        )
        return (CUDAGraphMode.FULL, new_descriptor) + result[2:]

    setattr(patched_determine, _WORKER_DETERMINE_PATCH_MARKER, True)
    return patched_determine


# ---------------------------------------------------------------------------
# Beam decode ACL Graph startup capture (FULL mode)
# ---------------------------------------------------------------------------

_BEAM_DUMMY_REQ_ID = "vllm-gr-beam-graph-dummy"

# Fake prefix length for the dummy capture session.  The xllm fused kernels
# read the true prefix extent from the fixed-address prefix_seqlens_kv_device
# buffer (refreshed each step outside the graph), so the captured graph's
# shape does not depend on the prefix length; 1024 is arbitrary.
_BEAM_CAPTURE_PREFIX_LEN = 1024

# Eager warmup forwards before the FULL-mode capture forward.  Primes the
# xllm cache_unshared_kv / x_attention operator shape selection.
_BEAM_CAPTURE_NUM_WARMUPS = 2


def _capture_create_dummy_context_npu(
    runner, gr_cfg, width: int, steps_upper: int, prefix_len: int
) -> tuple:
    """Create the BeamAttentionContext, dummy session, and beam_info for capture.

    Returns ``(context, session, beam_info, batch_descriptor)`` — the same
    contract as the GPU-side helper, so the runtime lazy-creation path and
    the capture path share pool sizing via ``_create_beam_context``.
    """
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor

    from vllm_gr.v1.beam.beam_search_session import BeamSearchSession

    configured_max_batch = int(getattr(gr_cfg, "beam_max_batch_size", 1))
    context = getattr(runner, "_beam_context", None)
    if context is None:
        context = _create_beam_context(
            runner,
            gr_cfg,
            max_batch=configured_max_batch,
        )
    elif int(context.max_batch) != configured_max_batch:
        raise RuntimeError(
            "captured Beam context capacity differs from beam_max_batch_size"
        )
    slot = context.registry.allocate(_BEAM_DUMMY_REQ_ID, beam_width=width, prefix_len=prefix_len)
    session = BeamSearchSession(session_id=_BEAM_DUMMY_REQ_ID, slot=slot, context=context)
    session.begin_session(width, prefix_len)

    runner._beam_context = context
    runner._beam_sessions = {_BEAM_DUMMY_REQ_ID: session}

    beam_data = {
        _BEAM_DUMMY_REQ_ID: {
            "prefix_len": prefix_len,
            "cache_len": prefix_len,
            "max_decode_steps": steps_upper,
            "is_beam_decode": True,
            "beam_width": width,
            "beam_decode_steps": 1,
            "parent_beam_ids": [0] * width,
            "decode_step": 0,
        }
    }
    beam_info = {
        "beam_data": beam_data,
        "req_ids": [_BEAM_DUMMY_REQ_ID],
        "beam_session": session,
    }

    batch_descriptor = BatchDescriptor(num_tokens=width, uniform=True)
    return context, session, beam_info, batch_descriptor


def _dummy_beam_forward_npu(
    runner,
    context,
    beam_info,
    batch_descriptor,
    width: int,
    prefix_len: int,
    capture: bool,
) -> None:
    """Execute one dummy forward for beam-decode ACL Graph capture (or warmup).

    Mirrors the GPU ``_dummy_beam_forward`` ordering contract:
    1. ``decode_step_buf`` / ``decode_step_1based_buf`` / ``xllm_block_table_buf``
       fills — device inputs refreshed outside the graph, must precede forward.
    2. ``input_ids`` / ``positions`` — runner persistent buffers.
    3. ``optimistic_seq_lens_cpu`` → ``seq_lens.copy_`` — CPU write before H2D.
    4. ``query_start_loc`` — CPU write → H2D copy.
    5. ``commit_block_table`` for 1 request.
    6. slot_mappings fill(-1) — skip paged KV writes for dummy tokens.
    7. ``BEAM_DATA_VAR`` scope around metadata build + forward.

    The NPU runner builds attention metadata inside ``_build_attention_metadata``
    (no separate per-request builder call), and the forward runs under
    ``set_ascend_forward_context`` so the ACLGraphWrapper (wrapping
    ``runner.model``) sees the FULL runtime mode + beam batch descriptor and
    captures the whole forward as one ACL graph.
    """
    from vllm.v1.cudagraph_dispatcher import CUDAGraphMode
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    from vllm_gr.v1.beam.context import BEAM_DATA_VAR

    # 1. Refresh the per-step device buffers (same as the runtime path).
    context.decode_step_buf.fill_(0)
    if hasattr(context, "decode_step_1based_buf"):
        context.decode_step_1based_buf.fill_(1)
    if hasattr(context, "xllm_block_table_buf"):
        context.xllm_block_table_buf.fill_(0)
        context.xllm_block_table_buf[0] = 0  # dummy session slot

    # 2. Dummy inputs in the runner's persistent buffers (fixed addresses).
    input_ids = runner.input_ids.gpu[:width]
    input_ids.fill_(1)
    positions = runner.positions[:width]
    positions.copy_(
        torch.arange(prefix_len, prefix_len + width, dtype=positions.dtype, device=runner.device)
    )

    # 3. Dummy batch state: 1 req with `width` scheduled tokens;
    #    seq_len = prefix_len + width.  CPU write precedes the H2D copy.
    runner.optimistic_seq_lens_cpu[:1] = prefix_len + width
    runner.optimistic_seq_lens_cpu[1:] = 0
    runner.seq_lens.copy_(runner.optimistic_seq_lens_cpu, non_blocking=True)

    # 4. query_start_loc: CPU write precedes copy_to_gpu.
    runner.query_start_loc.np[0] = 0
    runner.query_start_loc.np[1] = width
    runner.query_start_loc.np[2:] = 0
    runner.query_start_loc.copy_to_gpu()

    # 5. Commit block table for 1 request.
    runner.input_batch.block_table.commit_block_table(1)

    # 6. Dummy runs have no real slot assignments — fill with -1 so
    #    reshape_and_cache kernels skip the KV write.  (The NPU
    #    _build_attention_metadata derives slot mappings from the committed
    #    block table; priming the mappings here mirrors _dummy_run so any
    #    pre-fill of stale slot ids is overwritten.)
    slot_mappings_by_group, _ = runner._get_slot_mappings(
        num_tokens_padded=width,
        num_reqs_padded=1,
        num_tokens_unpadded=width,
    )
    if slot_mappings_by_group is not None:
        for sm in slot_mappings_by_group.values():
            sm.fill_(-1)

    # 7. Publish beam context, build metadata, run the model forward.
    #    NPU note: the ascend _build_attention_metadata derives slot mappings
    #    from the committed block table internally (no slot_mappings kwarg).
    token = BEAM_DATA_VAR.set(beam_info)
    try:
        metadata, _ = runner._build_attention_metadata(
            num_tokens=width,
            num_reqs=1,
            max_query_len=width,
            num_tokens_padded=width,
            num_reqs_padded=1,
            for_cudagraph_capture=capture,
        )
        with set_ascend_forward_context(
            metadata,
            runner.vllm_config,
            num_tokens=width,
            num_actual_tokens=width,
            aclgraph_runtime_mode=(CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE),
            batch_descriptor=batch_descriptor,
            model_instance=runner.model,
        ):
            runner.model(input_ids=input_ids, positions=positions)
    finally:
        BEAM_DATA_VAR.reset(token)


def _verify_and_record_captured_graph_npu(runner, width: int) -> bool:
    """Confirm an ACL graph was captured for the beam descriptor; record width.

    Walks ``ACLGraphWrapper._all_instances`` looking for a FULL-mode entry
    keyed by ``BatchDescriptor(num_tokens=width, uniform=True)``.  On a hit,
    records ``width`` in ``runner._beam_graph_keys`` for the O(1) runtime
    lookup used by ``_beam_full_graph_ready``.
    """
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

    desc = BatchDescriptor(num_tokens=width, uniform=True)
    for wrapper in ACLGraphWrapper._all_instances:
        if wrapper.runtime_mode != CUDAGraphMode.FULL:
            continue
        entry = wrapper.concrete_aclgraph_entries.get(desc)
        if entry is not None and entry.aclgraph is not None:
            beam_keys = getattr(runner, "_beam_graph_keys", None)
            if beam_keys is None:
                beam_keys = set()
                runner._beam_graph_keys = beam_keys
            beam_keys.add(width)
            return True
    return False


def capture_beam_decode_graph(runner) -> bool:
    """Startup dummy capture of the beam-decode FULL ACL graph (NPU).

    Runs once after ``capture_model()`` when the GRConfig beam graph is
    active (``beam_graph_active`` and ``beam_max_width`` set).  Mirrors the
    GPU-side flow: N eager warmup forwards (mode NONE — primes the xllm
    cache_unshared_kv / x_attention operators), then one FULL-mode forward
    which the ACLGraphWrapper captures under
    ``BatchDescriptor(num_tokens=beam_max_width, uniform=True)``.

    Every runtime-varying kernel input is a persistent fixed-address device
    buffer refreshed outside the graph each step (metadata builder's
    ``_fixed_buffers`` pool; the beam context's ``decode_step_buf`` /
    ``decode_step_1based_buf`` / ``xllm_block_table_buf``), so the single
    graph serves every decode step and every prompt length.  The capture
    output is discarded; real requests replay only
    (``patched_determine`` + ``_beam_full_graph_ready`` gate FULL).

    Returns True if a graph was captured.
    """
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled

    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    gr_cfg = _get_gr_config(runner)
    if not gr_cfg.beam_graph_active:
        return False

    # Only the fused xllm execution path is graph-safe on NPU (the cascade
    # path slices the beam pool with host-int decode_step values, which a
    # captured graph would freeze).
    if not _npu_xllm_decode_mode_active():
        logger.info(
            "vllm-gr: beam decode ACL graph requires the xllm decode execution "
            "mode; beam decode will run eager"
        )
        return False

    # ── TP / ubatching guards (same rationale as the GPU side) ──────
    tp_size = runner.parallel_config.tensor_parallel_size
    if tp_size > 1:
        logger.warning(
            "vllm-gr: beam decode ACL graph is not supported with "
            "tensor parallelism (TP=%d); beam decode will run eager",
            tp_size,
        )
        return False
    if runner.parallel_config.use_ubatching:
        logger.warning(
            "vllm-gr: beam decode ACL graph is not supported with "
            "DP ubatching; beam decode will run eager",
        )
        return False
    # ─────────────────────────────────────────────────────────────────

    # Re-entry guard: an existing graph references the current context's
    # buffer addresses; rebuilding the context would silently detach it.
    width = int(gr_cfg.beam_max_width)
    if _beam_full_graph_ready(runner, width):
        return True

    steps_upper = int(gr_cfg.beam_max_decode_steps)
    prefix_len = _BEAM_CAPTURE_PREFIX_LEN

    context, session, beam_info, batch_descriptor = _capture_create_dummy_context_npu(
        runner, gr_cfg, width, steps_upper, prefix_len
    )

    captured = False
    try:
        set_cudagraph_capturing_enabled(True)
        # The graph pool's tensors are inference tensors (the native
        # ``_dummy_run`` creates them under ``@torch.inference_mode()``), so
        # ``capture_begin``'s in-place updates are only legal inside
        # inference mode.  Mirror the native decorator here.
        with torch.inference_mode():
            # ACL graph capture requires a non-default stream (native
            # capture_model provides one; emulate here).
            side = torch.npu.Stream()
            side.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(side):
                for _ in range(_BEAM_CAPTURE_NUM_WARMUPS):
                    _dummy_beam_forward_npu(
                        runner,
                        context,
                        beam_info,
                        batch_descriptor,
                        width,
                        prefix_len,
                        capture=False,
                    )
                _dummy_beam_forward_npu(
                    runner,
                    context,
                    beam_info,
                    batch_descriptor,
                    width,
                    prefix_len,
                    capture=True,
                )
            torch.npu.current_stream().wait_stream(side)
            torch.npu.synchronize()

            captured = _verify_and_record_captured_graph_npu(runner, width)
    finally:
        set_cudagraph_capturing_enabled(False)
        # Drop the dummy session even if capture threw.  The context and
        # pools stay — a captured graph (if any) references their addresses.
        session.end_session()
        runner._beam_sessions.pop(_BEAM_DUMMY_REQ_ID, None)
        runner._current_beam_info = None

    if captured:
        logger.info(
            "vllm-gr: startup dummy capture of beam decode ACL graph succeeded "
            "(num_tokens=%d, uniform=True); runtime is replay-only",
            width,
        )
    else:
        logger.warning(
            "vllm-gr: beam dummy capture did NOT produce an ACL graph for "
            "num_tokens=%d; beam decode will run eager",
            width,
        )
    return captured


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


def make_patched_model_forward_npu(_orig_model_forward):
    """Wrap _model_forward to intercept and replay pure prefill NPU graphs."""
    if getattr(_orig_model_forward, _WORKER_MODEL_FORWARD_PATCH_MARKER, False):
        return _orig_model_forward

    @wraps(_orig_model_forward)
    def patched_model_forward(
        self,
        num_tokens_padded,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        scheduler_output = getattr(self, "_current_scheduler_output", None)
        if scheduler_output is not None:
            if is_prefill_graph_enabled() and is_prefill_phase_batch(self, scheduler_output):
                p_runner = get_prefill_graph_runner(self)

                raw_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", [])
                # The scheduler already computed the batch's token total; only
                # the request count is derived here (the runner re-derives the
                # per-request counts it needs from the dict itself).
                num_reqs = len(raw_scheduled_tokens)
                total_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
                if not total_tokens and raw_scheduled_tokens:
                    total_tokens = (
                        sum(raw_scheduled_tokens.values())
                        if isinstance(raw_scheduled_tokens, dict)
                        else sum(raw_scheduled_tokens)
                    )

                if 1 <= num_reqs <= SUPPORTED_NUM_REQS[-1] and total_tokens > 0:
                    bucket = p_runner.match_bucket(total_tokens)
                    if bucket is not None and p_runner.has_captured_graph(bucket, num_reqs):
                        # Extract the current batch's slot mapping from the
                        # attention metadata built by _prepare_inputs. The
                        # metadata is published in the forward context by
                        # execute_model; NPUModelRunner does not store it as
                        # a runner attribute.
                        slot_mapping = None
                        try:
                            from vllm.forward_context import get_forward_context

                            attn_metadata = getattr(get_forward_context(), "attn_metadata", None)
                            if isinstance(attn_metadata, dict) and attn_metadata:
                                slot_mapping = next(iter(attn_metadata.values())).slot_mapping
                        except Exception as e:  # noqa: BLE001
                            logger.debug("Could not get slot_mapping: %r", e)

                        output_buf = p_runner.execute_model_prefill(
                            bucket=bucket,
                            input_ids=input_ids,
                            positions=positions,
                            slot_mapping=slot_mapping,
                            num_scheduled_tokens=raw_scheduled_tokens,
                        )

                        if output_buf is not None:
                            p_runner.log_replay(bucket, num_reqs, total_tokens)
                            logger.debug(
                                "Prefill NPU Graph Replay: Executed captured graph "
                                "for bucket %d (reqs=%d, %d actual tokens).",
                                bucket,
                                num_reqs,
                                total_tokens,
                            )
                            return output_buf
        return _orig_model_forward(
            self,
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )

    setattr(patched_model_forward, _WORKER_MODEL_FORWARD_PATCH_MARKER, True)
    return patched_model_forward


def capture_prefill_graphs(runner) -> None:
    """Startup capture of the pure-prefill NPU graphs (one per bucket).

    Chained after the beam decode-graph capture in ``capture_model``; runs
    only when the prefill graph env gate is enabled. Every failure falls back
    to eager prefill (the interception simply never matches a bucket).
    """
    if not is_prefill_graph_enabled():
        return
    if getattr(runner, "_prefill_graphs_captured", False):
        return
    try:
        # Mirror the beam decode-graph capture: run the prefill capture on a
        # dedicated side stream ordered against the default stream (the
        # native capture_stream context is gone once capture_model returns).
        side = torch.npu.Stream(device=runner.device)
        side.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(side):
            p_runner = get_prefill_graph_runner(runner)
            p_runner.capture_graphs(runner)
        torch.npu.current_stream().wait_stream(side)
        torch.npu.synchronize()
        runner._prefill_graphs_captured = True
    except Exception:
        logger.exception(
            "vllm-gr: prefill NPU graph capture failed; prefill will fall back to eager"
        )


make_patched_model_forward = make_patched_model_forward_npu
