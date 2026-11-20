# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""NPU ModelRunner wrappers for grouped Beam execution."""

# Native NPU patch internals remain dynamically typed. Keep the helper
# signatures and caller-side keyword checks covered by the NPU worker group.
# mypy: disable-error-code="no-any-return,no-untyped-call,no-untyped-def,type-arg,union-attr,var-annotated"

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_gr.config.models import deployment_for
from vllm_gr.v1.worker.gr_context import (
    _beam_full_graph_ready,
    _beam_graph_enabled_for_run,
    _create_beam_context,
)
from vllm_gr.v1.worker.model_runner_contracts import (
    _WORKER_DETERMINE_PATCH_MARKER,
    _WORKER_MODEL_FORWARD_PATCH_MARKER,
    _WORKER_PREPARE_PATCH_MARKER,
    _WORKER_SAMPLE_TOKENS_PATCH_MARKER,
    is_worker_runner_patched,
)
from vllm_gr.v1.worker.npu.npu_model_runner_inputs import (
    InputBatchProxy,
)
from vllm_gr.v1.worker.prefill_graph_runner import (
    SUPPORTED_NUM_REQS,
    get_prefill_graph_runner,
    is_prefill_graph_enabled,
    is_prefill_phase_batch,
)

logger = init_logger(__name__)


def make_patched_load_model(original: Callable[..., Any]) -> Callable[..., Any]:
    """Install catalog-free SID state after the shared NPU model load."""
    marker = "_vllm_gr_npu_sid_load_patch"
    if getattr(original, marker, False):
        return original

    @wraps(original)
    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        # The inner shared loader closes the old projection before loading
        # native weights. Detach the NPU-owned maps at the same boundary.
        self._gr_npu_sid_stage_vocab = None
        projection = None
        base_memory_usage = None
        try:
            result = original(self, *args, **kwargs)

            from vllm_gr.config import GRConfig

            config = GRConfig.from_vllm_config(getattr(self, "vllm_config", None))
            if config.constrained_generation is not False or not config.beam_sid_stage_vocab:
                return result
            if str(getattr(self, "device", "")).split(":")[0] != "npu":
                raise ValueError("beam.sid_stage_vocab currently requires NPU")

            from vllm.tokenizers import cached_tokenizer_from_config

            from vllm_gr.ops.npu.sid_stage_vocab import build_sid_stage_vocab
            from vllm_gr.v1.worker.model_runner_patch import _runner_vocab_size
            from vllm_gr.v1.worker.npu.stage_projection import (
                install_npu_sid_stage_projection,
            )

            base_memory_usage = int(getattr(self, "model_memory_usage", 0))
            tokenizer = cached_tokenizer_from_config(self.model_config)
            if tokenizer is None:
                raise ValueError("beam.sid_stage_vocab requires tokenizer initialization")
            stage_vocab = build_sid_stage_vocab(tokenizer, _runner_vocab_size(self), self.device)
            projection = install_npu_sid_stage_projection(self, stage_vocab)
            memory_usage = (
                base_memory_usage
                + stage_vocab.device_memory_bytes
                + (projection.device_memory_bytes if projection is not None else 0)
            )
            stage_sizes = tuple(int(ids.numel()) for ids in stage_vocab.stage_token_ids)

            # Publish only when maps and the optional projection are complete.
            self._gr_npu_sid_stage_vocab = stage_vocab
            self._gr_npu_stage_projection = projection
            self.model_memory_usage = memory_usage
            logger.info(
                "NPU catalog-free SID vocabulary ready: stage sizes=%s, projection=%s",
                stage_sizes,
                "sliced" if projection is not None else "full",
            )
            return result
        except BaseException:
            self._gr_npu_sid_stage_vocab = None
            # A projection may be installed but not yet published if a later
            # preparation step fails. The shared closer owns published state.
            if (
                projection is not None
                and getattr(self, "_gr_npu_stage_projection", None) is not projection
            ):
                projection.close()
            from vllm_gr.v1.worker.model_runner_patch import close_constraint_backend

            close_constraint_backend(self)
            if base_memory_usage is not None:
                self.model_memory_usage = base_memory_usage
            raise

    setattr(load_model, marker, True)
    return load_model


def make_patched_shutdown(original: Callable[..., Any]) -> Callable[..., Any]:
    """Detach NPU SID maps before the shared projection/backend shutdown."""
    marker = "_vllm_gr_npu_sid_shutdown_patch"
    if getattr(original, marker, False):
        return original

    @wraps(original)
    def shutdown(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._gr_npu_sid_stage_vocab = None
        return original(self, *args, **kwargs)

    setattr(shutdown, marker, True)
    return shutdown


def make_patched_reload_weights(original):
    from vllm_gr.v1.worker.npu.stage_projection import make_npu_stage_reload_weights

    return make_npu_stage_reload_weights(original)


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

        return _orig_prepare_inputs(self, scheduler_output, num_scheduled_tokens)

    setattr(patched_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, True)
    return patched_prepare_inputs


def make_patched_sample_tokens_npu(_orig_sample_tokens):
    """Dispatch V1 stage sampling; ordinary requests use the native sampler."""
    if getattr(_orig_sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, False):
        return _orig_sample_tokens

    @wraps(_orig_sample_tokens)
    def patched_sample_tokens(self, grammar_output, *args, **kwargs):
        stage_runner = getattr(self, "_gr_npu_stage_runner", None)
        if stage_runner is not None and (
            stage_runner.pending_plan is not None or stage_runner.execution_error is not None
        ):
            return stage_runner.sample(grammar_output)
        return _orig_sample_tokens(self, grammar_output, *args, **kwargs)

    setattr(patched_sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, True)
    return patched_sample_tokens


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
    """Reject native FULL for Beam; preserve native PIECEWISE/NONE dispatch."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    if result[0] == CUDAGraphMode.FULL:
        return (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=result[1].num_tokens)) + result[2:]
    return result


def _npu_graph_fallback(runner, result, reason: str):
    """Count Beam FULL misses, not necessarily whole-forward eager execution."""

    runner._beam_graph_fallback_count = int(getattr(runner, "_beam_graph_fallback_count", 0)) + 1
    reasons = getattr(runner, "_beam_graph_fallback_reasons", None)
    if reasons is None:
        reasons = {}
        runner._beam_graph_fallback_reasons = reasons
    reasons[reason] = int(reasons.get(reason, 0)) + 1
    return _downgrade_to_none(result)


def _npu_beam_identity_layout_matches(runner, beam_info, num_tokens: int) -> bool:
    """Require the identity row layout baked into the NPU Beam FULL graph.

    Attention's Python slice/gather choice is frozen during capture. Check
    physical scheduler counts and complete cached prefixes before replay,
    rather than trusting the runtime metadata flag to change that choice.
    """
    session = beam_info.get("beam_session")
    batch = int(getattr(session, "batch_size", 1))
    width = int(getattr(session, "request_width", getattr(session, "beam_width", 0)))
    req_ids = tuple(beam_info.get("req_ids") or ())
    scheduler_output = getattr(runner, "_current_scheduler_output", None)
    counts = getattr(scheduler_output, "num_scheduled_tokens", None)
    beam_data = beam_info.get("beam_data", {})
    query_starts = getattr(getattr(runner, "query_start_loc", None), "np", None)
    if (
        batch < 1
        or width < 1
        or num_tokens != batch * width
        or getattr(session, "attention_width", None) != num_tokens
        or len(req_ids) != batch
        or len(set(req_ids)) != batch
        or not isinstance(counts, dict)
        or tuple(counts) != req_ids
        or set(beam_data) != set(req_ids)
        or query_starts is None
        or len(query_starts) < batch + 1
        or tuple(query_starts[: batch + 1]) != tuple(range(0, num_tokens + 1, width))
    ):
        return False
    for req_id in req_ids:
        mdata = beam_data[req_id]
        prefix_len = mdata.get("prefix_len", 0)
        if (
            counts[req_id] != width
            or not mdata.get("is_beam_decode", False)
            or mdata.get("beam_width") != width
            or type(prefix_len) is not int
            or prefix_len < 1
            or mdata.get("cache_len") != prefix_len
            or type(mdata.get("decode_step")) is not int
            or mdata["decode_step"] < 0
            or mdata.get("beam_decode_steps") != mdata["decode_step"] + 1
        ):
            return False
    return True


def make_patched_determine_batch_execution_and_padding_npu(_orig_determine):
    """Override cudagraph_mode to FULL for beam decode steps (NPU/ACL graph).

    Mirrors the GPU-side override (``make_patched_determine_batch_execution_
    and_padding_gpu``): beam decode batches never satisfy the native
    ``_is_uniform_decode`` predicate (beam_width tokens != query_len 1), so
    without this hook dispatch can only ever select PIECEWISE — 29 subgraph
    replays per forward with every beam attention call eager.  For a PURE
    beam-decode batch with an active BeamSearchSession, an exactly-matching
    startup-captured FULL graph and the graph-safe xllm execution mode, the
    dispatch result is rewritten to ``(FULL, BatchDescriptor(num_tokens=B*W,
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
            return _npu_graph_fallback(self, result, "inactive_session")

        # FULL is granted only for a PURE beam-decode batch: the captured
        # beam graph contains no standard-attention path.  ``req_ids``
        # (filled by the patched _prepare_inputs) enumerates the whole batch.
        batch_req_ids = set(beam_info.get("req_ids") or [])
        if batch_req_ids != beam_keys:
            return _npu_graph_fallback(self, result, "mixed_batch")

        # The startup beam graph is captured only for the default no-LoRA
        # descriptor and ``_beam_full_graph_ready`` is keyed by width alone,
        # so a beam batch with active LoRA adapters must not be granted FULL:
        # the rewritten descriptor would drop has_lora/num_active_loras and
        # silently route the batch to the LoRA-free graph. Keep native
        # PIECEWISE/NONE until LoRA-aware Beam FULL descriptors are captured.
        if getattr(result[1], "has_lora", False):
            return _npu_graph_fallback(self, result, "lora")

        # The captured graph bakes a contiguous request-ordered pool view that
        # starts at slot 0. NPUBeamExecutionBatch enforces all remaining slots.
        if getattr(session, "slot", 0) != 0:
            return _npu_graph_fallback(self, result, "nonzero_pool_slot")

        # Beam graphs are opt-in via GRConfig (beam_max_width) and can be
        # disabled for A/B comparison (beam_graph_enabled / --no-beam-graph).
        if not _beam_graph_enabled_for_run(self):
            return _npu_graph_fallback(self, result, "graph_disabled")

        # Only the fused xllm path is graph-safe on NPU (the cascade path
        # bakes host-int pool slices; see _npu_xllm_decode_mode_active).
        if not _npu_xllm_decode_mode_active():
            return _npu_graph_fallback(self, result, "unsupported_decode_mode")

        # Exact-width gate: compare the UNPADDED scheduled token count
        # against the captured graph's width (the beam width for a pure
        # batch).  num_tokens is argument 0 on the NPU runner contract.
        num_tokens_arg = kwargs.get("num_tokens")
        if num_tokens_arg is None and args:
            num_tokens_arg = args[0]
        real_num_tokens = (
            int(num_tokens_arg) if num_tokens_arg is not None else result[1].num_tokens
        )
        # FULL iff the exact (B, W) shape and physical token bucket were
        # captured; otherwise preserve native PIECEWISE/NONE dispatch, never
        # a native FULL graph that does not contain the Beam attention path.
        batch_size = int(getattr(session, "batch_size", 1))
        request_width = int(getattr(session, "request_width", getattr(session, "beam_width", 0)))
        captured_shapes = getattr(self, "_gr_npu_beam_graph_shapes", set())
        if (batch_size, request_width) not in captured_shapes:
            return _npu_graph_fallback(self, result, "shape_miss")
        if not _npu_beam_identity_layout_matches(self, beam_info, real_num_tokens):
            return _npu_graph_fallback(self, result, "identity_layout")
        if not _beam_full_graph_ready(self, real_num_tokens):
            return _npu_graph_fallback(self, result, "token_bucket_miss")

        from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

        new_descriptor = BatchDescriptor(num_tokens=real_num_tokens, uniform=True)
        self._beam_graph_full_dispatch_count = (
            int(getattr(self, "_beam_graph_full_dispatch_count", 0)) + 1
        )
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
    runner,
    gr_cfg,
    batch_size: int,
    width: int,
    steps_upper: int,
    prefix_len: int,
    context=None,
) -> tuple:
    """Create an exact-B context and production NPU attention facade.

    Real and dummy Decode share request ordering, pool views, and buffers.
    """
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor

    from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
    from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
    from vllm_gr.v1.worker.npu.beam_execution_batch import NPUBeamExecutionBatch

    request_ids = tuple(
        _BEAM_DUMMY_REQ_ID if batch_size == 1 else f"{_BEAM_DUMMY_REQ_ID}-{index}"
        for index in range(batch_size)
    )
    execution = BeamBatchExecutionView(
        session_id=_BEAM_DUMMY_REQ_ID,
        dispatch_id=1,
        stage_index=1,
        kind="decode",
        decode_step=0,
        owner_generation=0,
        request_ids=request_ids,
        num_prompt_tokens_by_request=(prefix_len,) * batch_size,
        beam_width=int(gr_cfg.beam_max_width),
        produces_output=True,
        dynamic_beam_widths=(width, width, width) if width != int(gr_cfg.beam_max_width) else None,
    )
    if context is None:
        context = _create_beam_context(runner, gr_cfg, max_batch=batch_size)
        if getattr(gr_cfg, "beam_graph_widths", None) is not None:
            from vllm_gr.v1.beam.dynamic_beam import (
                initialize_packed_stage_pools,
            )

            initialize_packed_stage_pools(context)
    sessions = []
    for request_id in request_ids:
        slot = context.registry.allocate(
            request_id,
            beam_width=int(gr_cfg.beam_max_width),
            prefix_len=prefix_len,
        )
        session = BeamSearchSession(session_id=request_id, slot=slot, context=context)
        session.begin_session(int(gr_cfg.beam_max_width), prefix_len)
        sessions.append(session)
    attention_batch = NPUBeamExecutionBatch(execution, sessions)

    runner._beam_context = context
    runner._beam_sessions = {session.session_id: session for session in sessions}

    beam_data = {
        request_id: {
            "prefix_len": prefix_len,
            "cache_len": prefix_len,
            "max_decode_steps": steps_upper,
            "is_beam_decode": True,
            "beam_width": width,
            "beam_decode_steps": 1,
            "parent_beam_ids": [0] * width,
            "decode_step": 0,
        }
        for request_id in request_ids
    }
    beam_info = {
        "beam_data": beam_data,
        "req_ids": list(request_ids),
        "beam_session": attention_batch,
    }

    batch_descriptor = BatchDescriptor(num_tokens=batch_size * width, uniform=True)
    return context, attention_batch, beam_info, batch_descriptor


def _dummy_beam_forward_npu(
    runner,
    context,
    beam_info,
    batch_descriptor,
    batch_size: int,
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
    5. ``commit_block_table`` for B logical requests.
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

    from vllm_gr.v1.beam import BEAM_DATA_VAR

    num_tokens = batch_size * width
    # 1. Refresh the per-step device buffers (same as the runtime path).
    context.decode_step_buf.fill_(0)
    if hasattr(context, "decode_step_1based_buf"):
        context.decode_step_1based_buf.fill_(1)
    if hasattr(context, "xllm_block_table_buf"):
        context.xllm_block_table_buf.fill_(0)
        context.xllm_block_table_buf[:batch_size].copy_(
            torch.arange(batch_size, dtype=torch.int32, device=runner.device)
        )

    # 2. Dummy inputs in the runner's persistent buffers (fixed addresses).
    input_ids = runner.input_ids.gpu[:num_tokens]
    input_ids.fill_(1)
    positions = runner.positions[:num_tokens]
    positions.fill_(prefix_len)

    # 3. Dummy batch state: B logical requests with W scheduled tokens each.
    runner.optimistic_seq_lens_cpu[:batch_size] = prefix_len + width
    runner.optimistic_seq_lens_cpu[batch_size:] = 0
    runner.seq_lens.copy_(runner.optimistic_seq_lens_cpu, non_blocking=True)

    # 4. query_start_loc: CPU write precedes copy_to_gpu.
    runner.query_start_loc.np[: batch_size + 1] = list(range(0, num_tokens + 1, width))
    runner.query_start_loc.np[batch_size + 1 :] = 0
    runner.query_start_loc.copy_to_gpu()

    # 5. Commit one prefix block table row per logical request.
    runner.input_batch.block_table.commit_block_table(batch_size)

    # 6. Dummy runs have no real slot assignments — fill with -1 so
    #    reshape_and_cache kernels skip the KV write.  (The NPU
    #    _build_attention_metadata derives slot mappings from the committed
    #    block table; priming the mappings here mirrors _dummy_run so any
    #    pre-fill of stale slot ids is overwritten.)
    slot_mappings_by_group, _ = runner._get_slot_mappings(
        num_tokens_padded=num_tokens,
        num_reqs_padded=batch_size,
        num_tokens_unpadded=num_tokens,
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
            num_tokens=num_tokens,
            num_reqs=batch_size,
            max_query_len=width,
            num_tokens_padded=num_tokens,
            num_reqs_padded=batch_size,
            for_cudagraph_capture=capture,
        )
        with set_ascend_forward_context(
            metadata,
            runner.vllm_config,
            num_tokens=num_tokens,
            num_actual_tokens=num_tokens,
            aclgraph_runtime_mode=(CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE),
            batch_descriptor=batch_descriptor,
            model_instance=runner.model,
        ):
            runner.model(input_ids=input_ids, positions=positions)
    finally:
        BEAM_DATA_VAR.reset(token)


def _verify_and_record_captured_graph_npu(runner, width: int) -> bool:
    """Record only graphs owned by this runner's installed Beam wrapper."""
    beam_model = getattr(runner, "_gr_npu_graph_model", None)
    if beam_model is None or getattr(runner, "model", None) is not beam_model:
        return False
    if not beam_model.graph_ready(width):
        return False
    keys = getattr(runner, "_beam_graph_keys", None)
    if keys is None:
        keys = set()
        runner._beam_graph_keys = keys
    keys.add(width)
    return True


def _capture_beam_decode_graph_width(runner, width: int) -> bool:
    """Startup dummy capture of the beam-decode FULL ACL graph (NPU).

    Runs once after ``capture_model()`` when the GRConfig beam graph is
    active (``beam_graph_active`` and ``beam_max_width`` set).  Mirrors the
    GPU-side flow: N eager warmup forwards (mode NONE — primes the xllm
    cache_unshared_kv / x_attention operators), then one FULL-mode forward
    which the ACLGraphWrapper captures under
    ``BatchDescriptor(num_tokens=B*beam_max_width, uniform=True)``.

    Every runtime-varying kernel input is a persistent fixed-address device
    buffer refreshed outside the graph each step (metadata builder's
    ``_fixed_buffers`` pool; the beam context's ``decode_step_buf`` /
    ``decode_step_1based_buf`` / ``xllm_block_table_buf``), so the single
    graph serves the configured exact B/W across decode steps and prompt
    lengths. The capture output is discarded; real requests replay only
    (``patched_determine`` + ``_beam_full_graph_ready`` gate FULL).

    Returns True if a graph was captured.
    """
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled

    from vllm_gr.v1.worker.gr_context import (
        _get_gr_config,
    )

    gr_cfg = _get_gr_config(runner)
    if not gr_cfg.beam_graph_active:
        return False

    # Only the fused xllm execution path is graph-safe on NPU (the cascade
    # path slices the beam pool with host-int decode_step values, which a
    # captured graph would freeze).
    if not _npu_xllm_decode_mode_active():
        logger.info(
            "vllm-gr: beam decode ACL graph requires the xllm decode execution "
            "mode; Beam FULL graph will not be used"
        )
        return False

    # ── TP / ubatching guards (same rationale as the GPU side) ──────
    tp_size = runner.parallel_config.tensor_parallel_size
    if tp_size > 1:
        logger.warning(
            "vllm-gr: beam decode ACL graph is not supported with "
            "tensor parallelism (TP=%d); Beam FULL graph will not be used",
            tp_size,
        )
        return False
    if runner.parallel_config.use_ubatching:
        logger.warning(
            "vllm-gr: beam decode ACL graph is not supported with "
            "DP ubatching; Beam FULL graph will not be used",
        )
        return False
    # ─────────────────────────────────────────────────────────────────

    # Capture one exact logical-B shape; other B values keep native
    # PIECEWISE/NONE dispatch. Never pad inactive rows into Beam state/KV ownership.
    batch_size = int(getattr(gr_cfg, "beam_graph_batch_size", 1))
    num_tokens = batch_size * width
    from vllm_gr.v1.worker.npu.npu_beam_decode_graph import install_npu_beam_graph_model

    # The implicit capacity-width capture needs the same isolation as explicit
    # stage widths. Install before the cached-shape check and verify ownership.
    install_npu_beam_graph_model(runner)
    captured_shapes = getattr(runner, "_gr_npu_beam_graph_shapes", set())
    if (batch_size, width) in captured_shapes and _verify_and_record_captured_graph_npu(
        runner, num_tokens
    ):
        return True

    scheduler_config = getattr(runner, "scheduler_config", None)
    max_num_seqs = int(getattr(scheduler_config, "max_num_seqs", batch_size))
    max_num_batched_tokens = int(getattr(scheduler_config, "max_num_batched_tokens", num_tokens))
    if batch_size > max_num_seqs or num_tokens > max_num_batched_tokens:
        logger.warning(
            "vllm-gr: NPU Beam graph shape B=%d, W=%d exceeds scheduler "
            "capacity (max_num_seqs=%d, max_num_batched_tokens=%d); "
            "Beam FULL graph will not be used",
            batch_size,
            width,
            max_num_seqs,
            max_num_batched_tokens,
        )
        return False

    steps_upper = int(gr_cfg.beam_max_decode_steps)
    model_capacity = int(
        getattr(
            getattr(runner, "model_config", None),
            "max_model_len",
            _BEAM_CAPTURE_PREFIX_LEN + width,
        )
    )
    if model_capacity <= width:
        logger.warning(
            "vllm-gr: NPU Beam graph needs max_model_len > W; Beam FULL graph will not be used"
        )
        return False
    prefix_len = min(_BEAM_CAPTURE_PREFIX_LEN, model_capacity - width)

    context, attention_batch, beam_info, batch_descriptor = _capture_create_dummy_context_npu(
        runner,
        gr_cfg,
        batch_size,
        width,
        steps_upper,
        prefix_len,
        getattr(runner, "_gr_npu_graph_context", None),
    )
    runner._gr_npu_graph_context = context
    runner._gr_npu_graph_batch_size = batch_size

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
                        batch_size,
                        width,
                        prefix_len,
                        capture=False,
                    )
                _dummy_beam_forward_npu(
                    runner,
                    context,
                    beam_info,
                    batch_descriptor,
                    batch_size,
                    width,
                    prefix_len,
                    capture=True,
                )
            torch.npu.current_stream().wait_stream(side)
            torch.npu.synchronize()

            captured = _verify_and_record_captured_graph_npu(runner, num_tokens)
            if captured:
                if getattr(context, "_npu_packed_width_pools", None) is not None:
                    context._npu_beam_graph_captured = True
                shapes = getattr(runner, "_gr_npu_beam_graph_shapes", None)
                if shapes is None:
                    shapes = set()
                    runner._gr_npu_beam_graph_shapes = shapes
                shapes.add((batch_size, width))
                contexts = getattr(runner, "_gr_npu_graph_contexts", None)
                if contexts is None:
                    contexts = {}
                    runner._gr_npu_graph_contexts = contexts
                contexts[(batch_size, width)] = context
    finally:
        set_cudagraph_capturing_enabled(False)
        # Drop the dummy session even if capture threw.  The context and
        # pools stay — a captured graph (if any) references their addresses.
        attention_batch.end_session()
        for dummy_session in attention_batch.sessions:
            runner._beam_sessions.pop(dummy_session.session_id, None)
        runner._current_beam_info = None

    if captured:
        logger.info(
            "vllm-gr: startup dummy capture of NPU Beam decode ACL graph "
            "succeeded (B=%d, W=%d, num_tokens=%d, uniform=True); "
            "runtime is replay-only",
            batch_size,
            width,
            num_tokens,
        )
    else:
        logger.warning(
            "vllm-gr: NPU Beam dummy capture did NOT produce an ACL graph "
            "for B=%d, W=%d, num_tokens=%d; Beam FULL graph will not be used",
            batch_size,
            width,
            num_tokens,
        )
    return captured


def _retained_npu_suffix_kv_bytes(runner: Any) -> int:
    """Count unique retained suffix-bank storages, excluding graph/scratch memory."""
    contexts = list(getattr(runner, "_gr_npu_graph_contexts", {}).values())
    context = getattr(runner, "_gr_npu_graph_context", None)
    if context is not None:
        contexts.append(context)
    storages: dict[tuple[Any, int], int] = {}
    for context in contexts:
        banks = [getattr(context, "pools", ())]
        banks.extend(getattr(context, "_npu_beam_width_pools", {}).values())
        for pools in banks:
            for pool in pools:
                for tensor in (pool.unshared_key, pool.unshared_value):
                    storage = tensor.untyped_storage()
                    storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def capture_beam_decode_graph(runner: Any) -> bool:
    """Capture configured widths over private graphs and stable suffix banks."""
    from vllm_gr.v1.worker.gr_context import (
        _get_gr_config,
    )

    config = _get_gr_config(runner)
    if not config.beam_graph_active:
        return False
    widths = set(getattr(config, "beam_graph_widths", None) or ())
    widths.add(int(config.beam_max_width))
    selected_widths = sorted(widths, reverse=True)
    logger.info(
        "vllm-gr: NPU Beam graph capture widths=%s (count=%d, B=%d)",
        selected_widths,
        len(selected_widths),
        int(getattr(config, "beam_graph_batch_size", 1)),
    )
    started = perf_counter()
    captured = False
    try:
        for width in selected_widths:
            captured = _capture_beam_decode_graph_width(runner, width) or captured
    finally:
        shapes = sorted(getattr(runner, "_gr_npu_beam_graph_shapes", ()))
        logger.info(
            "vllm-gr: NPU Beam graph capture finished in %.3f s; captured "
            "(B, W)=%s; retained suffix-KV bank storage=%.3f MiB "
            "(excludes graph and scratch memory)",
            perf_counter() - started,
            shapes,
            _retained_npu_suffix_kv_bytes(runner) / (1024 * 1024),
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
make_patched_update_states = make_patched_update_states_npu
# The dispatcher (model_runner_patch.py) looks up this exact name; without the
# alias the sample_tokens hook is silently skipped and the worker beam decision
# never reaches the ModelRunnerOutput (async scheduling keeps the decision path
# broken end-to-end: engine-driven chaining no-ops and the frontend keeps its RPC loop).
make_patched_sample_tokens = make_patched_sample_tokens_npu
make_patched_determine_batch_execution_and_padding = (
    make_patched_determine_batch_execution_and_padding_npu
)


#  Read actual packed row order; do not modify latest stage/decode hooks.
def _select_bsnd_spec(runner, scheduler_output, p_runner, num_reqs):
    """BSND eligibility: complete prompts without history, fitting each row.

    Every scheduled request must have num_computed_tokens == 0 (its full
    prompt is scheduled in this batch) and max_len <= P of a captured (T,B)
    spec. Returns (T, B, token_counts) or None.
    """
    #  computed==0 and scheduled==prompt
    num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(num_scheduled, dict) or len(num_scheduled) != num_reqs:
        return None
    input_batch = getattr(runner, "input_batch", None)
    computed = getattr(input_batch, "num_computed_tokens_cpu", None)
    prompt = getattr(input_batch, "num_prompt_tokens", None)
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if computed is None or prompt is None or req_id_to_index is None:
        return None
    # Packed model inputs follow input-batch row order,
    # not scheduler dictionary insertion order (the failure log contains 0,2,1,3).
    row_counts = []
    for req_id, n_sched in num_scheduled.items():
        row = req_id_to_index.get(req_id)
        if row is None or computed[row] != 0:
            return None
        if prompt[row] is None or n_sched != prompt[row]:
            return None
        row_counts.append((row, int(n_sched)))
    row_counts.sort()
    if [row for row, _ in row_counts] != list(range(num_reqs)):
        return None
    token_counts = [count for _, count in row_counts]
    if len(token_counts) != num_reqs or any(c <= 0 for c in token_counts):
        return None
    # Match request count and the longest per-request length.
    max_len = max(token_counts)
    spec = p_runner.match_bsnd_bucket(num_reqs, max_len)
    if spec is None:
        return None
    T, B = spec
    if not p_runner.has_bsnd_captured_graph(T, B):
        return None
    return (T, B, token_counts)


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
        p_runner = getattr(self, "_prefill_graph_runner", None)
        if p_runner is not None:
            # Check before every route, including native and BSND forwards.
            p_runner.raise_if_failed()
        scheduler_output = getattr(self, "_current_scheduler_output", None)
        # BSND/TND graph wrappers cannot consume non-token model payloads.
        if intermediate_tensors is not None or inputs_embeds is not None or model_kwargs:
            return _orig_model_forward(
                self,
                num_tokens_padded,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                **model_kwargs,
            )
        if (
            scheduler_output is not None
            and is_prefill_graph_enabled(self)
            and is_prefill_phase_batch(self, scheduler_output)
        ):
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
                # Try optional BSND before TND; preserve compact logits indices.
                # Each request must be a complete prompt without cached history.
                bsnd_spec = None
                if getattr(p_runner, "has_bsnd_captured_graph", None) is not None:
                    bsnd_spec = _select_bsnd_spec(self, scheduler_output, p_runner, num_reqs)
                if bsnd_spec is not None:
                    T, B, token_counts = bsnd_spec
                    slot_mapping = None
                    try:
                        from vllm.forward_context import get_forward_context

                        attn_metadata = getattr(get_forward_context(), "attn_metadata", None)
                        if isinstance(attn_metadata, dict) and attn_metadata:
                            slot_mapping = next(iter(attn_metadata.values())).slot_mapping
                    except Exception as e:  # noqa: BLE001
                        logger.debug("Could not get slot_mapping (BSND): %r", e)
                    output_buf = p_runner.execute_model_prefill_bsnd(
                        T=T,
                        B=B,
                        input_ids=input_ids,
                        positions=positions,
                        slot_mapping=slot_mapping,
                        token_counts=token_counts,
                    )
                    if output_buf is not None:
                        p_runner.log_replay_bsnd(T, B, total_tokens)
                        return output_buf
                    # BSND miss → fall through to the TND path below.

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
    only when the prefill graph env gate is enabled. With BSND disabled,
    failures fall back to eager prefill. Explicit BSND configuration makes
    capture failures fatal so deployment errors are not silently hidden.
    """
    if not is_prefill_graph_enabled(runner):
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
        # Propagate startup/capture failures when BSND is enabled.

        settings = deployment_for(getattr(runner, "vllm_config", None))
        specs = settings.npu_bsnd_capture_specs
        if specs.strip():
            raise
        logger.exception(
            "vllm-gr: prefill NPU graph capture failed; prefill will fall back to eager"
        )


make_patched_model_forward = make_patched_model_forward_npu


def is_patched(runner_class: type) -> bool:
    from vllm_gr.v1.worker.runner_lifecycle import has_runner_lifecycle

    return is_worker_runner_patched(runner_class) and has_runner_lifecycle(runner_class, __name__)


def install() -> None:
    """Install the Ascend runner after its native worker initializes device ops."""
    from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES
    from vllm_gr.v1.worker.model_runner_patch import install_v1_worker_hooks

    install_v1_worker_hooks(
        sys.modules[__name__],
        constraint_max_batch=max(SUPPORTED_BEAM_BATCH_SIZES),
        implementation_id=__name__,
    )
