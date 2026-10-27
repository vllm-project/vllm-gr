# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU ModelRunner wrappers for grouped Beam execution."""

from __future__ import annotations

from functools import wraps

import torch
from vllm.logger import init_logger

from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES
from vllm_gr.v1.beam.execution_batch import (
    BeamBatchExecutionView,
    BeamExecutionContractError,
)
from vllm_gr.v1.worker.beam_decode_graph import close_beam_decode_runtime
from vllm_gr.v1.worker.gpu_prefill_inputs import try_prepare_gr_prefill_inputs
from vllm_gr.v1.worker.model_runner_common import (
    _WORKER_MODEL_FORWARD_PATCH_MARKER,
    _WORKER_PREPARE_PATCH_MARKER,
    _WORKER_SAMPLE_TOKENS_PATCH_MARKER,
)
from vllm_gr.v1.worker.prefill_graph_runner import (
    SUPPORTED_NUM_REQS,
    get_prefill_graph_runner,
    is_prefill_graph_enabled,
    is_prefill_phase_batch,
)

logger = init_logger(__name__)
close_runtime = close_beam_decode_runtime

_BEAM_POSTPROCESS_LOAD_PATCH_MARKER = "_vllm_gr_beam_postprocess_load_patched"


def initialize_beam_postprocess_workspaces(
    runner,
    *,
    max_batch: int | None = None,
):
    """Prewarm the configured CUDA Beam postprocess bucket on ``runner``.

    V1 reserves the largest supported logical batch so every admitted size
    reuses the same fixed-address state and selection storage.
    """

    device = torch.device(getattr(runner, "device", "cpu"))
    if device.type != "cuda":
        return None

    from vllm_gr.config import BeamExecutionMode
    from vllm_gr.v1.beam.beam_decision import (
        BeamPostprocessWorkspace,
        BeamPostprocessWorkspaceKey,
        BeamPostprocessWorkspacePool,
    )
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    gr_cfg = _get_gr_config(runner)
    width = gr_cfg.beam_max_width
    if not bool(gr_cfg.beam_worker_decision) or width is None:
        return None

    context = getattr(runner, "_beam_context", None)
    # The Beam context is created later, on the first admitted batch, with
    # max_batch=max(SUPPORTED_BEAM_BATCH_SIZES); the workspace built here must
    # match it or the stage runner rejects the session as incompatible.
    default_max_batch = (
        max(SUPPORTED_BEAM_BATCH_SIZES) if gr_cfg.beam_execution_mode is BeamExecutionMode.V1 else 1
    )
    resolved_max_batch = (
        int(max_batch)
        if max_batch is not None
        else int(getattr(context, "max_batch", default_max_batch))
    )
    dtype = getattr(getattr(runner, "model_config", None), "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(
            f"CUDA Beam postprocess workspace requires model_config.dtype, got {dtype!r}"
        )

    key = BeamPostprocessWorkspaceKey(int(width), int(width), dtype)
    existing = getattr(runner, "_beam_postprocess_workspaces", None)
    if existing is not None:
        workspace = existing.require(key)
        if workspace.max_batch != resolved_max_batch or workspace.max_decode_steps != int(
            gr_cfg.beam_max_decode_steps
        ):
            raise RuntimeError(
                "prewarmed Beam postprocess workspace geometry changed after "
                "initialization; rebuilding fixed-address buffers is forbidden"
            )
        return existing

    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=resolved_max_batch,
        beam_width=int(width),
        top_k=int(width),
        max_decode_steps=int(gr_cfg.beam_max_decode_steps),
        device=device,
        logits_dtype=dtype,
        pin_host=True,
    )
    pool = BeamPostprocessWorkspacePool()
    pool.add(workspace)
    runner._beam_postprocess_workspaces = pool
    logger.info(
        "vllm-gr: prewarmed CUDA Beam postprocess workspace "
        "(batch=%d, width=%d, top_k=%d, dtype=%s, device_bytes=%d, pinned_bytes=%d)",
        resolved_max_batch,
        width,
        width,
        dtype,
        workspace.device_memory_bytes,
        workspace.pinned_memory_bytes,
    )
    return pool


def make_load_model_with_beam_postprocess_workspace(original):
    """Initialize fixed Beam postprocess storage after model weights load."""

    if getattr(original, _BEAM_POSTPROCESS_LOAD_PATCH_MARKER, False):
        return original

    @wraps(original)
    def load_model_with_beam_postprocess_workspace(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        workspace_pool = initialize_beam_postprocess_workspaces(self)
        if workspace_pool is not None:
            # The pool is allocated after native load_model has measured the
            # model weights, but before Worker memory profiling.  Treat its
            # device storage as model-owned memory so KV-cache capacity is not
            # computed from bytes that are already permanently resident.
            self.model_memory_usage = int(getattr(self, "model_memory_usage", 0)) + int(
                workspace_pool.device_memory_bytes
            )
        return result

    setattr(
        load_model_with_beam_postprocess_workspace,
        _BEAM_POSTPROCESS_LOAD_PATCH_MARKER,
        True,
    )
    return load_model_with_beam_postprocess_workspace


def make_patched_prepare_inputs_gpu(_orig_prepare_inputs):
    """Wrap the GPU two-item ``_prepare_inputs`` contract."""
    if getattr(_orig_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False):
        return _orig_prepare_inputs

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        # Publish the scheduler output so the _model_forward prefill-graph
        # interception can classify the batch and extract token counts.
        self._current_scheduler_output = scheduler_output

        logits_indices = try_prepare_gr_prefill_inputs(self, scheduler_output, num_scheduled_tokens)
        if logits_indices is None:
            logits_indices, spec_decode_metadata = _orig_prepare_inputs(
                self, scheduler_output, num_scheduled_tokens
            )
        else:
            spec_decode_metadata = None

        # GPU attention metadata is built after _prepare_inputs returns.
        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = self.input_batch.req_ids

        return logits_indices, spec_decode_metadata

    setattr(patched_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, True)
    return patched_prepare_inputs


def make_patched_sample_tokens_gpu(_orig_sample_tokens):
    """Attach a CUDA worker decision produced during async sampling."""
    if getattr(_orig_sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, False):
        return _orig_sample_tokens

    @wraps(_orig_sample_tokens)
    def patched_sample_tokens(self, grammar_output, *args, **kwargs):
        stage_runner = getattr(self, "_gr_gpu_stage_runner", None)
        if stage_runner is not None and stage_runner.pending is not None:
            return stage_runner.sample(grammar_output)
        return _orig_sample_tokens(self, grammar_output, *args, **kwargs)

    setattr(patched_sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, True)
    return patched_sample_tokens


def discover_worker_runners() -> tuple[tuple[str, type], ...]:
    """Discover the upstream GPU ModelRunner without importing NPU runtime."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as exc:
        logger.warning(
            "Failed to import vllm.v1.worker.gpu_model_runner.GPUModelRunner; "
            "skipping vllm-gr GPU beam-search runner patches for this attempt. "
            "ImportError: %s",
            exc,
        )
        logger.debug("GPUModelRunner import traceback", exc_info=True)
        return ()
    return (("GPUModelRunner", GPUModelRunner),)


make_patched_prepare_inputs = make_patched_prepare_inputs_gpu
make_patched_sample_tokens = make_patched_sample_tokens_gpu


def make_patched_model_forward_gpu(_orig_model_forward):
    """Wrap _model_forward to intercept and replay pure prefill CUDA graphs."""
    if getattr(_orig_model_forward, _WORKER_MODEL_FORWARD_PATCH_MARKER, False):
        return _orig_model_forward

    @wraps(_orig_model_forward)
    def patched_model_forward(self, input_ids=None, positions=None, *args, **kwargs):
        scheduler_output = getattr(self, "_current_scheduler_output", None)
        gr_stages = getattr(scheduler_output, "gr_stage_metadata", None)
        batch_view = None
        if gr_stages:
            # Decode uses its independent startup graph. A Batch Session has
            # exactly one stage descriptor whose request_ids define the
            # immutable logical order for this Prefill dispatch.
            if len(gr_stages) != 1:
                return _orig_model_forward(
                    self, input_ids=input_ids, positions=positions, *args, **kwargs
                )
            try:
                batch_view = BeamBatchExecutionView.from_stage_metadata(
                    next(iter(gr_stages.values()))
                )
            except BeamExecutionContractError:
                logger.debug("Prefill CUDA graph fallback: invalid_batch_contract")
                return _orig_model_forward(
                    self, input_ids=input_ids, positions=positions, *args, **kwargs
                )
            if batch_view.kind != "prefill":
                return _orig_model_forward(
                    self, input_ids=input_ids, positions=positions, *args, **kwargs
                )
            if getattr(self.model_config, "enforce_eager", False):
                logger.debug("Prefill CUDA graph fallback: enforce_eager")
                return _orig_model_forward(
                    self, input_ids=input_ids, positions=positions, *args, **kwargs
                )
            if is_prefill_graph_enabled() and not hasattr(self, "_prefill_graph_runner"):
                logger.debug("Prefill CUDA graph fallback: runner_not_initialized")
                return _orig_model_forward(
                    self, input_ids=input_ids, positions=positions, *args, **kwargs
                )
        if scheduler_output is not None:
            if is_prefill_graph_enabled() and is_prefill_phase_batch(self, scheduler_output):
                raw_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", [])
                # The scheduler already computed the batch's token total; only
                # the request count is derived here (the runner re-derives the
                # per-request counts it needs from the dict itself).
                if batch_view is not None:
                    if not isinstance(raw_scheduled_tokens, dict):
                        logger.debug(
                            "Prefill CUDA graph fallback: batch_session_missing_request_ids"
                        )
                        return _orig_model_forward(
                            self, input_ids=input_ids, positions=positions, *args, **kwargs
                        )
                    try:
                        batch_view.validate_request_order(tuple(raw_scheduled_tokens))
                    except BeamExecutionContractError:
                        logger.debug("Prefill CUDA graph fallback: batch_session_request_order")
                        return _orig_model_forward(
                            self, input_ids=input_ids, positions=positions, *args, **kwargs
                        )
                num_reqs = (
                    batch_view.batch_size if batch_view is not None else len(raw_scheduled_tokens)
                )
                total_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
                if not total_tokens and raw_scheduled_tokens:
                    total_tokens = (
                        sum(raw_scheduled_tokens.values())
                        if isinstance(raw_scheduled_tokens, dict)
                        else sum(raw_scheduled_tokens)
                    )

                if 1 <= num_reqs <= SUPPORTED_NUM_REQS[-1] and total_tokens > 0:
                    p_runner = get_prefill_graph_runner(self)
                    bucket = p_runner.match_bucket(total_tokens)
                    reason = "no_matching_bucket" if bucket is None else "graph_not_captured"
                    if gr_stages:
                        # Reuse the same padded geometry and static buffers as
                        # legacy Prefill. The runner keeps the real Attention
                        # lengths and masks padded KV slots. Async scheduling
                        # adds no exact-bucket requirement; adapters and invalid
                        # live input shapes still fall back to the original path.
                        if (
                            input_ids is None
                            or tuple(input_ids.shape) != (total_tokens,)
                            or positions is None
                            or tuple(positions.shape) != (total_tokens,)
                            or kwargs.get("inputs_embeds") is not None
                            or kwargs.get("intermediate_tensors") is not None
                        ):
                            reason = "unsupported_inputs"
                            bucket = None
                    if bucket is not None and p_runner.has_captured_graph(bucket, num_reqs):
                        # Extract slot mapping from active block table
                        slot_mapping = None
                        if (
                            hasattr(self, "input_batch")
                            and getattr(self.input_batch, "block_table", None) is not None
                        ):
                            try:
                                blk_table = self.input_batch.block_table[0]
                                if hasattr(blk_table, "slot_mapping") and hasattr(
                                    blk_table.slot_mapping, "gpu"
                                ):
                                    slot_mapping = blk_table.slot_mapping.gpu
                            except Exception as e:
                                logger.warning("Could not get slot_mapping: %r", e)
                        output_buf = p_runner.execute_model_prefill(
                            bucket=bucket,
                            input_ids=input_ids,
                            positions=positions,
                            slot_mapping=slot_mapping,
                            num_scheduled_tokens=raw_scheduled_tokens,
                        )

                        if output_buf is not None:
                            dispatches = getattr(self, "_gr_prefill_graph_dispatches", None)
                            if dispatches is None:
                                dispatches = self._gr_prefill_graph_dispatches = {}
                            key = f"replay:{num_reqs}"
                            dispatches[key] = dispatches.get(key, 0) + 1
                            logger.debug(
                                "Prefill CUDA graph dispatch: scheduled=%d requests=%d "
                                "bucket=%s outcome=replay",
                                total_tokens,
                                num_reqs,
                                bucket,
                            )
                            return output_buf
                        reason = getattr(p_runner, "last_replay_reason", "runner_rejected")
                    dispatches = getattr(self, "_gr_prefill_graph_dispatches", None)
                    if dispatches is None:
                        dispatches = self._gr_prefill_graph_dispatches = {}
                    key = f"fallback:{num_reqs}:{reason}"
                    dispatches[key] = dispatches.get(key, 0) + 1
                    logger.debug(
                        "Prefill CUDA graph dispatch: scheduled=%d requests=%d "
                        "bucket=%s outcome=%s",
                        total_tokens,
                        num_reqs,
                        bucket,
                        reason,
                    )
        return _orig_model_forward(self, input_ids=input_ids, positions=positions, *args, **kwargs)

    setattr(patched_model_forward, _WORKER_MODEL_FORWARD_PATCH_MARKER, True)
    return patched_model_forward


def capture_prefill_graphs(runner) -> None:
    """Startup capture of pure-prefill CUDA graphs keyed by bucket and request count.

    Chained after the beam decode-graph capture in ``capture_model``; runs
    only when the prefill graph env gate is enabled. Only explicitly
    recoverable capture failures fall back to eager prefill.
    """
    if not is_prefill_graph_enabled():
        return
    if getattr(runner, "_prefill_graphs_captured", False):
        return
    try:
        # CUDA graph capture requires a non-default stream; capture_model()
        # already returned, so its capture stream context is gone. Mirror the
        # beam decode-graph capture: a side stream, ordered against the
        # default stream before and after.
        side = torch.cuda.Stream()
        side.wait_stream(torch.accelerator.current_stream())
        with torch.cuda.stream(side):
            p_runner = get_prefill_graph_runner(runner)
            p_runner.capture_graphs(runner)
        torch.accelerator.current_stream().wait_stream(side)
        torch.accelerator.synchronize()
        runner._prefill_graphs_captured = True
    except Exception as error:
        from vllm_gr.v1.worker.cudagraph_capture import is_recoverable_capture_error

        if not isinstance(error, RuntimeError) or not is_recoverable_capture_error(error):
            raise
        torch.accelerator.synchronize()
        logger.exception(
            "vllm-gr: prefill CUDA graph capture failed; prefill will fall back to eager"
        )


make_patched_model_forward = make_patched_model_forward_gpu
