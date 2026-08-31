# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU ModelRunner wrappers for grouped Beam execution."""

from __future__ import annotations

from functools import wraps

import torch
from vllm.logger import init_logger

from vllm_gr.v1.worker.model_runner_common import (
    _WORKER_BOOKKEEPING_PATCH_MARKER,
    _WORKER_PREPARE_PATCH_MARKER,
    _beam_collapse_bookkeeping,
    _beam_remap_inputs,
)
from vllm_gr.v1.worker.prefill_graph_runner import (
    SUPPORTED_NUM_REQS,
    get_prefill_graph_runner,
    is_prefill_graph_enabled,
    is_prefill_phase_batch,
)

logger = init_logger(__name__)


def make_patched_prepare_inputs_gpu(_orig_prepare_inputs):
    """Wrap the GPU two-item ``_prepare_inputs`` contract."""
    if getattr(_orig_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False):
        return _orig_prepare_inputs

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        # Publish the scheduler output so the _model_forward prefill-graph
        # interception can classify the batch and extract token counts.
        self._current_scheduler_output = scheduler_output

        logits_indices, spec_decode_metadata = _orig_prepare_inputs(
            self, scheduler_output, num_scheduled_tokens
        )

        # GPU attention metadata is built after _prepare_inputs returns.
        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = self.input_batch.req_ids

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data:
            return logits_indices, spec_decode_metadata

        new_logits_indices = _beam_remap_inputs(self, scheduler_output, num_scheduled_tokens)
        return (
            torch.tensor(new_logits_indices, dtype=torch.int32, device=self.device),
            spec_decode_metadata,
        )

    setattr(patched_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, True)
    return patched_prepare_inputs


def make_patched_bookkeeping_sync_gpu(_orig_bookkeeping_sync):
    """Wrap the GPU seven-item ``_bookkeeping_sync`` contract."""
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
            return _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)

        request_row_configs = scheduler_output._gr_request_row_configs
        orig_input_batch = object.__getattribute__(self.input_batch, "_obj")

        try:
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
        finally:
            self.input_batch = orig_input_batch

        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = res

        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(
            self, orig_input_batch, request_row_configs, beam_data, sampler_output
        )

        return (
            num_nans_in_logits,
            [[] for _ in range(n)] if logprobs_lists is not None else None,
            [[] for _ in range(n)],
            prompt_logprobs_dict,
            collapsed_req_ids,
            collapsed_idx,
            invalid_req_indices,
        )

    setattr(patched_bookkeeping_sync, _WORKER_BOOKKEEPING_PATCH_MARKER, True)
    return patched_bookkeeping_sync


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


def make_patched_capture_cudagraphs_gpu(_orig_capture_cudagraphs):
    """Wrap _capture_cudagraphs to capture prefill graphs during warmup."""

    @wraps(_orig_capture_cudagraphs)
    def patched_capture_cudagraphs(
        self, batch_descriptors, cudagraph_runtime_mode, *args, **kwargs
    ):
        res = _orig_capture_cudagraphs(
            self, batch_descriptors, cudagraph_runtime_mode, *args, **kwargs
        )

        # Trigger prefill graph capture once when FULL decode graphs are being captured
        from vllm.config import CUDAGraphMode

        from vllm_gr.v1.worker.prefill_graph_runner import is_prefill_graph_enabled

        if (
            is_prefill_graph_enabled()
            and cudagraph_runtime_mode == CUDAGraphMode.FULL
            and not getattr(self, "_prefill_graphs_captured", False)
        ):
            self._prefill_graphs_captured = True
            p_runner = get_prefill_graph_runner(self)
            p_runner.capture_graphs(self)

        return res

    return patched_capture_cudagraphs


def make_patched_model_forward_gpu(_orig_model_forward):
    """Wrap _model_forward to intercept and replay pure prefill CUDA graphs."""

    @wraps(_orig_model_forward)
    def patched_model_forward(self, input_ids=None, positions=None, *args, **kwargs):
        scheduler_output = getattr(self, "_current_scheduler_output", None)
        if scheduler_output is not None:
            if is_prefill_graph_enabled() and is_prefill_phase_batch(self, scheduler_output):
                p_runner = get_prefill_graph_runner(self)

                raw_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", [])
                token_counts = (
                    list(raw_scheduled_tokens.values())
                    if isinstance(raw_scheduled_tokens, dict)
                    else list(raw_scheduled_tokens)
                )
                total_tokens = sum(token_counts) if token_counts else 0
                num_reqs = len(token_counts)

                if 1 <= num_reqs <= SUPPORTED_NUM_REQS[-1] and total_tokens > 0:
                    bucket = p_runner.match_bucket(total_tokens)
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
                        if slot_mapping is None:
                            logger.debug("Could not get slot_mapping: extraction unavailable.")

                        output_buf = p_runner.execute_model_prefill(
                            bucket=bucket,
                            input_ids=input_ids,
                            positions=positions,
                            slot_mapping=slot_mapping,
                            num_scheduled_tokens=raw_scheduled_tokens,
                        )

                        if output_buf is not None:
                            logger.debug(
                                "Prefill Graph Replay: Executed captured CUDA Graph for bucket %d (reqs=%d, %d actual tokens).",
                                bucket,
                                num_reqs,
                                total_tokens,
                            )
                            return output_buf

        return _orig_model_forward(self, input_ids=input_ids, positions=positions, *args, **kwargs)

    return patched_model_forward


make_patched_prepare_inputs = make_patched_prepare_inputs_gpu
make_patched_bookkeeping_sync = make_patched_bookkeeping_sync_gpu
make_patched_capture_cudagraphs = make_patched_capture_cudagraphs_gpu
make_patched_model_forward = make_patched_model_forward_gpu
