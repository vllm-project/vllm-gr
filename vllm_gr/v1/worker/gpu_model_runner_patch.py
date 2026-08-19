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

logger = init_logger(__name__)


def make_patched_prepare_inputs_gpu(_orig_prepare_inputs):
    """Wrap the GPU two-item ``_prepare_inputs`` contract."""
    if getattr(_orig_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False):
        return _orig_prepare_inputs

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
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

        # The scheduler contract requires LogprobsLists (with slice_request),
        # not a nested Python list. _beam_collapse_bookkeeping has already
        # remuxed and resized these tensors to one row per logical Request.
        remuxed_logprobs = [[] for _ in range(n)] if logprobs_lists is not None else None
        remuxed_sampled_token_ids = [[] for _ in range(n)]
        logprobs_tensors = getattr(sampler_output, "logprobs_tensors", None)
        if logprobs_lists is not None and logprobs_tensors is not None:
            remuxed_logprobs = logprobs_tensors.tolists()
        sampled_token_ids = getattr(sampler_output, "sampled_token_ids", None)
        if sampled_token_ids is not None:
            remuxed_sampled_token_ids = self._to_list(sampled_token_ids[:n])

        return (
            num_nans_in_logits,
            remuxed_logprobs,
            remuxed_sampled_token_ids,
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


make_patched_prepare_inputs = make_patched_prepare_inputs_gpu
make_patched_bookkeeping_sync = make_patched_bookkeeping_sync_gpu
