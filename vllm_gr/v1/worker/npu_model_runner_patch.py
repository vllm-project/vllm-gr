# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU ModelRunner wrappers for grouped Beam execution."""

from __future__ import annotations

from functools import wraps

import torch
from vllm.logger import init_logger

from vllm_gr.v1.worker.model_runner_common import (
    _WORKER_BOOKKEEPING_PATCH_MARKER,
    _WORKER_PREPARE_PATCH_MARKER,
    InputBatchProxy,
    _beam_collapse_bookkeeping,
    _beam_remap_inputs,
)

logger = init_logger(__name__)


def make_patched_prepare_inputs_npu(_orig_prepare_inputs):
    """Wrap the NPU four-item ``_prepare_inputs`` contract."""
    if getattr(_orig_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False):
        return _orig_prepare_inputs

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
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
        logits_indices = torch.tensor(new_logits_indices, dtype=torch.int32, device=self.device)

        return (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
            num_scheduled_tokens_compressed_list,
        )

    setattr(patched_prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, True)
    return patched_prepare_inputs


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

        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
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
