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
from vllm_gr.v1.worker.prefill_graph_runner import get_prefill_graph_runner

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

        # Publish the scheduler output so the _model_forward prefill-graph
        # interception can classify the batch and extract token counts.
        self._current_scheduler_output = scheduler_output

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


def make_patched_capture_cudagraphs_npu(_orig_capture_cudagraphs):
    """Wrap _capture_cudagraphs to capture prefill graphs during warmup (GPU parity)."""

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


def make_patched_model_forward_npu(_orig_model_forward):
    """Wrap _model_forward to intercept and replay pure prefill NPU graphs."""

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
            from vllm_gr.v1.worker.prefill_graph_runner import (
                SUPPORTED_NUM_REQS,
                is_prefill_graph_enabled,
                is_prefill_phase_batch,
            )

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
                        # execute_model (set_ascend_forward_context wraps the
                        # _model_forward call); NPUModelRunner does not store
                        # it as a runner attribute.
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

    return patched_model_forward


make_patched_prepare_inputs = make_patched_prepare_inputs_npu
make_patched_bookkeeping_sync = make_patched_bookkeeping_sync_npu
make_patched_capture_cudagraphs = make_patched_capture_cudagraphs_npu
make_patched_model_forward = make_patched_model_forward_npu
