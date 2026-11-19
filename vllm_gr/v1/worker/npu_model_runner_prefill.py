# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""NPU ModelRunner prefill/BSND forward and capture."""

# Native NPU runner internals remain dynamically typed; the npu-worker mypy
# group still checks facade re-exports and caller-side helper signatures.
# mypy: disable-error-code="no-any-return,no-untyped-call,no-untyped-def,type-arg,var-annotated"

from __future__ import annotations

from functools import wraps

import torch
from vllm.logger import init_logger

from vllm_gr.v1.worker.model_runner_contracts import (
    _WORKER_MODEL_FORWARD_PATCH_MARKER,
)
from vllm_gr.v1.worker.prefill_graph_runner import (
    SUPPORTED_NUM_REQS,
    get_prefill_graph_runner,
    is_prefill_graph_enabled,
    is_prefill_phase_batch,
)

logger = init_logger(__name__)


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
            and is_prefill_graph_enabled()
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
        # Propagate startup/capture failures when BSND is enabled.
        import os

        if os.getenv("VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS", "").strip():
            raise
        logger.exception(
            "vllm-gr: prefill NPU graph capture failed; prefill will fall back to eager"
        )
