# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Ascend Worker Beam decision for incremental Decode.

The V1 stage runner supplies candidate tensors directly. Device-resident
BeamDecisionBuffers own cumulative scores and sequence history through Prefill
initialization and subsequent Decode updates. Selected tokens and parent
indices return to the stage runner for the same-step KV reorder and terminal
result construction.

The fused xLLM path and the Torch fallback produce the same decision tensors
for the stage runner. The Torch path also supports host parity tests.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Kernel tiling constraint: beam_search_rec_final_select requires result_width
# to be a multiple of 32 (verified on device; see ops/npu/beam_ops.py ABI notes).
_FINAL_SELECT_ALIGNMENT = 32


def _reorder_kv_same_step(
    session: Any,
    beam_index: torch.Tensor,
    group_token_num: torch.Tensor,
) -> None:
    """Apply the beam decision's KV reorder inside the SAME step.

    Asynchronous scheduling runs ``select_unshared_kv`` immediately after
    ``beam_search_group`` — the next step's attention
    then reads an already-reordered pool with no frontend round-trip in the
    critical path.  ``beam_search_group`` output is grouped by source beam
    ascending (the torch reference's stable ``argsort(parent)`` locks the
    contract), so at full pool width the fused kernel's prefix-sum mapping
    applies directly — bypassing ``reorder_unshared_kv``'s per-step D2H
    sortedness check + bincount round-trip.  Sub-width pools, CPU hosts and
    unavailable ops fall back to the standard dispatcher.
    """
    context = session.context
    pools = context.pools
    state = session._state
    step = state.decode_step  # 0-based; the reorder covers [0, step]
    pool_beam_width = pools[0].unshared_key.shape[1]

    # The fused operator only executes on NPU tensors; CPU hosts (unit tests,
    # GPU-debug parity runs) go straight to the dispatcher fallback instead of
    # raising through the ACL layer first.
    if beam_index.numel() == pool_beam_width and str(beam_index.device.type) == "npu":
        try:
            from vllm_gr.ops.npu import beam_ops

            if beam_ops.is_available():
                # Reuse the persistent block-table buffer filled by this
                # step's execute_model (fixed address, slot already current).
                block_table = getattr(context, "xllm_block_table_buf", None)
                if block_table is None:
                    block_table = torch.tensor(
                        [session.slot], dtype=torch.int32, device=beam_index.device
                    )
                else:
                    block_table = block_table[session.slot : session.slot + 1]
                beam_ops.select_unshared_kv(
                    beam_index.contiguous(),
                    [p.unshared_key for p in pools],
                    [p.unshared_value for p in pools],
                    block_table,
                    group_token_num.contiguous(),
                    step + 1,
                    pool_beam_width,
                    len(pools),
                )
                state.decode_step += 1
                return
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("vllm-gr fused same-step KV reorder failed: %s; fallback", exc)

    # Sub-width pools / CPU hosts / op-unavailable: the dispatcher picks the
    # cascade kernel (arbitrary widths) and step_complete advances decode_step.
    # Runner-owned pools, aux buffers, and device pointer tables were created
    # and validated together with this persistent context.  Avoid rescanning
    # every K/V layer between BeamSearchGroup and SelectUnsharedKV.
    session.step_complete(beam_index, layout_prevalidated=True)


def _reorder_kv_batch_same_step(
    session: Any,
    sessions: tuple[Any, ...],
    beam_index: torch.Tensor,
    group_token_num: torch.Tensor,
) -> None:
    """Apply one request-major B*W KV reorder for an NPU Beam batch."""

    context = session.context
    pools = context.pools
    batch = len(sessions)
    beam_width = session.beam_width
    steps = {item._state.decode_step for item in sessions}
    if len(steps) != 1:
        raise RuntimeError("NPU Beam batch sessions left the same KV reorder step")
    step = steps.pop()
    expected = (batch, beam_width)
    if tuple(beam_index.shape) != expected or tuple(group_token_num.shape) != expected:
        raise RuntimeError("NPU Beam batch reorder tensors have an invalid shape")

    if str(beam_index.device.type) == "npu":
        try:
            from vllm_gr.ops.npu import beam_ops

            if beam_ops.is_available():
                block_table = getattr(context, "xllm_block_table_buf", None)
                if block_table is None:
                    block_table = torch.tensor(
                        [item.slot for item in sessions],
                        dtype=torch.int32,
                        device=beam_index.device,
                    )
                else:
                    block_table = block_table[:batch]
                # xllm consumes all B*W local parents and B pool slots in one
                # call. Per-request calls do not preserve its batch geometry
                # for a non-zero pool slot.
                beam_ops.select_unshared_kv(
                    beam_index.reshape(-1).contiguous(),
                    [pool.unshared_key for pool in pools],
                    [pool.unshared_value for pool in pools],
                    block_table,
                    group_token_num.reshape(-1).contiguous(),
                    step + 1,
                    beam_width,
                    len(pools),
                )
                for item in sessions:
                    item._state.decode_step += 1
                return
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "vllm-gr fused batch same-step KV reorder failed: %s; fallback",
                exc,
            )

    for row, item in enumerate(sessions):
        _reorder_kv_same_step(item, beam_index[row], group_token_num[row])


def _finalize_decision(
    session: Any,
    bufs: Any,
    out_ids: torch.Tensor,
    out_idx: torch.Tensor,
    out_lp: torch.Tensor,
    out_seq: torch.Tensor,
    prefix: torch.Tensor | None,
    batch: int,
    beam_width: int,
    result_width: int,
) -> dict[str, torch.Tensor]:
    """Persist cumulative state and apply the asynchronous Beam decision.

    Non-final steps (``prefix`` is not None): convert the operator's GLOBAL
    ``request * beam_width`` offsets to request-local form for both the
    parent mapping and the prefix sums, persist
    ``group_token_num`` for the fused ``select_unshared_kv`` kernel, and
    reorder the KV pool in the SAME step.  Final steps (``prefix is None``):
    no reorder (select_unshared_kv is skipped on the last step).
    """
    # Persist cumulative state back into the fixed-address buffers.
    write_w = min(out_seq.shape[1], bufs.sequence.shape[1])
    bufs.sequence[:batch, :write_w, : out_seq.shape[2]].copy_(out_seq[:, :write_w])
    bufs.beam_scores[:batch, :result_width].copy_(out_lp[:, :result_width])

    decision = {
        "out_token_ids": out_ids,
        "out_token_index": out_idx,
        "out_log_probs": out_lp,
        "out_sequence": out_seq,
    }
    if prefix is None:
        return decision

    request_offset = (
        torch.arange(batch, dtype=torch.int32, device=prefix.device).unsqueeze(1) * beam_width
    )
    local_idx = out_idx[:batch, :beam_width].to(torch.int32) - request_offset
    local_prefix = prefix[:batch, :beam_width].to(torch.int32) - request_offset
    bufs.group_token_num[:batch, :beam_width].copy_(local_prefix)

    sessions = getattr(session, "sessions", (session,))
    if len(sessions) != batch:
        raise RuntimeError("Beam decision batch does not match its request sessions")
    if not all(item._state.is_last_step for item in sessions):
        # NPU B>1 keeps the complete request-major geometry in one operator call.
        # Other devices and B=1 retain the established per-session path.
        if batch > 1 and local_idx.device.type == "npu":
            _reorder_kv_batch_same_step(session, sessions, local_idx, local_prefix)
        else:
            for row, item in enumerate(sessions):
                _reorder_kv_same_step(item, local_idx[row], local_prefix[row])
    return decision


def _run_npu_decision(
    session: Any,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    is_last_step: bool,
    result_width: int,
) -> dict[str, torch.Tensor] | None:
    """Run the fused xllm decision operators."""
    from vllm_gr.ops.npu import beam_ops

    bufs = session.context.decision
    width = bufs.worker_decode_steps
    beam_width = session.beam_width
    batch = top_tokens.shape[0] // beam_width
    step = width  # worker-side token count == next sequence slot to write

    log_probs = bufs.beam_scores[:batch, :beam_width].reshape(batch * beam_width, 1)
    # xllm derives the history stride from the physical last dimension and
    # requires exactly current_step + 1 slots. Passing the runtime N=3
    # capacity to an N=2 terminal group step misreads the second SID token.
    seq = bufs.sequence[:batch, :beam_width, : step + 1].contiguous()

    prefix: torch.Tensor | None = None
    if is_last_step and result_width % _FINAL_SELECT_ALIGNMENT == 0:
        out_ids, out_idx, out_lp, out_seq = beam_ops.beam_search_rec_final_select(
            log_probs,
            top_tokens,
            top_probs,
            seq,
            step,
            result_width,
        )
    else:
        out_ids, out_idx, out_lp, prefix, out_seq = beam_ops.beam_search_group(
            log_probs,
            top_tokens,
            top_probs,
            seq,
            step,
            beam_width,
        )
        result_width = beam_width

    out_ids = out_ids.reshape(batch, result_width)
    out_idx = out_idx.reshape(batch, result_width)
    out_lp = out_lp.reshape(batch, result_width)
    bufs.worker_decode_steps = step + 1
    return _finalize_decision(
        session, bufs, out_ids, out_idx, out_lp, out_seq, prefix, batch, beam_width, result_width
    )


def _run_torch_decision(
    session: Any,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    is_last_step: bool,
    result_width: int,
) -> dict[str, torch.Tensor] | None:
    """PyTorch reference decision (GPU/host parity oracle for the NPU path)."""
    from vllm_gr.ops.onerec_decode import (
        beam_search_group,
        beam_search_rec_final_select,
    )

    bufs = session.context.decision
    width = bufs.worker_decode_steps
    beam_width = session.beam_width
    batch = top_tokens.shape[0] // beam_width
    step = width

    log_probs = bufs.beam_scores[:batch, :beam_width].reshape(batch * beam_width, 1)
    seq = bufs.sequence[:batch, :beam_width, : step + 1].contiguous()

    prefix: torch.Tensor | None = None
    if is_last_step and (
        top_tokens.device.type == "cuda" or result_width % _FINAL_SELECT_ALIGNMENT == 0
    ):
        out_ids, out_idx, out_lp, out_seq = beam_search_rec_final_select(
            log_probs, top_tokens, top_probs, seq, step, result_width
        )
    else:
        out_ids, out_idx, out_lp, prefix, out_seq = beam_search_group(
            log_probs, top_tokens, top_probs, seq, step, beam_width
        )
        result_width = beam_width

    out_ids = out_ids.reshape(batch, result_width)
    out_idx = out_idx.reshape(batch, result_width)
    out_lp = out_lp.reshape(batch, result_width)
    bufs.worker_decode_steps = step + 1
    return _finalize_decision(
        session, bufs, out_ids, out_idx, out_lp, out_seq, prefix, batch, beam_width, result_width
    )


def run_beam_decision_candidates(
    runner: Any,
    session: Any,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    *,
    is_last_step: bool,
    result_width: int | None = None,
) -> dict[str, torch.Tensor]:
    """Run a worker decision from canonical fixed-width candidates.

    The NPU eager GR runner already owns ``ConstraintOutput`` tensors, so
    routing them through vLLM's sampler container would only reconstruct an
    object that the constrained backend has just deconstructed. This narrow
    entry point shares the persistent decision and KV-reorder machinery while
    keeping the canonical candidate ABI explicit.
    """
    width = session.beam_width
    if (
        top_tokens.ndim != 2
        or top_probs.shape != top_tokens.shape
        or top_tokens.shape[1] < 1
        or top_tokens.shape[0] == 0
        or top_tokens.shape[0] % width
    ):
        raise ValueError("Beam candidates must have shape [B*W, K] with K >= 1")
    if top_tokens.dtype is not torch.int32 or top_probs.dtype is not torch.float32:
        raise ValueError("Beam candidates must use int32 token IDs and float32 scores")
    if session.context.decision.worker_decode_steps == 0:
        raise RuntimeError("Beam decision state must be initialized from Prefill")

    result_width = result_width or width
    if not 1 <= result_width <= width:
        raise ValueError("result_width must be in [1, beam_width]")
    device_type = str(getattr(getattr(runner, "device", None), "type", ""))
    if device_type.startswith("npu"):
        from vllm_gr.ops.npu import beam_ops

        if not beam_ops.is_available():
            raise RuntimeError("NPU Beam decision operators are unavailable")
        if top_tokens.shape[1] != width:
            raise NotImplementedError(
                "NPU beam decision requires candidate_width == beam_width, got "
                f"candidate_width={top_tokens.shape[1]}, beam_width={width}"
            )
        decision = _run_npu_decision(session, top_tokens, top_probs, is_last_step, result_width)
    else:
        decision = _run_torch_decision(session, top_tokens, top_probs, is_last_step, result_width)
    if decision is None:
        raise RuntimeError("Beam decision did not produce a result")
    batch = top_tokens.shape[0] // width
    decision["probs"] = session.context.decision.beam_scores[:batch, :width].clone()
    return decision


__all__ = ["run_beam_decision_candidates"]
