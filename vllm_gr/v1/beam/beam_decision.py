# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Worker-side beam decision for incremental beam decode (asynchronous scheduling).

After the accelerator model forward produces top-k candidates (from the sampler's
logprob tensors, or the Constraint Table ``rec_constrained_topk`` override),
the worker can run the beam search decision itself instead of round-tripping
the full ``W * K`` logprob tensor to the frontend for a NumPy ``argpartition``:

- non-final steps: ``beam_search_group`` selects the surviving ``W`` beams and
  rewrites the sequence history;
- final step: ``beam_search_rec_final_select`` returns ``result_width`` beams
  directly (the 32-row tiling constraint applies only to the NPU kernel).

Cumulative scores and sequence history live in the device-resident
:class:`~vllm_gr.v1.beam.beam_attention_context.BeamDecisionBuffers`
(``context.decision``), which the worker treats as the single source of truth,
through Prefill initialization and subsequent asynchronous Decode updates.
The result dict (``out_token_ids`` / ``out_token_index`` /
``out_log_probs`` / ``out_sequence``) is attached to
``ModelRunnerOutput.beam_search_result`` and consumed by the serving frontend.

The runner's stage service binds the group and final selection operators before dispatch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vllm_gr.v1.beam.decision_ops import DecisionOps, reorder_sessions
from vllm_gr.v1.beam.gpu_beam_postprocess_workspace import _workspace_candidate_width
from vllm_gr.v1.worker.runner_lifecycle import get_stage_execution


def _decision_buffers(runner: Any):
    """Return the active beam session's BeamDecisionBuffers (or None)."""
    beam_info = getattr(runner, "_current_beam_info", None)
    if beam_info is None:
        return None
    session = beam_info.get("beam_session")
    if session is None or not getattr(session, "is_active", False):
        return None
    context = getattr(session, "context", None)
    if context is None or not hasattr(context, "decision"):
        return None
    return context.decision


def _is_beam_decode_step(runner: Any) -> bool:
    """True when the current run is an incremental beam decode step."""
    beam_info = getattr(runner, "_current_beam_info", None)
    if beam_info is None:
        return False
    for mdata in (beam_info.get("beam_data") or {}).values():
        if mdata.get("is_beam_decode", False) and "decode_step" in mdata:
            return True
    return False


def worker_decision_enabled(runner: Any) -> bool:
    """Return whether worker-side beam decision is active for this run.

    Requires the GRConfig opt-in, an active BeamSearchSession, an incremental
    beam-decode step, an NPU device, and worker-side candidate filtering. It
    additionally requires the fused xllm decode mode. CUDA V1 decisions are
    handled by the stage runner and never enter this path. Trie filtering
    remains frontend-side and therefore keeps the frontend NumPy decision path.
    """
    from vllm_gr.config import ConstraintBackend
    from vllm_gr.v1.worker.gr_context import (
        _get_gr_config,
    )

    gr_cfg = _get_gr_config(runner)
    if not bool(gr_cfg.beam_worker_decision):
        return False
    if _is_beam_decode_step(runner) is False:
        return False
    if _decision_buffers(runner) is None:
        return False
    if not get_stage_execution(runner).worker_decision:
        return False
    # Trie-catalog filtering runs frontend-side; Constraint Table candidates
    # are already filtered by the worker's rec_constrained_topk.  Only decide
    # worker-side when no frontend-side catalog pruning applies.
    if (
        gr_cfg.catalog_path is not None
        and gr_cfg.constraint_backend != ConstraintBackend.CONSTRAINT_TABLE
    ):
        return False
    return True


def init_beam_decision_state(
    session: Any,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
) -> None:
    """Seed the persistent decision buffers from the first candidate round.

    At the first
    decision step each beam (one row of ``[B*W, W]`` candidates) contributes
    its top candidate token and score to ``sequence[..., 0]`` and
    ``beam_scores``, so the subsequent ``beam_search_group`` steps read
    cumulative state entirely from device-resident buffers.  Runs once per
    session before any ``decode_select``-style step.

    Args:
        session: the active BeamSearchSession (its ``context.decision`` buffers).
        top_tokens / top_logprobs: ``[B * W, W]`` candidate token ids / probs.
    """
    bufs = getattr(session.context, "decision", None)
    if bufs is None:
        return
    width = session.beam_width
    n_rows = top_tokens.shape[0]
    if width == 0 or n_rows == 0 or n_rows % width != 0:
        return
    batch = n_rows // width
    # [B, W, K]: the last dim holds the top-candidate ordering per beam, and K
    # is the candidate width, which is independent of the beam width.
    # Column-0 views are already contiguous per-row, and the dtype guards
    # keep the constrained path (int32 ids / float32 probs straight from
    # rec_constrained_topk) copy-free — the profiled ~12ms-per-copy host
    # stalls (aclnnInplaceCopyGetWorkspaceSize) came from casting first and
    # copying the cast result; matching dtypes lets copy_ take the
    # contiguous fast path directly.
    candidate_width = top_tokens.shape[1]
    if candidate_width < 1:
        return
    best_token = top_tokens.reshape(batch, width, candidate_width)[:, :, 0]
    best_score = top_logprobs.reshape(batch, width, candidate_width)[:, :, 0]
    if best_token.dtype is not bufs.sequence.dtype:
        best_token = best_token.to(bufs.sequence.dtype)
    if best_score.dtype is not bufs.beam_scores.dtype:
        best_score = best_score.to(bufs.beam_scores.dtype)
    n = min(width, bufs.sequence.shape[1])
    bufs.sequence[:batch, :n, 0].copy_(best_token[:, :n])
    bufs.beam_scores[:batch, :n].copy_(best_score[:, :n])
    # Worker-side token-count for the decision: one committed token per beam.
    bufs.worker_decode_steps = 1


def init_beam_decision_from_parents(
    session: Any,
    parent_sequences: Sequence[Sequence[int]],
    parent_cumulative_logprobs: Sequence[float],
) -> None:
    """Restore persistent decision state from already-selected parents.

    The first grouped decode evaluates candidates after the prefill winner
    (for OneRec, ``s_b`` after ``s_a``).  Seeding from the current candidate
    matrix would lose the prefill score and count the current local score
    twice, changing beam rank order.
    """
    import math

    width = session.beam_width
    if len(parent_sequences) != width or len(parent_cumulative_logprobs) != width:
        raise RuntimeError(
            f"worker beam decision parent state must contain exactly beam_width={width} rows"
        )
    lengths = {len(sequence) for sequence in parent_sequences}
    if len(lengths) != 1:
        raise RuntimeError("worker beam decision parent sequences must have equal lengths")
    steps = next(iter(lengths), 0)
    bufs = session.context.decision
    if steps < 1 or steps > bufs.sequence.shape[2]:
        raise RuntimeError(
            f"worker beam decision parent sequence length {steps} is outside "
            f"[1, {bufs.sequence.shape[2]}]"
        )
    scores = [float(score) for score in parent_cumulative_logprobs]
    # -inf is a legitimate parent-score sentinel at the EOS-mask protocol
    # boundary. Only NaN/+inf are protocol errors.
    if any(math.isnan(score) or score == math.inf for score in scores):
        raise RuntimeError(
            "worker beam decision parent scores must not be NaN or +inf; "
            "-inf marks an EOS-masked beam and is a valid score"
        )
    device = bufs.sequence.device
    sequence_tensor = torch.tensor(parent_sequences, dtype=torch.int32, device=device)
    score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
    bufs.sequence[0, :width, :steps].copy_(sequence_tensor)
    bufs.beam_scores[0, :width].copy_(score_tensor)
    bufs.worker_decode_steps = steps


def _select_candidates(
    sampler_output: Any,
    width: int,
    v2_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extract ``[B*W, K]`` (token_ids int32, probs fp32) from sampler output.

    ``width`` here is the candidate width ``K`` (columns per beam), which is
    independent of the beam width ``W``. The row count is always ``B * W``.

    The sampler logprob tensors carry ``[rows, max_logprobs + 1]`` with the
    sampled token in column 0; the constrained override already replaced the
    constrained rows with the fixed-width ``K`` operator output.  For
    unconstrained rows we drop the sampled-token column to get exactly ``K``
    candidates per beam.

    ``v2_layout`` selects an explicit layout contract instead of inferring it
    from ``shape[1]``: ``False`` is the V1 layout with the sampled token in
    column 0; ``True`` declares the vLLM 0.30 V2 pure top-K layout with no
    sampled column. The flag is threaded from the V2 sample wrapper so the
    layout is never guessed from the tensor width alone.

    dtype: the ``.to(...)`` casts are skipped when the source tensor already
    carries the target dtype (``torch.Tensor.to`` short-circuits, but the
    check avoids even the dispatch — profiling bw128_1k_t3 attributed a
    ~30ms host stall to these casts when they ran per candidate row).  The
    constrained paths already emit int32 token ids / float32 probs, so in
    the worker-decision flow the casts are no-ops; the unconstrained
    sampler path (int64 ids from upstream topk) still converts once per
    step here instead of per row.
    """
    lp_tensors = getattr(sampler_output, "logprobs_tensors", None)
    if lp_tensors is None:
        return None
    token_ids = getattr(lp_tensors, "logprob_token_ids", None)
    logprobs = getattr(lp_tensors, "logprobs", None)
    if token_ids is None or logprobs is None:
        return None
    if token_ids.dim() != 2 or logprobs.dim() != 2:
        return None
    if token_ids.shape[0] != logprobs.shape[0]:
        return None
    # Constrained rows already have K columns; unconstrained rows have
    # max_logprobs + 1.  Use min() so slicing can't overflow.
    n_cols = min(token_ids.shape[1], logprobs.shape[1])
    if v2_layout:
        # Pure top-K: every column is a candidate (no sampled column to drop).
        if n_cols < width:
            return None
        top_tokens = token_ids[:, :width]
        top_probs = logprobs[:, :width]
    elif n_cols > width:
        top_tokens = token_ids[:, 1 : width + 1]
        top_probs = logprobs[:, 1 : width + 1]
    elif n_cols == width:
        top_tokens = token_ids[:, :width]
        top_probs = logprobs[:, :width]
    else:
        return None
    if top_tokens.dtype is not torch.int32:
        top_tokens = top_tokens.to(torch.int32)
    if top_probs.dtype is not torch.float32:
        top_probs = top_probs.to(torch.float32)
    if not top_tokens.is_contiguous():
        top_tokens = top_tokens.contiguous()
    if not top_probs.is_contiguous():
        top_probs = top_probs.contiguous()
    return top_tokens, top_probs


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
    *,
    reorder: Any = reorder_sessions,
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
        reorder(session, sessions, local_idx, local_prefix)
        # The applied mapping is NOT materialized here: reading
        # ``local_idx[0].tolist()`` is a device→host sync that stalls the
        # pipeline every step (~30ms on NPU, profiling bw128_1k_t3).  The
        # return channel (``_maybe_run_worker_beam_decision``) already copies
        # ``out_token_index`` to host for the frontend, and its request-0 row
        # equals this request-local mapping (global offset 0), so the record
        # is derived there from that copy — no second D2H.
    return decision


def _run_decision(
    ops: DecisionOps,
    session: Any,
    tokens: torch.Tensor,
    probs: torch.Tensor,
    last: bool,
    result_width: int,
    *,
    strict: bool,
) -> dict[str, torch.Tensor]:
    bufs = session.context.decision
    width = session.beam_width
    batch = tokens.shape[0] // width
    step = bufs.worker_decode_steps
    scores = bufs.beam_scores[:batch, :width].reshape(batch * width, 1)
    history = bufs.sequence[:batch, :width, : step + 1].contiguous()
    ids, parents, log_probs, prefix, sequence, result_width = ops.select(
        scores, tokens, probs, history, step, width, result_width, last, strict
    )
    bufs.worker_decode_steps = step + 1
    return _finalize_decision(
        session,
        bufs,
        ids.reshape(batch, result_width),
        parents.reshape(batch, result_width),
        log_probs.reshape(batch, result_width),
        sequence,
        prefix,
        batch,
        width,
        result_width,
        reorder=ops.reorder,
    )


def run_beam_decision(
    runner: Any,
    session: Any,
    sampler_output: Any,
    is_last_step: bool,
    result_width: int | None = None,
    *,
    v2_layout: bool = False,
) -> dict[str, torch.Tensor] | None:
    """Run the worker-side beam decision for one decode step.

    Args:
        runner: the model runner (must expose ``_current_beam_info``).
        session: the active BeamSearchSession.
        sampler_output: the sampler output whose logprob tensors carry the
            per-beam top-k candidates (post constraint override).
        is_last_step: whether this is the final decode step.
        result_width: output width for the final step; defaults to beam width.
        v2_layout: explicit sampler layout contract (see ``_select_candidates``).

    Returns a dict with ``out_token_ids`` / ``out_token_index`` /
    ``out_log_probs`` / ``out_sequence`` (device tensors, ``[B, W_eff]`` /
    ``[B, W_eff]`` / ``[B, W_eff]`` / ``[B, W_eff, steps]``), plus a copy of
    the cumulative ``beam_scores`` (``probs`` key), or ``None`` when the step
    cannot be decided worker-side (caller keeps the frontend path).
    """
    width = session.beam_width
    # Candidate width K is independent of the beam width W; the sampler emits
    # one row per beam and K columns per row.
    candidate_width = _workspace_candidate_width(runner, width)
    candidates = _select_candidates(sampler_output, candidate_width, v2_layout=v2_layout)
    if candidates is None:
        return None

    top_tokens, top_probs = candidates[:2]
    n_rows = top_tokens.shape[0]
    if n_rows == 0 or n_rows % width != 0:
        return None
    # Keep the fused mapping well-defined: require one survivor per beam.
    beam_info = getattr(runner, "_current_beam_info", None)
    for mdata in (beam_info.get("beam_data") or {}).values():
        if mdata.get("is_beam_decode", False):
            parent = mdata.get("parent_beam_ids")
            if parent is not None and len(parent) != width:
                return None

    result_width = result_width or width
    if result_width <= 0 or result_width < width:
        result_width = width

    bufs = session.context.decision
    if bufs.worker_decode_steps == 0:
        # A decode candidate matrix cannot reconstruct the scores of its
        # already-selected parents. Fall back to frontend selection instead
        # of silently dropping prefill scores and counting this local score
        # twice. Normal engine paths seed from request metadata at creation.
        return None

    decision = _run_decision(
        get_stage_execution(runner).decision,
        session,
        top_tokens,
        top_probs,
        is_last_step,
        result_width,
        strict=False,
    )
    decision["probs"] = session.context.decision.beam_scores[:1, :width].clone()
    return decision


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
    decision = _run_decision(
        get_stage_execution(runner).decision,
        session,
        top_tokens,
        top_probs,
        is_last_step,
        result_width,
        strict=True,
    )
    batch = top_tokens.shape[0] // width
    decision["probs"] = session.context.decision.beam_scores[:batch, :width].clone()
    return decision


def maybe_run_worker_beam_decision(
    runner: Any,
    sampler_output: Any,
    *,
    v2_layout: bool = False,
) -> None:
    """Run and publish the NPU worker-side beam decision.

    ``v2_layout`` is threaded to :func:`run_beam_decision` so the V2 sample
    wrapper can declare its sampler layout explicitly (see
    :func:`_select_candidates`).
    """
    if not worker_decision_enabled(runner):
        runner._beam_decision_result = None
        return
    beam_info = getattr(runner, "_current_beam_info", None)
    session = beam_info.get("beam_session") if beam_info else None
    if session is None:
        runner._beam_decision_result = None
        return

    state = session._state
    result_width = None
    for mdata in (beam_info.get("beam_data") or {}).values():
        if mdata.get("is_beam_decode", False):
            result_width = mdata.get("result_width")
            break

    decision = run_beam_decision(
        runner,
        session,
        sampler_output,
        is_last_step=state.is_last_step,
        result_width=result_width,
        v2_layout=v2_layout,
    )
    if decision is not None:
        # NPU keeps its established plain-container ABI.  Its host-copy and
        # xllm lifecycle have separate performance contracts from the retired
        # CUDA transport path.
        keys = (
            ("out_token_ids", "out_token_index", "out_log_probs")
            if not state.is_last_step
            else ("out_token_ids", "out_token_index", "out_log_probs", "out_sequence")
        )
        payload = {}
        for key in keys:
            value = decision.get(key)
            if value is None:
                continue
            payload[key] = value.detach().to("cpu").tolist() if torch.is_tensor(value) else value
        payload["is_last_step"] = bool(state.is_last_step)
        if not state.is_last_step:
            out_idx_host = payload.get("out_token_index")
            if out_idx_host:
                state.kv_reorder = [int(parent) for parent in out_idx_host[0]]
        decision = payload
    if decision is not None:
        # Ownership markers (review P5): mixed batches must never let session
        # B's decision chain onto session A's step update downstream.
        decision["stage"] = "decode"
        decode_req_id = next(
            (
                req_id
                for req_id, mdata in (beam_info.get("beam_data") or {}).items()
                if mdata.get("is_beam_decode", False)
            ),
            None,
        )
        if decode_req_id is not None:
            decision["session_id"] = decode_req_id
    runner._beam_decision_result = decision


__all__ = [
    "init_beam_decision_from_parents",
    "init_beam_decision_state",
    "maybe_run_worker_beam_decision",
    "run_beam_decision",
    "run_beam_decision_candidates",
    "worker_decision_enabled",
]
