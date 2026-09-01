# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side beam decision for incremental beam decode (ACS-aligned).

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
mirroring the ACS ``BeamSearchContext.decode_select`` + ``init_from_prefill``
save flow.  The result dict (``out_token_ids`` / ``out_token_index`` /
``out_log_probs`` / ``out_sequence``) is attached to
``ModelRunnerOutput.beam_search_result`` and consumed by the serving frontend.

The public ``onerec_decode`` operators dispatch to the active accelerator:
CUDA uses the fused CUDA kernels, NPU uses the xllm operators, and CPU keeps a
PyTorch reference path for parity tests.
"""

from __future__ import annotations

import struct
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Kernel tiling constraint: beam_search_rec_final_select requires result_width
# to be a multiple of 32 (verified on device; see ops/npu/beam_ops.py ABI notes).
_FINAL_SELECT_ALIGNMENT = 32
_BEAM_RESULT_BUFFER_FORMAT = "beam_result_i32_f32_v1"


def _encode_cuda_beam_result(
    decision: dict[str, Any],
    *,
    is_last_step: bool,
) -> tuple[dict[str, Any], list[int]] | None:
    """Encode CUDA winners with one device-to-host synchronization.

    The canonical result channel is an untyped ``dict[str, Any]``.  Sending
    tensors through it loses msgpack auxiliary frames, while independently
    calling ``tensor.to('cpu').tolist()`` for ids, parents and scores forces
    multiple synchronizations and creates thousands of Python objects.

    Store the three equal-width arrays in one int32 transport buffer (float32
    scores are bit-reinterpreted, not converted), perform one D2H copy, and
    send the resulting bytes in ``beam_search_result``.  This is intentionally
    distinct from the removed FlatLogprobs packed-wire path: token ids and
    parents remain separate arrays and no sentinel or token-bit packing is
    involved.
    """
    token_ids = decision.get("out_token_ids")
    parents = decision.get("out_token_index")
    scores = decision.get("out_log_probs")
    if not all(torch.is_tensor(value) for value in (token_ids, parents, scores)):
        return None
    if token_ids.ndim != 2 or parents.shape != token_ids.shape or scores.shape != token_ids.shape:
        return None

    token_ids_i32 = token_ids if token_ids.dtype is torch.int32 else token_ids.to(torch.int32)
    parents_i32 = parents if parents.dtype is torch.int32 else parents.to(torch.int32)
    scores_f32 = scores if scores.dtype is torch.float32 else scores.to(torch.float32)
    pieces = (
        token_ids_i32.contiguous().reshape(-1),
        parents_i32.contiguous().reshape(-1),
        scores_f32.contiguous().reshape(-1).view(torch.int32),
    )
    host = torch.cat(pieces).to(device="cpu")
    count = token_ids.numel()
    payload = {
        "buffer_format": _BEAM_RESULT_BUFFER_FORMAT,
        "shape": [int(token_ids.shape[0]), int(token_ids.shape[1])],
        "buffer": host.numpy().tobytes(),
        "is_last_step": bool(is_last_step),
    }
    return payload, [int(parent) for parent in host[count : 2 * count].tolist()]


def unpack_beam_result_buffer(
    decision: Any,
) -> tuple[list[int], list[float], list[int]] | None:
    """Decode the compact ``beam_search_result`` bytes on EngineCore/frontend."""
    if not isinstance(decision, dict):
        return None
    if decision.get("buffer_format") != _BEAM_RESULT_BUFFER_FORMAT:
        return None
    shape = decision.get("shape")
    data = decision.get("buffer")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in shape)
        or not isinstance(data, (bytes, bytearray, memoryview))
    ):
        return None
    count = int(shape[0]) * int(shape[1])
    if len(data) != count * 3 * 4:
        return None
    view = memoryview(data)
    token_ids = list(struct.unpack_from(f"<{count}i", view, 0))
    parents = list(struct.unpack_from(f"<{count}i", view, count * 4))
    scores = list(struct.unpack_from(f"<{count}f", view, count * 8))
    return token_ids, scores, parents


def flatten_decision_value(value: Any) -> list[Any] | None:
    """Flatten one worker beam-decision payload value to a flat Python list.

    The return-channel payload crosses the EngineCore IPC boundary inside
    ``kv_transfer_params`` (``dict[str, Any]``): msgpack only round-trips
    plain containers through an untyped mapping, so the worker ships plain
    nested lists (see ``_maybe_run_worker_beam_decision``).  In-process
    consumers may still hand over the original host tensors, and a payload
    that skipped normalization arrives as the raw encoder triple
    ``('int64', shape, data)`` — the ``data`` slot is either an inline
    memoryview or an aux-buffer INDEX whose backing frame never reaches this
    side, i.e. unrecoverable — so return ``None`` and let the caller fall
    back to its legacy path.
    """
    if torch.is_tensor(value):
        return value.detach().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        # Serialized-tensor triple: ('<dtype>', shape, memoryview | index).
        if len(value) == 3 and isinstance(value[0], str) and isinstance(value[1], (list, tuple)):
            return None
        flat: list[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                nested = flatten_decision_value(item)
                if nested is None:
                    return None
                flat.extend(nested)
            else:
                flat.append(item)
        return flat
    return None


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


def _cuda_worker_decision_width_supported(beam_width: object) -> bool:
    """Whether the fused CUDA decision kernels support ``beam_width``."""
    if isinstance(beam_width, bool) or not isinstance(beam_width, int):
        return False
    from vllm_gr.config.models import MAX_BEAM_WIDTH

    return 1 <= beam_width <= MAX_BEAM_WIDTH


def _xllm_decode_mode_active() -> bool:
    """True iff the NPU decode execution mode resolves to the fused xllm path.

    The worker-side beam decision is the post-processing half of the xllm
    pipeline: its same-step ``select_unshared_kv`` reorder assumes the fused
    pool layout, and running it under ``cascade`` mode would mix the two
    execution paths.  Resolved lazily through the same per-process cache the
    attention backend uses (``_get_npu_decode_execution``; unset config
    defaults to ``cascade``, NPU-with-library defaults to ``xllm``).
    """
    try:
        from vllm_gr.v1.attention.backends.beam_attn import _get_npu_decode_execution

        return _get_npu_decode_execution() == "xllm"
    except Exception:
        return False


def worker_decision_enabled(runner: Any) -> bool:
    """Return whether worker-side beam decision is active for this run.

    Requires the GRConfig opt-in, an active BeamSearchSession, an incremental
    beam-decode step, a CUDA or NPU device, and worker-side candidate
    filtering. NPU additionally requires the fused xllm decode mode; CUDA
    dispatches through the canonical CUDA operators. Trie filtering remains
    frontend-side and therefore keeps the frontend NumPy decision path.
    """
    from vllm_gr.config import ConstraintBackend
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    gr_cfg = _get_gr_config(runner)
    if not bool(gr_cfg.beam_worker_decision):
        return False
    if _is_beam_decode_step(runner) is False:
        return False
    if _decision_buffers(runner) is None:
        return False
    device = getattr(runner, "device", None)
    device_type = str(getattr(device, "type", ""))
    if device_type not in ("cuda", "npu"):
        return False
    if device_type == "cuda":
        beam_info = getattr(runner, "_current_beam_info", None)
        session = beam_info.get("beam_session") if beam_info else None
        if not _cuda_worker_decision_width_supported(getattr(session, "beam_width", None)):
            return False
    # NPU same-step KV selection is defined only for fused xllm mode. CUDA
    # uses the generic session dispatcher and has no xllm dependency.
    if device_type == "npu" and not _xllm_decode_mode_active():
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


def _run_cuda_prefill_decision(runner: Any, sampler_output: Any) -> dict[str, torch.Tensor] | None:
    """Publish constrained prefill winners from the CUDA worker.

    Constraint prefill has one candidate row and no ``BeamSearchSession`` yet.
    The backend row is already deterministically ordered, so its W entries are
    the first W beam survivors (all with parent 0).  Publishing them through
    the canonical result channel avoids the remaining frontend NumPy select.

    The decode session initializes independently from its first per-parent
    candidate matrix; prefill scores must not seed its persistent buffers.
    """
    from vllm_gr.config import ConstraintBackend
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    device = getattr(runner, "device", None)
    if str(getattr(device, "type", "")) != "cuda":
        return None
    gr_cfg = _get_gr_config(runner)
    if not bool(gr_cfg.beam_worker_decision):
        return None
    if gr_cfg.constraint_backend != ConstraintBackend.CONSTRAINT_TABLE:
        return None

    beam_info = getattr(runner, "_current_beam_info", None)
    beam_data = (beam_info.get("beam_data") or {}) if beam_info else {}
    prefills = [mdata for mdata in beam_data.values() if mdata.get("is_constraint_prefill", False)]
    # The result ABI is one logical request. Multi-request prefill keeps the
    # established frontend path until the result channel becomes batched.
    if len(prefills) != 1:
        return None
    mdata = prefills[0]
    width = mdata.get("beam_width")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        return None
    if not _cuda_worker_decision_width_supported(width):
        return None

    lp_tensors = getattr(sampler_output, "logprobs_tensors", None)
    token_ids = getattr(lp_tensors, "logprob_token_ids", None)
    scores = getattr(lp_tensors, "logprobs", None)
    if (
        not torch.is_tensor(token_ids)
        or not torch.is_tensor(scores)
        or token_ids.ndim != 2
        or scores.shape != token_ids.shape
        or token_ids.shape[0] != 1
        or token_ids.shape[1] < width
    ):
        return None

    out_ids = token_ids[:, :width].to(dtype=torch.int32).contiguous()
    out_scores = scores[:, :width].to(dtype=torch.float32).contiguous()
    out_parents = torch.zeros_like(out_ids, dtype=torch.int32)
    return {
        "out_token_ids": out_ids,
        "out_token_index": out_parents,
        "out_log_probs": out_scores,
    }


def init_beam_decision_state(
    session: Any,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
) -> None:
    """Seed the persistent decision buffers from the first candidate round.

    Mirrors ACS ``BeamSearchContext.init_from_prefill``: at the first
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
    # [B, W, W]: the last dim holds the top-candidate ordering per beam.
    # Column-0 views are already contiguous per-row, and the dtype guards
    # keep the constrained path (int32 ids / float32 probs straight from
    # rec_constrained_topk) copy-free — the profiled ~12ms-per-copy host
    # stalls (aclnnInplaceCopyGetWorkspaceSize) came from casting first and
    # copying the cast result; matching dtypes lets copy_ take the
    # contiguous fast path directly.
    best_token = top_tokens.reshape(batch, width, width)[:, :, 0]
    best_score = top_logprobs.reshape(batch, width, width)[:, :, 0]
    if best_token.dtype is not bufs.sequence.dtype:
        best_token = best_token.to(bufs.sequence.dtype)
    if best_score.dtype is not bufs.beam_scores.dtype:
        best_score = best_score.to(bufs.beam_scores.dtype)
    n = min(width, bufs.sequence.shape[1])
    bufs.sequence[:batch, :n, 0].copy_(best_token[:, :n])
    bufs.beam_scores[:batch, :n].copy_(best_score[:, :n])
    # Worker-side token-count for the decision: one committed token per beam.
    bufs.worker_decode_steps = 1


def _select_candidates(
    sampler_output: Any,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extract ``[B*W, W]`` (token_ids int32, probs fp32) from sampler output.

    The sampler logprob tensors carry ``[rows, max_logprobs + 1]`` with the
    sampled token in column 0; the constrained override already replaced the
    constrained rows with the fixed-width ``W`` operator output.  For
    unconstrained rows we drop the sampled-token column to get exactly ``W``
    candidates per beam, matching ACS ``model_runner_v1``.

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
    # Constrained rows already have width == W; unconstrained rows have
    # width == max_logprobs + 1.  Use min() so slicing can't overflow.
    n_cols = min(token_ids.shape[1], logprobs.shape[1])
    if n_cols > width:
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


def _reorder_kv_same_step(
    session: Any,
    beam_index: torch.Tensor,
    group_token_num: torch.Tensor,
) -> None:
    """Apply the beam decision's KV reorder inside the SAME step (ACS timing).

    ACS ``decode_select`` runs ``select_unshared_kv`` immediately after
    ``beam_search_group`` (context.py:471-480) — the next step's attention
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
    session.step_complete(beam_index)


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
    """Persist cumulative state + run the ACS decode_select post-processing.

    Non-final steps (``prefix`` is not None): convert the operator's GLOBAL
    ``request * beam_width`` offsets to request-local form for both the
    parent mapping and the prefix sums (ACS context.py:458-468), persist
    ``group_token_num`` for the fused ``select_unshared_kv`` kernel, and
    reorder the KV pool in the SAME step.  Final steps (``prefix is None``):
    no reorder (ACS skips select_unshared_kv on the last step).
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

    state = session._state
    if not state.is_last_step:
        # The shared beam context serves a single active session (max_batch=1
        # design): its operator row is batch row 0.  Only that row's mapping
        # may feed the pool reorder — flattening every batch row would push a
        # [B*W] parent mapping into a [W]-wide pool.
        _reorder_kv_same_step(session, local_idx[0], local_prefix[0])
        # The applied mapping is NOT materialized here: reading
        # ``local_idx[0].tolist()`` is a device→host sync that stalls the
        # pipeline every step (~30ms on NPU, profiling bw128_1k_t3).  The
        # return channel (``_maybe_run_worker_beam_decision``) already copies
        # ``out_token_index`` to host for the frontend, and its request-0 row
        # equals this request-local mapping (global offset 0), so the record
        # is derived there from that copy — no second D2H.
    return decision


def _run_npu_decision(
    session: Any,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    is_last_step: bool,
    result_width: int,
) -> dict[str, torch.Tensor] | None:
    """Run the fused xllm decision operators (ACS decode_select equivalent)."""
    from vllm_gr.ops.npu import beam_ops

    bufs = session.context.decision
    width = bufs.worker_decode_steps
    beam_width = session.beam_width
    batch = top_tokens.shape[0] // beam_width
    step = width  # worker-side token count == next sequence slot to write

    log_probs = bufs.beam_scores[:batch, :beam_width].reshape(batch * beam_width, 1)
    seq = bufs.sequence[:batch, :beam_width, :].contiguous()

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
    seq = bufs.sequence[:batch, :beam_width, :].contiguous()

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


def run_beam_decision(
    runner: Any,
    session: Any,
    sampler_output: Any,
    is_last_step: bool,
    result_width: int | None = None,
) -> dict[str, torch.Tensor] | None:
    """Run the worker-side beam decision for one decode step.

    Args:
        runner: the model runner (must expose ``_current_beam_info``).
        session: the active BeamSearchSession.
        sampler_output: the sampler output whose logprob tensors carry the
            per-beam top-k candidates (post constraint override).
        is_last_step: whether this is the final decode step.
        result_width: output width for the final step; defaults to beam width.

    Returns a dict with ``out_token_ids`` / ``out_token_index`` /
    ``out_log_probs`` / ``out_sequence`` (device tensors, ``[B, W_eff]`` /
    ``[B, W_eff]`` / ``[B, W_eff]`` / ``[B, W_eff, steps]``), plus a copy of
    the cumulative ``beam_scores`` (``probs`` key), or ``None`` when the step
    cannot be decided worker-side (caller keeps the frontend path).
    """
    width = session.beam_width
    run_device = getattr(runner, "device", None)
    device_type = str(run_device.type) if run_device is not None else ""
    if device_type == "cuda" and not _cuda_worker_decision_width_supported(width):
        return None
    candidates = _select_candidates(sampler_output, width)
    if candidates is None:
        return None

    top_tokens, top_probs = candidates
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
        # First candidate round: seed the persistent buffers from the
        # prefill/initial sampler output so the decision starts from a
        # device-resident cumulative state (ACS init_from_prefill equivalent).
        init_beam_decision_state(session, top_tokens, top_probs)

    try:
        from vllm_gr.ops.npu import beam_ops

        # The fused xllm operators execute on NPU tensors; on CPU/CUDA hosts
        # (unit-test / GPU-debug environments) always take the PyTorch
        # reference so the decision logic stays verifiable off-device.
        if beam_ops.is_available() and device_type.startswith("npu"):
            decision = _run_npu_decision(session, top_tokens, top_probs, is_last_step, result_width)
            if decision is not None:
                decision["probs"] = session.context.decision.beam_scores[:1, :width].clone()
                return decision
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("vllm-gr worker beam decision (NPU) failed: %s; fallback", exc)

    decision = _run_torch_decision(session, top_tokens, top_probs, is_last_step, result_width)
    if decision is not None:
        decision["probs"] = session.context.decision.beam_scores[:1, :width].clone()
    return decision


def maybe_run_worker_beam_decision(runner: Any, sampler_output: Any) -> None:
    """Run and publish the active accelerator's worker-side beam decision."""
    prefill_decision = _run_cuda_prefill_decision(runner, sampler_output)
    if prefill_decision is not None:
        runner._beam_decision_tensors = prefill_decision
        compact = _encode_cuda_beam_result(prefill_decision, is_last_step=False)
        runner._beam_decision_result = compact[0] if compact is not None else None
        return

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
    )
    if decision is not None:
        # Keep the device winners alive until bookkeeping has replaced the
        # redundant W-by-W sampler payload with its O(W) compatibility row.
        runner._beam_decision_tensors = decision
        device_type = str(getattr(getattr(runner, "device", None), "type", ""))
        compact = (
            _encode_cuda_beam_result(decision, is_last_step=state.is_last_step)
            if device_type == "cuda"
            else None
        )
        if compact is not None:
            payload, parent_host = compact
            if not state.is_last_step:
                state.kv_reorder = parent_host[: session.beam_width]
            decision = payload
        else:
            # NPU keeps its established plain-container ABI.  Its host-copy
            # and xllm lifecycle have separate performance contracts; do not
            # change them as part of the CUDA transport optimization.
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
                payload[key] = (
                    value.detach().to("cpu").tolist() if torch.is_tensor(value) else value
                )
            payload["is_last_step"] = bool(state.is_last_step)
            if not state.is_last_step:
                out_idx_host = payload.get("out_token_index")
                if out_idx_host:
                    state.kv_reorder = [int(parent) for parent in out_idx_host[0]]
            decision = payload
    runner._beam_decision_result = decision


__all__ = [
    "flatten_decision_value",
    "init_beam_decision_state",
    "maybe_run_worker_beam_decision",
    "run_beam_decision",
    "unpack_beam_result_buffer",
    "worker_decision_enabled",
]
