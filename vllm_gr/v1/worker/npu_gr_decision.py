# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict device-resident decision boundary for persistent NPU ModelRunner V1.

This module is deliberately not wired into ``NPUModelRunner`` yet.  The pinned
vllm-ascend V1 prepare/output contracts must be inspected and tested on an NPU
installation before the adapter can safely publish physical rows or transport
the final result.  These helpers isolate the already-verifiable xllm decision
contract: they never fall back to Torch, materialize tensors on the Host, or
reorder KV implicitly.

The canonical copies below reuse session-owned fixed buffers.  The current
xllm operators still allocate their output tensors, including
``out_sequence``; fixed operator-output workspace is intentionally left for
the later, hardware-verified ModelRunner adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class NPUDecisionStep:
    """One validated decision over the canonical session-owned buffers.

    ``sequence`` and ``scores`` are active views of
    ``session.context.decision`` and therefore share its fixed storage.
    Non-final ``reorder_parents`` / ``reorder_prefix`` remain device tensors
    returned by xllm and must be consumed exactly once by the future NPU
    backend.  A final result never requests KV reorder.
    """

    device_step: int
    token_ids: torch.Tensor
    parent_ids: torch.Tensor
    scores: torch.Tensor
    sequence: torch.Tensor
    reorder_parents: torch.Tensor | None = None
    reorder_prefix: torch.Tensor | None = None


def _resolve_ops(ops: Any | None) -> tuple[Any, bool]:
    injected = ops is not None
    if ops is None:
        from vllm_gr.ops.npu import beam_ops
        from vllm_gr.ops.npu._xllm_loader import custom_ops, has_op

        ops = beam_ops
        required = ("beam_search_group", "beam_search_rec_final_select")
        missing = [name for name in required if not has_op(name)]
        if custom_ops() is None or missing:
            detail = ", ".join(missing) if missing else "custom_ops_lib"
            raise RuntimeError("persistent NPU decision operators are unavailable: " + detail)
    else:
        available = getattr(ops, "is_available", None)
        if not callable(available) or available() is not True:
            reason = getattr(ops, "unavailable_reason", lambda: "unknown reason")()
            raise RuntimeError(f"persistent NPU decision operators are unavailable: {reason}")
    for name in ("beam_search_group", "beam_search_rec_final_select"):
        if not callable(getattr(ops, name, None)):
            raise RuntimeError(f"persistent NPU decision operator {name} is unavailable")
    return ops, injected


def _require_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    require_npu: bool,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype is not dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if require_npu and value.device.type != "npu":
        raise RuntimeError(f"{name} must be NPU-resident, got {value.device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def _decision_buffers(
    session: Any,
    width: int,
    *,
    require_npu: bool,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError(f"expected_width must be a positive integer, got {width!r}")
    if getattr(session, "beam_width", None) != width:
        raise ValueError("persistent NPU decision width does not match its session")
    if getattr(session, "is_active", None) is not True:
        raise RuntimeError("persistent NPU decision requires an active session")
    if getattr(session, "slot", None) != 0:
        raise RuntimeError("persistent NPU decision currently requires isolated B=1 slot zero")
    context = getattr(session, "context", None)
    buffers = getattr(context, "decision", None)
    if buffers is None:
        raise RuntimeError("persistent NPU session has no canonical decision buffers")
    sequence = getattr(buffers, "sequence", None)
    scores = getattr(buffers, "beam_scores", None)
    prefix = getattr(buffers, "group_token_num", None)
    if not torch.is_tensor(sequence) or sequence.ndim != 3:
        raise RuntimeError("canonical NPU sequence buffer must be a 3-D Tensor")
    if sequence.shape[0] != 1:
        raise RuntimeError("persistent NPU decision does not support B>1 buffers")
    if sequence.shape[1] != width:
        raise RuntimeError("canonical NPU sequence buffer must use the fixed request beam width")
    max_steps = int(sequence.shape[2])
    if max_steps < 1:
        raise RuntimeError("canonical NPU sequence buffer has no generation capacity")
    device = sequence.device
    _require_tensor(
        sequence,
        name="canonical sequence",
        shape=(1, width, max_steps),
        dtype=torch.int32,
        device=device,
        require_npu=require_npu,
    )
    _require_tensor(
        scores,
        name="canonical scores",
        shape=(1, width),
        dtype=torch.float32,
        device=device,
        require_npu=require_npu,
    )
    _require_tensor(
        prefix,
        name="canonical group_token_num",
        shape=(1, width),
        dtype=torch.int32,
        device=device,
        require_npu=require_npu,
    )
    return buffers, sequence, scores, prefix, max_steps


def bootstrap_from_constrained_prefill(
    session: Any,
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    *,
    expected_width: int,
    ops: Any | None = None,
) -> NPUDecisionStep:
    """Seed canonical state directly from constrained Prefill device winners.

    Supplying ``ops`` is an explicit CPU-test backend contract.  Production
    callers omit it and consequently require real NPU tensors and available
    xllm decision operators.
    """

    _, injected = _resolve_ops(ops)
    buffers, sequence, canonical_scores, parent_scratch, _ = _decision_buffers(
        session,
        expected_width,
        require_npu=not injected,
    )
    if getattr(buffers, "worker_decode_steps", None) != 0:
        raise RuntimeError("persistent NPU Prefill bootstrap is not at device step zero")
    if getattr(session, "decode_step", None) != 0:
        raise RuntimeError("persistent NPU Prefill session is not at decode step zero")
    device = sequence.device
    _require_tensor(
        token_ids,
        name="Prefill token_ids",
        shape=(1, expected_width),
        dtype=torch.int32,
        device=device,
        require_npu=not injected,
    )
    _require_tensor(
        scores,
        name="Prefill scores",
        shape=(1, expected_width),
        dtype=torch.float32,
        device=device,
        require_npu=not injected,
    )

    # Mutation starts only after every input and canonical buffer is validated.
    sequence[:, :, 0].copy_(token_ids)
    canonical_scores.copy_(scores)
    parent_scratch.zero_()
    buffers.worker_decode_steps = 1
    return NPUDecisionStep(
        device_step=1,
        token_ids=sequence[:, :, 0],
        parent_ids=parent_scratch,
        scores=canonical_scores,
        sequence=sequence,
    )


def advance_fused_decision(
    session: Any,
    candidate_token_ids: torch.Tensor,
    candidate_scores: torch.Tensor,
    *,
    expected_device_step: int,
    is_last_step: bool,
    result_width: int,
    ops: Any | None = None,
) -> NPUDecisionStep:
    """Run exactly one fused xllm decision without fallback or KV reorder.

    ``expected_device_step`` is the number of already-completed generation
    stages.  The returned step is always ``expected_device_step + 1``.
    """

    ops, injected = _resolve_ops(ops)
    width = getattr(session, "beam_width", None)
    buffers, sequence, canonical_scores, canonical_prefix, max_steps = _decision_buffers(
        session, width, require_npu=not injected
    )
    if (
        isinstance(expected_device_step, bool)
        or not isinstance(expected_device_step, int)
        or expected_device_step < 1
    ):
        raise ValueError("expected_device_step must be a positive integer")
    if expected_device_step >= max_steps:
        raise ValueError("persistent NPU decision exceeds fixed max generation steps")
    if getattr(buffers, "worker_decode_steps", None) != expected_device_step:
        raise RuntimeError("canonical NPU decision/device step mismatch")
    if getattr(session, "decode_step", None) != expected_device_step - 1:
        raise RuntimeError("persistent NPU session/device step mismatch")
    if not isinstance(is_last_step, bool):
        raise TypeError("is_last_step must be a boolean")
    if isinstance(result_width, bool) or not isinstance(result_width, int):
        raise TypeError("result_width must be an integer")
    if not 1 <= result_width <= width:
        raise ValueError("result_width must be in [1, beam_width]")
    if not is_last_step and result_width != width:
        raise ValueError("non-final NPU result_width must equal the fixed beam width")
    if is_last_step and result_width % 32 != 0:
        raise ValueError("final NPU result_width must be a multiple of 32")
    device = sequence.device
    _require_tensor(
        candidate_token_ids,
        name="candidate_token_ids",
        shape=(width, width),
        dtype=torch.int32,
        device=device,
        require_npu=not injected,
    )
    _require_tensor(
        candidate_scores,
        name="candidate_scores",
        shape=(width, width),
        dtype=torch.float32,
        device=device,
        require_npu=not injected,
    )
    if candidate_token_ids.shape[1] % 8 != 0:
        raise ValueError("NPU candidate width must be a multiple of 8")

    cumulative = canonical_scores.view(width, 1)
    if is_last_step:
        raw = ops.beam_search_rec_final_select(
            cumulative,
            candidate_token_ids,
            candidate_scores,
            sequence,
            expected_device_step,
            result_width,
        )
        if not isinstance(raw, tuple) or len(raw) != 4:
            raise RuntimeError("NPU final-select returned an invalid output protocol")
        out_tokens, out_parents, out_scores, out_sequence = raw
        _require_tensor(
            out_tokens,
            name="final token_ids",
            shape=(result_width, 1),
            dtype=torch.int32,
            device=device,
            require_npu=not injected,
        )
        _require_tensor(
            out_parents,
            name="final parent_ids",
            shape=(result_width, 1),
            dtype=torch.int32,
            device=device,
            require_npu=not injected,
        )
        _require_tensor(
            out_scores,
            name="final scores",
            shape=(result_width, 1),
            dtype=torch.float32,
            device=device,
            require_npu=not injected,
        )
        _require_tensor(
            out_sequence,
            name="final sequence",
            shape=(1, result_width, max_steps),
            dtype=torch.int32,
            device=device,
            require_npu=not injected,
        )
        # No canonical mutation occurs until the complete operator tuple passes.
        sequence[:, :result_width].copy_(out_sequence)
        canonical_scores[:, :result_width].copy_(out_scores.view(1, result_width))
        buffers.worker_decode_steps = expected_device_step + 1
        return NPUDecisionStep(
            device_step=expected_device_step + 1,
            token_ids=out_tokens.view(1, result_width),
            parent_ids=out_parents.view(1, result_width),
            scores=canonical_scores[:, :result_width],
            sequence=sequence[:, :result_width],
        )

    raw = ops.beam_search_group(
        cumulative,
        candidate_token_ids,
        candidate_scores,
        sequence,
        expected_device_step,
        width,
    )
    if not isinstance(raw, tuple) or len(raw) != 5:
        raise RuntimeError("NPU beam-search-group returned an invalid output protocol")
    out_tokens, out_parents, out_scores, out_prefix, out_sequence = raw
    for name, tensor, dtype in (
        ("group token_ids", out_tokens, torch.int32),
        ("group parent_ids", out_parents, torch.int32),
        ("group scores", out_scores, torch.float32),
        ("group prefix", out_prefix, torch.int32),
    ):
        _require_tensor(
            tensor,
            name=name,
            shape=(1, width),
            dtype=dtype,
            device=device,
            require_npu=not injected,
        )
    _require_tensor(
        out_sequence,
        name="group sequence",
        shape=(1, width, max_steps),
        dtype=torch.int32,
        device=device,
        require_npu=not injected,
    )
    # B=1 makes the xllm global parent/prefix offsets request-local already.
    sequence.copy_(out_sequence)
    canonical_scores.copy_(out_scores)
    canonical_prefix.copy_(out_prefix)
    buffers.worker_decode_steps = expected_device_step + 1
    return NPUDecisionStep(
        device_step=expected_device_step + 1,
        token_ids=out_tokens,
        parent_ids=out_parents,
        scores=canonical_scores,
        sequence=sequence,
        reorder_parents=out_parents,
        reorder_prefix=out_prefix,
    )


__all__ = [
    "NPUDecisionStep",
    "advance_fused_decision",
    "bootstrap_from_constrained_prefill",
]
