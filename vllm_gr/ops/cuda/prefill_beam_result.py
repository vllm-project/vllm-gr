# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA prefill Beam validation/bootstrap/result packing.

The public Beam wire format is ``token_ids | parent_ids | score_bits``.  The
status words used to defer constrained-candidate validation are an internal
Worker transport and are never included in the public payload.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

STATUS_OK = 0
STATUS_NO_VALID_CANDIDATE = 1


@triton.jit
def _prefill_beam_result_kernel(
    input_token_ids,
    input_scores,
    input_candidate_counts,
    output_status,
    output_payload,
    active_count,
    invalid_id,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    lane = rank < W
    count = tl.load(input_candidate_counts + row)
    valid = lane & (rank < count)
    source_offset = row * W + rank

    token = tl.load(
        input_token_ids + source_offset,
        mask=valid,
        other=invalid_id,
    ).to(tl.int32)
    score = tl.load(
        input_scores + source_offset,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    score_bits = score.to(tl.int32, bitcast=True)

    tl.store(output_payload + source_offset, token, mask=lane)
    tl.store(output_payload + active_count + source_offset, 0, mask=lane)
    tl.store(output_payload + 2 * active_count + source_offset, score_bits, mask=lane)
    tl.store(output_status + row, tl.where(count > 0, 0, 1))


def _validate_prefill_beam_result_inputs(
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    candidate_counts: torch.Tensor,
    out_status: torch.Tensor,
    out_payload: torch.Tensor,
) -> tuple[int, int]:
    if token_ids.device.type != "cuda":
        raise ValueError(
            f"prefill Beam result packing requires CUDA tensors, got {token_ids.device}"
        )
    if token_ids.dtype != torch.int32 or token_ids.ndim != 2:
        raise TypeError("token_ids must be a 2-D CUDA int32 tensor")
    if scores.dtype != torch.float32 or tuple(scores.shape) != tuple(token_ids.shape):
        raise TypeError("scores must be CUDA float32 with the same shape as token_ids")
    if scores.device != token_ids.device:
        raise ValueError("scores.device must match token_ids.device")
    batch_size, beam_width = token_ids.shape
    if batch_size < 1 or beam_width < 1:
        raise ValueError("token_ids must have positive batch and Beam dimensions")
    specs = (
        ("candidate_counts", candidate_counts, torch.int32, 1),
        ("out_status", out_status, torch.int32, 1),
        ("out_payload", out_payload, torch.int32, 1),
    )
    for name, tensor, dtype, ndim in specs:
        if tensor.device != token_ids.device:
            raise ValueError(f"{name}.device must match token_ids.device")
        if tensor.dtype != dtype or tensor.ndim != ndim:
            raise TypeError(f"{name} must be a {ndim}-D CUDA {dtype} tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not token_ids.is_contiguous() or not scores.is_contiguous():
        raise ValueError("token_ids and scores must be contiguous")
    if candidate_counts.numel() < batch_size or out_status.numel() < batch_size:
        raise ValueError("candidate_counts/out_status capacity is smaller than batch_size")
    if out_payload.numel() < 3 * batch_size * beam_width:
        raise ValueError("out_payload capacity is smaller than 3 * batch_size * beam_width")
    return batch_size, beam_width


def _launch_prefill_beam_result(
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    candidate_counts: torch.Tensor,
    out_status: torch.Tensor,
    out_payload: torch.Tensor,
    invalid_token_id: int,
) -> None:
    batch_size, beam_width = token_ids.shape
    block = triton.next_power_of_2(beam_width)
    _prefill_beam_result_kernel[(batch_size,)](
        token_ids,
        scores,
        candidate_counts,
        out_status,
        out_payload,
        batch_size * beam_width,
        invalid_token_id,
        W=beam_width,
        BLOCK=block,
        num_warps=4 if block >= 64 else 1,
    )


def prefill_beam_result_out(
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    candidate_counts: torch.Tensor,
    out_status: torch.Tensor,
    out_payload: torch.Tensor,
    invalid_token_id: int,
) -> None:
    """Validate constrained prefill rows and pack the existing Beam wire ABI.

    ``out_payload`` is a flat int32 buffer containing three active-size planes:
    ``token_ids[B*W] | parent_ids[B*W] | score_bits[B*W]``. Scores are bitcast,
    never numerically converted. Invalid ranks receive a safe token sentinel,
    ``-inf`` score, and parent zero. A row with no candidates is reported via
    ``out_status``; callers must check it before publishing the payload.
    """

    _validate_prefill_beam_result_inputs(
        token_ids,
        scores,
        candidate_counts,
        out_status,
        out_payload,
    )
    if isinstance(invalid_token_id, bool) or not isinstance(invalid_token_id, int):
        raise TypeError("invalid_token_id must be an int")
    if not -(2**31) <= invalid_token_id < 2**31:
        raise ValueError("invalid_token_id must fit int32")
    _launch_prefill_beam_result(
        token_ids,
        scores,
        candidate_counts,
        out_status,
        out_payload,
        invalid_token_id,
    )


def _prefill_beam_result_out_unchecked(
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    candidate_counts: torch.Tensor,
    out_status: torch.Tensor,
    out_payload: torch.Tensor,
    invalid_token_id: int,
) -> None:
    """Launch on a startup-validated fixed workspace (Worker-internal only)."""

    _launch_prefill_beam_result(
        token_ids,
        scores,
        candidate_counts,
        out_status,
        out_payload,
        invalid_token_id,
    )


__all__ = [
    "STATUS_NO_VALID_CANDIDATE",
    "STATUS_OK",
    "prefill_beam_result_out",
]
