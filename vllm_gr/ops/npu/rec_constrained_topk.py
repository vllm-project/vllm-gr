# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU dispatch wrapper for the fused constrained Top-K operator."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch

from vllm_gr.ops.npu._xllm_loader import custom_ops as _custom_ops

MAX_TOP_K = 256
NPUConstraintOp = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def is_available() -> bool:
    """Return whether the NPU extension exposes the fused operator."""
    custom_ops = _custom_ops()
    return custom_ops is not None and callable(getattr(custom_ops, "rec_constrained_topk", None))


def unavailable_reason() -> str:
    """Describe why the fused NPU operator cannot be called."""
    custom_ops = _custom_ops()
    if custom_ops is None:
        return "custom_ops_lib cannot be imported"
    if not callable(getattr(custom_ops, "rec_constrained_topk", None)):
        return "custom_ops_lib does not export rec_constrained_topk"
    return ""


def resolve_rec_constrained_topk_npu() -> NPUConstraintOp:
    """Resolve the real fused callable once, without a Torch fallback."""

    custom_ops = _custom_ops()
    if custom_ops is None:
        raise RuntimeError(
            "NPU rec_constrained_topk is unavailable: custom_ops_lib cannot be imported"
        )
    op = getattr(custom_ops, "rec_constrained_topk", None)
    if not callable(op):
        raise RuntimeError(
            "NPU rec_constrained_topk is unavailable: "
            "custom_ops_lib does not export rec_constrained_topk"
        )
    return cast(NPUConstraintOp, op)


def rec_constrained_topk_npu(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_values: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    prefix2_values: torch.Tensor,
    temperatures: torch.Tensor,
    current_step: int,
    top_k: int,
    max_prefix1_degree: int = 0,
    max_prefix2_degree: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the fused NPU operator using the shared operator signature."""
    custom_ops = _custom_ops()
    if custom_ops is None or not callable(getattr(custom_ops, "rec_constrained_topk", None)):
        raise RuntimeError(
            f"NPU rec_constrained_topk is unavailable ({unavailable_reason()}); "
            "use the Torch implementation instead"
        )
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        custom_ops.rec_constrained_topk(
            logits,
            sequence_group,
            first_token_ids,
            prefix1_offsets,
            prefix1_values,
            prefix1_pair_keys,
            prefix2_value_offsets,
            prefix2_values,
            temperatures,
            current_step,
            top_k,
            max_prefix1_degree,
            max_prefix2_degree,
        ),
    )


__all__ = [
    "MAX_TOP_K",
    "is_available",
    "rec_constrained_topk_npu",
    "resolve_rec_constrained_topk_npu",
    "unavailable_reason",
]
