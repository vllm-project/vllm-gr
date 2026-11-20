# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Test-side view helpers for ``BeamPostprocessWorkspace``.

These mirror the CUDA Beam kernel input geometry (``[rows, K]`` contiguous
views, compact step-0 bootstrap).  Production code slices the workspace inline,
so the helpers live with their only consumers: tests and benchmarks.
"""

from __future__ import annotations

import torch

from vllm_gr.v1.worker.cuda.beam_postprocess_workspace import BeamPostprocessWorkspace


def validate_batch_size(workspace: BeamPostprocessWorkspace, batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if batch_size > workspace.max_batch:
        raise ValueError(f"batch_size must be in [1, {workspace.max_batch}], got {batch_size}")


def candidate_rows(
    workspace: BeamPostprocessWorkspace,
    tensor: torch.Tensor,
    batch_size: int,
    *,
    compact: bool = False,
) -> torch.Tensor:
    """Return an exact contiguous ``[rows, K]`` view without allocating.

    Decode uses ``rows=B*W``.  The compact bootstrap uses the first ``B`` rows
    of the same fixed storage, matching the CUDA Beam step-0 ABI.
    """

    validate_batch_size(workspace, batch_size)
    rows = batch_size if compact else batch_size * workspace.key.beam_width
    return tensor.view(workspace.max_batch * workspace.key.beam_width, workspace.key.top_k)[:rows]


def active_beams(
    workspace: BeamPostprocessWorkspace,
    tensor: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Return the leading contiguous ``[B, W, ...]`` workspace slice."""

    validate_batch_size(workspace, batch_size)
    return tensor[:batch_size]


def result_count(
    workspace: BeamPostprocessWorkspace,
    batch_size: int,
    result_width: int | None = None,
) -> int:
    validate_batch_size(workspace, batch_size)
    width = workspace.key.beam_width if result_width is None else result_width
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("result_width must be an int")
    if not 1 <= width <= workspace.key.beam_width:
        raise ValueError(f"result_width must be in [1, {workspace.key.beam_width}], got {width}")
    return int(batch_size * width)
