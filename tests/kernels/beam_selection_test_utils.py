# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Deterministic inputs and exact comparisons for CUDA Beam selection tests."""

from collections.abc import Sequence

import torch


def make_selection_inputs(
    batch_size: int,
    beam_width: int,
    current_step: int,
    *,
    max_steps: int = 3,
    candidate_width: int | None = None,
) -> tuple[torch.Tensor, ...]:
    # Candidate width K is independent of Beam width W; default to the square case.
    candidates = beam_width if candidate_width is None else candidate_width
    generator = torch.Generator(device="cuda").manual_seed(20260827)
    cumulative = torch.randn(
        (batch_size * beam_width, 1),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    top_tokens = torch.arange(
        batch_size * beam_width * candidates,
        dtype=torch.int32,
        device="cuda",
    ).reshape(batch_size * beam_width, candidates)
    top_logprobs = torch.randn(
        (batch_size * beam_width, candidates),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    sequence = torch.arange(
        batch_size * beam_width * max_steps,
        dtype=torch.int32,
        device="cuda",
    ).reshape(batch_size, beam_width, max_steps)
    assert current_step < max_steps
    return cumulative, top_tokens, top_logprobs, sequence


def assert_selection_equal(
    actual: Sequence[torch.Tensor], expected: Sequence[torch.Tensor]
) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
