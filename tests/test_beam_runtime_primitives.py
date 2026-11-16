# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Tests for platform-independent Beam runtime primitives."""

from __future__ import annotations

import pytest

from vllm_gr.v1.beam.geometry import _compute_beam_bounds


@pytest.mark.parametrize(  # type: ignore[misc]
    ("args", "expected"),
    [
        ((0, 10, 6, 4, 99), (8, 8, 0, 0)),
        ((0, 10, 12, 4, 99), (2, 4, 2, 0)),
        ((1, 10, 16, 4, 99), (2, 4, 2, 1)),
        ((0, 10, 6, 4, 2), (2, 2, 0, 0)),
    ],
    ids=[
        "prefix-incomplete",
        "first-beam-suffix",
        "later-beam-partial-cache",
        "chunk-budget-smaller-than-remaining",
    ],
)
def test_compute_beam_bounds_behavior(
    args: tuple[int, int, int, int, int],
    expected: tuple[int, int, int, int],
) -> None:
    assert _compute_beam_bounds(*args) == expected


pytestmark = pytest.mark.cpu_test
