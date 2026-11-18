# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Explicit output checks shared by the real-model multi-batch acceptance run."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any


def validate_output(
    output: list[Any], *, width: int, steps: int, legal: set[tuple[int, ...]]
) -> None:
    assert len(output) == width, "incorrect candidate count"
    keys = [tuple(tokens) for tokens, _ in output]
    assert len(set(keys)) == width, "duplicate candidates"
    assert all(len(key) == steps and key in legal for key in keys), "illegal candidate"
    assert all(math.isfinite(score) for _, score in output), "non-finite score"
    assert all(a[1] >= b[1] for a, b in pairwise(output)), "unsorted scores"


def compare_output(reference: list[Any], actual: list[Any]) -> dict[str, Any]:
    """Report exact equality separately from the existing candidate criterion.

    Exact cumulative scores are a strict diagnostic gate. No unapproved
    numerical tolerance is inferred from the observed differences.
    """
    left = {tuple(tokens): score for tokens, score in reference}
    right = {tuple(tokens): score for tokens, score in actual}
    assert left and len(left) == len(reference) == len(right) == len(actual)
    assert all(math.isfinite(score) for score in (*left.values(), *right.values()))
    common = left.keys() & right.keys()
    top1 = tuple(actual[0][0])
    overlap = len(common) / len(left)
    tied_top1 = left.get(top1) == reference[0][1]
    errors = [abs(left[key] - right[key]) for key in common]
    return {
        "exact": reference == actual,
        "ordered_candidates_equal": list(left) == list(right),
        "top1_equal": reference[0][0] == actual[0][0],
        "reference_top1_tie_equivalent": tied_top1,
        "candidate_overlap": overlap,
        "max_score_abs_error": max(errors) if errors else None,
        "mean_score_abs_error": sum(errors) / len(errors) if errors else None,
        "reference_top1_gap": reference[0][1] - left[top1] if top1 in left else None,
        "candidate_gate_passed": overlap >= 0.95 and tied_top1,
    }
