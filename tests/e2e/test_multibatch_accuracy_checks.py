# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Acceptance checks must not hide numeric, ordering or validity failures."""

from typing import Any

import pytest

from tests.e2e.multibatch_accuracy_checks import (
    compare_output,
    validate_output,
)


def test_exact_ties_are_reported_without_claiming_exact_equality() -> None:
    ref = [([1], -1.0), ([2], -1.0)]
    got = list(reversed(ref))
    result = compare_output(ref, got)
    assert result["candidate_gate_passed"] and not result["exact"]
    assert not result["top1_equal"] and result["reference_top1_tie_equivalent"]


def test_lower_scoring_top1_is_not_accepted_as_a_tie() -> None:
    ref = [([1], -1.0), ([2], -1.125)]
    result = compare_output(ref, list(reversed(ref)))
    assert not result["candidate_gate_passed"]
    assert result["reference_top1_gap"] == 0.125


def test_score_drift_remains_visible_when_candidates_match() -> None:
    result = compare_output([([1], -1.0)], [([1], -1.25)])
    assert result["candidate_gate_passed"] and not result["exact"]
    assert result["max_score_abs_error"] == 0.25


@pytest.mark.parametrize(  # type: ignore[misc]
    "output,reason",
    [
        ([([1], -1.0)], "count"),
        ([([1], -1.0), ([1], -2.0)], "duplicate"),
        ([([1], -1.0), ([3], -2.0)], "illegal"),
        ([([1], -1.0), ([2], float("nan"))], "non-finite"),
        ([([1], -2.0), ([2], -1.0)], "unsorted"),
    ],
)
def test_invalid_results_fail(output: list[Any], reason: str) -> None:
    with pytest.raises(AssertionError, match=reason):
        validate_output(output, width=2, steps=1, legal={(1,), (2,)})


pytestmark = pytest.mark.cpu_test
