# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from pathlib import Path
from typing import Any

import pytest

from tests.helpers.beam_candidates import describe_beam_candidate_mismatch

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("offline", "online", "messages"),
    [
        (
            [[dict(tokens=[1], score=-1.0)]],
            [[dict(tokens=[1], score=-1.25)]],
            [
                "First mismatch at rank 0",
                "Offline-only candidates: 0",
                "Online-only candidates: 0",
                "shared candidates: 0.25",
            ],
        ),
        (
            [[dict(tokens=[1], score=-1.0), dict(tokens=[2], score=-2.0)]],
            [[dict(tokens=[2], score=-2.0), dict(tokens=[1], score=-1.0)]],
            [
                "First mismatch at rank 0",
                "Offline-only candidates: 0",
                "Online-only candidates: 0",
                "shared candidates: 0.0",
            ],
        ),
        (
            [[dict(tokens=[1], score=-1.0)]],
            [[dict(tokens=[2], score=-1.0)]],
            [
                "Offline-only candidates: 1",
                "Online-only candidates: 1",
                "((1,), 1)",
                "((2,), 1)",
                "shared candidates: n/a",
            ],
        ),
        (
            [[dict(tokens=[1], score=-1.0), dict(tokens=[1], score=-2.0)]],
            [[dict(tokens=[1], score=-1.0)]],
            [
                "First mismatch at rank 1",
                "online=None",
                "Offline-only candidates: 1",
                "Online-only candidates: 0",
            ],
        ),
        (
            [],
            [[dict(tokens=[1], score=-1.0)]],
            [
                "Prompt groups: offline=0, online=1",
                "Group 0: offline=0, online=1",
                "Offline-only candidates: 0",
                "Online-only candidates: 1",
            ],
        ),
        (
            [[dict(tokens=[1], score=-1.0)], [dict(tokens=[2], score=-2.0)]],
            [[dict(tokens=[1], score=-1.0)], []],
            ["Group 1: offline=1, online=0", "First mismatch at rank 0", "online=None"],
        ),
    ],
    ids=["scores", "order", "membership", "duplicates", "extra-group", "missing-candidate"],
)
def test_beam_mismatch_reports_candidate_differences(
    offline: list[list[dict[str, Any]]],
    online: list[list[dict[str, Any]]],
    messages: list[str],
) -> None:
    artifact = Path("results/completion-pair-1.json")
    diagnostic = describe_beam_candidate_mismatch(offline, online, artifact)
    assert str(artifact) in diagnostic
    for message in messages:
        assert message in diagnostic


def test_beam_mismatch_bounds_examples_without_losing_counts() -> None:
    offline = [[dict(tokens=[i], score=-float(i)) for i in range(128)]]
    diagnostic = describe_beam_candidate_mismatch(offline, [[]], Path("pair.json"))
    assert "Offline-only candidates: 128" in diagnostic
    assert "token/count examples: [((0,), 1), ((1,), 1), ((2,), 1)]" in diagnostic
    assert "((127,), 1)" not in diagnostic
