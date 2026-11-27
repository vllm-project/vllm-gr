# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Failure diagnostics for exact offline/online Beam candidate comparisons."""

from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any


def describe_beam_candidate_mismatch(
    offline: list[list[dict[str, Any]]],
    online: list[list[dict[str, Any]]],
    artifact: Path,
) -> str:
    lines = [
        f"Offline/online Beam candidates differ; full evidence: {artifact}",
        f"Prompt groups: offline={len(offline)}, online={len(online)}",
    ]
    empty: list[dict[str, Any]] = []
    for group, (expected, actual) in enumerate(zip_longest(offline, online, fillvalue=empty)):
        if expected == actual:
            continue
        lines.append(f"Group {group}: offline={len(expected)}, online={len(actual)} candidates")
        for rank, (left, right) in enumerate(zip_longest(expected, actual)):
            if left != right:
                lines.append(f"First mismatch at rank {rank}: offline={left!r}, online={right!r}")
                break
        expected_tokens = Counter(tuple(candidate["tokens"]) for candidate in expected)
        actual_tokens = Counter(tuple(candidate["tokens"]) for candidate in actual)
        for name, counts in (
            ("Offline-only", expected_tokens - actual_tokens),
            ("Online-only", actual_tokens - expected_tokens),
        ):
            lines.append(
                f"{name} candidates: {counts.total()}; "
                f"token/count examples: {list(counts.items())[:3]}"
            )
        expected_scores: dict[tuple[int, ...], list[float]] = defaultdict(list)
        actual_scores: dict[tuple[int, ...], list[float]] = defaultdict(list)
        for candidates, scores in ((expected, expected_scores), (actual, actual_scores)):
            for candidate in candidates:
                scores[tuple(candidate["tokens"])].append(candidate["score"])
        # Compare by token identity so a rank change does not imply a score change.
        deltas = [
            abs(left - right)
            for tokens in expected_scores.keys() & actual_scores.keys()
            for left, right in zip(sorted(expected_scores[tokens]), sorted(actual_scores[tokens]))
        ]
        lines.append(
            f"Max absolute score delta for shared candidates: {max(deltas) if deltas else 'n/a'}"
        )
    return "\n".join(lines)
