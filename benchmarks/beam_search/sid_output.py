# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Extract, validate, and write CPU SID token IDs and cumulative scores."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def extract_candidates(
    output: Any, mode: str, prompt_length: int, begin_id: int, end_id: int
) -> list[tuple[tuple[int, ...], float]]:
    """Consume only public fields and return independent CPU SID token IDs/scores.

    SID values are model-vocabulary token IDs with begin/end markers removed.
    """
    if len(output) != 1:
        raise ValueError("expected one request")
    candidates = []
    for sequence in output[0].sequences:
        if mode == "compact":
            generated = sequence.token_ids
            score = sequence.cumulative_logprob
        else:
            generated = sequence.tokens[prompt_length:]
            score = sequence.cum_logprob
        if not generated or generated[0] != begin_id or generated[-1] != end_id:
            raise ValueError("expected configured SID begin/end markers")
        candidates.append((tuple(generated[1:-1]), float(score)))
    return candidates


def validate_candidates(candidates: list[tuple[tuple[int, ...], float]], beam_width: int) -> None:
    """Require beam_width SID=3 candidates with finite cumulative scores."""
    if len(candidates) != beam_width:
        raise ValueError(f"expected {beam_width} candidates, got {len(candidates)}")
    if any(len(ids) != 3 or not math.isfinite(score) for ids, score in candidates):
        raise ValueError("expected three SID token IDs and a finite cumulative score")


def write_candidates(path: Path, candidates: list[tuple[tuple[int, ...], float]]) -> None:
    """Write raw SID/score pairs in candidate order as JSON."""
    path.write_text(json.dumps(candidates, indent=2, allow_nan=False) + "\n")
