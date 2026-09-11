# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from benchmarks.beam_search.sid_output import extract_candidates, validate_candidates
from vllm_gr.outputs import CompactBeamSearchOutput, CompactBeamSearchSequence


def test_sid_extraction_matches_across_public_output_modes() -> None:
    compact = [
        CompactBeamSearchOutput(
            (
                CompactBeamSearchSequence((1, 100, 200, 300, 2), -0.25),
                CompactBeamSearchSequence((1, 101, 201, 301, 2), -0.5),
            )
        )
    ]
    full = [
        SimpleNamespace(
            sequences=[
                SimpleNamespace(tokens=[77, 88, 1, 100, 200, 300, 2], cum_logprob=-0.25),
                SimpleNamespace(tokens=[77, 88, 1, 101, 201, 301, 2], cum_logprob=-0.5),
            ]
        )
    ]
    rows = extract_candidates(compact, "compact", 2, 1, 2)
    assert rows == [((100, 200, 300), -0.25), ((101, 201, 301), -0.5)]
    other = extract_candidates(full, "full", 2, 1, 2)
    assert rows == other
    validate_candidates(rows, 2)


@pytest.mark.parametrize(  # type: ignore[misc]
    "tokens", [(1, 100, 200, 300), (100, 200, 300, 2), ()]
)
def test_sid_extraction_rejects_missing_markers(tokens: tuple[int, ...]) -> None:
    output = [CompactBeamSearchOutput((CompactBeamSearchSequence(tokens, -0.25),))]
    with pytest.raises(ValueError, match="markers"):
        extract_candidates(output, "compact", 2, 1, 2)


def test_fixed_sid3_benchmark_rejects_early_stop_and_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="three SID"):
        validate_candidates([((100, 200), -0.25)], 1)
    with pytest.raises(ValueError, match="finite"):
        validate_candidates([((100, 200, 300), float("nan"))], 1)
