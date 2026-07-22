# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Candidate pruning operations for beam search."""

from collections.abc import Iterable
from heapq import nsmallest
from math import isfinite

from vllm_gr.beam.domain import BeamCandidate
from vllm_gr.beam.ranking import candidate_rank_key


def prune_candidates(
    candidates: Iterable[BeamCandidate], beam_width: int
) -> tuple[BeamCandidate, ...]:
    """Keep the best finite candidates up to ``beam_width``."""

    if isinstance(beam_width, bool) or not isinstance(beam_width, int):
        raise TypeError("beam_width must be an integer")
    if beam_width <= 0:
        raise ValueError("beam_width must be greater than zero")
    viable = (candidate for candidate in candidates if isfinite(candidate.cumulative_logprob))
    # Preserve the deterministic full-ranking order without sorting candidates
    # that will be discarded. This keeps pruning O(n log beam_width).
    return tuple(nsmallest(beam_width, viable, key=candidate_rank_key))
