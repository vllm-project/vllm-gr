# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure scoring and ordering operations for beam search."""

from collections.abc import Iterable
from math import pow

from vllm_gr.beam.domain import BeamCandidate, BeamState


def candidate_rank_key(candidate: BeamCandidate) -> tuple[float, int, int]:
    """Return the deterministic ordering key for a beam candidate."""

    return (
        -candidate.cumulative_logprob,
        candidate.parent_index,
        candidate.token_id,
    )


def length_penalized_score(
    token_ids: tuple[int, ...],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
    *,
    prompt_length: int = 0,
) -> float:
    """Calculate a beam score using vLLM's sequence-length convention.

    EOS is excluded from the effective length. ``prompt_length`` may be used
    by adapters that want generated-token-only normalization; its default of
    zero preserves the upstream vLLM behavior.
    """

    effective_length = len(token_ids) - prompt_length
    if token_ids and token_ids[-1] == eos_token_id:
        effective_length -= 1
    if effective_length <= 0:
        raise ValueError("a beam must contain at least one scored token")
    normalization = pow(effective_length, length_penalty)
    return cumulative_logprob / normalization


def rank_candidates(candidates: Iterable[BeamCandidate]) -> tuple[BeamCandidate, ...]:
    """Rank candidates by score with deterministic source-order tie breaks."""

    return tuple(sorted(candidates, key=candidate_rank_key))


def rank_states(
    states: Iterable[BeamState],
    *,
    eos_token_id: int,
    length_penalty: float = 1.0,
    prompt_length: int = 0,
) -> tuple[BeamState, ...]:
    """Rank final beams by length-penalized score, deterministically."""

    return tuple(
        sorted(
            states,
            key=lambda state: (
                -length_penalized_score(
                    state.token_ids,
                    state.cumulative_logprob,
                    eos_token_id,
                    length_penalty,
                    prompt_length=prompt_length,
                ),
                state.token_ids,
            ),
        )
    )
