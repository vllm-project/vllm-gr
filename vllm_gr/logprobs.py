# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized FlatLogprobs.append_fast: uses extend instead of
per-element append."""

import itertools
from collections.abc import Iterable
from typing import Any

__all__ = [
    "append_fast",
    "append_logprobs_for_next_position",
    "reconstruct_beam_logprobs",
]


def append_logprobs_for_next_position(
    request_logprobs: Any,
    token_ids: list[int],
    logprobs: list[float],
    decoded_tokens: Iterable[str | None],
    rank: Any,
    num_logprobs: int,
) -> None:
    """Appends logprobs for the next position, supporting list/array of ranks."""
    if isinstance(rank, (list, tuple)):
        ranks: Iterable[Any] = rank
    elif hasattr(rank, "tolist"):
        ranks = rank.tolist()
    else:
        if num_logprobs == -1:
            num_logprobs = len(logprobs)
        topk_ranks = range(1, num_logprobs + 1)
        ranks = itertools.chain((rank,), topk_ranks)

    from vllm.logprobs import FlatLogprobs, Logprob

    if isinstance(request_logprobs, FlatLogprobs):
        request_logprobs.append_fast(token_ids, logprobs, ranks, decoded_tokens)
    else:
        request_logprobs.append(
            {
                token_id: Logprob(
                    logprob=logprob,
                    rank=r,
                    decoded_token=token,
                )
                for token_id, logprob, r, token in zip(token_ids, logprobs, ranks, decoded_tokens)
            }
        )


def append_fast(
    self,
    token_ids: list[int],
    logprobs: list[float],
    ranks: "Iterable[int | None] | list[int | None]",
    decoded_tokens: "Iterable[str | None]",
) -> None:
    self.start_indices.append(len(self.logprobs))
    self.token_ids.extend(token_ids)
    self.logprobs.extend(logprobs)
    # ranks/decoded_tokens may be lazy or infinite iterators
    # (e.g. itertools.chain, itertools.repeat) -- use islice to bound them.
    # When they are already lists, extend directly (C-level memcpy).
    if ranks is None:
        self.ranks.extend(itertools.repeat(None, len(token_ids)))
    elif isinstance(ranks, list):
        self.ranks.extend(ranks)
    else:
        self.ranks.extend(itertools.islice(ranks, len(token_ids)))
    if decoded_tokens is None:
        self.decoded_tokens.extend(itertools.repeat(None, len(token_ids)))
    elif isinstance(decoded_tokens, list):
        self.decoded_tokens.extend(decoded_tokens)
    else:
        self.decoded_tokens.extend(itertools.islice(decoded_tokens, len(token_ids)))
    self.end_indices.append(len(self.logprobs))


def reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id=None):
    """Walk parent-pointer chain to build full FlatLogprobs for a beam.

    Replaces ``beam.logprobs`` in-place with a newly constructed
    ``FlatLogprobs`` containing the initial logprobs plus all step data
    collected via ``_lp_parent`` / ``_lp_step_data`` pointers.  If
    ``sid_end_token_id`` is given and the beam did not finish with EOS,
    an extra entry is appended for the session-end token.
    """
    from vllm.logprobs import FlatLogprobs

    steps = []
    ancestor = beam
    while getattr(ancestor, "_lp_step_data", None) is not None:
        steps.append(ancestor._lp_step_data)
        ancestor = ancestor._lp_parent
    steps.reverse()
    lp = FlatLogprobs(
        start_indices=list(initial_logprobs.start_indices),
        end_indices=list(initial_logprobs.end_indices),
        token_ids=list(initial_logprobs.token_ids),
        logprobs=list(initial_logprobs.logprobs),
        ranks=list(initial_logprobs.ranks),
        decoded_tokens=list(initial_logprobs.decoded_tokens),
    )
    for sd in steps:
        t_ids = sd[0].tolist() if hasattr(sd[0], "tolist") else sd[0]
        lps = sd[1].tolist() if hasattr(sd[1], "tolist") else sd[1]
        r = sd[2].tolist() if hasattr(sd[2], "tolist") else sd[2]
        d = sd[3].tolist() if hasattr(sd[3], "tolist") else sd[3]
        lp.append_fast(t_ids, lps, r, d)
    if sid_end_token_id is not None and beam.finish_reason != "stop":
        lp.append_fast([sid_end_token_id], [0.0], [None], [None])
    beam.logprobs = lp
