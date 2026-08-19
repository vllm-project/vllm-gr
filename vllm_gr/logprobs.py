# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized FlatLogprobs.append_fast: uses extend instead of
per-element append."""

import itertools
from collections.abc import Iterable

__all__ = [
    "append_fast",
    "project_flat_logprobs",
    "reconstruct_beam_logprobs",
]


def project_flat_logprobs(flat, selected_tokens, logprobs_mode, top_logprobs=0):
    """Project a candidate history to selected-token or top-k output."""
    from vllm.logprobs import FlatLogprobs

    projected = FlatLogprobs()
    for position, (start, end) in enumerate(zip(flat.start_indices, flat.end_indices)):
        token_ids = list(flat.token_ids[start:end])
        logprobs = list(flat.logprobs[start:end])
        ranks = list(flat.ranks[start:end])
        decoded = list(flat.decoded_tokens[start:end])
        selected_token = selected_tokens[position] if position < len(selected_tokens) else None
        selected_index = (
            token_ids.index(selected_token)
            if selected_token is not None and selected_token in token_ids
            else None
        )
        if logprobs_mode == "selected":
            indices = [] if selected_index is None else [selected_index]
        else:
            indices = sorted(range(len(logprobs)), key=logprobs.__getitem__, reverse=True)[
                :top_logprobs
            ]
            if selected_index is not None:
                indices = [selected_index, *[i for i in indices if i != selected_index]]
        projected.append_fast(
            [token_ids[i] for i in indices],
            [logprobs[i] for i in indices],
            [ranks[i] for i in indices],
            [decoded[i] for i in indices],
        )
    return projected


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
    if isinstance(ranks, list):
        self.ranks.extend(ranks)
    else:
        self.ranks.extend(itertools.islice(ranks, len(token_ids)))
    if isinstance(decoded_tokens, list):
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
        lp.append_fast(sd[0], sd[1], sd[2], sd[3])
    if sid_end_token_id is not None and beam.finish_reason != "stop":
        lp.append_fast([sid_end_token_id], [0.0], [None], [None])
    beam.logprobs = lp
