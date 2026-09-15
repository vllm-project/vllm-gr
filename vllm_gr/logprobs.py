# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized FlatLogprobs.append_fast: uses extend instead of
per-element append."""

import itertools
from collections.abc import Iterable

from vllm.logprobs import FlatLogprobs

__all__ = [
    "append_fast",
    "reconstruct_beam_logprobs",
    "reconstruct_beam_logprobs_batch",
]


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
    # A fused operator may drop the optional metadata entirely (ranks /
    # decoded_tokens == None): pad with None entries so every candidate has a
    # rank slot, matching the FlatLogprobs per-token contract.
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


def _reconstruct_one(
    beam,
    init_token_ids,
    init_logprobs,
    init_ranks,
    init_decoded,
    init_start,
    init_end,
    sid_end_token_id,
) -> None:
    """Build the full FlatLogprobs for one beam by walking its parent chain.

    Inlined equivalent of repeatedly calling ``append_fast``: the per-step
    candidate segments (``_lp_step_data``) only ever carry ``ranks`` /
    ``decoded_tokens`` that are either ``None`` or plain lists (produced by
    ``extract_and_dedup_segment`` / ``extract_constrained_segment``), so the
    lazy-iterator ``islice`` branch of ``append_fast`` is unnecessary here.

    The six flat lists are pre-allocated to their final length and filled via
    slice assignment (a single C-level memcpy per segment) instead of repeated
    ``extend`` calls; ``ranks`` / ``decoded_tokens`` default to ``None`` slots
    and are only overwritten for segments that actually carry metadata.
    """
    # Walk the parent-pointer chain, collecting steps newest-first.  The
    # traversal doubles as the length accumulation: a plain loop replaces
    # ``sum(len(seg[0]) for seg in steps)`` (a generator frame per beam was
    # visible in the per-request tail profile).
    steps = []
    ancestor = beam
    step_data = getattr(ancestor, "_lp_step_data", None)
    tail_total = 0
    while step_data is not None:
        steps.append(step_data)
        tail_total += len(step_data[0])
        ancestor = ancestor._lp_parent
        step_data = getattr(ancestor, "_lp_step_data", None)

    add_sid_end = sid_end_token_id is not None and beam.finish_reason != "stop"
    init_len = len(init_token_ids)
    total = init_len + tail_total + (1 if add_sid_end else 0)

    # Pre-allocate the flat arrays at their final size. ranks/decoded_tokens
    # are filled with None (the common hot-path case where metadata is None);
    # only list-carrying segments overwrite their slice below.
    token_ids = [None] * total
    logprobs = [0.0] * total
    ranks = [None] * total
    decoded_tokens = [None] * total
    if init_len:
        token_ids[:init_len] = init_token_ids
        logprobs[:init_len] = init_logprobs
        ranks[:init_len] = init_ranks
        decoded_tokens[:init_len] = init_decoded
    start_indices = list(init_start)
    end_indices = list(init_end)

    # Replay steps oldest-first (steps were collected newest-first).
    pos = init_len
    for seg_token_ids, seg_logprobs, seg_ranks, seg_decoded in reversed(steps):
        n = len(seg_token_ids)
        start_indices.append(pos)
        token_ids[pos : pos + n] = seg_token_ids
        logprobs[pos : pos + n] = seg_logprobs
        if seg_ranks is not None:
            ranks[pos : pos + n] = seg_ranks
        if seg_decoded is not None:
            decoded_tokens[pos : pos + n] = seg_decoded
        pos += n
        end_indices.append(pos)

    if add_sid_end:
        start_indices.append(pos)
        token_ids[pos] = sid_end_token_id
        logprobs[pos] = 0.0
        pos += 1
        end_indices.append(pos)

    beam.logprobs = FlatLogprobs(
        start_indices=start_indices,
        end_indices=end_indices,
        token_ids=token_ids,
        logprobs=logprobs,
        ranks=ranks,
        decoded_tokens=decoded_tokens,
    )


def reconstruct_beam_logprobs_batch(
    beams,
    initial_logprobs,
    sid_end_token_id=None,
) -> None:
    """Batched variant of :func:`reconstruct_beam_logprobs`.

    Reconstructs full ``FlatLogprobs`` for every beam in *beams* in a single
    pass, writing the result back to ``beam.logprobs`` in place.  Produces
    results field-by-field identical to calling
    :func:`reconstruct_beam_logprobs` on each beam, but reads the shared
    read-only ``initial_logprobs`` fields once and inlines the per-segment
    ``append_fast`` logic to avoid hundreds of Python call frames,
    ``isinstance`` checks and ``len()`` calls on the hot tail path.
    """
    init_token_ids = initial_logprobs.token_ids
    init_logprobs = initial_logprobs.logprobs
    init_ranks = initial_logprobs.ranks
    init_decoded = initial_logprobs.decoded_tokens
    init_start = initial_logprobs.start_indices
    init_end = initial_logprobs.end_indices
    for beam in beams:
        _reconstruct_one(
            beam,
            init_token_ids,
            init_logprobs,
            init_ranks,
            init_decoded,
            init_start,
            init_end,
            sid_end_token_id,
        )


def reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id=None):
    """Walk parent-pointer chain to build full FlatLogprobs for a beam.

    Replaces ``beam.logprobs`` in-place with a newly constructed
    ``FlatLogprobs`` containing the initial logprobs plus all step data
    collected via ``_lp_parent`` / ``_lp_step_data`` pointers.  If
    ``sid_end_token_id`` is given and the beam did not finish with EOS,
    an extra entry is appended for the session-end token.

    Terminal worker-decision beams (worker-sinking Stage 0) carry no
    parent pointers and are built with ``logprobs=None`` by design (the
    settled degradation: cum_logprob only).  They are left untouched so
    the degradation stays visible instead of being silently replaced by
    a prompt-only FlatLogprobs that would misrepresent the beam's steps.

    Thin wrapper around :func:`reconstruct_beam_logprobs_batch` kept for
    backward compatibility with existing call sites and tests.
    """
    if getattr(beam, "_lp_step_data", None) is None and beam.logprobs is None:
        return
    reconstruct_beam_logprobs_batch([beam], initial_logprobs, sid_end_token_id)
