# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized FlatLogprobs.append_fast: uses extend instead of
per-element append."""

import itertools
from collections.abc import Iterable


def append_fast(
    self,
    token_ids: list[int],
    logprobs: list[float],
    ranks: "Iterable[int | None] | list[int | None]",
    decoded_tokens: "Iterable[str | None]",
) -> None:
    # V2 uses -1 as the structural padding token for rectangular batched
    # logprob tensors. Token IDs are otherwise non-negative, so trim the
    # padding before storing the position in FlatLogprobs.
    entry_count = next(
        (i for i, token_id in enumerate(token_ids) if token_id == -1),
        len(token_ids),
    )

    self.start_indices.append(len(self.logprobs))
    self.token_ids.extend(token_ids[:entry_count])
    self.logprobs.extend(logprobs[:entry_count])
    # ranks/decoded_tokens may be lazy or infinite iterators
    # (e.g. itertools.chain, itertools.repeat) -- use islice to bound them.
    bounded_ranks = list(itertools.islice(ranks, entry_count))
    bounded_decoded = list(itertools.islice(decoded_tokens, entry_count))
    self.ranks.extend(bounded_ranks)
    self.ranks.extend([None] * (entry_count - len(bounded_ranks)))
    self.decoded_tokens.extend(bounded_decoded)
    self.decoded_tokens.extend([None] * (entry_count - len(bounded_decoded)))
    self.end_indices.append(len(self.logprobs))


def extract_and_dedup_flat_logprobs(flat, logprobs_num):
    """Extract position-0 logprobs from FlatLogprobs, dedup sampled token.

    Flat arrays store [sampled, top1, ..., topK] — the only possible
    duplicate is position 0 (sampled token) when it also appears in the
    top-K (positions 1..K).  Returns exactly ``logprobs_num`` entries.
    """
    start = flat.start_indices[0]
    end = flat.end_indices[0]
    token_ids = flat.token_ids[start:end]
    lps = flat.logprobs[start:end]
    ranks = flat.ranks[start:end]
    decoded = flat.decoded_tokens[start:end]
    if len(token_ids) > logprobs_num:
        if any(token_ids[0] == token_ids[j] for j in range(1, logprobs_num + 1)):
            s = slice(1, logprobs_num + 1)
        else:
            s = slice(logprobs_num)
        token_ids, lps, ranks, decoded = (token_ids[s], lps[s], ranks[s], decoded[s])
    return token_ids, lps, ranks, decoded


def grouped_logprobs_stride(flat, beam_count: int, logprobs_num: int) -> int:
    """Validate and return the per-beam width of grouped flat logprobs.

    A grouped beam decode returns one contiguous top-K block for every
    active beam. Both the offline and OpenAI entrypoints consume that
    layout, so keep its contract in one place rather than letting one path
    turn malformed runner output into an indexing error.
    """
    if flat is None:
        raise ValueError(
            "Beam search generation expected non-empty logprobs for each beam, but received 'None'."
        )
    if beam_count <= 0:
        raise ValueError("Grouped beam search received an empty beam update.")

    total_logprobs = len(flat.token_ids)
    if len(flat.logprobs) != total_logprobs:
        raise ValueError(
            "Grouped beam search returned inconsistent logprobs tensors: "
            f"{total_logprobs} token ids but {len(flat.logprobs)} logprobs."
        )
    for name in ("ranks", "decoded_tokens"):
        values = getattr(flat, name, None)
        if values is not None and len(values) != total_logprobs:
            raise ValueError(
                "Grouped beam search returned inconsistent logprobs tensors: "
                f"{total_logprobs} token ids but {len(values)} {name}."
            )

    # Batched V2 sampling uses one rectangular tensor for requests with
    # different beam widths. The runner marks structural tail padding with
    # token ID -1; it is not part of any beam's top-K block.
    effective_logprobs = total_logprobs
    while effective_logprobs > 0 and flat.token_ids[effective_logprobs - 1] == -1:
        effective_logprobs -= 1

    if effective_logprobs % beam_count != 0:
        raise ValueError(
            "Grouped beam search returned an invalid logprobs shape: "
            f"received {effective_logprobs} entries for {beam_count} beams "
            f"({total_logprobs - effective_logprobs} structural padding entries). "
            "This usually means the active vLLM model runner was not "
            "patched for grouped beam requests."
        )

    stride_k = effective_logprobs // beam_count
    if stride_k < logprobs_num:
        raise ValueError(
            "Grouped beam search returned too few logprobs per beam: "
            f"received {stride_k}, expected at least {logprobs_num}."
        )
    return stride_k


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
