# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Compact, eagerly available results for offline beam search."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompactBeamSearchSequence:
    """Generated tokens and the unchanged cumulative search score.

    Field names follow vLLM CompletionOutput.
    ``token_ids`` excludes the original prompt and includes configured
    begin/end markers, matching the generated suffix of a full beam output.
    Tokens and scores are ordinary CPU values; reading them performs no deferred
    work. No prompt, tokenizer, device tensor, or search history is retained.
    """

    token_ids: tuple[int, ...]
    cumulative_logprob: float
    finish_reason: str | None = None
    stop_reason: int | None = None


@dataclass(frozen=True, slots=True)
class CompactBeamSearchOutput:
    """Candidates in the same order as the full beam-search output.

    As in the full output, there can be fewer valid candidates than beam width.
    ``prompt_token_ids`` is the accepted input (without the configured begin
    token) and is shared by every candidate of the request.
    """

    sequences: tuple[CompactBeamSearchSequence, ...]
    # Accepted original input, once per request (not once per candidate).
    # None allows older callers constructing these records by hand.
    prompt_token_ids: tuple[int, ...] | None = None
