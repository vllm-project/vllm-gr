# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Results for offline beam search."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import NDArray


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


@dataclass(frozen=True, slots=True, eq=False)
class FlatBeamSearchSequences(Sequence[CompactBeamSearchSequence]):
    """Read-only token and score columns for fixed-length Beam candidates.

    Numeric indexing and iteration materialize compact candidates on demand.
    """

    token_ids: NDArray[np.int32] | NDArray[np.int64]
    cumulative_logprobs: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.cumulative_logprobs)

    @overload
    def __getitem__(self, index: int) -> CompactBeamSearchSequence: ...

    @overload
    def __getitem__(self, index: slice) -> FlatBeamSearchSequences: ...

    def __getitem__(
        self, index: int | slice
    ) -> CompactBeamSearchSequence | FlatBeamSearchSequences:
        if isinstance(index, slice):
            return FlatBeamSearchSequences(self.token_ids[index], self.cumulative_logprobs[index])
        return CompactBeamSearchSequence(
            tuple(self.token_ids[index].tolist()), float(self.cumulative_logprobs[index]), "length"
        )

    def __iter__(self) -> Iterator[CompactBeamSearchSequence]:
        return (self[index] for index in range(len(self)))


@dataclass(frozen=True, slots=True, eq=False)
class FlatBeamSearchOutput:
    """An offline result with columns for eligible fixed-length candidates.

    A tuple can also represent short candidate lists, variable lengths, or explicit stops.
    """

    sequences: FlatBeamSearchSequences | tuple[CompactBeamSearchSequence, ...]
    prompt_token_ids: tuple[int, ...]

    def as_compact(self) -> CompactBeamSearchOutput:
        """Materialize the same candidates as the offline compact API."""
        sequences = self.sequences
        if isinstance(sequences, FlatBeamSearchSequences):
            sequences = tuple(
                CompactBeamSearchSequence(tuple(tokens), score, "length")
                for tokens, score in zip(
                    sequences.token_ids.tolist(),
                    sequences.cumulative_logprobs.tolist(),
                    strict=True,
                )
            )
        return CompactBeamSearchOutput(sequences, self.prompt_token_ids)
