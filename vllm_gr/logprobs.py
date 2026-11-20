# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Optimized FlatLogprobs.append_fast: uses extend instead of
per-element append."""

import itertools
from collections.abc import Iterable

__all__ = [
    "append_fast",
    "patch_flat_logprobs",
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


def patch_flat_logprobs() -> None:
    from vllm.logprobs import FlatLogprobs

    FlatLogprobs.append_fast = append_fast
