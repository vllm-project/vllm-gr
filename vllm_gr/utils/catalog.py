# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Catalog-constraint utilities for beam search."""

import json
from collections.abc import Sequence
from typing import Any

from vllm.tokenizers import TokenizerLike

from vllm_gr.logger import init_logger

logger = init_logger(__name__)


class Catalog:
    # Sentinel key used to mark nodes that are valid terminal completions
    # (end of a catalog sequence). Token IDs are integers, so a string key
    # cannot collide with actual trie entries.
    _TERMINAL_MARKER = "__catalog_terminal__"

    def __init__(self, tokenizer: TokenizerLike):
        self.tokenizer = tokenizer
        self.trie: dict[int, Any] = {}

    def load(self, path: str):
        try:
            with open(path) as f:
                catalog_data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Catalog file not found: {path}")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from catalog file: {path}")

        for seq_idx, seq in enumerate(catalog_data):
            token_ids = self.tokenizer.convert_tokens_to_ids(seq)
            node = self.trie
            for i, (token_str, token_id) in enumerate(zip(seq, token_ids)):
                if token_id is None or token_id == -1:
                    raise ValueError(
                        f"Failed to map token {token_str!r} to a valid token ID using the tokenizer "
                        f"(got {token_id!r}). Token occurs at index {i} of sequence #{seq_idx} in catalog {path!r}. "
                        "Please verify that special/semantic tokens are registered in the model's tokenizer."
                    )
                if token_id not in node:
                    node[token_id] = {}
                node = node[token_id]
            node[self._TERMINAL_MARKER] = True

    def is_terminal(self, token_ids: list[int]) -> bool:
        """Returns True if token_ids exactly matches a complete catalog item."""
        node = self.trie
        for tid in token_ids:
            if tid in node:
                node = node[tid]
            else:
                return False
        return self._TERMINAL_MARKER in node

    def valid(self, token_ids: Sequence[int]) -> set[int]:
        node = self.trie
        for tid in token_ids:
            if tid in node:
                node = node[tid]
            else:
                return set()
        return {k for k in node if k != self._TERMINAL_MARKER}

    def apply_catalog_filter(
        self,
        token_ids_pos: list[int],
        logprobs_pos: list[float],
        ranks_pos: list[int | None] | None,
        decoded_pos: list[str | None] | None,
        valid_tokens_set: set[int],
        catalog_strategy: str,
    ) -> tuple[list[int], list[float], list[int | None] | None, list[str | None] | None]:
        """Apply catalog constraint to a single beam's candidate window.

        This function filters and potentially modifies a beam's candidate tokens to
        satisfy catalog constraints, preventing invalid continuations in the output.

        Args:
            token_ids_pos: Token IDs in the candidate window. Length N.
            logprobs_pos: Log probabilities for each token. Length N, must align with
                         token_ids_pos.
            ranks_pos: Optional per-token ranking metadata (or None if not tracked).
            decoded_pos: Optional per-token decoded string metadata (or None if not tracked).
            valid_tokens_set: Set of token IDs that are valid continuations per the
                             catalog constraint. Empty set means beam is invalid.
            catalog_strategy: Either "fill" or "null":
                - "fill": Keep catalog-valid entries with their original logprobs;
                         replace invalid entries with a catalog-valid token and
                         logprob of -10000. ranks/decoded are set to None for
                         replaced positions to prevent metadata misattribution to
                         injected tokens.
                - "null": Mask invalid tokens to -inf (dropped by finite-only top-K);
                         if all entries are masked, eliminate the beam entirely
                         and return empty lists.

        Returns:
            Tuple of (token_ids, logprobs, ranks, decoded) with length invariants:
            - token_ids and logprobs always have equal length (may be 0 if beam eliminated)
            - ranks is None iff input ranks_pos was None
            - decoded is None iff input decoded_pos was None
        """
        if not valid_tokens_set:
            return (
                [],
                [],
                ([] if ranks_pos is not None else None),
                ([] if decoded_pos is not None else None),
            )

        if catalog_strategy == "null":
            n = len(logprobs_pos)
            filtered_tids = token_ids_pos
            filtered_lps = logprobs_pos[:]
            all_inf = True
            for i in range(n):
                if token_ids_pos[i] not in valid_tokens_set:
                    filtered_lps[i] = -float("inf")
                else:
                    all_inf = False
            if all_inf:
                return (
                    [],
                    [],
                    ([] if ranks_pos is not None else None),
                    ([] if decoded_pos is not None else None),
                )
            return filtered_tids, filtered_lps, ranks_pos, decoded_pos

        valid_list = sorted(valid_tokens_set)
        n_valid = len(valid_list)
        n = len(token_ids_pos)
        new_tids = token_ids_pos[:]
        new_lps = logprobs_pos[:]
        new_ranks: list | None = None
        new_decoded: list | None = None
        if ranks_pos is not None:
            new_ranks = [ranks_pos[i] if i < len(ranks_pos) else None for i in range(n)]
        if decoded_pos is not None:
            new_decoded = [decoded_pos[i] if i < len(decoded_pos) else None for i in range(n)]
        fill_used = 0
        for j, tid in enumerate(token_ids_pos):
            if tid not in valid_tokens_set:
                ft = valid_list[fill_used % n_valid]
                new_tids[j] = ft
                new_lps[j] = -10000.0 - (fill_used % n_valid) * 0.1
                fill_used += 1
                if new_ranks is not None:
                    new_ranks[j] = None
                if new_decoded is not None:
                    new_decoded[j] = None
        return new_tids, new_lps, new_ranks, new_decoded
