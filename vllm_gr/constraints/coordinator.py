# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared constraint orchestration for online and offline beam search."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_gr.constraints.models import ParentCandidateBuilder
from vllm_gr.constraints.strategies import (
    ConstraintTableStrategy,
    TrieConstraintStrategy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm_gr.constraints.strategies import ConstraintStrategy
    from vllm_gr.utils.catalog import Catalog


class BeamConstraintCoordinator:
    """Coordinate prefix lookup, candidate filtering, and parent mapping."""

    def __init__(self, strategy: ConstraintStrategy | None) -> None:
        self.strategy = strategy

    @classmethod
    def from_catalog(cls, catalog: Catalog | None) -> BeamConstraintCoordinator:
        return cls(TrieConstraintStrategy(catalog) if catalog is not None else None)

    @property
    def enabled(self) -> bool:
        return self.strategy is not None

    def resolve_catalog_strategy(self, value: str | None) -> str | None:
        """Resolve the frontend catalog-strategy for this coordinator's backend.

        - Trie: ``None`` means the historical default ``"fill"``; an explicit
          value passes through unchanged.
        - Constraint Tables: candidate validity is enforced by the Worker's
          raw-logits hook, so a frontend ``fill``/``null`` strategy is a
          contradiction and is rejected loudly; ``None`` (the new default)
          passes through.
        - No strategy: the value passes through untouched.
        """
        if isinstance(self.strategy, ConstraintTableStrategy):
            if value is not None:
                raise ValueError(
                    "catalog_strategy is not applicable to the constraint_table "
                    f"backend; got {value!r} (candidate validity is enforced in "
                    "the Worker)"
                )
            return None
        if isinstance(self.strategy, TrieConstraintStrategy):
            return "fill" if value is None else value
        return value

    def valid_next_token_ids(self, generated_token_ids: Sequence[int]) -> set[int] | None:
        if self.strategy is None:
            return None
        return self.strategy.valid_next_token_ids(generated_token_ids)

    def filter_candidates(
        self,
        *,
        token_ids: Sequence[int],
        logprobs: Sequence[float],
        ranks: Sequence[int | None] | None,
        decoded_tokens: Sequence[str | None] | None,
        valid_token_ids: set[int] | None,
        candidate_strategy: str,
    ) -> tuple[
        Sequence[int],
        Sequence[float],
        Sequence[int | None] | None,
        Sequence[str | None] | None,
    ]:
        if self.strategy is None or valid_token_ids is None:
            return token_ids, logprobs, ranks, decoded_tokens
        return self.strategy.filter_frontend_candidates(
            token_ids,
            logprobs,
            ranks,
            decoded_tokens,
            valid_token_ids,
            candidate_strategy,
        )

    def add_parent_candidates(
        self,
        builder: ParentCandidateBuilder,
        *,
        parent_index: int,
        parent_cumulative_logprob: float,
        token_ids: Sequence[int],
        logprobs: Sequence[float],
        ranks: Sequence[int | None] | None,
        decoded_tokens: Sequence[str | None] | None,
        valid_token_ids: set[int] | None,
        candidate_strategy: str,
    ) -> None:
        token_ids, logprobs, ranks, decoded_tokens = self.filter_candidates(
            token_ids=token_ids,
            logprobs=logprobs,
            ranks=ranks,
            decoded_tokens=decoded_tokens,
            valid_token_ids=valid_token_ids,
            candidate_strategy=candidate_strategy,
        )
        builder.add_parent(
            parent_index=parent_index,
            parent_cumulative_logprob=parent_cumulative_logprob,
            token_ids=token_ids,
            local_logprobs=logprobs,
            ranks=ranks,
            decoded_tokens=decoded_tokens,
        )
