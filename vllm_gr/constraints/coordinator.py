# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resource-free frontend bridge retained until the full Beam flow takes over."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_gr.constraints.models import ParentCandidateBuilder
from vllm_gr.constraints.strategies import ConstraintTableStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence


class BeamConstraintCoordinator:
    """Carry only whether the current service/request enables constraints."""

    __slots__ = ("strategy",)

    def __init__(self, enabled: bool = False) -> None:
        self.strategy = ConstraintTableStrategy() if enabled else None

    @property
    def enabled(self) -> bool:
        return self.strategy is not None

    def resolve_catalog_strategy(self, value: str | None) -> None:
        if self.enabled and value is not None:
            raise ValueError(
                f"catalog_strategy is not applicable to constraint_table; got {value!r}"
            )
        return None

    def valid_next_token_ids(self, generated_token_ids: Sequence[int]) -> None:
        del generated_token_ids
        if self.enabled:
            raise RuntimeError("constraint candidates are selected by the Worker Backend")
        return None

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
        candidate_strategy: str | None,
    ) -> None:
        del valid_token_ids, candidate_strategy
        builder.add_parent(
            parent_index=parent_index,
            parent_cumulative_logprob=parent_cumulative_logprob,
            token_ids=token_ids,
            local_logprobs=logprobs,
            ranks=ranks,
            decoded_tokens=decoded_tokens,
        )
