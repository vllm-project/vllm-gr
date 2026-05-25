# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Request-level output types for generative recommendation.

- RecommendedItem: a single candidate with score metadata
- GRRequestOutput: full recommendation response, extends PoolingRequestOutput
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.outputs import PoolingOutput, PoolingRequestOutput


@dataclass
class RecommendedItem:
    """Single candidate item with score metadata."""

    item_id: str | int
    score: float
    features: dict[str, Any] | None = None
    task_scores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "item_id": self.item_id,
            "score": self.score,
            "features": self.features,
        }
        if self.task_scores is not None:
            result["task_scores"] = self.task_scores
        return result


class GRRequestOutput(PoolingRequestOutput):
    """Recommendation output extending PoolingRequestOutput.

    Adds recommendation-specific fields (user_id, candidate scores, raw
    scores) while remaining compatible with vLLM's output pipeline.
    """

    def __init__(
        self,
        request_id: str,
        user_id: Any = None,
        recommendations: list[RecommendedItem] | None = None,
        finished: bool = True,
        all_scores: torch.Tensor | None = None,
        prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int = 0,
    ):
        outputs = PoolingOutput(data=all_scores if all_scores is not None
                                else torch.tensor([]))
        super().__init__(
            request_id=request_id,
            outputs=outputs,
            prompt_token_ids=prompt_token_ids or [],
            num_cached_tokens=num_cached_tokens,
            finished=finished,
        )

        self.user_id = user_id
        self.recommendations = recommendations or []
        self.all_scores = all_scores

    def __repr__(self) -> str:
        return (
            f"GRRequestOutput(request_id={self.request_id!r}, "
            f"user_id={self.user_id!r}, "
            f"num_candidates={len(self.recommendations)}, "
            f"finished={self.finished})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "finished": self.finished,
        }

    @property
    def top_recommendation(self) -> RecommendedItem | None:
        if not self.recommendations:
            return None
        return max(self.recommendations, key=lambda item: item.score)

    @property
    def item_ids(self) -> list[str | int]:
        return [item.item_id for item in self.recommendations]

    @property
    def scores(self) -> list[float]:
        return [item.score for item in self.recommendations]
