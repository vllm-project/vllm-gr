# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time

from pydantic import Field
from vllm.entrypoints.openai.protocol import OpenAIBaseModel
from vllm.pooling_params import PoolingParams
from vllm.utils import random_uuid

from vllm_gr.inputs.data import GRPrompt


class GRRecommendRequest(OpenAIBaseModel):
    model: str | None = None
    request_id: str = Field(default_factory=random_uuid)
    prompt: GRPrompt
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )

    def to_pooling_params(self) -> PoolingParams:
        return PoolingParams(task="embed")


class GRRecommendResult(OpenAIBaseModel):
    index: int
    item_id: str | int
    score: float
    task_scores: dict[str, float] | None = None


class GRRecommendResponse(OpenAIBaseModel):
    id: str
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    user_id: str | int | None = None
    recommendations: list[GRRecommendResult]
