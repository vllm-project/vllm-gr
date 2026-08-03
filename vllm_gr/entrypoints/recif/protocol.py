# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request and response models for RECIF online beam search."""

from pydantic import BaseModel, Field


class RecifBeamSearchRequest(BaseModel):
    history_sids: list[int]
    beam_width: int = Field(default=64, gt=0)
    branch_factors: tuple[int, int, int] = (8, 8, 8)


class RecifBeamResult(BaseModel):
    rank: int
    sid: int
    cumulative_logprob: float


class RecifBeamSearchResponse(BaseModel):
    id: str
    object: str = "recif.beam_search"
    results: list[RecifBeamResult]


__all__ = [
    "RecifBeamResult",
    "RecifBeamSearchRequest",
    "RecifBeamSearchResponse",
]
