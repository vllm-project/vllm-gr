# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patch: AsyncLLM.engine_beam_search method."""

from __future__ import annotations

from vllm_gr.v1.engine.types import BeamSearchRequest, BeamSearchResult


async def engine_beam_search(
    self, request: BeamSearchRequest
) -> BeamSearchResult:
    return await self.engine_core.beam_search_async(request)
