# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Submit-once/wait-once frontend control flow for Beam Search V1."""

from __future__ import annotations

from typing import Any

from vllm_gr.entrypoints.beam_search_v1.protocol import (
    AsyncBeamSearchFrontend,
    BeamSearchFrontend,
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)


def run_beam_search_v1(
    frontend: BeamSearchFrontend,
    call: OfflineBeamSearchRequest,
) -> list[Any]:
    """Execute the permanent submit-once/wait-once V1 frontend flow."""
    request = frontend.prepare_request(call)
    session = frontend.submit_once(request)
    try:
        result = frontend.wait_final(session)
        return frontend.build_output(call, result)
    finally:
        frontend.close(session)


async def run_beam_search_v1_async(
    frontend: AsyncBeamSearchFrontend,
    call: OnlineBeamSearchRequest,
) -> list[Any]:
    """Execute the online submit-once/wait-once flow without blocking its loop."""
    request = await frontend.prepare_request(call)
    session = await frontend.submit_once(request)
    try:
        result = await frontend.wait_final(session)
        return frontend.build_output(call, result)
    finally:
        await frontend.close(session)


__all__ = ["run_beam_search_v1", "run_beam_search_v1_async"]
