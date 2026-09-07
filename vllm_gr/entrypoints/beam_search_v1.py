# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Server-side contracts and fixed control flow for Beam Search V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest, BeamBatchResult


@dataclass(frozen=True, slots=True)
class OfflineBeamSearchV1Call:
    """Arguments received by the offline V1 entry point."""

    prompts: list[Any]
    params: Any
    lora_request: Any = None
    use_tqdm: bool = False
    concurrency_limit: int | None = None


@dataclass(frozen=True, slots=True)
class OnlineBeamSearchV1Call:
    """Arguments received by the online V1 entry point."""

    prompt: Any
    request_id: str
    params: Any
    lora_request: Any = None
    trace_headers: Any = None


@dataclass(frozen=True, slots=True)
class BeamSearchV1Session:
    """Frontend handle for one submitted EngineCore Beam batch."""

    batch_id: str
    handle: object


class BeamSearchV1Server(Protocol):
    """Synchronous server boundary for EngineCore-owned Beam execution.

    The frontend submits one logical request and waits once. Prefill and all
    Decode scheduling happen behind ``submit_once``/``wait_final`` in EngineCore and the
    Worker; no per-step callback is exposed here.
    """

    def prepare_request(self, call: OfflineBeamSearchV1Call) -> BeamBatchRequest:
        """Validate and convert frontend arguments to the shared V1 request."""
        ...

    def submit_once(self, request: BeamBatchRequest) -> BeamSearchV1Session:
        """Submit one persistent request and return its frontend session handle."""
        ...

    def wait_final(self, session: BeamSearchV1Session) -> BeamBatchResult:
        """Wait for the terminal EngineCore result."""
        ...

    def build_output(self, call: OfflineBeamSearchV1Call, result: BeamBatchResult) -> list[Any]:
        """Convert the terminal result to frontend BeamSearchOutput objects."""
        ...

    def close(self, session: BeamSearchV1Session) -> None:
        """Release frontend session resources after success or failure."""
        ...


class AsyncBeamSearchV1Server(Protocol):
    """Asynchronous online adapter using the same EngineCore request type."""

    def prepare_request(self, call: OnlineBeamSearchV1Call) -> BeamBatchRequest:
        ...

    async def submit_once(self, request: BeamBatchRequest) -> BeamSearchV1Session:
        ...

    async def wait_final(self, session: BeamSearchV1Session) -> BeamBatchResult:
        ...

    def build_output(self, call: OnlineBeamSearchV1Call, result: BeamBatchResult) -> list[Any]:
        ...

    async def close(self, session: BeamSearchV1Session) -> None:
        ...


def run_beam_search_v1(
    server: BeamSearchV1Server,
    call: OfflineBeamSearchV1Call,
) -> list[Any]:
    """Execute the permanent submit-once/wait-once V1 frontend flow."""
    request = server.prepare_request(call)
    session = server.submit_once(request)
    try:
        result = server.wait_final(session)
        return server.build_output(call, result)
    finally:
        server.close(session)


async def run_beam_search_v1_async(
    server: AsyncBeamSearchV1Server,
    call: OnlineBeamSearchV1Call,
) -> list[Any]:
    """Execute the online submit-once/wait-once flow without blocking its loop."""
    request = server.prepare_request(call)
    session = await server.submit_once(request)
    try:
        result = await server.wait_final(session)
        return server.build_output(call, result)
    finally:
        await server.close(session)


__all__ = [
    "AsyncBeamSearchV1Server",
    "BeamSearchV1Server",
    "BeamSearchV1Session",
    "OfflineBeamSearchV1Call",
    "OnlineBeamSearchV1Call",
    "run_beam_search_v1",
    "run_beam_search_v1_async",
]
