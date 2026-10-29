# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Frontend requests, session handles, and transport contracts for beam search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest, BeamBatchResult


@dataclass(frozen=True, slots=True)
class OfflineBeamSearchRequest:
    """Offline arguments for one caller-formed, fixed-membership batch.

    Submit-once means exactly once for the complete batch session. The runtime
    preserves input order and never adds, removes, or dynamically pairs items.
    """

    prompts: list[Any]
    params: Any
    lora_request: Any = None
    use_tqdm: bool = False
    concurrency_limit: int | None = None
    output_mode: Literal["full", "compact"] = "compact"

    def __post_init__(self) -> None:
        if self.output_mode not in ("full", "compact"):
            raise ValueError("output_mode must be full or compact")


@dataclass(frozen=True, slots=True)
class OnlineBeamSearchRequest:
    """Arguments received by the online V1 entry point.

    ``prompts`` is a sequence for the same reason the offline request carries
    one: a caller may submit k prompts as a single logical batch, and the whole
    sequence is submitted once and runs as one forward.
    """

    prompts: tuple[Any, ...]
    request_id: str
    params: Any
    lora_request: Any = None
    trace_headers: Any = None
    priority: int = 0
    data_parallel_rank: int | None = None

    # Native caller affinity hint, distinct from the internal Beam batch/session ID.
    session_id: str | None = None

    def __post_init__(self) -> None:
        # The field is a sequence on the wire; pinning the concrete type here
        # keeps a caller's list from being mutated under a running session.
        object.__setattr__(self, "prompts", tuple(self.prompts))


@dataclass(slots=True)
class BeamSearchSession:
    """Frontend handle for one submitted EngineCore Beam batch."""

    batch_id: str
    session_id: str
    handle: object
    data_parallel_rank: int | None = None
    terminal: bool = False
    failed: bool = False
    closed: bool = False


class BeamSearchFrontend(Protocol):
    """Synchronous frontend boundary for EngineCore-owned Beam execution.

    The frontend submits one logical request and waits once. Prefill and all
    Decode scheduling happen behind ``submit_once``/``wait_final`` in EngineCore and the
    Worker; no per-step callback is exposed here.
    """

    def prepare_request(self, call: OfflineBeamSearchRequest) -> BeamBatchRequest:
        """Validate and convert frontend arguments to the shared V1 request."""
        ...

    def submit_once(self, request: BeamBatchRequest) -> BeamSearchSession:
        """Submit once and transfer cleanup ownership with the returned handle.

        A2 must own cleanup by request.batch_id before sending. If submission
        fails after acceptance, roll it back or hand it to reliable cleanup;
        the caller has no session handle until this method returns.
        """
        ...

    def wait_final(self, session: BeamSearchSession) -> BeamBatchResult:
        """Wait for the terminal EngineCore result."""
        ...

    def build_output(
        self,
        call: OfflineBeamSearchRequest,
        request: BeamBatchRequest,
        result: BeamBatchResult,
    ) -> list[Any]:
        """Convert the result using the exact request accepted by EngineCore."""
        ...

    def close(self, session: BeamSearchSession) -> None:
        """Idempotently release a terminal session or abort an unfinished one.

        A2 must request EngineCore abort on abnormal exit, not just discard
        the frontend handle. Device resources stay alive until work is drained.
        Synchronous cleanup retries on the calling thread until acknowledged.
        """
        ...


class AsyncBeamSearchFrontend(Protocol):
    """Asynchronous online adapter using the same EngineCore request type."""

    async def prepare_request(self, call: OnlineBeamSearchRequest) -> BeamBatchRequest: ...

    async def submit_once(self, request: BeamBatchRequest) -> BeamSearchSession:
        """Same ownership contract as sync submit, including cancellation.

        A2 must roll back or retain reliable cleanup ownership if cancelled
        after acceptance but before returning, using the preexisting batch ID.
        Cleanup must survive repeated cancellation and fence delayed acceptance.
        Failed rollback remains owned and is retried until abort is acknowledged.
        """
        ...

    async def wait_final(self, session: BeamSearchSession) -> BeamBatchResult: ...

    def build_output(
        self,
        call: OnlineBeamSearchRequest,
        request: BeamBatchRequest,
        result: BeamBatchResult,
    ) -> list[Any]: ...

    async def close(self, session: BeamSearchSession) -> None:
        """Retain abort/release retries through errors and caller cancellation.

        The first cleanup error may return while the retained owner retries.
        Mark the session closed only after acknowledgement.
        """
        ...
