# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A1 contracts for EngineCore-owned asynchronous Beam Search V1 execution.

This module defines state and extension points only. A2-A5 provide the
request, scheduler, executor, Worker-state, and terminal-result implementations.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest, BeamBatchResult


class GRStageKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class GRSessionStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHING = "finishing"
    FINISHED = "finished"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass(slots=True)
class GRSessionState:
    """EngineCore scheduling state for one B=1 GR session.

    ``issued_stages`` advances when a Prefill/Decode stage is dispatched.
    ``completed_stages`` advances when its Future is consumed. The difference
    is the number of in-flight stages. ``output_placeholders`` reserves results
    that have been scheduled but are not yet confirmed on the host.
    """

    session_id: str
    request: BeamBatchRequest
    max_decode_steps: int
    status: GRSessionStatus = GRSessionStatus.WAITING
    issued_stages: int = 0
    completed_stages: int = 0
    output_placeholders: int = 0
    terminal_result_pending: bool = False

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if len(self.request.items) != 1:
            raise ValueError("Beam Search V1 currently requires batch size B=1")
        if self.max_decode_steps < 0:
            raise ValueError("max_decode_steps must be non-negative")
        if not 0 <= self.completed_stages <= self.issued_stages <= self.total_stages:
            raise ValueError(
                "stage counters must satisfy 0 <= completed <= issued <= total"
            )
        if self.output_placeholders < 0:
            raise ValueError("output_placeholders must be non-negative")

    @property
    def total_stages(self) -> int:
        return 1 + self.max_decode_steps

    @property
    def in_flight_stages(self) -> int:
        return self.issued_stages - self.completed_stages

    @property
    def next_stage_kind(self) -> GRStageKind | None:
        """Return the next logical stage; stage 0 is the only Prefill."""
        if self.issued_stages >= self.total_stages:
            return None
        if self.issued_stages == 0:
            return GRStageKind.PREFILL
        return GRStageKind.DECODE


@dataclass(frozen=True, slots=True)
class GRStageCommand:
    """One EngineCore-to-Worker dispatch for a B=1 session.

    Stage 0 is Prefill. Stages 1..N are Decode steps 0..N-1. A3 must issue
    each stage exactly once; A4 keeps the dependency data on the device.
    """

    session_id: str
    stage_index: int
    kind: GRStageKind
    decode_step: int | None
    is_last_stage: bool
    scheduler_output: object


@dataclass(frozen=True, slots=True)
class GRStageCompletion:
    """One Worker stage result consumed by EngineCore/Scheduler."""

    session_id: str
    stage_index: int
    finished: bool
    model_output: object | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GRBatchQueueEntry:
    """A non-blocking Worker Future paired with its scheduling command."""

    command: GRStageCommand
    future: Future[Any]


class GRAsyncSchedulerInterface(Protocol):
    """GR additions implemented by A2/A3 on top of vLLM SchedulerInterface."""

    def add_gr_session(self, request: BeamBatchRequest) -> GRSessionState:
        """Create one persistent scheduler state and add it once."""
        ...

    def can_issue_gr_stage(self, session: GRSessionState) -> bool:
        """Whether another stage may be scheduled without consuming a Future.

        A3 implements the bounded look-ahead rule with output placeholders.
        This is what permits Prefill-to-Decode and Decode-to-Decode overlap.
        """
        ...

    def schedule_gr_stage(self, session: GRSessionState) -> GRStageCommand | None:
        """Schedule exactly one Prefill or Decode stage."""
        ...

    def update_gr_from_output(
        self,
        session: GRSessionState,
        command: GRStageCommand,
        model_output: object,
    ) -> GRStageCompletion:
        """Consume one completed Future and update placeholders/session state."""
        ...

    def finish_gr_session(
        self,
        session: GRSessionState,
        status: GRSessionStatus,
    ) -> BeamBatchResult | None:
        """Finish or abort a session and release Scheduler-owned resources."""
        ...


class GRAsyncBatchQueueInterface(Protocol):
    """Queue used to overlap CPU scheduling with non-blocking device work."""

    def has_capacity(self) -> bool:
        ...

    def push(self, entry: GRBatchQueueEntry) -> None:
        ...

    def pop_oldest(self, *, block: bool) -> GRBatchQueueEntry | None:
        ...


class GRModelExecutorInterface(Protocol):
    """EngineCore-to-Worker stage dispatch implemented by A3/A4."""

    def execute_gr_stage(
        self,
        command: GRStageCommand,
        *,
        non_block: bool,
    ) -> Future[Any]:
        """Dispatch one stage; asynchronous execution returns a Future."""
        ...


class GREngineCoreInterface(Protocol):
    """Shared EngineCore entry used by both online and offline frontends."""

    def submit_gr_session(self, request: BeamBatchRequest) -> str:
        """Enqueue one frontend submission and return its session ID.

        Both online and offline adapters call this same entry exactly once.
        """
        ...

    def step_gr_with_batch_queue(self) -> list[BeamBatchResult]:
        """Advance all GR sessions by one non-blocking EngineCore iteration.

        The implementation drains new sessions, fills available queue slots
        with stage Futures, then consumes the oldest ready Future. Terminal
        results are returned while other sessions continue using the device.
        """
        ...

    def issue_gr_stage(self, session_id: str) -> GRBatchQueueEntry | None:
        """Schedule and dispatch one stage without waiting for its result."""
        ...

    def complete_gr_stage(self, entry: GRBatchQueueEntry) -> GRStageCompletion:
        """Consume one completed Worker Future."""
        ...

    def finish_gr_session(self, session_id: str) -> BeamBatchResult:
        """Return the terminal result and release EngineCore session state."""
        ...

    def abort_gr_session(self, session_id: str) -> None:
        ...


__all__ = [
    "GRAsyncBatchQueueInterface",
    "GRAsyncSchedulerInterface",
    "GRBatchQueueEntry",
    "GREngineCoreInterface",
    "GRModelExecutorInterface",
    "GRSessionState",
    "GRSessionStatus",
    "GRStageCommand",
    "GRStageCompletion",
    "GRStageKind",
]
