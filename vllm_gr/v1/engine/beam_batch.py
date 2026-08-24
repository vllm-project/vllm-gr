# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the VLLM project

"""EngineCore-side Beam Batch ingress and final result router.

This module implements the shared Online/Offline entry boundary from issue #37.
It receives either a single native ``EngineCoreRequest`` or an already-formed
``BeamBatchRequest``, materializes the B native child ``Request`` objects and one
``BeamBatch`` host object, records the batch as one atomic unit, and resolves a
final-only ``BeamBatchHandle`` when a terminal ``BeamBatchResult`` arrives.

It does not allocate PromptKV, BeamKV, or RunSlot, and it does not implement
all-or-none Scheduler admission (issue #39).  The ``enqueue`` / ``cancel``
callbacks are the seams where the EngineCore scheduler boundary is injected.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Callable

from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchFailure,
    BeamBatchItem,
    BeamBatchLimits,
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputOptions,
    validate_beam_batch_request,
    validate_beam_batch_request_limits,
    validate_beam_batch_result,
)

_MAX_TERMINAL_TOMBSTONES = 4096


class BeamBatchState(str, Enum):
    """Lifecycle of one registered Beam Batch."""

    WAITING = "waiting"
    ACTIVE = "active"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"


class BeamBatchIngressError(RuntimeError):
    """Base error for EngineCore Beam Batch ingress failures."""


class BeamBatchDuplicateBatchError(BeamBatchIngressError):
    """Raised when a batch_id or terminal result is registered twice."""


class BeamBatchMixedModeError(BeamBatchIngressError):
    """Raised when ordinary Requests and Beam Batches are mixed."""


@dataclass
class BeamBatch:
    """EngineCore host object for one atomic Beam Batch.

    ``children`` are native vLLM ``Request`` objects, one per
    ``BeamBatchRequest.item``.  They carry independent prompt histories and are
    used for standard Prefill / Paged KV / Prefix Cache; they are not submitted
    to the public waiting queue as independent nodes.
    """

    request: BeamBatchRequest
    children: tuple[Request, ...]
    batch_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.batch_id = self.request.batch_id


@dataclass
class BeamBatchRecord:
    """Internal mutable state for one registered Beam Batch."""

    request: BeamBatchRequest
    future: Future[BeamBatchResult]
    batch: BeamBatch | None = None
    state: BeamBatchState = BeamBatchState.WAITING
    terminal: BeamBatchResult | None = None
    enqueued: bool = False


@dataclass(frozen=True)
class BeamBatchHandle:
    """Final-only handle returned to Online/Offline callers."""

    batch_id: str
    record: BeamBatchRecord

    def result(self, timeout: float | None = None) -> BeamBatchResult:
        """Block until the final :class:`BeamBatchResult` is available."""
        return self.record.future.result(timeout=timeout)

    def done(self) -> bool:
        return self.record.future.done()

    def cancelled(self) -> bool:
        return self.record.future.cancelled()


ChildBuilder = Callable[[BeamBatchRequest], BeamBatch]
EnqueueCallback = Callable[[BeamBatchRecord], None]
CancelCallback = Callable[[BeamBatchRecord], None]
BatchIdFactory = Callable[[], str]


def default_child_builder(request: BeamBatchRequest) -> BeamBatch:
    """Build a ``BeamBatch`` using native vLLM Request construction."""
    children = tuple(Request.from_engine_core_request(item.request, None) for item in request.items)
    return BeamBatch(request=request, children=children)


def build_beam_batch_request(
    requests: list[EngineCoreRequest] | tuple[EngineCoreRequest, ...],
    *,
    beam_params: BeamBatchParams,
    output_options: BeamOutputOptions,
    batch_id: str | None = None,
) -> BeamBatchRequest:
    """Build a ``BeamBatchRequest`` from native requests and batch parameters."""
    if not requests:
        raise ValueError("beam batch requires at least one request")
    resolved_batch_id = batch_id or f"beam-batch-{uuid.uuid4().hex}"
    items = tuple(
        BeamBatchItem(item_index=index, request=request) for index, request in enumerate(requests)
    )
    request = BeamBatchRequest(
        protocol_version=1,
        batch_id=resolved_batch_id,
        items=items,
        beam_params=beam_params,
        output_options=output_options,
    )
    validate_beam_batch_request(request)
    return request


class EngineCoreBeamBatchIngress:
    """Submit, cancel, and route final results for Beam Batches.

    This class is intentionally scheduler-free. It materializes native child
    Requests through ``child_builder``, registers one batch record, and exposes
    a final-only future. The ``enqueue`` and ``cancel`` callbacks are injection
    points for the EngineCore scheduler boundary.
    """

    def __init__(
        self,
        *,
        enqueue: EnqueueCallback | None = None,
        cancel: CancelCallback | None = None,
        child_builder: ChildBuilder | None = None,
        limits: BeamBatchLimits | None = None,
        max_num_seqs: int | None = None,
        batch_id_factory: BatchIdFactory | None = None,
        reject_mixed_requests: bool = False,
    ) -> None:
        self._enqueue = enqueue or (lambda record: None)
        self._cancel = cancel or (lambda record: None)
        self._child_builder = child_builder or default_child_builder
        self._limits = limits
        self._max_num_seqs = max_num_seqs
        self._batch_id_factory = batch_id_factory or (lambda: f"beam-batch-{uuid.uuid4().hex}")
        self._reject_mixed_requests = reject_mixed_requests
        self._beam_runtime_enabled = False
        self._lock = RLock()
        self._records_by_batch_id: dict[str, BeamBatchRecord] = {}
        self._terminal_tombstones: OrderedDict[str, BeamBatchState] = OrderedDict()
        self._ordinary_request_ids: set[str] = set()

    def _retire_record(self, record: BeamBatchRecord) -> None:
        """Move a terminal record out of the live registry into a bounded tombstone.

        Completed / cancelled records no longer need live lookups, but their
        batch_id must be remembered (bounded) so a duplicate terminal result
        still fails with the duplicate-completion error instead of "unknown
        batch".
        """
        with self._lock:
            self._records_by_batch_id.pop(record.request.batch_id, None)
            self._terminal_tombstones[record.request.batch_id] = record.state
            self._terminal_tombstones.move_to_end(record.request.batch_id)
            while len(self._terminal_tombstones) > _MAX_TERMINAL_TOMBSTONES:
                self._terminal_tombstones.popitem(last=False)

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(
                record.state in (BeamBatchState.WAITING, BeamBatchState.ACTIVE)
                for record in self._records_by_batch_id.values()
            )

    def set_beam_runtime_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._beam_runtime_enabled = enabled

    def note_ordinary_request(self, request_id: str) -> None:
        """Record an ordinary Request for mixed-mode enforcement."""
        with self._lock:
            if self._beam_runtime_enabled and self.active_count:
                raise BeamBatchMixedModeError(
                    "ordinary Request cannot be added while a Beam Batch is active"
                )
            self._ordinary_request_ids.add(request_id)

    def submit_single(
        self,
        request: EngineCoreRequest,
        *,
        beam_params: BeamBatchParams,
        output_options: BeamOutputOptions,
        batch_id: str | None = None,
    ) -> BeamBatchHandle:
        """Submit one native request as a single-item Beam Batch."""
        resolved_batch_id = batch_id or self._batch_id_factory()
        batch_request = build_beam_batch_request(
            [request],
            beam_params=beam_params,
            output_options=output_options,
            batch_id=resolved_batch_id,
        )
        return self.submit_batch(batch_request)

    def submit_batch(self, request: BeamBatchRequest) -> BeamBatchHandle:
        """Atomically validate, materialize, register, and enqueue one Beam Batch."""
        validate_beam_batch_request(request)
        if self._limits is not None:
            max_num_seqs = self._max_num_seqs or self._limits.max_num_seqs
            validate_beam_batch_request_limits(
                request,
                self._limits,
                max_num_seqs=max_num_seqs,
            )

        # Materialize all native child Requests before taking the registry lock.
        # This is the "build all host objects first" linearization point.
        try:
            batch = self._child_builder(request)
        except Exception as exc:
            raise BeamBatchIngressError(f"failed to build beam batch {request.batch_id!r}") from exc

        with self._lock:
            if self._beam_runtime_enabled and self._ordinary_request_ids:
                raise BeamBatchMixedModeError(
                    "Beam Batch cannot be added while ordinary Requests are present"
                )
            if request.batch_id in self._records_by_batch_id:
                raise BeamBatchDuplicateBatchError(
                    f"beam batch_id already exists: {request.batch_id!r}"
                )
            if request.batch_id in self._terminal_tombstones:
                raise BeamBatchDuplicateBatchError(
                    f"beam batch_id was already used and completed: {request.batch_id!r}"
                )
            future: Future[BeamBatchResult] = Future()
            record = BeamBatchRecord(
                request=request,
                future=future,
                batch=batch,
            )
            self._records_by_batch_id[request.batch_id] = record

        handle = BeamBatchHandle(
            batch_id=request.batch_id,
            record=record,
        )

        # Re-check under the lock right before enqueue so a concurrent cancel
        # that arrived after registration cannot lead to a cancelled batch
        # being enqueued (the enqueue seam is also required to honor this).
        with self._lock:
            if record.state != BeamBatchState.WAITING:
                return handle

        try:
            self._enqueue(record)
        except Exception as exc:
            with self._lock:
                self._records_by_batch_id.pop(request.batch_id, None)
                if record.state not in (
                    BeamBatchState.COMPLETE,
                    BeamBatchState.CANCELLED,
                ):
                    record.state = BeamBatchState.FAILED
            # A concurrent cancel may have already cancelled the future; do not
            # overwrite that terminal outcome with the enqueue failure.
            try:
                if not record.future.done():
                    record.future.set_exception(exc)
            except Exception:
                pass
            raise BeamBatchIngressError(
                f"failed to enqueue beam batch {request.batch_id!r}"
            ) from exc

        record.enqueued = True
        return handle

    def cancel(self, handle_or_batch_id: BeamBatchHandle | str) -> bool:
        """Cancel a waiting/active Beam Batch and forward cancellation."""
        batch_id = (
            handle_or_batch_id.batch_id
            if isinstance(handle_or_batch_id, BeamBatchHandle)
            else handle_or_batch_id
        )
        with self._lock:
            record = self._records_by_batch_id.get(batch_id)
            if record is None:
                return False
            if record.state in (BeamBatchState.COMPLETE, BeamBatchState.CANCELLED):
                return False
            record.state = BeamBatchState.CANCELLED

        try:
            self._cancel(record)
        finally:
            if not record.future.done():
                record.future.cancel()
            self._retire_record(record)
        return True

    def route_terminal_result(self, result: BeamBatchResult) -> BeamBatchHandle:
        """Route exactly one final result to its waiting caller."""
        validate_beam_batch_result(result)
        with self._lock:
            record = self._records_by_batch_id.get(result.batch_id)
            if record is None:
                state = self._terminal_tombstones.get(result.batch_id)
                if state == BeamBatchState.COMPLETE:
                    raise BeamBatchDuplicateBatchError(
                        f"beam batch {result.batch_id!r} already completed"
                    )
                raise BeamBatchIngressError(
                    f"terminal result for unknown batch_id {result.batch_id!r}"
                )
            if record.state in (BeamBatchState.CANCELLED, BeamBatchState.FAILED):
                raise BeamBatchIngressError(
                    f"beam batch {result.batch_id!r} is {record.state.value}; "
                    "a terminal result cannot be routed"
                )
            if record.terminal is not None:
                raise BeamBatchDuplicateBatchError(
                    f"beam batch {result.batch_id!r} already completed"
                )
            record.terminal = result
            record.state = BeamBatchState.COMPLETE
            record.future.set_result(result)
            self._retire_record(record)
            return BeamBatchHandle(
                batch_id=record.request.batch_id,
                record=record,
            )

    def fail_beam_batch(
        self,
        batch_id: str,
        *,
        code: str,
        message: str,
        item_index: int | None = None,
    ) -> BeamBatchHandle:
        """Route a whole-Batch failure result."""
        record = self.get_record(batch_id)
        if record is None:
            raise BeamBatchIngressError(f"unknown beam batch_id {batch_id!r}")
        result = BeamBatchResult(
            protocol_version=record.request.protocol_version,
            batch_id=batch_id,
            output=None,
            failure=BeamBatchFailure(
                code=code,
                message=message,
                item_index=item_index,
            ),
        )
        return self.route_terminal_result(result)

    def get_record(self, batch_id: str) -> BeamBatchRecord | None:
        with self._lock:
            return self._records_by_batch_id.get(batch_id)

    def get_batch(self, batch_id: str) -> BeamBatch | None:
        record = self.get_record(batch_id)
        return record.batch if record is not None else None
