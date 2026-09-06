# SPDX-License-Identifier: Apache-2.0
"""CUDA-private ownership for two fixed control-output storage slots.

The runner, not a GR request, must retain this pool. Reserve a slot and call
begin_device_use BEFORE writing its device tensor. Registration follows the
last copy on the notification stream. Only successful completion AND Host
consumption/discard allow reuse. No CUDA API, copy, polling thread or second
Future queue is implemented here; the GPU V1 adapter supplies those operations.
This boundary is not yet installed in ModelRunner.
"""

import threading
from dataclasses import dataclass
from typing import Any, Callable

from vllm_gr.v1.worker.cuda_completion import (
    CompletionDeviceError,
    CompletionRegistrationError,
    CompletionTicket,
    NativeCompletionAPI,
)


class ControlSlotBusy(RuntimeError):
    """Admission did not acquire storage and must not write it."""


@dataclass(frozen=True, slots=True)
class ControlLease:
    slot: int
    generation: int
    request_id: str
    session_epoch: int
    device_step: int


@dataclass(slots=True)
class _Entry:
    lease: ControlLease
    ticket: CompletionTicket
    device_started: bool = False
    producer_active: bool = False
    registered: bool = False
    reader_active: bool = False
    host_done: bool = False


class ControlSlotPool:
    """Bounded, nonblocking admission; busy slots never wait for EngineCore.

    Slot selection is explicit so an adapter can preserve the two fixed
    Tensor identities required by GRModelRunnerState. A new request/session
    does not reset ownership. Generation guards also reject late Host acks.
    An unsafe notification error quarantines the entire pool until shutdown
    successfully drains the device; cancelling delivery is not a drain.
    """

    def __init__(self, api: NativeCompletionAPI, resources: tuple[Any, Any]):
        if len(resources) != 2 or resources[0] is resources[1]:
            raise ValueError("control pool requires two distinct fixed storage slots")
        self._api = api
        self._resources = tuple(resources)
        self._entries: list[_Entry | None] = [None, None]
        self._lock = threading.RLock()
        self._generation = 0
        self._poison: BaseException | None = None
        self._closed = False
        self._drained = False

    def _check_live(self):
        if self._poison is not None:
            raise RuntimeError("control pool quarantined; device drain is required") from self._poison
        if self._closed:
            raise RuntimeError("control pool is closed")

    def _entry(self, lease: ControlLease) -> _Entry:
        if not isinstance(lease, ControlLease) or lease.slot not in (0, 1):
            raise RuntimeError("invalid control lease")
        entry = self._entries[lease.slot]
        # Identity rejects forged/cross-pool tokens even when fields match.
        if entry is None or entry.lease is not lease:
            raise RuntimeError("stale or foreign control lease")
        return entry

    def _reap(self):
        for slot, entry in enumerate(self._entries):
            if entry is None or not entry.registered:
                continue
            snapshot = entry.ticket.snapshot()
            if snapshot.registration_failed or (snapshot.device_done and snapshot.device_status):
                self._poison = CompletionDeviceError("control completion did not prove safe device drain")
                self._check_live()
            if snapshot.device_succeeded and entry.host_done and not entry.reader_active:
                entry.ticket.close()
                self._entries[slot] = None

    def acquire(self, slot: int, request_id: str, session_epoch: int, device_step: int) -> ControlLease:
        if type(slot) is not int or slot not in (0, 1):
            raise ValueError("control slot must be 0 or 1")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("control lease requires a request ID")
        if type(session_epoch) is not int or session_epoch < 0:
            raise ValueError("control lease requires a nonnegative session epoch")
        if type(device_step) is not int or device_step < 1:
            raise ValueError("control lease requires a positive device step")
        with self._lock:
            self._check_live()
            self._reap()
            if self._entries[slot] is not None:
                raise ControlSlotBusy("control slot is busy; device writes are forbidden")
            generation = self._generation + 1
            ticket = self._api.new_ticket(generation)
            lease = ControlLease(slot, generation, request_id, session_epoch, device_step)
            self._entries[slot] = _Entry(lease, ticket)
            self._generation = generation
            return lease

    def begin_device_use(self, lease: ControlLease) -> Any:
        """Return retained storage once, before the first device fill/copy."""
        with self._lock:
            self._check_live()
            entry = self._entry(lease)
            if entry.device_started or entry.host_done:
                raise RuntimeError("control device use already started or discarded")
            entry.device_started = True
            entry.producer_active = True
            return self._resources[lease.slot]

    def register_completion(self, lease: ControlLease, stream_pointer: int):
        with self._lock:
            entry = self._entry(lease)
            if not entry.device_started or entry.registered:
                raise RuntimeError("control completion registration is out of order")
            try:
                # Even a quarantine raised by another slot must close this
                # producer's submission phase, so a terminal drain can run.
                self._check_live()
                entry.ticket.register(stream_pointer)
            except BaseException as error:
                self.quarantine(error)
                raise
            finally:
                entry.producer_active = False
            entry.registered = True

    def fail_submission(self, lease: ControlLease, error: BaseException):
        """Producer has stopped enqueueing after failure; retain all storage."""
        with self._lock:
            entry = self._entry(lease)
            if not entry.producer_active:
                raise RuntimeError("control submission is not active")
            entry.producer_active = False
            self.quarantine(error)

    def consume(self, lease: ControlLease, read_host: Callable[[Any], Any], *, timeout_ms: int | None):
        """Read actual Host storage after notification; ack only after read.

        A reader excludes discard and shutdown. A timeout/reader exception
        releases that exclusion but never acknowledges or frees the storage.
        The caller can then retry or explicitly discard the delivery.
        read_host must return a detached result, never a view/alias of the
        reusable storage. The adapter must validate storage non-aliasing at
        pool construction; Python object identity alone cannot establish it.
        """
        with self._lock:
            self._check_live()
            entry = self._entry(lease)
            if not entry.registered or entry.reader_active or entry.host_done:
                raise RuntimeError("control delivery is not available for consumption")
            entry.reader_active = True
        failed = False
        try:
            try:
                ready = entry.ticket.wait_delivery(timeout_ms)
            except (CompletionDeviceError, CompletionRegistrationError) as error:
                self.quarantine(error)
                raise
            if not ready:
                raise TimeoutError("control delivery remains in flight")
            result = read_host(self._resources[lease.slot])
            with self._lock:
                entry.host_done = True
            return result
        except BaseException:
            failed = True
            raise
        finally:
            with self._lock:
                entry.reader_active = False
                # Preserve the original reader/device exception on failure.
                if self._poison is None:
                    try:
                        self._reap()
                    except RuntimeError:
                        if not failed:
                            raise

    def discard(self, lease: ControlLease) -> bool:
        """Cancel Host delivery only; never return an in-flight DMA slot.

        False means a reader owns the delivery or it was already discarded.
        A stale/foreign lease is an error, not permission to affect a new one.
        """
        with self._lock:
            entry = self._entry(lease)
            if entry.reader_active or entry.host_done:
                return False
            entry.host_done = True
            entry.ticket.cancel_delivery()
            if not entry.device_started:
                entry.ticket.close()
                self._entries[lease.slot] = None
            elif self._poison is None:
                self._reap()
            return True

    def quarantine(self, error: BaseException):
        """Prevent further writes after failed enqueue/notification/drain."""
        with self._lock:
            if self._poison is None:
                self._poison = error

    @property
    def occupied_slots(self) -> int:
        with self._lock:
            return sum(entry is not None for entry in self._entries)

    def shutdown(self, drain_device: Callable[[], None]):
        """Terminal release only AFTER backend drain succeeds, never on error.

        The caller must supply a real backend drain of all streams touching
        these resources. It must retain this pool if that drain raises. The
        pool deliberately provides no force-free/timeout-free alternative.
        """
        with self._lock:
            if self._drained:
                return False
            if any(entry is not None and entry.reader_active for entry in self._entries):
                raise RuntimeError("cannot shut down control storage with an active Host reader")
            if any(entry is not None and entry.producer_active for entry in self._entries):
                raise RuntimeError("cannot shut down control storage with an active device producer")
            self._closed = True
            try:
                drain_device()
            except BaseException as error:
                self.quarantine(error)
                raise
            for entry in self._entries:
                if entry is not None:
                    entry.ticket.close()
            self._entries = [None, None]
            self._resources = ()
            self._drained = True
            return True
