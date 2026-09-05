# SPDX-License-Identifier: Apache-2.0
"""CUDA-private native completion handles; no model/stream work at import.

This module does not enqueue copies or own device/pinned buffers. A backend
must retain those resources until successful device completion AND delivery
consumption/discard. Closing a ticket only drops its native owner reference;
it is never permission to reuse DMA storage. No ModelRunner path uses this
module until the output adapter and its resource leases are installed.
"""

from contextlib import contextmanager
from concurrent.futures import CancelledError
import ctypes
from dataclasses import dataclass
from importlib import import_module
import threading


API_VERSION = 2
_loaded_api = None
_load_lock = threading.Lock()


class CompletionRegistrationError(RuntimeError):
    """Registration failed; earlier device work can still be in flight."""


class CompletionDeviceError(RuntimeError):
    """Notification arrived with a device error, not safe buffer completion."""


@dataclass(frozen=True, slots=True)
class CompletionSnapshot:
    generation: int
    device_status: int
    device_done: bool
    delivery_cancelled: bool
    registration_failed: bool
    # Native wait has claimed the result; it does NOT prove the caller has
    # finished reading a pinned payload. A storage lease needs its own ack.
    delivery_claimed: bool

    @property
    def device_succeeded(self) -> bool:
        return self.device_done and self.device_status == 0 and not self.registration_failed


def _unsigned(value, bits, name):
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer")
    return value


class NativeCompletionAPI:
    """Keep both native code and the PyTorch-linked CUDA runtime alive."""

    def __init__(self, library, runtime):
        self.lib = library
        self.runtime = runtime
        signatures = {
            "gr_ticket_api_version": ([], ctypes.c_int),
            "gr_ticket_create": ([ctypes.c_uint64], ctypes.c_void_p),
            "gr_ticket_retain": ([ctypes.c_void_p], None),
            "gr_ticket_release": ([ctypes.c_void_p], None),
            "gr_ticket_live": ([], ctypes.c_int),
            "gr_ticket_register": ([ctypes.c_void_p] * 3, ctypes.c_int),
            "gr_ticket_cancel_delivery": ([ctypes.c_void_p], None),
            "gr_ticket_wait": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
            "gr_ticket_snapshot": (
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int)],
                ctypes.c_int,
            ),
        }
        try:
            for name, (arguments, result) in signatures.items():
                function = getattr(library, name)
                function.argtypes, function.restype = arguments, result
            self.register_address = ctypes.cast(runtime.cudaStreamAddCallback, ctypes.c_void_p)
        except AttributeError as error:
            raise RuntimeError("native GR completion ABI/CUDA callback capability is missing") from error
        if library.gr_ticket_api_version() != API_VERSION:
            raise RuntimeError("native GR completion ABI version mismatch; rebuild the CUDA package")

    def new_ticket(self, generation: int):
        return CompletionTicket(self, generation)


def load_native_completion_api() -> NativeCompletionAPI:
    """Load once, explicitly; never compile at Worker startup or fall back.

    Resolve libcudart symbols through torch._C's dependency handle instead of
    searching arbitrary system/wheel paths for a second CUDA runtime. Keep the
    singleton for process lifetime, including after a ticket owner is closed
    but its native callback has not returned. CPU/NPU paths must not call this.
    """
    global _loaded_api
    with _load_lock:
        if _loaded_api is not None:
            return _loaded_api
        torch = import_module("torch")
        if torch.version.cuda is None:
            raise RuntimeError("native GR CUDA completion requires a CUDA-enabled PyTorch build")
        try:
            extension = import_module("vllm_gr.ops.cuda._completion")
        except ImportError as error:
            raise RuntimeError(
                "native GR completion is not installed; build/install the CUDA vllm-gr package"
            ) from error
        if type(getattr(extension, "API_VERSION", None)) is not int or extension.API_VERSION != API_VERSION:
            raise RuntimeError("native GR completion module version mismatch")
        try:
            library = ctypes.CDLL(extension.__file__)
            runtime = ctypes.CDLL(torch._C.__file__)
            api = NativeCompletionAPI(library, runtime)
        except (OSError, AttributeError) as error:
            raise RuntimeError("cannot load native GR completion and PyTorch's CUDA runtime") from error
        _loaded_api = api
        return api


class CompletionTicket:
    def __init__(self, api: NativeCompletionAPI, generation: int):
        self.generation = _unsigned(generation, 64, "generation")
        self.api = api
        self._lock = threading.Lock()
        self._pointer = api.lib.gr_ticket_create(self.generation)
        if not self._pointer:
            raise MemoryError("native GR completion ticket allocation failed")

    @contextmanager
    def _borrow(self):
        # Every call gets an independent strong native ref before releasing
        # the owner lock. close() cannot destroy a concurrent waiter's CV.
        with self._lock:
            pointer = self._pointer
            if pointer is None:
                raise RuntimeError("native GR completion ticket is closed")
            self.api.lib.gr_ticket_retain(pointer)
        try:
            yield pointer
        finally:
            self.api.lib.gr_ticket_release(pointer)

    def register(self, stream_pointer: int) -> None:
        stream_pointer = _unsigned(stream_pointer, ctypes.sizeof(ctypes.c_void_p) * 8, "stream pointer")
        with self._borrow() as pointer:
            status = self.api.lib.gr_ticket_register(pointer, stream_pointer, self.api.register_address)
        if status != 0:
            raise CompletionRegistrationError(
                f"CUDA completion registration failed ({status}); retain buffers until proven drain"
            )

    def _snapshot(self, pointer) -> CompletionSnapshot:
        generation, status = ctypes.c_uint64(), ctypes.c_int()
        flags = self.api.lib.gr_ticket_snapshot(pointer, ctypes.byref(generation), ctypes.byref(status))
        if flags & ~15 or generation.value != self.generation:
            raise RuntimeError("native GR completion generation/flags mismatch")
        return CompletionSnapshot(
            generation.value, status.value, bool(flags & 1), bool(flags & 2),
            bool(flags & 4), bool(flags & 8),
        )

    def snapshot(self) -> CompletionSnapshot:
        with self._borrow() as pointer:
            return self._snapshot(pointer)

    def _wait(self, timeout_ms: int | None, *, device: bool) -> bool:
        # Reject overflow rather than letting ctypes silently truncate to int.
        # None preserves the upstream unbounded output-consumption wait; it
        # does not create a new request deadline. Explicit numeric timeouts
        # remain useful for cancellation/drain diagnostics and unit tests.
        native_timeout = -1 if timeout_ms is None else _unsigned(timeout_ms, 31, "timeout_ms")
        with self._borrow() as pointer:
            result = self.api.lib.gr_ticket_wait(pointer, native_timeout, int(device))
            if result == 1:
                return False
            snapshot = self._snapshot(pointer)
        if result == 3 or snapshot.registration_failed:
            raise CompletionRegistrationError("completion registration failed; device drain is unproven")
        if snapshot.device_done and snapshot.device_status != 0:
            raise CompletionDeviceError(f"CUDA completion reported device error {snapshot.device_status}")
        if not device and (result == 2 or snapshot.delivery_cancelled):
            raise CancelledError("GR control delivery cancelled; this does not release DMA storage")
        if result != 0 or not snapshot.device_succeeded:
            raise RuntimeError("native GR completion returned an inconsistent wait result")
        return True

    def wait_device(self, timeout_ms: int | None) -> bool:
        """True only for successful actual completion; False for timeout."""
        return self._wait(timeout_ms, device=True)

    def wait_delivery(self, timeout_ms: int | None) -> bool:
        """Claim a completed delivery, not acknowledge reading its Host payload."""
        return self._wait(timeout_ms, device=False)

    def cancel_delivery(self) -> None:
        with self._borrow() as pointer:
            self.api.lib.gr_ticket_cancel_delivery(pointer)

    def close(self) -> bool:
        """Drop this ticket owner, not its callback ref or any DMA lease."""
        with self._lock:
            pointer, self._pointer = self._pointer, None
        if pointer is None:
            return False
        self.api.lib.gr_ticket_release(pointer)
        return True
