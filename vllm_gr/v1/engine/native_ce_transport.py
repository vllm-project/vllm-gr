# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private, versioned C ABI for CE ZMQ/shared-memory transport.

CDLL releases the GIL during calls. Send copies buffers into Rust-owned storage
before returning, allowing the existing Python pool to reuse them immediately.
CE pulls RX frames on its scheduler thread. Optional diagnostic callbacks run
on the native receiver with the GIL acquired by ctypes. Frame copies own their
memory before native storage is released.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np


class _FrameView(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("length", ctypes.c_size_t)]


_RxCallback = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(_FrameView), ctypes.c_size_t)
# A CFUNCTYPE pointer alone does not keep its Python callable alive. Pin callback
# bridges even if the caller drops its last reference, until native RX is joined.
_CALLBACK_OWNERS = {}


@atexit.register
def _close_callback_bridges():
    # atexit runs before interpreter teardown; no foreign thread may call Python
    # after finalization starts. Normal CE shutdown closes these earlier.
    for bridge in list(_CALLBACK_OWNERS.values()):
        if bridge._pid == os.getpid():
            bridge.close()


def _library_name() -> str:
    if sys.platform == "win32":
        return "ce_transport.dll"
    if sys.platform == "darwin":
        return "libce_transport.dylib"
    return "libce_transport.so"


def _load_library():
    package = Path(__file__).resolve().parents[2]
    override = os.environ.get("VLLM_GR_CE_TRANSPORT_LIBRARY")
    candidates = (
        [Path(override)]
        if override
        else [
            package / "bin" / _library_name(),
            package.parent / "rust" / "target" / "release" / _library_name(),
        ]
    )
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise RuntimeError(
            "Rust CE transport library not found; run cargo build --release --lib "
            "--manifest-path rust/Cargo.toml or install a wheel containing it"
        )
    lib = ctypes.CDLL(str(path))
    lib.ce_transport_abi.restype = ctypes.c_uint32
    if lib.ce_transport_abi() != 3:
        raise RuntimeError("Incompatible Rust CE transport ABI (expected 3); rebuild the library")
    signatures = {
        "abi": ([], ctypes.c_uint32),
        "open": (
            [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.c_int32,
                ctypes.c_uint32,
                _RxCallback,
            ],
            ctypes.c_uint64,
        ),
        "open_direct": (
            [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_uint32],
            ctypes.c_uint64,
        ),
        "send": ([ctypes.c_uint64, ctypes.POINTER(_FrameView), ctypes.c_size_t], ctypes.c_int32),
        "send_batch": (
            [ctypes.c_uint64, ctypes.POINTER(_FrameView), ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "receive": (
            [ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)],
            ctypes.c_int32,
        ),
        "frame": (
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)],
            ctypes.c_void_p,
        ),
        "message_free": ([ctypes.c_void_p], None),
        "check": ([ctypes.c_uint64], ctypes.c_int32),
        "start_receive": ([ctypes.c_uint64], ctypes.c_int32),
        "close": ([ctypes.c_uint64], ctypes.c_int32),
        "error": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, f"ce_transport_{name}")
        fn.argtypes, fn.restype = args, result
    return lib


class NativeBridge:
    """One CE transport, constructed after process startup, explicitly closed.

    CE uses receive(0) to drain the native inbox on the scheduler thread.
    Optional on_receive delivery is retained for diagnostics and comparisons;
    that mode has no RX queue and requires start_receive before callbacks run.
    Close joins native RX before releasing the callback and is safe alongside
    send/receive. Callbacks must finish in bounded time and cannot call close.
    The object must not be inherited across fork or serialized to another process.

    With caller_owned=True, iceoryx2 send/receive execute synchronously on the
    calling threads, with no native workers or native inbox/outbox. Python TX
    supplies prepared frames; send_batch and callbacks are unavailable. Close
    interrupts and waits for in-flight port operations. CDLL releases the GIL.
    """

    def __init__(
        self,
        tx_addr: str,
        rx_addr: str,
        capacity: int = 64,
        hwm: int = 64,
        timeout_ms: int = 5000,
        *,
        on_receive=None,
        caller_owned=False,
    ):
        if not 1 <= capacity <= 65536 or not 1 <= hwm <= 2**31 - 1:
            raise ValueError("CE queue capacity must be 1..65536 and HWM positive int32")
        if not 1 <= timeout_ms <= 2**32 - 1:
            raise ValueError("CE send timeout must be positive uint32 milliseconds")
        if "\0" in tx_addr or "\0" in rx_addr:
            raise ValueError("CE socket addresses cannot contain NUL")
        self._pid = os.getpid()
        self._lib = _load_library()
        self._close_lock = threading.Lock()
        self._closing = threading.Event()
        self._closed = False
        self._callback_error = None
        self._callback_thread = None
        self._on_receive = on_receive
        self._callback = _RxCallback(self._dispatch) if on_receive is not None else _RxCallback()
        if caller_owned:
            if on_receive is not None:
                raise ValueError("caller-owned transport cannot use a native callback")
            self._handle = self._lib.ce_transport_open_direct(
                tx_addr.encode(), rx_addr.encode(), hwm, timeout_ms
            )
        else:
            self._handle = self._lib.ce_transport_open(
                tx_addr.encode(),
                rx_addr.encode(),
                capacity,
                hwm,
                timeout_ms,
                self._callback,
            )
        if not self._handle:
            self._raise_error()
        if on_receive is not None:
            _CALLBACK_OWNERS[id(self)] = self

    def start_receive(self):
        self._check_process()
        if self._lib.ce_transport_start_receive(self._handle) < 0:
            self._raise_error()

    def _dispatch(self, views, count):
        # ctypes acquires the GIL here on the existing Rust RX OS thread. Copy
        # before application because requests/NumPy views can outlive this call.
        try:
            self._callback_thread = threading.get_ident()
            if self._closing.is_set():
                return 0
            frames = [
                SimpleNamespace(
                    buffer=memoryview(ctypes.string_at(views[index].data, views[index].length))
                )
                for index in range(count)
            ]
            self._on_receive(frames)
            return 0
        except BaseException as exc:  # noqa: BLE001
            # Exceptions must not escape CFUNCTYPE (ctypes would swallow them
            # and leave the C return value undefined). CE health checks chain it.
            self._callback_error = exc
            return -1
        finally:
            self._callback_thread = None

    def _raise_error(self):
        buffer = ctypes.create_string_buffer(2048)
        size = self._lib.ce_transport_error(buffer, len(buffer))
        raise RuntimeError(
            buffer.raw[:size].decode("utf-8", errors="replace")
        ) from self._callback_error

    def _check_process(self):
        if os.getpid() != self._pid:
            raise RuntimeError("Rust CE transport cannot be used after fork")

    def check(self):
        self._check_process()
        if self._callback_error is not None:
            raise RuntimeError("CE RX callback failed") from self._callback_error
        if self._lib.ce_transport_check(self._handle) < 0:
            self._raise_error()

    def send(self, frames):
        self._submit(self._lib.ce_transport_send, frames)

    def send_batch(self, metadata, buffers):
        """Snapshot metadata and buffers; Rust TX builds the v6 wire message.

        Encoding this small descriptor is synchronous. Candidate packing, row
        expansion, token conversion and final header encoding run on Rust TX.
        """
        self._submit(
            self._lib.ce_transport_send_batch, [msgspec.msgpack.encode(metadata), *buffers]
        )

    def _submit(self, submit, frames):
        self._check_process()
        # np.frombuffer supports read-only memoryviews without copying. References
        # live through the copying FFI call; callers must not concurrently mutate.
        arrays = [np.frombuffer(frame, dtype=np.uint8) for frame in frames]
        views = (_FrameView * len(arrays))(
            *[_FrameView(array.ctypes.data, array.nbytes) for array in arrays]
        )
        if submit(self._handle, views, len(arrays)) < 0:
            self._raise_error()

    def receive(self, timeout_ms: int = 100):
        self._check_process()
        if not 0 <= timeout_ms <= 2**32 - 1:
            raise ValueError("CE receive timeout must be uint32 milliseconds")
        message = ctypes.c_void_p()
        result = self._lib.ce_transport_receive(self._handle, timeout_ms, ctypes.byref(message))
        if result < 0:
            self._raise_error()
        if result == 0:
            return None
        try:
            frames = []
            for index in range(64):
                size = ctypes.c_size_t()
                data = self._lib.ce_transport_frame(message, index, ctypes.byref(size))
                if data is None:
                    break
                # Match PyZMQ Frame.buffer so existing wire validation and numeric
                # views can stay unchanged. The bytes owner outlives native storage.
                frames.append(
                    SimpleNamespace(buffer=memoryview(ctypes.string_at(data, size.value)))
                )
            return frames
        finally:
            self._lib.ce_transport_message_free(message)

    def close(self):
        self._check_process()
        # Reject before acquiring the lock: another thread can already be
        # closing and waiting for this callback to return.
        if self._callback_thread == threading.get_ident():
            raise RuntimeError("cannot close CE transport from an RX callback")
        with self._close_lock:
            if self._closed:
                return
            self._closing.set()
            # CDLL releases the GIL while native close joins RX. Callback remains
            # pinned until this returns, including any in-flight application.
            if self._lib.ce_transport_close(self._handle) < 0:
                self._raise_error()
            self._closed = True
            _CALLBACK_OWNERS.pop(id(self), None)
            self._callback = None
            self._on_receive = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
