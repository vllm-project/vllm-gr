# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real local ZMQ peers exercise the native ABI and CE integration (CPU only)."""

import queue
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import numpy as np
import pytest
import zmq

from vllm_gr.v1.engine.native_ce_transport import NativeBridge

Peer = tuple[zmq.Socket, zmq.Socket, str, str]


def start_test_transport(
    state: SimpleNamespace, *, python_threads: bool = False, capacity: int = 1
) -> None:
    """Exercise internal adapters independently of the public mode selector."""
    from vllm_gr.v1.engine import core

    state.pb_queue = queue.Queue(maxsize=capacity)
    state.beamproc_outbound_queue = queue.Queue(maxsize=capacity) if python_threads else None
    state.beamproc_correlation_id = 0
    state.beamproc_pending_plan_count = 0
    state.scheduler._gr_beamproc_pending_plans = 0
    start = core._start_beamproc_python_shm_io if python_threads else core._start_beamproc_native_io
    start(
        state,
        state.BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR,
        state.BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR,
        capacity,
    )


@pytest.fixture  # type: ignore[misc]
def peer(monkeypatch: pytest.MonkeyPatch) -> Iterator[Peer]:
    # These peers speak ZMQ even when the suite also enables shared-memory tests.
    monkeypatch.setenv("VLLM_GR_CE_BP_TRANSPORT", "zmq")
    with tempfile.TemporaryDirectory(prefix="ce-native-test-") as directory:
        tx_addr, rx_addr = f"ipc://{directory}/tx", f"ipc://{directory}/rx"
        with (
            zmq.Context() as ctx,
            ctx.socket(zmq.PULL) as receiver,
            ctx.socket(zmq.PUSH) as sender,
        ):
            for socket in (receiver, sender):
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 2000)
                socket.setsockopt(zmq.SNDTIMEO, 2000)
            receiver.bind(tx_addr)
            sender.bind(rx_addr)
            yield receiver, sender, tx_addr, rx_addr


def test_multipart_ownership_order_and_empty_frames(peer: Peer) -> None:
    receiver, sender, tx_addr, rx_addr = peer
    with NativeBridge(tx_addr, rx_addr) as bridge:
        for index in range(25):
            array = np.arange(8192, dtype=np.int32) + index
            expected = array.tobytes()
            bridge.send([b"ENGINE_OUTPUT_BATCH", str(index).encode(), memoryview(array), b""])
            array.fill(-1)  # Native queues may still be sending; input is reusable now.
            assert receiver.recv_multipart() == [
                b"ENGINE_OUTPUT_BATCH",
                str(index).encode(),
                expected,
                b"",
            ]
        sender.send_multipart([b"BEAM_PLAN", b"header", bytes(range(256)), b""])
        frames = bridge.receive(2000)
        assert frames is not None
    # Every returned frame remains valid after both native message and bridge free.
    assert [bytes(frame.buffer) for frame in frames] == [
        b"BEAM_PLAN",
        b"header",
        bytes(range(256)),
        b"",
    ]
    bridge.close()
    with pytest.raises(RuntimeError, match="closed"):
        bridge.send([b"late"])


def test_idle_receive_timeout_and_concurrent_close(peer: Peer) -> None:
    _, _, tx_addr, rx_addr = peer
    bridge = NativeBridge(tx_addr, rx_addr)
    assert bridge.receive(10) is None
    result: queue.Queue[str] = queue.Queue()

    def receive() -> None:
        try:
            bridge.receive(5000)
        except RuntimeError as exc:
            result.put(str(exc))

    worker = threading.Thread(target=receive)
    worker.start()
    time.sleep(0.02)
    start = time.monotonic()
    bridge.close()
    worker.join(1)
    assert not worker.is_alive()
    assert time.monotonic() - start < 1
    assert "closed" in result.get_nowait()


def test_invalid_configuration_and_startup_error(peer: Peer) -> None:
    _, _, tx_addr, rx_addr = peer
    with pytest.raises(ValueError):
        NativeBridge(tx_addr, rx_addr, capacity=0)
    with pytest.raises(ValueError):
        NativeBridge(tx_addr + "\0", rx_addr)
    with pytest.raises(RuntimeError, match="ce-zmq-tx"):
        NativeBridge("invalid-protocol://socket", rx_addr, timeout_ms=200)
    with NativeBridge(tx_addr, rx_addr) as bridge:
        with pytest.raises(RuntimeError, match="frame count"):
            bridge.send([])
        with pytest.raises(RuntimeError, match="frame count"):
            bridge.send([b""] * 65)
        bridge.check()  # Invalid caller input does not poison the worker.


def test_disconnected_peer_backpressure_is_bounded() -> None:
    with tempfile.TemporaryDirectory(prefix="ce-native-absent-") as directory:
        bridge = NativeBridge(
            f"ipc://{directory}/tx", f"ipc://{directory}/rx", capacity=1, hwm=1, timeout_ms=100
        )
        start = time.monotonic()
        try:
            with pytest.raises(RuntimeError, match="timed out|remained full"):
                for _ in range(100):
                    bridge.send([b"message"])
        finally:
            bridge.close()
        assert time.monotonic() - start < 2


def test_close_when_rx_queue_full(peer: Peer) -> None:
    _, sender, tx_addr, rx_addr = peer
    bridge = NativeBridge(tx_addr, rx_addr, capacity=1)
    for _ in range(10):
        sender.send_multipart([b"queued"])
    time.sleep(0.02)
    start = time.monotonic()
    bridge.close()
    assert time.monotonic() - start < 1


def test_rust_mode_delivers_on_scheduler_without_pb_handoff(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_gr.v1.engine import core

    receiver, sender, tx_addr, rx_addr = peer
    state = SimpleNamespace(
        BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR=tx_addr,
        BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR=rx_addr,
        scheduler=SimpleNamespace(),
        engine_index=0,
    )
    start_test_transport(state)
    try:
        core._enqueue_beamproc_message(state, b"ABORT", {"request_ids": ["root"]})
        frames = receiver.recv_multipart()
        assert frames[0] == b"ABORT"
        assert msgspec.msgpack.decode(frames[1]) == {"request_ids": ["root"]}
        sender.send_multipart(
            [
                b"BEAM_FINISHED",
                msgspec.msgpack.encode(
                    {
                        "version": 1,
                        "root_request_id": "root",
                    }
                ),
            ]
        )
        received = []
        scheduler_thread = threading.get_ident()
        monkeypatch.setattr(
            core,
            "_handle_beamproc_plan",
            lambda engine, item: received.append((threading.get_ident(), item)),
        )
        deadline = time.monotonic() + 2
        while not received and time.monotonic() < deadline:
            core._drain_pb_queue(state)
            time.sleep(core._BEAMPROC_DUAL_QUEUE_POLL_SLEEP_S)
        assert received == [
            (
                scheduler_thread,
                {
                    "version": 1,
                    "root_request_id": "root",
                    "action": "beamproc_finished",
                },
            )
        ]
        assert state.pb_queue.empty()
        assert state.beamproc_native_bridge._on_receive is None
        core._check_beamproc_native_io(state)
    finally:
        core._stop_beamproc_native_io(state)
    assert state.beamproc_io_bp_tx_thread is None
    assert state.beamproc_outbound_queue is None
    assert state.beamproc_io_bp_rx_thread is None


def test_buffer_pool_release_after_native_copy(peer: Peer) -> None:
    from vllm_gr.v1.engine.core import _send_beamproc_frames

    receiver, _, tx_addr, rx_addr = peer
    with NativeBridge(tx_addr, rx_addr) as bridge:
        buffer = {"_beamproc_tx_buffer_set": True, "array": np.arange(128, dtype=np.int32)}
        pool: list[dict[str, Any]] = []
        pending: deque[Any] = deque()
        original = buffer["array"].tobytes()
        _send_beamproc_frames(
            bridge, [memoryview(buffer["array"])], pending, refs=buffer, output_buffer_pool=pool
        )
        assert not pending and pool[0] is buffer
        pool.pop()["array"].fill(0)
        assert receiver.recv_multipart() == [original]


def test_native_worker_failure_reaches_engine_loop(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_gr.v1.engine import core

    _, _, tx_addr, rx_addr = peer
    state = SimpleNamespace(
        BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR=tx_addr,
        BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR=rx_addr,
        scheduler=SimpleNamespace(),
        engine_index=0,
    )
    start_test_transport(state)
    try:
        state.beamproc_native_bridge.close()
        with pytest.raises(RuntimeError):
            core._drain_pb_queue(state)
    finally:
        core._stop_beamproc_native_io(state)


def test_fork_guard(peer: Peer, monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_gr.v1.engine.native_ce_transport as native

    _, _, tx_addr, rx_addr = peer
    with NativeBridge(tx_addr, rx_addr) as bridge, monkeypatch.context() as patch:
        patch.setattr(native.os, "getpid", lambda: bridge._pid + 1)
        with pytest.raises(RuntimeError, match="fork"):
            bridge.send([b"unsafe"])


def test_explicit_missing_library_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLLM_GR_CE_TRANSPORT_LIBRARY", str(tmp_path / "missing.so"))
    with pytest.raises(RuntimeError, match="not found"):
        NativeBridge("ipc:///tmp/unused-tx", "ipc:///tmp/unused-rx")


def test_live_v5_plan_through_native_rx(peer: Peer, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_beamproc_scheduler_unit import (
        _beamproc_plan_engine,
        _mock_request,
        _two_child_plan,
    )
    from vllm_gr.v1.engine import core

    _, sender, tx_addr, rx_addr = peer
    monkeypatch.setenv(core._BEAMPROC_ENABLED_ENV, "1")
    monkeypatch.setattr("vllm.v1.request.Request", _mock_request)
    state = _beamproc_plan_engine()
    state.scheduler = SimpleNamespace()
    state.BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR = tx_addr
    state.BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR = rx_addr
    payload = _two_child_plan()
    payload.pop("token_ids")
    payload.pop("beam_tokens")
    payload.update(
        version=5,
        action="observed_full",
        finished=False,
        token_ids_len=2,
        beam_tokens_len=2,
        beam_token_stride=1,
    )
    tokens = np.asarray([100, 200], dtype="<i4").tobytes()
    start_test_transport(state)
    try:
        sender.send_multipart([b"BEAM_PLAN", msgspec.msgpack.encode(payload), tokens, tokens])
        requests = []
        scheduler_thread = threading.get_ident()
        state.add_request = lambda req, wave: requests.append((threading.get_ident(), req, wave))
        deadline = time.monotonic() + 2
        admitted = 0
        while not requests and time.monotonic() < deadline:
            admitted += core._drain_pb_queue(state)
            time.sleep(core._BEAMPROC_DUAL_QUEUE_POLL_SLEEP_S)
        assert admitted == 1
        assert len(requests) == 1
        tid, req, wave = requests[0]
        assert tid == scheduler_thread
        assert wave == 7
        assert req.prompt_token_ids == [10, 20, 100, 200]
        assert req.is_beam_decode
        assert state.pb_queue.empty()
        core._check_beamproc_native_io(state)
    finally:
        core._stop_beamproc_native_io(state)
