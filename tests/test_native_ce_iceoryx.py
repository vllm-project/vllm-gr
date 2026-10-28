# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional real two-process iceoryx2 tests. Build --features iceoryx --examples first."""

import os
import select
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_native_ce_transport import start_test_transport
from vllm_gr.v1.engine.native_ce_transport import NativeBridge

IpcPeer = tuple[str, str, Callable[..., subprocess.Popen[str]]]

PEER = Path(__file__).resolve().parents[1] / "rust/target/release/examples/iceoryx-peer"
pytestmark = pytest.mark.skipif(
    os.getenv("VLLM_GR_CE_BP_TRANSPORT") != "sharedmem",
    reason="requires VLLM_GR_CE_BP_TRANSPORT=sharedmem",
)


@pytest.fixture(params=[False, True], ids=["rust-workers", "python-callers"])  # type: ignore[misc]
def bridge_factory(request: pytest.FixtureRequest) -> Callable[..., NativeBridge]:
    def create(*args: Any, **kwargs: Any) -> NativeBridge:
        return NativeBridge(*args, **kwargs, caller_owned=request.param)

    return create


@pytest.fixture  # type: ignore[misc]
def ipc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[IpcPeer]:
    monkeypatch.setenv("VLLM_GR_CE_BP_TRANSPORT", "sharedmem")
    tx, rx = f"ipc://{tmp_path}/tx", f"ipc://{tmp_path}/rx"
    children = []

    def start(count: int, capacity: int = 1, paused: bool = False) -> subprocess.Popen[str]:
        child = subprocess.Popen(
            [str(PEER), tx, rx, str(capacity), str(count)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        children.append(child)
        assert child.stdout is not None
        assert child.stdin is not None
        assert select.select([child.stdout], [], [], 10)[0], "peer startup timeout"
        assert child.stdout.readline().strip() == "READY", "peer failed to start"
        if not paused:
            child.stdin.write("GO\n")
            child.stdin.flush()
        return child

    yield tx, rx, start
    for child in children:
        if child.poll() is None:
            child.terminate()
        try:
            child.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate(timeout=3)


def test_bidirectional_order_ownership_and_restart(
    ipc: IpcPeer, bridge_factory: Callable[..., NativeBridge]
) -> None:
    tx, rx, start = ipc
    for _ in range(2):
        child = start(30)
        assert child.stdin is not None
        with bridge_factory(tx, rx, capacity=2, hwm=1) as bridge:
            for index in range(30):
                data = bytearray(bytes([index]) * (66560 + index))
                expected = [b"ENGINE_OUTPUT_BATCH", str(index).encode(), bytes(data), b""]
                bridge.send([expected[0], expected[1], data, b""])
                data[:] = bytes(len(data))
                frames = bridge.receive(2000)
                assert frames is not None
                actual = [bytes(f.buffer) for f in frames]
                assert actual == expected
            bridge.check()
            child.stdin.write("EXIT\n")
            child.stdin.flush()
            assert child.wait(timeout=3) == 0
        assert actual == expected


@pytest.mark.parametrize("present", [False, True])  # type: ignore[misc]
def test_absent_or_full_receiver_times_out(
    ipc: IpcPeer, present: bool, bridge_factory: Callable[..., NativeBridge]
) -> None:
    tx, rx, start = ipc
    if present:
        start(100, paused=True)
    started = time.monotonic()
    with (
        bridge_factory(tx, rx, capacity=1, hwm=1, timeout_ms=200) as bridge,
        pytest.raises(RuntimeError, match="timed out|remained full"),
    ):
        for _ in range(100):
            bridge.send([b"blocked"])
    assert time.monotonic() - started < 3


def test_close_interrupts_absent_peer_send(
    ipc: IpcPeer, bridge_factory: Callable[..., NativeBridge]
) -> None:
    tx, rx, _ = ipc
    bridge = bridge_factory(tx, rx, capacity=1, hwm=1, timeout_ms=5000)

    def produce() -> None:
        try:
            for _ in range(100):
                bridge.send([b"blocked"])
        except RuntimeError:
            pass

    worker = threading.Thread(target=produce)
    worker.start()
    time.sleep(0.02)
    started = time.monotonic()
    bridge.close()
    worker.join(1)
    assert not worker.is_alive()
    assert time.monotonic() - started < 1


def test_peer_death_becomes_transport_error(
    ipc: IpcPeer, bridge_factory: Callable[..., NativeBridge]
) -> None:
    tx, rx, start = ipc
    child = start(100)
    with bridge_factory(tx, rx, capacity=1, hwm=1, timeout_ms=200) as bridge:
        bridge.send([b"before"])
        assert bridge.receive(2000) is not None
        child.terminate()
        child.wait(timeout=3)
        with pytest.raises(RuntimeError, match="timed out|remained full|iceoryx"):
            for _ in range(100):
                bridge.send([b"after"])


def test_backpressure_resumes_without_loss_or_duplicates(
    ipc: IpcPeer, bridge_factory: Callable[..., NativeBridge]
) -> None:
    tx, rx, start = ipc
    child = start(30, paused=True)
    assert child.stdin is not None
    errors = []
    with bridge_factory(tx, rx, capacity=1, hwm=1) as bridge:

        def produce() -> None:
            try:
                for i in range(30):
                    bridge.send([str(i).encode(), bytes(66560)])
            except RuntimeError as exc:
                errors.append(exc)

        producer = threading.Thread(target=produce)
        producer.start()
        time.sleep(0.05)
        assert producer.is_alive(), "paused receiver did not apply backpressure"
        child.stdin.write("GO\n")
        child.stdin.flush()
        for i in range(30):
            frames = bridge.receive(2000)
            assert frames is not None
            assert [bytes(f.buffer) for f in frames] == [str(i).encode(), bytes(66560)]
        producer.join(2)
        assert not producer.is_alive()
        assert not errors
        assert bridge.receive(20) is None
        child.stdin.write("EXIT\n")
        child.stdin.flush()
        assert child.wait(timeout=3) == 0


def test_sharedmem_automatically_starts_rust_workers(
    ipc: IpcPeer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_gr.v1.engine import core

    tx, rx, start = ipc
    start(1)
    monkeypatch.delenv("VLLM_GR_CE_TRANSPORT", raising=False)
    monkeypatch.setenv("VLLM_GR_BEAMPROC_ZMQ_HWM", "1")
    state = SimpleNamespace(
        BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR=tx,
        BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR=rx,
        scheduler=SimpleNamespace(),
    )
    core._start_beamproc_io_thread(state)
    try:
        bridge = state.beamproc_native_bridge
        assert state.beamproc_io_bp_tx_thread is None
        assert state.beamproc_io_bp_rx_thread is None
        assert state.beamproc_outbound_queue is None
        assert bridge._on_receive is None
        core._enqueue_beamproc_message(state, b"ABORT", {"request_ids": ["root"]})
        frames = bridge.receive(2000)
        assert frames is not None
        assert bytes(frames[0].buffer) == b"ABORT"
    finally:
        core._stop_beamproc_native_io(state)


def test_python_threads_own_io_and_close_with_full_admission_queue(
    ipc: IpcPeer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_gr.v1.engine import core

    tx, rx, start = ipc
    start(3)
    monkeypatch.setenv("VLLM_GR_BEAMPROC_QUEUE_CAPACITY", "1")
    monkeypatch.setenv("VLLM_GR_BEAMPROC_ZMQ_HWM", "1")
    state = SimpleNamespace(
        BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR=tx,
        BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR=rx,
        scheduler=SimpleNamespace(),
        engine_index=0,
    )
    rx_calls, tx_calls = [], []
    original_send = core._send_beamproc_outbound_message

    def send(*args: Any) -> None:
        tx_calls.append(threading.get_ident())
        original_send(*args)

    def receive(engine: SimpleNamespace, frames: list[SimpleNamespace]) -> None:
        rx_calls.append(threading.get_ident())
        assert bytes(frames[0].buffer) == b"ABORT"
        core._publish_beamproc_item(engine, ("decoded", None), "test queue")

    monkeypatch.setattr(core, "_send_beamproc_outbound_message", send)
    monkeypatch.setattr(core, "_process_beamproc_plan_frames", receive)
    before = {p.name for p in Path("/proc/self/task").iterdir()}
    start_test_transport(state, python_threads=True)
    try:
        assert getattr(state, "beamproc_native_bridge", None) is None
        after = {p.name for p in Path("/proc/self/task").iterdir()}
        assert after - before == {
            str(state.beamproc_io_bp_tx_thread.native_id),
            str(state.beamproc_io_bp_rx_thread.native_id),
        }, "shared-memory Python mode started extra native threads"
        for i in range(3):
            core._enqueue_beamproc_message(state, b"ABORT", {"index": i})
        deadline = time.monotonic() + 3
        while len(rx_calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(rx_calls) >= 2
        assert set(rx_calls) == {state.beamproc_io_bp_rx_thread.ident}
        assert set(tx_calls) == {state.beamproc_io_bp_tx_thread.ident}
        core._check_beamproc_native_io(state)
    finally:
        started = time.monotonic()
        core._stop_beamproc_native_io(state)
        assert time.monotonic() - started < 1
        assert not state.beamproc_io_bp_tx_thread.is_alive()
        assert not state.beamproc_io_bp_rx_thread.is_alive()


def test_direct_receive_close_interrupts_wait(ipc: IpcPeer) -> None:
    tx, rx, _ = ipc
    bridge = NativeBridge(tx, rx, caller_owned=True)
    errors = []

    def receive() -> None:
        try:
            bridge.receive(5000)
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=receive)
    worker.start()
    time.sleep(0.02)
    bridge.close()
    worker.join(1)
    assert not worker.is_alive()
    assert len(errors) == 1 and "closed" in errors[0]


def test_python_rx_failure_reaches_scheduler(ipc: IpcPeer, monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.engine import core

    tx, rx, start = ipc
    start(1)
    monkeypatch.setenv("VLLM_GR_BEAMPROC_ZMQ_HWM", "1")
    state = SimpleNamespace(
        BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR=tx,
        BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR=rx,
        scheduler=SimpleNamespace(),
        engine_index=0,
    )
    failure = ValueError("invalid reply")

    def receive(*args: Any) -> None:
        raise failure

    monkeypatch.setattr(core, "_process_beamproc_plan_frames", receive)
    start_test_transport(state, python_threads=True)
    try:
        core._enqueue_beamproc_message(state, b"ABORT", {})
        assert state.beamproc_native_stop.wait(3)
        with pytest.raises(RuntimeError, match="adapter failed") as caught:
            core._drain_pb_queue(state)
        assert caught.value.__cause__ is failure
    finally:
        core._stop_beamproc_native_io(state)
