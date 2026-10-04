# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Callback ownership, execution, errors and shutdown. Real IPC, CPU only."""

import gc
import queue
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_native_ce_transport import Peer
from tests.test_native_ce_transport import peer as _peer
from vllm_gr.v1.engine.native_ce_transport import NativeBridge

peer = _peer


def test_callback_activation_order_owner_and_frame_lifetime(peer: Peer) -> None:
    _, sender, tx, rx = peer
    received: queue.Queue[tuple[str, list[SimpleNamespace]]] = queue.Queue()

    def apply(frames: list[SimpleNamespace]) -> None:
        tid = threading.get_native_id()
        name = Path(f"/proc/self/task/{tid}/comm").read_text().strip()
        received.put((name, frames))

    with NativeBridge(tx, rx, on_receive=apply) as bridge:
        sender.send_multipart([b"0", b"", b"before-activation"])
        with pytest.raises(queue.Empty):
            received.get(timeout=0.05)
        with pytest.raises(RuntimeError, match="callback mode"):
            bridge.receive(0)
        bridge.start_receive()
        for index in range(1, 10):
            sender.send_multipart([str(index).encode(), b"", b"payload"])
        messages = [received.get(timeout=2) for _ in range(10)]
    assert [name for name, _ in messages] == ["ce-zmq-rx"] * 10
    assert [bytes(frames[0].buffer) for _, frames in messages] == [
        str(i).encode() for i in range(10)
    ]
    assert all(bytes(frames[1].buffer) == b"" for _, frames in messages)
    assert bytes(messages[0][1][2].buffer) == b"before-activation"


def test_callback_exception_is_chained_and_close_from_callback_is_rejected(peer: Peer) -> None:
    _, sender, tx, rx = peer
    attempted = threading.Event()

    def apply(_: Any) -> None:
        try:
            bridge.close()
        finally:
            attempted.set()

    with NativeBridge(tx, rx, on_receive=apply) as bridge:
        bridge.start_receive()
        sender.send_multipart([b"trigger"])
        assert attempted.wait(2)
        # close synchronizes with callback completion before inspecting its error.
        bridge.close()
        with pytest.raises(RuntimeError, match="RX callback failed") as error:
            bridge.check()
        assert "cannot close" in str(error.value.__cause__)


def test_concurrent_close_waits_for_callback_and_releases_pin(peer: Peer) -> None:
    from vllm_gr.v1.engine.native_ce_transport import _CALLBACK_OWNERS

    _, sender, tx, rx = peer
    entered, release = threading.Event(), threading.Event()
    results: queue.Queue[Exception | None] = queue.Queue()

    def apply(_: Any) -> None:
        entered.set()
        assert release.wait(2)

    bridge = NativeBridge(tx, rx, on_receive=apply)
    bridge.start_receive()
    sender.send_multipart([b"trigger"])
    assert entered.wait(2)
    ident = id(bridge)
    reference = weakref.ref(bridge)
    del bridge
    gc.collect()
    assert reference() is _CALLBACK_OWNERS[ident]

    def close() -> None:
        try:
            owner = reference()
            assert owner is not None
            owner.close()
            results.put(None)
        except Exception as exc:  # noqa: BLE001
            results.put(exc)

    workers = [threading.Thread(target=close) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        with pytest.raises(queue.Empty):
            results.get(timeout=0.05)
        assert ident in _CALLBACK_OWNERS
    finally:
        release.set()
        for worker in workers:
            worker.join(2)
    assert all(not worker.is_alive() for worker in workers)
    assert [results.get(timeout=1) for _ in workers] == [None, None]
    assert ident not in _CALLBACK_OWNERS
    gc.collect()
    assert reference() is None


def test_full_pb_queue_callback_failure_reaches_ce(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Callback mode remains available for diagnostics; production CE uses pull RX.
    from types import SimpleNamespace

    import msgspec

    from vllm_gr.v1.engine import core

    _, sender, tx, rx = peer
    monkeypatch.setattr(core, "_BEAMPROC_QUEUE_PUT_TIMEOUT_S", 0.05)
    state = SimpleNamespace(pb_queue=queue.Queue(maxsize=1))
    state.pb_queue.put_nowait("occupied")
    failed = threading.Event()

    def apply(frames: list[SimpleNamespace]) -> None:
        try:
            core._process_beamproc_plan_frames(state, frames)
        finally:
            failed.set()

    with NativeBridge(tx, rx, on_receive=apply) as bridge:
        bridge.start_receive()
        sender.send_multipart(
            [
                b"BEAM_FINISHED",
                msgspec.msgpack.encode({"version": 1, "root_request_id": "root"}),
            ]
        )
        assert failed.wait(2)
        bridge.close()
        with pytest.raises(RuntimeError) as error:
            bridge.check()
        assert "remained full" in str(error.value.__cause__)
