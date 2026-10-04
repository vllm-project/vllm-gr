# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned native RX: FIFO, idle progress, failures and AS priority."""

import queue
import threading
import time
from collections.abc import Iterator
from itertools import chain, repeat
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest

from tests.test_native_ce_transport import Peer
from tests.test_native_ce_transport import peer as _peer
from tests.test_native_ce_transport import start_test_transport
from vllm_gr.v1.engine import core

peer = _peer


@pytest.fixture  # type: ignore[misc]
def engine(peer: Peer, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    _, _, tx, rx = peer
    monkeypatch.setenv("VLLM_GR_BEAMPROC_QUEUE_CAPACITY", "1")
    state = SimpleNamespace(
        BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR=tx,
        BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR=rx,
        scheduler=SimpleNamespace(has_requests=lambda: False),
        engine_index=0,
        engines_running=False,
        batch_queue=[],
        input_queue=queue.Queue(),
    )
    start_test_transport(state)
    try:
        yield state
    finally:
        core._stop_beamproc_native_io(state)


def finish(sender: Any, root: str) -> None:
    sender.send_multipart(
        [
            b"BEAM_FINISHED",
            msgspec.msgpack.encode({"version": 1, "root_request_id": root}),
        ]
    )


def test_native_fifo_bypasses_python_queue_and_precedes_as(
    peer: Peer, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, sender, _, _ = peer
    calls = []
    owner = threading.get_ident()
    monkeypatch.setattr(
        core,
        "_handle_beamproc_plan",
        lambda state, plan: calls.append((threading.get_ident(), plan["root_request_id"])),
    )
    # Any attempted PB re-enqueue fails; native replies must be consumed directly.
    monkeypatch.setattr(engine.pb_queue, "put", lambda *a, **kw: pytest.fail("PB re-enqueue"))
    for i in range(8):
        finish(sender, str(i))
    deadline = time.monotonic() + 2
    while len(calls) < 8 and time.monotonic() < deadline:
        core._drain_pb_queue(engine)
        time.sleep(core._BEAMPROC_DUAL_QUEUE_POLL_SLEEP_S)
    assert calls == [(owner, str(i)) for i in range(8)]
    # Inject an already available native reply to make the AS ordering assertion
    # deterministic, independently of IPC arrival scheduling.
    frames = [
        SimpleNamespace(buffer=memoryview(x))
        for x in (
            b"BEAM_FINISHED",
            msgspec.msgpack.encode({"version": 1, "root_request_id": "last"}),
        )
    ]
    replies = chain([frames], repeat(None))
    monkeypatch.setattr(engine.beamproc_native_bridge, "receive", lambda timeout: next(replies))
    handler = core.make_patched_handle_client_request(lambda *args: calls.append((owner, "AS")))
    handler(engine, object(), None)
    assert calls[-2:] == [(owner, "last"), (owner, "AS")]


def test_idle_scheduler_pulls_reply_without_callback(
    peer: Peer, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, sender, _, _ = peer
    owner = threading.get_ident()
    seen = []

    def apply(state: SimpleNamespace, plan: Any) -> None:
        seen.append((threading.get_ident(), plan["root_request_id"]))
        state.engines_running = True

    monkeypatch.setattr(core, "_handle_beamproc_plan", apply)
    finish(sender, "wake")
    process = core.make_patched_process_input_queue(lambda state: "awake")
    assert process(engine) == "awake"
    assert seen == [(owner, "wake")]
    assert engine.beamproc_native_bridge._on_receive is None


def test_decode_application_failure_propagates_and_clears_delivery(
    peer: Peer, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, sender, _, _ = peer
    failure = ValueError("scheduler application failed")

    def apply(*args: Any) -> None:
        raise failure

    monkeypatch.setattr(core, "_handle_beamproc_plan", apply)
    finish(sender, "bad")
    deadline = time.monotonic() + 2
    with pytest.raises(ValueError, match="scheduler application failed") as error:
        while time.monotonic() < deadline:
            core._drain_pb_queue(engine)
            time.sleep(core._BEAMPROC_DUAL_QUEUE_POLL_SLEEP_S)
    assert error.value is failure
    assert engine.beamproc_scheduler_delivery is None


def test_idle_scheduler_yields_until_as_input_arrives(
    engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def yield_to_input() -> None:
        calls.append(True)
        engine.input_queue.put_nowait(("AS", None))

    monkeypatch.setattr(core.os, "sched_yield", yield_to_input)
    process = core.make_patched_process_input_queue(lambda state: "awake")
    assert process(engine) == "awake"
    assert calls == [True]


def test_idle_scheduler_exits_on_shutdown_without_work(engine: SimpleNamespace) -> None:
    engine.is_running = lambda: False
    process = core.make_patched_process_input_queue(
        lambda state: pytest.fail("processed after shutdown")
    )
    assert process(engine) is None
