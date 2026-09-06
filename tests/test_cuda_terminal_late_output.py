# SPDX-License-Identifier: Apache-2.0
"""Regression for resource ownership after successful CUDA transport teardown."""

from threading import Event, Thread

import pytest

from tests.test_cuda_async_output import (
    _activate,
    _FakeAPI,
    _queued_control,
    _runner,
    _workspace,
    cuda_fakes,  # noqa: F401
)
from vllm_gr.v1.worker import cuda_async_output as adapter


def test_late_output_after_successful_shutdown_does_not_restore_quarantine(cuda_fakes):  # noqa: F811
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, transport = _runner(workspace, api)
    _activate(runner, workspace)
    entry, output = _queued_control(cuda_fakes, runner, workspace, "gr", 103)
    pool = output._gr_control_frame.pool
    assert pool.occupied_slots == 1 and not entry[0].done()

    transport.shutdown()
    assert pool.occupied_slots == 0 and api.tickets[0].closed
    assert transport not in adapter._QUARANTINED
    waits_before = sum(
        isinstance(call, tuple) and call[0] == "wait_delivery"
        for call in cuda_fakes.calls
    )

    # A retained native Future must reject late delivery, not resurrect the
    # global resource-retention owner after a successful terminal device drain.
    with pytest.raises(RuntimeError, match="control pool is closed"):
        entry[0].result()

    assert entry[0].done() and isinstance(entry[0].exception(), RuntimeError)
    assert transport.consumed == 0 and pool.occupied_slots == 0
    assert sum(
        isinstance(call, tuple) and call[0] == "wait_delivery"
        for call in cuda_fakes.calls
    ) == waits_before
    assert transport not in adapter._QUARANTINED


def test_failed_drain_keeps_quarantine_until_successful_retry(cuda_fakes, monkeypatch):  # noqa: F811
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    runner, transport = _runner(workspace, api)
    _activate(runner, workspace)
    entry, output = _queued_control(cuda_fakes, runner, workspace, "gr", 107)
    pool = output._gr_control_frame.pool
    original_synchronize = adapter.torch.cuda.synchronize
    failure = RuntimeError("injected terminal drain failure")

    def fail_drain(device):
        raise failure

    monkeypatch.setattr(adapter.torch.cuda, "synchronize", fail_drain)
    with pytest.raises(RuntimeError, match="injected terminal drain failure") as caught:
        transport.shutdown()
    assert caught.value is failure
    assert transport._closed and not transport._terminal_drained
    assert transport in adapter._QUARANTINED
    assert pool.occupied_slots == 1 and not api.tickets[0].closed
    with pytest.raises(RuntimeError, match="control pool quarantined"):
        entry[0].result()
    assert transport in adapter._QUARANTINED
    assert pool.occupied_slots == 1

    monkeypatch.setattr(adapter.torch.cuda, "synchronize", original_synchronize)
    transport.shutdown()
    assert transport._terminal_drained and pool.occupied_slots == 0
    assert api.tickets[0].closed and transport not in adapter._QUARANTINED


def test_other_transport_quarantine_can_progress_during_device_drain(cuda_fakes, monkeypatch):  # noqa: F811
    workspace = _workspace(cuda_fakes.calls)
    api = _FakeAPI(cuda_fakes.calls)
    _, retiring = _runner(workspace, api)
    retiring.prepare_workspace(workspace)
    other_workspace = _workspace(cuda_fakes.calls, base=8192)
    _, other = _runner(other_workspace, api)
    other.prepare_workspace(other_workspace)
    retiring.quarantine(RuntimeError("retiring transport failure"))
    drain_entered, release_drain, other_done = Event(), Event(), Event()
    errors = []

    def blocked_drain(device):
        drain_entered.set()
        if not release_drain.wait(5):
            raise RuntimeError("test did not release simulated device drain")

    def close_retiring():
        try:
            retiring.shutdown()
        except BaseException as error:
            errors.append(error)

    def quarantine_other():
        try:
            other.quarantine(RuntimeError("other transport failure"))
        except BaseException as error:
            errors.append(error)
        finally:
            other_done.set()

    monkeypatch.setattr(adapter.torch.cuda, "synchronize", blocked_drain)
    closing = Thread(target=close_retiring)
    quarantining = Thread(target=quarantine_other)
    closing.start()
    try:
        assert drain_entered.wait(5)
        quarantining.start()
        # A global lock held across GPU drain would prevent this completion.
        assert other_done.wait(5)
        assert retiring in adapter._QUARANTINED and other in adapter._QUARANTINED
    finally:
        release_drain.set()
        closing.join(5)
        if quarantining.ident is not None:
            quarantining.join(5)
    assert not closing.is_alive() and not quarantining.is_alive()
    assert not errors
    assert retiring._terminal_drained and retiring not in adapter._QUARANTINED
    assert other in adapter._QUARANTINED  # Removing A must not lose B's owner.
    retiring.quarantine(RuntimeError("late error after successful drain"))
    assert retiring not in adapter._QUARANTINED and other in adapter._QUARANTINED
    other.shutdown()
    assert other not in adapter._QUARANTINED
