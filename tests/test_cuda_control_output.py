# SPDX-License-Identifier: Apache-2.0
"""CPU ownership tests using the real native Ticket completion primitive."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import pytest

from tests.test_cuda_completion import api_factory  # noqa: F401
from tests.test_gr_completion_build import built_completion  # noqa: F401
from vllm_gr.v1.worker.cuda_completion import CompletionDeviceError, CompletionRegistrationError
from vllm_gr.v1.worker.cuda_control_output import ControlSlotPool


@pytest.fixture
def pool_factory(api_factory):  # noqa: F811
    owned = []

    def create(**kwargs):
        api, pending, finish = api_factory(**kwargs)
        resources = (object(), object())
        pool = ControlSlotPool(api, resources)
        owned.append((pool, pending, finish))
        return pool, resources, pending, finish

    yield create
    for pool, pending, finish in owned:
        # CPU fake enqueues have no DMA. Finish all actual native callbacks
        # before the test-only no-op drain, including notification failures.
        while pending:
            finish()
        pool.shutdown(lambda: None)


def issue(pool, slot=1, epoch=1, step=1):
    lease = pool.acquire(slot, "request", epoch, step)
    resource = pool.begin_device_use(lease)
    pool.register_completion(lease, 0)
    return lease, resource


def test_device_completion_without_host_read_does_not_free_slot(pool_factory):
    pool, resources, _, finish = pool_factory()
    lease, resource = issue(pool)
    assert resource is resources[1]
    finish()
    with pytest.raises(RuntimeError, match="busy"):
        pool.acquire(1, "request", 1, 3)
    seen = []
    assert pool.consume(lease, lambda storage: seen.append(storage) or 42, timeout_ms=0) == 42
    assert seen == [resources[1]] and pool.occupied_slots == 0
    second, same_resource = issue(pool, step=3)
    assert same_resource is resource and second.generation > lease.generation


def test_abort_new_session_and_late_ack_cannot_overwrite_old_slot(pool_factory):
    pool, _, _, finish = pool_factory()
    old, _ = issue(pool)
    assert pool.discard(old)
    assert not pool.discard(old)
    with pytest.raises(RuntimeError, match="busy"):
        pool.acquire(1, "request", 2, 1)
    # The other slot can be reserved without blocking the EngineCore thread.
    other = pool.acquire(0, "request", 2, 2)
    finish()
    new = pool.acquire(1, "request", 2, 1)
    assert new.session_epoch == 2 and new.generation > old.generation
    for operation in (pool.discard, pool.begin_device_use):
        with pytest.raises(RuntimeError, match="stale"):
            operation(old)
    assert pool.occupied_slots == 2
    pool.discard(other)
    pool.discard(new)


def test_two_slots_are_bound_and_third_issue_never_waits(pool_factory):
    pool, _, _, _ = pool_factory()
    first, _ = issue(pool)
    second, _ = issue(pool, slot=0, step=2)
    with pytest.raises(RuntimeError, match="busy"):
        pool.acquire(1, "request", 1, 3)
    assert first.slot != second.slot and pool.occupied_slots == 2


def test_timeout_does_not_ack_and_retry_reads_completed_control(pool_factory):
    pool, resources, _, finish = pool_factory()
    lease, _ = issue(pool)
    with pytest.raises(TimeoutError):
        pool.consume(lease, lambda _: pytest.fail("Host read before device completion"), timeout_ms=0)
    assert pool.occupied_slots == 1
    finish()

    def read(storage):
        assert storage is resources[1]
        return 42

    assert pool.consume(lease, read, timeout_ms=0) == 42
    assert pool.occupied_slots == 0


def test_reader_error_requires_explicit_discard_or_retry(pool_factory):
    pool, _, _, finish = pool_factory()
    lease, _ = issue(pool)
    finish()

    def broken_reader(_):
        raise ValueError("decode Host control")

    with pytest.raises(ValueError, match="decode Host"):
        pool.consume(lease, broken_reader, timeout_ms=0)
    assert pool.occupied_slots == 1
    assert pool.discard(lease)
    assert pool.occupied_slots == 0


def test_late_other_slot_error_does_not_mask_host_reader_exception(pool_factory):
    pool, _, _, finish = pool_factory()
    first, _ = issue(pool)
    issue(pool, slot=0, step=2)
    finish()
    finish(status=719)

    def broken_reader(_):
        raise ValueError("original Host reader failure")

    with pytest.raises(ValueError, match="original Host"):
        pool.consume(first, broken_reader, timeout_ms=0)
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire(1, "request", 2, 1)
    assert pool.occupied_slots == 2


def test_host_read_ack_is_not_native_delivery_claim(pool_factory):
    pool, _, _, finish = pool_factory()
    lease, _ = issue(pool)
    finish()
    entered, release = threading.Event(), threading.Event()

    def read(_):
        entered.set()
        assert release.wait(5)
        return 17

    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(pool.consume, lease, read, timeout_ms=0)
        try:
            assert entered.wait(5)
            assert not pool.discard(lease)
            with pytest.raises(RuntimeError, match="busy"):
                pool.acquire(1, "request", 1, 3)
            with pytest.raises(RuntimeError, match="active Host reader"):
                pool.shutdown(lambda: pytest.fail("drain during Host read"))
        finally:
            release.set()
        assert task.result(timeout=5) == 17
    assert pool.occupied_slots == 0


def test_discard_before_enqueue_can_release_but_after_write_cannot(pool_factory):
    pool, _, _, finish = pool_factory()
    unused = pool.acquire(1, "request", 1, 1)
    assert pool.discard(unused) and pool.occupied_slots == 0
    orphan = pool.acquire(1, "request", 2, 1)
    pool.begin_device_use(orphan)
    assert pool.discard(orphan) and pool.occupied_slots == 1
    with pytest.raises(RuntimeError, match="busy"):
        pool.acquire(1, "request", 3, 1)
    pool.register_completion(orphan, 0)
    finish()
    replacement = pool.acquire(1, "request", 3, 1)
    assert replacement.generation > orphan.generation


@pytest.mark.parametrize("kind", ["registration", "device"])
def test_failed_completion_quarantines_both_slots(pool_factory, kind):
    pool, _, _, finish = pool_factory(failure=801 if kind == "registration" else 0)
    lease = pool.acquire(1, "request", 1, 1)
    pool.begin_device_use(lease)
    if kind == "registration":
        with pytest.raises(CompletionRegistrationError):
            pool.register_completion(lease, 0)
    else:
        pool.register_completion(lease, 0)
        finish(status=719)
        with pytest.raises(CompletionDeviceError):
            pool.consume(lease, lambda _: pytest.fail("read after CUDA error"), timeout_ms=0)
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire(0, "request", 2, 1)
    assert pool.occupied_slots == 1


def test_shutdown_failed_drain_retains_storage_and_blocks_admission(pool_factory):
    pool, _, pending, finish = pool_factory()
    lease, _ = issue(pool)

    def failed_drain():
        raise RuntimeError("device drain failed")

    with pytest.raises(RuntimeError, match="device drain failed"):
        pool.shutdown(failed_drain)
    assert pool.occupied_slots == 1 and len(pending) == 1
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire(0, "request", 2, 1)
    pool.shutdown(finish)
    assert pool.occupied_slots == 0 and not pending
    with pytest.raises(RuntimeError, match="stale"):
        pool.discard(lease)


def test_immediate_callback_and_duplicate_registration(pool_factory):
    pool, _, pending, _ = pool_factory(immediate=True)
    lease, _ = issue(pool)
    assert not pending
    with pytest.raises(RuntimeError, match="out of order"):
        pool.register_completion(lease, 0)
    assert pool.consume(lease, lambda _: 1, timeout_ms=0) == 1


def test_lease_identity_and_submission_order(pool_factory):
    pool, _, _, _ = pool_factory()
    lease = pool.acquire(1, "request", 1, 1)
    with pytest.raises(RuntimeError, match="foreign"):
        pool.begin_device_use(replace(lease))
    with pytest.raises(RuntimeError, match="out of order"):
        pool.register_completion(lease, 0)
    pool.begin_device_use(lease)
    with pytest.raises(RuntimeError, match="already started"):
        pool.begin_device_use(lease)
    pool.register_completion(lease, 0)


def test_shutdown_cannot_drain_before_active_producer_enqueues(pool_factory):
    pool, _, _, finish = pool_factory()
    lease = pool.acquire(1, "request", 1, 1)
    entered, proceed = threading.Event(), threading.Event()

    def submit():
        pool.begin_device_use(lease)
        entered.set()
        assert proceed.wait(5)
        pool.register_completion(lease, 0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(submit)
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="active device producer"):
                pool.shutdown(lambda: pytest.fail("drain before producer enqueues"))
            assert pool.discard(lease)  # No Host consumer; producer still owns DMA.
            assert pool.occupied_slots == 1
        finally:
            proceed.set()
        task.result(timeout=5)
    pool.shutdown(finish)
    assert pool.occupied_slots == 0


def test_failed_enqueue_must_close_producer_before_drain(pool_factory):
    pool, _, _, _ = pool_factory()
    lease = pool.acquire(1, "request", 1, 1)
    pool.begin_device_use(lease)
    error = RuntimeError("copy enqueue failed")
    pool.quarantine(error)
    with pytest.raises(RuntimeError, match="active device producer"):
        pool.shutdown(lambda: pytest.fail("producer has not stopped"))
    pool.fail_submission(lease, error)
    assert pool.occupied_slots == 1
    pool.shutdown(lambda: None)  # This CPU test never enqueued real DMA.
    assert pool.occupied_slots == 0


def test_successful_shutdown_is_idempotent(pool_factory):
    pool, _, _, finish = pool_factory()
    issue(pool)
    assert pool.shutdown(finish)
    assert not pool.shutdown(lambda: pytest.fail("duplicate drain"))
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire(1, "request", 2, 1)


def test_other_slot_quarantine_during_enqueue_does_not_strand_producer(pool_factory):
    pool, _, pending, _ = pool_factory()
    lease = pool.acquire(1, "request", 1, 1)
    pool.begin_device_use(lease)
    pool.quarantine(RuntimeError("other slot failed"))
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.register_completion(lease, 0)
    assert not pending and pool.occupied_slots == 1
    drained = []
    assert pool.shutdown(lambda: drained.append(True))
    assert drained == [True] and pool.occupied_slots == 0


@pytest.mark.parametrize("field,value", [("slot", True), ("slot", 2), ("request_id", ""),
                                         ("session_epoch", -1), ("device_step", 0)])
def test_invalid_identity_fails_before_ticket_allocation(pool_factory, field, value):
    pool, _, _, _ = pool_factory()
    values = dict(slot=1, request_id="request", session_epoch=1, device_step=1)
    values[field] = value
    with pytest.raises(ValueError):
        pool.acquire(**values)
    assert pool.occupied_slots == 0
