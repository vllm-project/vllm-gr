# SPDX-License-Identifier: Apache-2.0
"""Real CUDA control-slot transport tests, not ModelRunner integration tests."""

import ctypes
from types import SimpleNamespace

import pytest
import torch

from tests.test_gr_completion_build import built_completion  # noqa: F401
from vllm_gr.v1.worker.cuda_completion import NativeCompletionAPI
from vllm_gr.v1.worker.cuda_control_output import ControlSlotPool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA")


@pytest.fixture
def transport(built_completion):  # noqa: F811
    api = NativeCompletionAPI(ctypes.CDLL(str(built_completion)), ctypes.CDLL(torch._C.__file__))
    slots = tuple(SimpleNamespace(
        device=torch.empty((1, 1), dtype=torch.int32, device="cuda"),
        host=torch.empty((1, 1), dtype=torch.int32, pin_memory=True),
        ready=torch.cuda.Event(), produced=torch.cuda.Event(),
    ) for _ in range(2))
    assert slots[0].device.data_ptr() != slots[1].device.data_ptr()
    assert slots[0].host.data_ptr() != slots[1].host.data_ptr()
    pool = ControlSlotPool(api, slots)
    streams = (torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream())
    torch.cuda.synchronize()  # Fixture setup only, before the tested transport.
    try:
        yield pool, slots, streams
    finally:
        pool.shutdown(torch.cuda.synchronize)  # Terminal resource drain only.
        assert api.lib.gr_ticket_live() == 0


def enqueue(pool, slots, streams, stage, *, epoch=1, delay=False):
    lease = pool.acquire(stage % 2, "request", epoch, stage + 1)
    storage = pool.begin_device_use(lease)
    compute, copy, notify = streams
    try:
        with torch.cuda.stream(compute):
            if delay:
                torch.cuda._sleep(300_000_000)
            storage.device.fill_(stage + 31)
            storage.produced.record()
        with torch.cuda.stream(copy):
            copy.wait_event(storage.produced)
            storage.host.copy_(storage.device, non_blocking=True)
            storage.ready.record()
        with torch.cuda.stream(notify):
            notify.wait_event(storage.ready)
    except BaseException as error:
        pool.fail_submission(lease, error)
        raise
    pool.register_completion(lease, notify.cuda_stream)
    assert storage is slots[stage % 2]
    return lease


def read_control(storage):
    # A detached actual Host control; no Tensor.item/cpu/tolist or fake token.
    return ctypes.c_int32.from_address(storage.host.data_ptr()).value


def test_native_notification_reuses_exactly_two_fixed_cuda_control_slots(transport):
    pool, slots, streams = transport
    addresses = [(slot.device.data_ptr(), slot.host.data_ptr()) for slot in slots]
    pending = []
    for stage in range(8):
        if len(pending) == 2:
            previous, lease = pending.pop(0)
            assert pool.consume(lease, read_control, timeout_ms=10000) == previous + 31
        lease = enqueue(pool, slots, streams, stage)
        pending.append((stage, lease))
        assert pool.occupied_slots <= 2
    for stage, lease in pending:
        assert pool.consume(lease, read_control, timeout_ms=10000) == stage + 31
    assert pool.occupied_slots == 0
    assert addresses == [(slot.device.data_ptr(), slot.host.data_ptr()) for slot in slots]


def test_cancelled_cuda_control_retains_slot_through_late_dma(transport):
    pool, slots, streams = transport
    old = enqueue(pool, slots, streams, 0, delay=True)
    with pytest.raises(TimeoutError):
        pool.consume(old, read_control, timeout_ms=0)
    assert pool.discard(old)
    with pytest.raises(RuntimeError, match="busy"):
        pool.acquire(0, "request", 2, 1)
    # This is an explicit late-completion test boundary, not a runtime wait.
    torch.cuda.synchronize()
    new = enqueue(pool, slots, streams, 0, epoch=2)
    assert new.generation > old.generation
    with pytest.raises(RuntimeError, match="stale"):
        pool.discard(old)
    assert pool.consume(new, read_control, timeout_ms=10000) == 31
    assert pool.occupied_slots == 0


@pytest.mark.parametrize("release_error", [False, True])
def test_real_cuda_control_survives_uniproc_abort_and_shutdown(
    built_completion, release_error,  # noqa: F811
):
    """Real CUDA/adapter/RPC boundary, with fake request state and Worker.

    This does not load a model or validate full Beam KV/session release.
    """
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import AsyncOutputFuture, UniProcExecutor
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    from vllm_gr.v1.engine.engine_core_patch import make_patched_abort_requests
    from vllm_gr.v1.worker.cuda_async_output import (
        CudaOutputTransport, fill_control, install_async_output_adapter, sampling_scope,
    )
    from vllm_gr.v1.worker.gpu_model_runner_patch import make_patched_shutdown

    device = torch.device("cuda", torch.cuda.current_device())
    workspace = SimpleNamespace(
        gr_control_output_0=torch.empty((1, 1), dtype=torch.int32, device=device),
        gr_control_output_1=torch.empty((1, 1), dtype=torch.int32, device=device),
    )
    api = NativeCompletionAPI(
        ctypes.CDLL(str(built_completion)), ctypes.CDLL(torch._C.__file__),
    )
    runner = SimpleNamespace(
        device=device, use_async_scheduling=True, routed_experts_initialized=False,
        input_batch=SimpleNamespace(
            sampling_metadata=SimpleNamespace(output_token_ids=[]),
        ),
        _gr_model_runner_states={},
    )
    output_transport = CudaOutputTransport(runner, api)
    runner._gr_cuda_output_transport = output_transport
    pool = output_transport.prepare_workspace(workspace)
    copy_stream = torch.cuda.Stream(device=device)
    install_async_output_adapter(AsyncGPUModelRunnerOutput)
    calls = []

    def publish(request_id, token_id):
        runner._gr_model_runner_states[request_id] = SimpleNamespace(
            device_step=0,
            backend_state=SimpleNamespace(
                workspace=workspace, control_epoch=None, control_phase=None,
            ),
        )
        scheduled = SimpleNamespace(beam_data={request_id: {
            "persistent_runner_state": True, "is_beam_decode": True,
        }})
        with sampling_scope(runner, scheduled):
            torch.cuda._sleep(100_000_000)
            sampled = fill_control(workspace, 1, token_id)
            output = AsyncGPUModelRunnerOutput(
                ModelRunnerOutput(req_ids=[request_id], req_id_to_index={request_id: 0}),
                sampled, None, [], copy_stream, 32000,
            )
        return AsyncOutputFuture(output, single_value=True)

    failure = RuntimeError("injected Worker release failure")

    def release(request_ids):
        calls.append(("release", tuple(request_ids)))
        if release_error:
            raise failure
        for request_id in request_ids:
            runner._gr_model_runner_states.pop(request_id, None)
        return len(request_ids)

    def common_shutdown(current_runner):
        # The actual GPU shutdown adapter must drain before common cleanup.
        assert pool.occupied_slots == 0 and api.lib.gr_ticket_live() == 0
        current_runner._gr_model_runner_states.clear()
        calls.append("common_shutdown")

    shutdown = make_patched_shutdown(common_shutdown)
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor.driver_worker = SimpleNamespace(
        release_gr_model_runner_states=release,
        shutdown=lambda: shutdown(runner),
    )
    requests = {"old": SimpleNamespace(gr_persistent_runner_state=True)}

    def finish(request_ids, status):
        for request_id in request_ids:
            requests.pop(request_id, None)

    engine = SimpleNamespace(
        scheduler=SimpleNamespace(requests=requests, finish_requests=finish),
        model_executor=executor,
    )
    abort = make_patched_abort_requests(EngineCore.abort_requests)
    try:
        old = publish("old", 7)
        assert pool.occupied_slots == 1 and not old.done()
        if release_error:
            with pytest.raises(RuntimeError, match="injected Worker release failure") as caught:
                abort(engine, ["old"])
            assert caught.value is failure
            for action in (
                lambda: engine.step_fn(),
                lambda: executor.collective_rpc("release_gr_model_runner_states", args=(["old"],)),
            ):
                with pytest.raises(RuntimeError, match="shutdown required"):
                    action()
            assert pool.occupied_slots == 1 and not old.done()
        else:
            abort(engine, ["old"])
            # Release acknowledgement does not consume the old control Future.
            assert not requests and not runner._gr_model_runner_states
            assert pool.occupied_slots == 1 and output_transport.consumed == 0
            new = publish("new", 11)
            assert pool.occupied_slots == 2
            assert old.result().sampled_token_ids == [[7]]
            assert new.result().sampled_token_ids == [[11]]
            assert pool.occupied_slots == 0 and output_transport.consumed == 2
    finally:
        # Actual UniProc teardown still reaches the GPU adapter after retirement.
        executor.shutdown()
    assert calls == [("release", ("old",)), "common_shutdown"]
    assert output_transport.stats()["control_slots_occupied"] == 0
    if release_error:
        from vllm_gr.v1.worker import cuda_async_output as adapter

        # Retain the real native Future past the actual CUDA drain. It must
        # fail before reading the old Host slot, without restoring its owner.
        with pytest.raises(RuntimeError, match="control pool is closed"):
            old.result()
        assert output_transport.consumed == 0
        assert output_transport not in adapter._QUARANTINED
        assert api.lib.gr_ticket_live() == 0
