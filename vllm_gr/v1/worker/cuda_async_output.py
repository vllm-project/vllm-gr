# SPDX-License-Identifier: Apache-2.0
"""Thin GPU ModelRunner V1 fixed-control transport adapter.

Keep the upstream class and Future/BatchQueue contracts. Only an explicitly
scoped persistent sampling stage intercepts its output constructor. Normal
requests and partial Prefill delegate to the original implementation.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Any

import torch

from vllm_gr.v1.worker.cuda_control_output import ControlSlotBusy, ControlSlotPool

_FRAME: ContextVar[Any] = ContextVar("gr_cuda_output_frame", default=None)
_PATCH_MARKER = "_gr_fixed_control_output_adapter"
# Error quarantine must survive output/state GC until a successful drain or
# Worker/context teardown. This is not a queue and never advances generation.
_QUARANTINED: list[Any] = []
# Shared ownership spans transports. Hold this only for Host bookkeeping,
# never while draining the device or waiting for an individual pool lock.
_QUARANTINE_LOCK = Lock()


@dataclass(slots=True)
class _Storage:
    device: Any
    host: Any
    ready: Any


def persistent_output_request(scheduler_output):
    stages = [(req_id, data) for req_id, data in
              (getattr(scheduler_output, "beam_data", {}) or {}).items()
              if data.get("persistent_runner_state")]
    if not stages:
        return None
    if len(stages) != 1:
        raise RuntimeError("persistent CUDA control output requires one request")
    req_id, data = stages[0]
    if data.get("is_beam_decode") or (
        data.get("is_constraint_prefill") and data.get("is_last_prefill")
    ):
        return req_id
    return None


def _check_host_consumers(runner):
    batch = getattr(runner.input_batch, "_obj", runner.input_batch)
    if batch.sampling_metadata.output_token_ids:
        raise RuntimeError("persistent CUDA control does not support Host output-token history consumers")
    if getattr(runner, "routed_experts_initialized", False):
        raise RuntimeError("persistent CUDA control does not support routed expert output")


class CudaOutputTransport:
    def __init__(self, runner, api):
        self.runner = runner
        self._device = runner.device
        self.api = api
        self._pools = {}
        self._failure = None
        self._closed = False
        self._terminal_drained = False
        self._notification_stream = None
        self._epoch = 0
        self.copies = 0
        self.consumed = 0
        self.high_watermark = 0

    def check_live(self):
        if self._failure is not None:
            raise RuntimeError("persistent CUDA output is quarantined; Worker cannot execute") from self._failure
        if self._closed:
            raise RuntimeError("persistent CUDA output transport is closed")

    def prepare_workspace(self, workspace):
        self.check_live()
        key = id(workspace)
        if key in self._pools:
            return self._pools[key][1]
        devices = (workspace.gr_control_output_0, workspace.gr_control_output_1)
        for device in devices:
            if (device.device.type != "cuda" or device.dtype != torch.int32
                    or tuple(device.shape) != (1, 1) or not device.is_contiguous()):
                raise RuntimeError("persistent CUDA control requires contiguous int32 [1,1] slots")
        if devices[0].device != devices[1].device:
            raise RuntimeError("control slots must share the runner device")
        if devices[0].device != self._device or self.runner.device != self._device:
            raise RuntimeError("control workspace device does not match the runner device")
        if abs(devices[0].data_ptr() - devices[1].data_ptr()) < 4:
            raise RuntimeError("control device slots overlap")
        storages = tuple(_Storage(
            device, torch.empty((1, 1), dtype=torch.int32, device="cpu", pin_memory=True),
            torch.cuda.Event(),
        ) for device in devices)
        if abs(storages[0].host.data_ptr() - storages[1].host.data_ptr()) < 4:
            raise RuntimeError("control Host slots overlap")
        if self._notification_stream is None:
            self._notification_stream = torch.cuda.Stream(device=devices[0].device)
        pool = ControlSlotPool(self.api, storages)
        # Also retain the complete workspace on error, not only control slots.
        self._pools[key] = (workspace, pool)
        return pool

    def reserve(self, frame, workspace, device_step):
        self.check_live()
        state = getattr(self.runner, "_gr_model_runner_states", {}).get(frame.request_id)
        if state is None or state.device_step + 1 != device_step:
            raise RuntimeError("control request/device generation mismatch")
        backend_state = state.backend_state
        if backend_state.workspace is not workspace:
            raise RuntimeError("control workspace does not belong to the active request")
        pool = self.prepare_workspace(workspace)
        if backend_state.control_epoch is None:
            self._epoch += 1
            backend_state.control_epoch = self._epoch
        epoch = backend_state.control_epoch
        phase = backend_state.control_phase
        slots = (device_step % 2, 1 - device_step % 2) if phase is None else ((device_step % 2) ^ phase,)
        for slot in slots:
            try:
                lease = pool.acquire(slot, frame.request_id, epoch, device_step)
            except ControlSlotBusy:
                if slot == slots[-1]:
                    raise
                continue
            break
        if phase is None:
            backend_state.control_phase = slot ^ (device_step % 2)
        frame.pool, frame.lease = pool, lease
        frame.storage = pool.begin_device_use(lease)
        frame.producer_active = True
        self.high_watermark = max(self.high_watermark, pool.occupied_slots)
        return frame.storage.device

    def quarantine(self, error):
        with _QUARANTINE_LOCK:
            # Closed is not sufficient: a failed drain also closes admission
            # and MUST retain resources. Only successful terminal drain makes
            # a late delivery error incapable of resurrecting ownership.
            if self._terminal_drained:
                return
            if self._failure is None:
                self._failure = error
                _QUARANTINED.append(self)
            pools = tuple(self._pools.values())
        for _, pool in pools:
            pool.quarantine(error)

    def shutdown(self):
        # GPU adapter invokes this before the common shutdown returns Beam KV.
        with _QUARANTINE_LOCK:
            if self._terminal_drained:
                return
            self._closed = True
            pools = tuple(self._pools.values())
        try:
            for _, pool in pools:
                pool.shutdown(lambda: torch.cuda.synchronize(self._device))
        except BaseException as error:
            self.quarantine(error)
            raise
        with _QUARANTINE_LOCK:
            self._terminal_drained = True
            self._pools.clear()
            _QUARANTINED[:] = [transport for transport in _QUARANTINED if transport is not self]

    def stats(self):
        return {
            "control_d2h_count": self.copies,
            "control_d2h_bytes": self.copies * 4,
            "control_consumed": self.consumed,
            "control_slots_high_watermark": self.high_watermark,
            "control_slots_occupied": sum(pool.occupied_slots for _, pool in self._pools.values()),
            "control_pinned_bytes": len(self._pools) * 8,
        }


@dataclass(slots=True)
class _SamplingFrame:
    runner: Any
    request_id: str
    transport: CudaOutputTransport
    pool: Any = None
    lease: Any = None
    storage: Any = None
    producer_active: bool = False
    constructed: bool = False

    def fail(self, error):
        self.transport.quarantine(error)
        if self.lease is not None:
            if self.producer_active:
                self.pool.fail_submission(self.lease, error)
                self.producer_active = False
            try:
                self.pool.discard(self.lease)
            except RuntimeError:
                # consume may have reaped this slot before discovering an
                # error in the other slot. Never mask the original failure.
                pass


@contextmanager
def sampling_scope(runner, scheduler_output):
    request_id = persistent_output_request(scheduler_output)
    if request_id is None:
        yield None
        return
    if _FRAME.get() is not None:
        raise RuntimeError("persistent CUDA sampling cannot recurse")
    transport = getattr(runner, "_gr_cuda_output_transport", None)
    if transport is None or not getattr(runner, "use_async_scheduling", False):
        raise RuntimeError("persistent CUDA output transport was not initialized for ASYNC")
    transport.check_live()
    _check_host_consumers(runner)
    frame = _SamplingFrame(runner, request_id, transport)
    token = _FRAME.set(frame)
    try:
        yield frame
        if frame.lease is None or not frame.constructed:
            raise RuntimeError("persistent CUDA stage did not publish one control output")
    except BaseException as error:
        frame.fail(error)
        raise
    finally:
        _FRAME.reset(token)


def fill_control(workspace, device_step, token_id):
    frame = _FRAME.get()
    if frame is None or frame.lease is not None:
        raise RuntimeError("control write requires one active persistent sampling lease")
    output = frame.transport.reserve(frame, workspace, device_step)
    # The pool acquired storage BEFORE this write, including across sessions.
    output.fill_(token_id)
    return output


def install_async_output_adapter(cls):
    if getattr(cls, _PATCH_MARKER, False):
        return
    original_init, original_get = cls.__init__, cls.get_output

    @wraps(original_init)
    def initialize(self, model_runner_output, sampled_token_ids, logprobs_tensors,
                   invalid_req_indices, async_output_copy_stream, vocab_size,
                   routed_experts=None):
        frame = _FRAME.get()
        if frame is None:
            return original_init(self, model_runner_output, sampled_token_ids, logprobs_tensors,
                                 invalid_req_indices, async_output_copy_stream, vocab_size,
                                 routed_experts=routed_experts)
        _check_host_consumers(frame.runner)
        if frame.lease is None or frame.constructed or sampled_token_ids is not frame.storage.device:
            raise RuntimeError("control constructor does not own the sampled device slot")
        if (model_runner_output.req_ids != [frame.request_id]
                or model_runner_output.req_id_to_index != {frame.request_id: 0}
                or logprobs_tensors is not None or routed_experts is not None
                or any(type(i) is not int or i != 0 for i in invalid_req_indices)):
            raise RuntimeError("persistent CUDA control output ABI mismatch")
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = tuple(invalid_req_indices)
        self.vocab_size = vocab_size
        self._sampled_token_ids = sampled_token_ids
        self.sampled_token_ids_cpu = frame.storage.host
        self.async_copy_ready_event = frame.storage.ready  # A real upstream-compatible Event.
        if (frame.runner.device != frame.storage.device.device
                or async_output_copy_stream.device != frame.storage.device.device
                or frame.transport._notification_stream.device != frame.storage.device.device):
            raise RuntimeError("control copy/notification stream device does not match the runner device")
        producer_stream = torch.cuda.current_stream(sampled_token_ids.device)
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(producer_stream)
            self.sampled_token_ids_cpu.copy_(sampled_token_ids, non_blocking=True)
            self.async_copy_ready_event.record()
        notify = frame.transport._notification_stream
        # Both operations target an explicit stream. wait_event applies its
        # own device guard; switching the caller's current stream is needless.
        notify.wait_event(self.async_copy_ready_event)
        try:
            frame.pool.register_completion(frame.lease, notify.cuda_stream)
        finally:
            frame.producer_active = False
        frame.constructed = True
        frame.transport.copies += 1
        self._gr_control_frame = frame
        self._gr_control_consumed = False

    @wraps(original_get)
    def get_output(self):
        frame = getattr(self, "_gr_control_frame", None)
        if frame is None:
            return original_get(self)
        if self._gr_control_consumed:
            return self._model_runner_output

        def read(storage):
            control = ctypes.c_int32.from_address(storage.host.data_ptr()).value
            return [[]] if self._invalid_req_indices else [[control]]

        try:
            values = frame.pool.consume(frame.lease, read, timeout_ms=None)
        except BaseException as error:
            frame.fail(error)
            raise
        output = self._model_runner_output
        output.sampled_token_ids = values
        output.logprobs = None
        self._gr_control_consumed = True
        del self._sampled_token_ids
        frame.transport.consumed += 1
        return output

    cls.__init__, cls.get_output = initialize, get_output
    setattr(cls, _PATCH_MARKER, True)
