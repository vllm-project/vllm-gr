# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import uuid

import msgspec
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.engine.types import (
    BeamControllerCancelRequest,
    BeamControllerStartRequest,
    BeamConstraintResourceRegisterRequest,
    BeamRequestStepUpdate,
)

# ---------------------------------------------------------------------------
# AsyncMPClient.add_requests_async  (new method — ADD_BATCH path)
# ---------------------------------------------------------------------------


async def add_requests_async(self, requests, force_batch=False):
    """Send multiple requests to EngineCore.

    If single and not forced, delegate to add_request_async.
    Otherwise send as ADD_BATCH.
    """
    if not requests:
        return
    if len(requests) == 1 and not force_batch:
        return await self.add_request_async(requests[0])

    # DP path: ensure stats task is alive (lazy init for DPAsyncMPClient).
    if hasattr(self, "_ensure_stats_update_task"):
        self._ensure_stats_update_task()

    for request in requests:
        request.client_index = self.client_index
        if hasattr(self, "current_wave"):
            request.current_wave = self.current_wave

    # Route to the correct engine based on data_parallel_rank (DP case).
    # get_core_engine_for_request uses request.data_parallel_rank when set,
    # otherwise falls back to load-balancing. It also registers the first
    # request in reqs_in_flight; we register the rest manually.
    if hasattr(self, "get_core_engine_for_request"):
        chosen_engine = self.get_core_engine_for_request(requests[0])
        if hasattr(self, "reqs_in_flight"):
            for request in requests[1:]:
                self.reqs_in_flight[request.request_id] = chosen_engine
        to_await = self._send_input(EngineCoreRequestType.ADD_BATCH, requests, chosen_engine)
        # Notify coordinator when engines are idle (mirrors DPAsyncMPClient.add_request_async).
        if hasattr(self, "first_req_send_socket") and not self.engines_running:
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)
        await to_await
    else:
        await self._send_input(EngineCoreRequestType.ADD_BATCH, requests)
    self._ensure_output_queue_task()


# ---------------------------------------------------------------------------
# AsyncMPClient.beam_request_step_update_async  (BEAM_REQUEST_STEP_UPDATE path)
# ---------------------------------------------------------------------------


async def beam_request_step_update_async(self, update: BeamRequestStepUpdate) -> None:
    """Send BEAM_REQUEST_STEP_UPDATE to EngineCore.

    Transmits a single logical message for all branches of a decode step.
    """
    update.client_index = self.client_index
    if hasattr(self, "current_wave"):
        update.current_wave = self.current_wave

    engine = None
    if hasattr(self, "get_core_engine_for_request") and update.data_parallel_rank is not None:
        engine = self.core_engines[update.data_parallel_rank]
        if hasattr(self, "reqs_in_flight"):
            for child_id in update.child_beam_ids:
                self.reqs_in_flight[child_id] = engine
            for pruned_id in update.pruned_ids:
                self.reqs_in_flight.pop(pruned_id, None)

    to_await = self._send_input(EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE, update, engine)
    if hasattr(self, "first_req_send_socket") and not self.engines_running:
        req_msg = msgspec.msgpack.encode(("FIRST_REQ", engine))
        await self.first_req_send_socket.send(req_msg)
    await to_await
    self._ensure_output_queue_task()


async def run_beam_controller_async(
    self,
    start: BeamControllerStartRequest,
) -> dict:
    """Submit one logical beam session and await its terminal result."""
    call_id = uuid.uuid1().int >> 64
    start.call_id = call_id
    start.client_index = self.client_index
    start.engine_request.client_index = self.client_index
    if hasattr(self, "current_wave"):
        start.current_wave = self.current_wave
        start.engine_request.current_wave = self.current_wave

    future = asyncio.get_running_loop().create_future()
    self.utility_results[call_id] = future

    engine = None
    if hasattr(self, "get_core_engine_for_request"):
        engine = self.get_core_engine_for_request(start.engine_request)

    to_await = self._send_input(EngineCoreRequestType.BEAM_CONTROLLER_START, start, engine)
    if hasattr(self, "first_req_send_socket") and not self.engines_running:
        req_msg = msgspec.msgpack.encode(("FIRST_REQ", engine))
        await self.first_req_send_socket.send(req_msg)
    await to_await
    self._ensure_output_queue_task()
    try:
        return await future
    finally:
        if hasattr(self, "reqs_in_flight"):
            self.reqs_in_flight.pop(start.engine_request.request_id, None)


async def ensure_beam_constraint_resource_async(
    self,
    resource_id: str,
    constraint_resource: dict[str, object],
    *,
    data_parallel_rank: int | None = None,
    force: bool = False,
) -> dict:
    """Register a resource once per EngineCore generation/rank/digest.

    Concurrent callers share one registration task so a startup burst cannot
    duplicate a large resource payload.
    """
    current_wave = int(getattr(self, "current_wave", 0))
    key = (current_wave, data_parallel_rank, resource_id)
    registered = getattr(self, "_beam_registered_constraint_resources", None)
    if registered is None:
        registered = set()
        self._beam_registered_constraint_resources = registered
    if force:
        registered.discard(key)
    if key in registered:
        return {
            "resource_id": resource_id,
            "status": "already_registered",
            "error_code": None,
            "error": None,
        }

    tasks = getattr(self, "_beam_constraint_registration_tasks", None)
    if tasks is None:
        tasks = {}
        self._beam_constraint_registration_tasks = tasks
    task = tasks.get(key)
    if task is None:

        async def register() -> dict:
            engine = None
            core_engines = getattr(self, "core_engines", ())
            if len(core_engines) > 1:
                if data_parallel_rank is None:
                    raise ValueError(
                        "data_parallel_rank is required to register a resource "
                        "with a data-parallel EngineCore"
                    )
                engine = core_engines[data_parallel_rank]
            elif data_parallel_rank is not None and core_engines:
                engine = core_engines[data_parallel_rank]

            call_id = uuid.uuid1().int >> 64
            request = BeamConstraintResourceRegisterRequest(
                resource_id=resource_id,
                constraint_resource=constraint_resource,
                call_id=call_id,
                client_index=self.client_index,
                current_wave=current_wave,
                data_parallel_rank=data_parallel_rank,
            )
            future = asyncio.get_running_loop().create_future()
            self.utility_results[call_id] = future

            try:
                to_await = self._send_input(
                    EngineCoreRequestType.BEAM_CONSTRAINT_RESOURCE_REGISTER,
                    request,
                    engine,
                )
                if hasattr(self, "first_req_send_socket") and not self.engines_running:
                    req_msg = msgspec.msgpack.encode(("FIRST_REQ", engine))
                    await self.first_req_send_socket.send(req_msg)
                await to_await
                self._ensure_output_queue_task()
                result = await future
            except BaseException:
                pending = self.utility_results.pop(call_id, None)
                if pending is not None and not pending.done():
                    pending.cancel()
                raise
            if result.get("error") is not None:
                raise RuntimeError(
                    f"EngineCore constraint resource registration failed: {result['error']}"
                )
            if result.get("resource_id") != resource_id:
                raise RuntimeError("EngineCore acknowledged a different constraint resource")
            registered.add(key)
            return result

        task = asyncio.create_task(register())
        tasks[key] = task
    try:
        return await asyncio.shield(task)
    finally:
        if task.done() and tasks.get(key) is task:
            tasks.pop(key, None)


async def cancel_beam_controller_async(
    self,
    cancel: BeamControllerCancelRequest,
) -> None:
    cancel.client_index = self.client_index
    engine = None
    if hasattr(self, "core_engines") and cancel.data_parallel_rank is not None:
        engine = self.core_engines[cancel.data_parallel_rank]
    await self._send_input(EngineCoreRequestType.BEAM_CONTROLLER_CANCEL, cancel, engine)
