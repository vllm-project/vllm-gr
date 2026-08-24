# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.engine.types import BeamRequestStepUpdate

_MAX_BEAM_BATCH_ENGINE_MAPPINGS = 4096

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


# ---------------------------------------------------------------------------
# Beam Batch submit/abort (BEAM_BATCH_SUBMIT / BEAM_BATCH_ABORT)
# ---------------------------------------------------------------------------


async def submit_beam_batch_async(self, request):
    """Send a single immutable BeamBatchRequest to EngineCore."""
    engine = None
    if hasattr(self, "get_core_engine_for_request") and request.items:
        # Preserve data-parallel affinity: anchor on the first child's rank and
        # retain the chosen engine for abort/result routing.
        anchor = request.items[0].request
        engine = self.get_core_engine_for_request(anchor)
        engines = getattr(self, "_beam_batch_engines", None)
        if engines is None:
            from collections import OrderedDict

            engines = OrderedDict()
            self._beam_batch_engines = engines
        engines[request.batch_id] = engine
        engines.move_to_end(request.batch_id)
        # Normal-completion cleanup is owned by the result adapter (#41), so
        # bound the mapping: evicting the oldest entry only drops abort routing
        # for a long-idle batch (mirrors the ingress tombstone cap).
        while len(engines) > _MAX_BEAM_BATCH_ENGINE_MAPPINGS:
            engines.popitem(last=False)
        if hasattr(self, "first_req_send_socket") and not self.engines_running:
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", engine))
            await self.first_req_send_socket.send(req_msg)
    await self._send_input(EngineCoreRequestType.BEAM_BATCH_SUBMIT, request, engine)
    self._ensure_output_queue_task()
    return request.batch_id


async def abort_beam_batch_async(self, batch_id: str) -> None:
    """Send a Beam Batch abort to EngineCore."""
    engine = None
    engines = getattr(self, "_beam_batch_engines", None)
    if engines:
        engine = engines.pop(batch_id, None)
    await self._send_input(EngineCoreRequestType.BEAM_BATCH_ABORT, batch_id, engine)
    self._ensure_output_queue_task()


def submit_beam_batch(self, request):
    """Send a single immutable BeamBatchRequest to EngineCore.

    Returns a local :class:`BeamBatchHandle` so the offline entry can block on
    the final result; the handle's future is resolved when the result adapter
    routes the terminal :class:`BeamBatchResult` back (issue #41).
    """
    self._send_input(EngineCoreRequestType.BEAM_BATCH_SUBMIT, request)
    from concurrent.futures import Future

    from vllm_gr.v1.engine.beam_batch import BeamBatchHandle, BeamBatchRecord

    record = BeamBatchRecord(request=request, future=Future())
    return BeamBatchHandle(batch_id=request.batch_id, record=record)


def abort_beam_batch(self, batch_id: str) -> None:
    """Send a Beam Batch abort to EngineCore."""
    self._send_input(EngineCoreRequestType.BEAM_BATCH_ABORT, batch_id)


def inproc_submit_beam_batch(self, request):
    """Submit a BeamBatchRequest directly to the in-process ingress.

    Delegates to the EngineCore handler so the in-process and wire paths share
    one implementation; the returned handle lets callers block on the final
    result.
    """
    from vllm_gr.v1.engine.core import _handle_beam_batch_submit

    return _handle_beam_batch_submit(self.engine_core, request)


def inproc_abort_beam_batch(self, batch_id: str) -> None:
    """Abort a BeamBatchRequest directly through the in-process ingress."""
    ingress = getattr(self.engine_core, "_beam_batch_ingress", None)
    if ingress is None:
        raise RuntimeError("EngineCore beam batch ingress is not initialized")
    record = ingress.get_record(batch_id)
    if record is not None:
        ingress.cancel(batch_id)
