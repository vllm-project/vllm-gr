# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.engine.types import BeamRequestStepUpdate
from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


async def submit_gr_session_async(self, batch_request: BeamBatchRequest) -> None:
    """Submit the whole GR session once; Decode stages stay inside EngineCore."""
    request = batch_request.items[0].request
    request.client_index = self.client_index
    if hasattr(self, "current_wave"):
        request.current_wave = self.current_wave
    engine = None
    if hasattr(self, "get_core_engine_for_request"):
        engine = self.get_core_engine_for_request(request)
    to_await = self._send_input(
        EngineCoreRequestType.SUBMIT_GR_SESSION, batch_request, engine
    )
    if hasattr(self, "first_req_send_socket") and not self.engines_running:
        import msgspec

        await self.first_req_send_socket.send(msgspec.msgpack.encode(("FIRST_REQ", engine)))
    await to_await
    self._ensure_output_queue_task()

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
# AsyncMPClient.beam_session_template_async  (BEAM_SESSION_TEMPLATE path)
# ---------------------------------------------------------------------------


async def beam_session_template_async(self, template: BeamRequestStepUpdate) -> None:
    """Send BEAM_SESSION_TEMPLATE to EngineCore (fire-and-forget).

    Caches the session's step-update template on the EngineCore so its
    engine-driven prefill chain can construct the step-0 update in-process.
    Sent right after the session's ADD_BATCH on the same socket (FIFO
    guarantees it lands in beam_cache before the prefill output iteration).
    No reqs_in_flight bookkeeping: the template carries no live child ids.
    """
    template.client_index = self.client_index
    if hasattr(self, "current_wave"):
        template.current_wave = self.current_wave

    engine = None
    if hasattr(self, "get_core_engine_for_request") and template.data_parallel_rank is not None:
        engine = self.core_engines[template.data_parallel_rank]

    await self._send_input(EngineCoreRequestType.BEAM_SESSION_TEMPLATE, template, engine)
    self._ensure_output_queue_task()


# ---------------------------------------------------------------------------
# AsyncMPClient.submit_session_async  (SUBMIT_SESSION path)
# ---------------------------------------------------------------------------


async def submit_session_async(self, requests, template: BeamRequestStepUpdate) -> None:
    """Send SUBMIT_SESSION to EngineCore (ADD_BATCH + template in one message).

    Combines the session's prefill batch submission with its step template:
    the EngineCore handler runs the ADD_BATCH handling per request and then
    caches the template, so the template's arrival is atomic with the
    prefill submission (no FIFO reliance across two messages).  Routing and
    reqs_in_flight bookkeeping mirror add_requests_async; the template rides
    the same engine as the requests.
    """
    if not requests:
        return

    # DP path: ensure stats task is alive (lazy init for DPAsyncMPClient).
    if hasattr(self, "_ensure_stats_update_task"):
        self._ensure_stats_update_task()

    for request in requests:
        request.client_index = self.client_index
        if hasattr(self, "current_wave"):
            request.current_wave = self.current_wave
    template.client_index = self.client_index
    if hasattr(self, "current_wave"):
        template.current_wave = self.current_wave

    engine = None
    if hasattr(self, "get_core_engine_for_request"):
        engine = self.get_core_engine_for_request(requests[0])
        if hasattr(self, "reqs_in_flight"):
            for request in requests[1:]:
                self.reqs_in_flight[request.request_id] = engine
        to_await = self._send_input(
            EngineCoreRequestType.SUBMIT_SESSION, (requests, template), engine
        )
        # Notify coordinator when engines are idle (mirrors add_requests_async).
        if hasattr(self, "first_req_send_socket") and not self.engines_running:
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", engine))
            await self.first_req_send_socket.send(req_msg)
        await to_await
    else:
        await self._send_input(EngineCoreRequestType.SUBMIT_SESSION, (requests, template))
    self._ensure_output_queue_task()
