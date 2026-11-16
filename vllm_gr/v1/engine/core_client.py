# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


async def submit_gr_session_async(self, batch_request: BeamBatchRequest) -> None:
    """Submit the whole GR session once; Decode stages stay inside EngineCore."""
    requests = tuple(item.request for item in batch_request.items)
    if not requests:
        raise ValueError("GR batch session must not be empty")
    for request in requests:
        request.client_index = self.client_index
        if hasattr(self, "current_wave"):
            request.current_wave = self.current_wave
    ensure_stats_task = getattr(self, "_ensure_stats_update_task", None)
    if ensure_stats_task is not None:
        ensure_stats_task()
    engine = None
    if hasattr(self, "get_core_engine_for_request"):
        engine = self.get_core_engine_for_request(requests[0])
    to_await = self._send_input(EngineCoreRequestType.SUBMIT_GR_SESSION, batch_request, engine)
    if hasattr(self, "first_req_send_socket") and not self.engines_running:
        import msgspec

        await self.first_req_send_socket.send(msgspec.msgpack.encode(("FIRST_REQ", engine)))
    await to_await
    self._ensure_output_queue_task()
