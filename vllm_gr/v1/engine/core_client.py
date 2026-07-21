# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.engine.types import BeamRequestStepUpdate


def _get_beam_session_engines(client) -> dict[str, object]:
    """Return the client-local session affinity table."""
    sessions = client.__dict__.get("_vllm_gr_beam_session_engines")
    if sessions is None:
        sessions = {}
        client._vllm_gr_beam_session_engines = sessions
    return sessions


def _validate_batch_dp_ranks(requests) -> None:
    ranks = {getattr(request, "data_parallel_rank", None) for request in requests}
    if len(ranks) > 1:
        raise ValueError(
            "ADD_BATCH requests must target the same data_parallel_rank; "
            f"got {sorted(ranks, key=lambda rank: -1 if rank is None else rank)}"
        )


def _get_engine_for_rank(client, data_parallel_rank: int):
    if isinstance(data_parallel_rank, bool) or not isinstance(data_parallel_rank, int):
        raise ValueError(f"data_parallel_rank must be an integer, got {data_parallel_rank!r}")
    if data_parallel_rank < 0 or data_parallel_rank >= len(client.core_engines):
        raise ValueError(
            f"data_parallel_rank={data_parallel_rank} is outside the managed "
            f"engine range [0, {len(client.core_engines)})"
        )
    return client.core_engines[data_parallel_rank]


# ---------------------------------------------------------------------------
# AsyncMPClient.add_requests_async  (new method — ADD_BATCH path)
# ---------------------------------------------------------------------------


async def add_requests_async(self, requests, force_batch=False):
    """Send multiple requests to EngineCore.

    Normal requests use the upstream ADD path. ``force_batch`` selects the
    GR ADD_BATCH path, which also establishes the prefill cache and beam
    session affinity.
    """
    if not requests:
        return
    if not force_batch:
        for request in requests:
            await self.add_request_async(request)
        return

    _validate_batch_dp_ranks(requests)

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
        session_engines = _get_beam_session_engines(self)
        for request in requests:
            session_engines[request.request_id] = chosen_engine
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
    session_engines = _get_beam_session_engines(self)
    mapped_engine = session_engines.get(update.session_id)
    if hasattr(self, "get_core_engine_for_request"):
        if update.data_parallel_rank is not None:
            rank_engine = _get_engine_for_rank(self, update.data_parallel_rank)
            if mapped_engine is not None and mapped_engine != rank_engine:
                raise ValueError(
                    f"Beam session {update.session_id!r} is pinned to {mapped_engine!r}, "
                    f"not data_parallel_rank={update.data_parallel_rank}"
                )
            engine = rank_engine
        elif mapped_engine is not None:
            engine = mapped_engine
        elif len(getattr(self, "core_engines", ())) > 1:
            raise ValueError(
                f"Beam session {update.session_id!r} has no engine affinity and "
                "data_parallel_rank was not provided"
            )

        if engine is not None:
            session_engines[update.session_id] = engine

        if hasattr(self, "reqs_in_flight"):
            # The EngineCore executes and finishes a single logical request
            # whose ID is session_id. Child IDs are frontend beam lineage and
            # must not be registered as EngineCore requests.
            if update.beam_width > 0:
                self.reqs_in_flight[update.session_id] = engine
            for pruned_id in update.pruned_ids:
                self.reqs_in_flight.pop(pruned_id, None)

        # The first update identifies which cached prefill request is the
        # persistent session; discard affinity candidates for sibling beams.
        for parent_id in update.parent_beam_ids:
            if parent_id != update.session_id:
                session_engines.pop(parent_id, None)

    to_await = self._send_input(EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE, update, engine)
    if hasattr(self, "first_req_send_socket") and not self.engines_running:
        req_msg = msgspec.msgpack.encode(("FIRST_REQ", engine))
        await self.first_req_send_socket.send(req_msg)
    await to_await
    if update.beam_width == 0:
        session_engines.pop(update.session_id, None)
        if hasattr(self, "reqs_in_flight"):
            self.reqs_in_flight.pop(update.session_id, None)
    self._ensure_output_queue_task()
