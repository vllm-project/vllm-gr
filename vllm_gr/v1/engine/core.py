# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)

logger = init_logger(__name__)

_BEAM_STATE_INIT_LOCK = threading.Lock()


def _ensure_beam_state(self) -> None:
    """Initialize per-engine beam state, including for late plugin loading."""
    if (
        getattr(self, "beam_cache", None) is not None
        and getattr(self, "beam_cache_lock", None) is not None
        and getattr(self, "_gr_beam_sessions_lock", None) is not None
    ):
        return

    # General plugins may first load from inside EngineCore.__init__. In that
    # case, wrapping the class cannot affect the constructor already in flight,
    # so the request handlers also need a thread-safe lazy fallback.
    with _BEAM_STATE_INIT_LOCK:
        if getattr(self, "beam_cache", None) is None:
            self.beam_cache: dict[str, dict] = {}
        if getattr(self, "beam_cache_lock", None) is None:
            self.beam_cache_lock = threading.Lock()
        if getattr(self, "_gr_beam_sessions_lock", None) is None:
            # Serializes the beam-session map: the input thread mutates it
            # (BEAM_SEARCH / BEAM_SEARCH_ABORT) while the step loop snapshots
            # and pops from it in _route_beam_session_outputs.
            self._gr_beam_sessions_lock = threading.Lock()


# ------------------------------------------------------------------------------------
# _cache_beam_request / _handle_beam_request_step_update  (ported from source core.py)
# ------------------------------------------------------------------------------------


def _cache_beam_request(self, request) -> None:
    """Cache Request state for later beam forking.

    Thread-safe: uses beam_cache_lock to prevent data races.

    When the prompt length is not a multiple of the KV-cache block size,
    the last partial block would normally remain unhashed and therefore
    uncacheable.  We temporarily extend ``request._all_token_ids`` in-place
    to the next block boundary so that ``update_block_hashes()`` (which reads
    via ConstantList wrapper ``request.all_token_ids``) sees a complete last
    block and produces its hash.  We then revert ``request._all_token_ids``
    in-place so prefill execution runs strictly on original tokens.
    """
    _ensure_beam_state(self)

    # Make a clean, un-mutable copy of the original prompt token IDs
    raw_token_ids = list(request._all_token_ids)
    full_block_hashes = list(request.block_hashes)
    block_hasher = getattr(self, "request_block_hasher", None)
    vllm_config = getattr(self, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None) if vllm_config else None
    block_size = getattr(cache_config, "block_size", 16) if cache_config else 16
    pad_len = (-len(raw_token_ids)) % block_size

    partial_block_hash = None
    if pad_len > 0 and block_hasher is not None:
        # Extend in-place with sentinel token ID (-1) to avoid collision with real prompt tokens.
        # Set cache_salt to request.request_id so block_hasher hashes the full string request ID
        # directly into the partial tail block hash pipeline, guaranteeing 0% cross-request collision.
        orig_salt = request.cache_salt
        request._all_token_ids.extend([-1] * pad_len)
        request.cache_salt = request.request_id
        request._block_hasher = block_hasher
        try:
            request.update_block_hashes()
            if len(request.block_hashes) > len(full_block_hashes):
                partial_block_hash = request.block_hashes[-1]
        finally:
            request.cache_salt = orig_salt
            # Remove pad tokens in-place before prefill execution
            del request._all_token_ids[len(raw_token_ids) :]

    with self.beam_cache_lock:
        self.beam_cache[request.request_id] = {
            "all_token_ids": list(raw_token_ids),
            "full_block_hashes": full_block_hashes,
            "partial_block_hash": partial_block_hash,
            "block_hashes": list(request.block_hashes),
            "num_pad_tokens": pad_len,
            "eos_token_id": request.sampling_params.eos_token_id
            if request.sampling_params is not None
            else None,
            "lora_request": request.lora_request,
            "cache_salt": request.cache_salt,
            "mm_features": request.mm_features,
            "prompt_embeds": request.prompt_embeds,
            "prompt_is_token_ids": request.prompt_is_token_ids,
            "num_prompt_tokens": request.num_prompt_tokens,
        }


def _handle_beam_request_step_update(self, update) -> None:
    """Apply a grouped step update using only the prefill cache entry."""
    from vllm.v1.request import Request

    _ensure_beam_state(self)
    session_id = update.session_id
    beam_width = update.beam_width

    # Clear the session completely upon generation termination / cleanup message
    if beam_width == 0:
        with self.beam_cache_lock:
            self.beam_cache.pop(session_id, None)

        # We MUST abort the running beam-request session_id itself!
        self.aborts_queue.put_nowait(session_id)
        for abort_id in update.pruned_ids:
            self.aborts_queue.put_nowait(abort_id)
        return

    # Validate parallel arrays
    if (
        len(update.beam_tokens) != beam_width
        or len(update.parent_beam_ids) != beam_width
        or len(update.child_beam_ids) != beam_width
    ):
        logger.error("BEAM_REQUEST_STEP_UPDATE: session=%s length mismatch", session_id)
        for child_id in update.child_beam_ids:
            self.output_queue.put_nowait(
                (
                    update.client_index,
                    EngineCoreOutputs(
                        engine_index=self.engine_index,
                        finished_requests={child_id},
                        outputs=[
                            EngineCoreOutput(
                                request_id=child_id,
                                new_token_ids=[],
                                finish_reason=FinishReason.ERROR,
                            )
                        ],
                    ),
                )
            )
        return

    # Fetch the shared prefill cache entry
    with self.beam_cache_lock:
        session_cached = self.beam_cache.get(session_id)

    if session_cached is None and beam_width > 0:
        logger.error("BEAM_REQUEST_STEP_UPDATE: session %s not in cache", session_id)
        for child_id in update.child_beam_ids:
            self.output_queue.put_nowait(
                (
                    update.client_index,
                    EngineCoreOutputs(
                        engine_index=self.engine_index,
                        finished_requests={child_id},
                        outputs=[
                            EngineCoreOutput(
                                request_id=child_id,
                                new_token_ids=[],
                                finish_reason=FinishReason.ERROR,
                            )
                        ],
                    ),
                )
            )
        return

    arrival = time.time()
    prefill_tokens = session_cached["all_token_ids"]
    prefill_block_hashes = list(session_cached["block_hashes"])
    lora_request = session_cached["lora_request"]
    cache_salt = session_cached["cache_salt"]
    mm_features = session_cached["mm_features"]
    prompt_embeds = session_cached["prompt_embeds"]
    prompt_is_token_ids = session_cached["prompt_is_token_ids"]
    eos = session_cached["eos_token_id"]

    eos_token_id = update.eos_token_id if update.eos_token_id is not None else eos
    if eos_token_id is not None:
        update.sampling_params.update_from_generation_config({}, eos_token_id)

    # Pack the shared prefill and all beam suffixes into a single contiguous Beam Request
    beam_token_ids = list(prefill_tokens)
    decode_steps = len(update.beam_tokens[0]) if update.beam_tokens else 0
    assert all(len(t) == decode_steps for t in update.beam_tokens)
    for gen_tokens in update.beam_tokens:
        beam_token_ids.extend(gen_tokens)

    req = Request(
        request_id=session_id,  # Single request ID representing all beams
        prompt_token_ids=beam_token_ids,
        sampling_params=update.sampling_params,
        pooling_params=None,
        client_index=update.client_index,
        arrival_time=arrival,
        lora_request=(update.lora_request or lora_request),
        cache_salt=update.cache_salt or cache_salt,
        priority=update.priority,
        trace_headers=update.trace_headers,
        prompt_embeds=prompt_embeds,
        prompt_is_token_ids=prompt_is_token_ids,
        mm_features=mm_features,
        block_hasher=None,
    )

    # Start with the prefill block hashes; block_hasher extends it incrementally
    req.block_hashes = prefill_block_hashes
    if self.request_block_hasher is not None:
        req._block_hasher = self.request_block_hasher
        req.update_block_hashes()

    # Metadata Enrichment
    req.is_beam = True
    req.beam_width = beam_width
    req.is_beam_decode = True  # Flag indicating it's the decode stage
    req.beam_decode_steps = decode_steps
    req.prefix_len = update.prefix_len

    self.input_queue.put_nowait((EngineCoreRequestType.ADD, (req, update.current_wave)))

    for abort_id in update.pruned_ids:
        self.aborts_queue.put_nowait(abort_id)


# ---------------------------------------------------------------------------
# BEAM_SEARCH / BEAM_SEARCH_ABORT handlers (engine-side beam loop)
# ---------------------------------------------------------------------------


_CATALOG_CACHE_MAX_VERSIONS = 8
# Rough per-token memory estimate (trie dict node + sequence-list entry) used
# to bound the engine-side catalog cache by bytes instead of versions alone.
_CATALOG_BYTES_PER_TOKEN = 128


def _catalog_sequences_bytes(catalog_token_sequences) -> int:
    """Approximate retained bytes of a cached catalog trie (heuristic)."""
    return sum(len(seq) for seq in catalog_token_sequences) * _CATALOG_BYTES_PER_TOKEN


def _insert_gr_catalog(self, catalog_hash, catalog_token_sequences, catalog) -> None:
    """Insert a catalog into the byte- and version-bounded cache.

    Evicts the oldest entries BEFORE inserting so the cache never exceeds
    ``_gr_catalog_cache_budget`` bytes nor ``_CATALOG_CACHE_MAX_VERSIONS``
    entries (no 9th-version peak during replacement).  Called with
    ``_gr_beam_sessions_lock`` held.
    """
    new_size = _catalog_sequences_bytes(catalog_token_sequences)
    cache = self._gr_catalog_cache
    sizes = self._gr_catalog_cache_sizes
    while cache and (
        len(cache) >= _CATALOG_CACHE_MAX_VERSIONS
        or self._gr_catalog_cache_size + new_size > self._gr_catalog_cache_budget
    ):
        oldest_hash, _ = next(iter(cache.items()))
        cache.pop(oldest_hash)
        self._gr_catalog_cache_size -= sizes.pop(oldest_hash, 0)
    cache[catalog_hash] = catalog
    sizes[catalog_hash] = new_size
    self._gr_catalog_cache_size += new_size


def _get_gr_engine_catalog(self, catalog_hash, catalog_token_sequences):
    """Return the engine-side Catalog trie for a catalog identity.

    The cache is keyed by the frontend-computed ``catalog_hash`` (O(1)
    lookup — the sequences themselves are never re-hashed or rebuilt here).
    The cache is bounded by an approximate byte budget (and a version cap),
    so multiple large catalog versions cannot exhaust host memory.
    Returns ``None`` when the trie is not cached and no sequences were
    provided to build it.
    """
    _ensure_beam_state(self)
    if catalog_token_sequences is None:
        if catalog_hash is None:
            return None
        return getattr(self, "_gr_catalog_cache", {}).get(catalog_hash)

    # The cache is shared with the background catalog-build thread and the
    # input thread; guard the read-modify-write with the lifecycle lock.  The
    # expensive trie build itself stays outside the lock.
    with self._gr_beam_sessions_lock:
        if not getattr(self, "_gr_catalog_cache", None):
            self._gr_catalog_cache = {}
            self._gr_catalog_cache_sizes: dict = {}
            self._gr_catalog_cache_size = 0
        if getattr(self, "_gr_catalog_cache_budget", None) is None:
            self._gr_catalog_cache_budget = (
                int(os.getenv("VLLM_GR_CATALOG_CACHE_BUDGET_MB", "2048")) * 1024 * 1024
            )
        if catalog_hash is not None and catalog_hash in self._gr_catalog_cache:
            return self._gr_catalog_cache[catalog_hash]

    from vllm_gr.utils.catalog import Catalog

    catalog = Catalog()
    catalog.load_from_sequences(catalog_token_sequences)
    if catalog_hash is not None:
        with self._gr_beam_sessions_lock:
            _insert_gr_catalog(self, catalog_hash, catalog_token_sequences, catalog)
    return catalog


def _handle_beam_search_request(self, request) -> None:
    """Start a full engine-side beam-search session."""
    from vllm_gr.v1.beam.search_engine import BeamSearchSession

    _ensure_beam_state(self)
    sessions = getattr(self, "_gr_beam_sessions", None)
    if sessions is None:
        self._gr_beam_sessions = {}
        sessions = self._gr_beam_sessions

    # Catalog-only preload: build the engine-side catalog trie synchronously.
    # The preload arrives during idle startup (before any traffic), so
    # blocking the input thread here guarantees the cache is warm before the
    # first beam-search request — no race on the critical path.
    if request.preload_only:
        if request.catalog_token_sequences is not None:
            try:
                _get_gr_engine_catalog(self, request.catalog_hash, request.catalog_token_sequences)
            except Exception:
                logger.exception("BEAM_SEARCH: catalog preload build failed")
        return

    # Requests whose token budget covers only the reserved begin/end tokens
    # have no generation waves: mirror the frontend (which performs no model
    # step in that case) and finish immediately, without a prefill wave or
    # catalog work.
    pre_calc = (1 if request.sid_begin_token_id is not None else 0) + (
        1 if request.sid_end_token_id is not None else 0
    )
    if request.max_tokens - pre_calc <= 0:
        session = BeamSearchSession(self, request, None, catalog_pending=False)
        final = session._finish_session()
        if hasattr(self, "output_queue"):
            self.output_queue.put_nowait((request.client_index, final))
        return

    catalog = _get_gr_engine_catalog(self, request.catalog_hash, request.catalog_token_sequences)
    catalog_pending = catalog is None and request.catalog_token_sequences is not None
    session = BeamSearchSession(
        self,
        request,
        catalog,
        catalog_pending=catalog_pending,
    )
    # Register the session BEFORE submitting the prefill: the step loop may
    # produce the prefill output immediately, and it must find the session in
    # the map to route it back instead of treating it as a normal output.
    with self._gr_beam_sessions_lock:
        sessions[session.session_id] = session
    try:
        session._submit_prefill_wave()
    except Exception:
        logger.exception("BEAM_SEARCH: session %s prefill submission failed", request.session_id)
        with self._gr_beam_sessions_lock:
            sessions.pop(session.session_id, None)
        error_output = session._finish_with_error()
        if hasattr(self, "output_queue"):
            self.output_queue.put_nowait((request.client_index, error_output))
        return

    # The catalog trie is large (hundreds of thousands of sequences); build
    # it in a background thread while the prefill wave runs on the GPU.  The
    # wave-0 decision (the only catalog consumer before the final sort) waits
    # for readiness, so the build cost is hidden behind the model execution.
    if catalog_pending:
        threading.Thread(
            target=_build_gr_catalog_for_session,
            args=(self, session, request.catalog_hash, request.catalog_token_sequences),
            daemon=True,
        ).start()


def _build_gr_catalog_for_session(self, session, catalog_hash, catalog_token_sequences) -> None:
    """Build the catalog trie and attach it to a running session."""
    try:
        catalog = _get_gr_engine_catalog(self, catalog_hash, catalog_token_sequences)
    except Exception:
        logger.exception("BEAM_SEARCH: catalog build failed for session %s", session.session_id)
        catalog = None
    with self.beam_cache_lock:
        session.catalog = catalog
        session._catalog_ready.set()


def _handle_beam_search_abort(self, abort) -> None:
    """Abort an engine-side beam-search session."""
    sessions = getattr(self, "_gr_beam_sessions", None)
    if not sessions:
        return
    with self._gr_beam_sessions_lock:
        session = sessions.pop(abort.session_id, None)
    if session is not None:
        session.abort()


def _route_beam_session_outputs(self, outputs):
    """Split session outputs out of the step outputs and run the beam loop.

    Called from the patched step loop.  Returns the outputs dict that should
    be forwarded to the frontend output queue.
    """
    sessions = getattr(self, "_gr_beam_sessions", None)
    if not sessions or not outputs:
        return outputs

    # Snapshot the session map under the lifecycle lock: the input thread
    # mutates it concurrently (BEAM_SEARCH / BEAM_SEARCH_ABORT) and the
    # routing below iterates and pops from it.
    with self._gr_beam_sessions_lock:
        sessions_snapshot = list(sessions.values())

    # Pre-build a reverse lookup so every request_id resolves in O(1)
    # instead of scanning all sessions per output.
    rid_to_session: dict[str, Any] = {}
    for s in sessions_snapshot:
        rid_to_session[s.session_id] = s
        for rid in s.beam_request_ids:
            rid_to_session[rid] = s

    forwarded: dict[int, EngineCoreOutputs] = {}
    finals_by_client: dict[int, list[EngineCoreOutput]] = {}
    finals_finished_by_client: dict[int, set[str]] = {}
    for client_index, engine_core_outputs in outputs.items():
        session_outs: dict[int, list[EngineCoreOutput]] = {}
        normal_outs: list[EngineCoreOutput] = []
        for output in engine_core_outputs.outputs:
            request_id = output.request_id
            session = rid_to_session.get(request_id)
            if session is None:
                normal_outs.append(output)
            else:
                session_outs.setdefault(id(session), []).append(output)

        if (
            normal_outs
            or engine_core_outputs.scheduler_stats is not None
            or engine_core_outputs.finished_requests
        ):
            # Forward the step's outputs plus its scheduler stats.  The stats
            # carry the (reset-on-read) prefix cache hit counters, so they
            # must reach the frontend even when every output of this step was
            # intercepted by a beam session.  Mirrors vLLM emitting
            # stats-only EngineCoreOutputs for steps without request outputs.
            # finished_requests carries the terminal bookkeeping of normal
            # requests that finished in this step (possibly without an output
            # row), so it must survive the rebuild as well.
            forwarded[client_index] = EngineCoreOutputs(
                engine_index=engine_core_outputs.engine_index,
                outputs=normal_outs,
                finished_requests=engine_core_outputs.finished_requests,
                scheduler_stats=engine_core_outputs.scheduler_stats,
                timestamp=engine_core_outputs.timestamp,
                utility_output=engine_core_outputs.utility_output,
            )

        # Drive the beam sessions with their wave outputs (from the snapshot).
        for session in sessions_snapshot:
            outs = session_outs.get(id(session))
            if not outs:
                continue
            final = session.on_wave_output(
                EngineCoreOutputs(
                    engine_index=engine_core_outputs.engine_index,
                    outputs=outs,
                    timestamp=engine_core_outputs.timestamp,
                )
            )
            if final is not None:
                finals_by_client.setdefault(session.client_index, []).extend(final.outputs)
                finals_finished_by_client.setdefault(session.client_index, set()).update(
                    final.finished_requests or set()
                )
                with self._gr_beam_sessions_lock:
                    sessions.pop(session.session_id, None)

    for client_index, final_outs in finals_by_client.items():
        existing = forwarded.get(client_index)
        if existing is not None:
            # Merge the session finals into this step's forwarded batch so
            # normal outputs and beam finals produced by the same step are
            # all delivered, while the step's scheduler_stats ride along.
            existing.outputs.extend(final_outs)
            existing.finished_requests = (
                (existing.finished_requests or set())
                | finals_finished_by_client.get(client_index, set())
                | {out.request_id for out in final_outs}
            )
        else:
            forwarded[client_index] = EngineCoreOutputs(
                engine_index=self.engine_index,
                outputs=final_outs,
                finished_requests=(
                    finals_finished_by_client.get(client_index, set())
                    | {out.request_id for out in final_outs}
                ),
                timestamp=time.time(),
            )
    return forwarded


# ---------------------------------------------------------------------------
# EngineCoreProc.process_input_sockets replacement
# ---------------------------------------------------------------------------


def process_input_sockets(
    self,
    input_addresses,
    coord_input_address,
    identity,
    ready_event,
):
    """Replacement for EngineCoreProc.process_input_sockets with
    ADD_BATCH and BEAM_REQUEST_STEP_UPDATE decoders."""
    from contextlib import ExitStack

    import msgspec
    import zmq
    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine import (
        EngineCoreReadyResponse,
        EngineCoreRequest,
    )

    from vllm_gr.v1.engine.codec import (
        build_engine_core_decoders,
        decode_engine_core_payload,
    )
    from vllm_gr.v1.engine.types import BeamRequestStepUpdate

    decoders = build_engine_core_decoders(self.tensor_ipc_receiver)

    with ExitStack() as stack, zmq.Context() as ctx:
        input_sockets = [
            stack.enter_context(
                make_zmq_socket(ctx, input_address, zmq.DEALER, identity=identity, bind=False)
            )
            for input_address in input_addresses
        ]
        if coord_input_address is None:
            coord_socket = None
        else:
            coord_socket = stack.enter_context(
                make_zmq_socket(
                    ctx,
                    coord_input_address,
                    zmq.XSUB,
                    identity=identity,
                    bind=False,
                )
            )
            # Send subscription message to coordinator.
            coord_socket.send(b"\x01")

        # Register sockets with poller.
        poller = zmq.Poller()
        ready_response = EngineCoreReadyResponse(
            max_model_len=self.vllm_config.model_config.max_model_len,
            num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
            dp_stats_address=self.frontend_stats_publish_address,
            dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
        )
        ready_payload = msgspec.msgpack.encode(ready_response)
        for input_socket in input_sockets:
            input_socket.send(ready_payload)
            poller.register(input_socket, zmq.POLLIN)

        if coord_socket is not None:
            assert coord_socket.recv() == b"READY"
            poller.register(coord_socket, zmq.POLLIN)

        ready_event.set()
        del ready_event
        while True:
            for input_socket, _ in poller.poll():
                type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                if type_frame.buffer == b"READY":
                    assert input_socket == coord_socket
                    continue
                request_type = EngineCoreRequestType(bytes(type_frame.buffer))
                decoded_payload = decode_engine_core_payload(
                    decoders,
                    request_type,
                    data_frames,
                )

                request: Any
                if request_type == EngineCoreRequestType.ADD:
                    req: EngineCoreRequest = decoded_payload
                    try:
                        request = self.preprocess_add_request(req)
                    except Exception:
                        self._handle_request_preproc_error(req)
                        continue

                elif request_type == EngineCoreRequestType.ADD_BATCH:
                    batch: list[EngineCoreRequest] = decoded_payload
                    for req in batch:
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                        self._cache_beam_request(request[0])
                        self.input_queue.put_nowait((EngineCoreRequestType.ADD, request))
                    continue

                elif request_type == EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE:
                    beam_update: BeamRequestStepUpdate = decoded_payload
                    self._handle_beam_request_step_update(beam_update)
                    continue

                elif request_type == EngineCoreRequestType.BEAM_SEARCH:
                    beam_search_request = decoded_payload
                    self._handle_beam_search_request(beam_search_request)
                    continue

                elif request_type == EngineCoreRequestType.BEAM_SEARCH_ABORT:
                    beam_search_abort = decoded_payload
                    self._handle_beam_search_abort(beam_search_abort)
                    continue

                else:
                    request = decoded_payload

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
