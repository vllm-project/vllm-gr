# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import os
import threading
import time

import msgspec
import numpy as np
import zmq
import zmq.asyncio
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.utils.env import is_truthy_env
from vllm_gr.v1.engine.types import BeamRequestStepUpdate

logger = init_logger(__name__)

BEAM_PROC_APISERVER_OUTPUT_ENV = "VLLM_GR_BEAM_PROC_APISERVER_OUTPUT_ADDR"
BEAM_PROC_COLLECT_APISERVER_OUTPUT_ENV = "VLLM_GR_BEAM_PROC_COLLECT_APISERVER_OUTPUT"
BEAM_PROC_ENABLED_ENV = "VLLM_GR_BEAM_PROC_ENABLED"


def _decode_beamproc_final_frames(frames) -> dict:
    """Decode legacy msgpack or numeric BEAM_FINAL frames."""
    if len(frames) < 2:
        raise ValueError("BEAM_FINAL requires at least two frames")

    payload = msgspec.msgpack.decode(frames[1].buffer)
    version = payload.get("version")
    if version not in (2, 3):
        return payload
    expected_frames = 5 if version == 2 else 9
    if len(frames) != expected_frames:
        raise ValueError(
            f"BEAM_FINAL v{version} requires {expected_frames} frames, received {len(frames)}"
        )

    flat_tokens = np.frombuffer(frames[2].buffer, dtype="<i4")
    offsets = np.frombuffer(frames[3].buffer, dtype="<u8")
    scores = np.frombuffer(frames[4].buffer, dtype="<f4")
    expected_tokens = int(payload.get("final_token_ids_len", -1))
    expected_offsets = int(payload.get("final_token_offsets_len", -1))
    expected_scores = int(payload.get("final_scores_len", -1))
    expected_beams = int(payload.get("final_beams", -1))
    if flat_tokens.size != expected_tokens:
        raise ValueError(
            f"BEAM_FINAL v{version} token count {flat_tokens.size} != {expected_tokens}"
        )
    if offsets.size != expected_offsets:
        raise ValueError(f"BEAM_FINAL v{version} offset count {offsets.size} != {expected_offsets}")
    if scores.size != expected_scores:
        raise ValueError(f"BEAM_FINAL v{version} score count {scores.size} != {expected_scores}")
    if offsets.size == 0 or offsets[0] != 0 or offsets[-1] != flat_tokens.size:
        raise ValueError(f"BEAM_FINAL v{version} offsets do not cover the token frame")
    if offsets.size - 1 != expected_beams or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"BEAM_FINAL v{version} offsets do not describe the declared beams")

    payload["final_token_ids"] = [
        flat_tokens[int(offsets[index]) : int(offsets[index + 1])].tolist()
        for index in range(expected_beams)
    ]
    payload["final_scores"] = scores.tolist()
    if version == 3:
        logprob_token_ids = np.frombuffer(frames[5].buffer, dtype="<i4")
        logprob_values = np.frombuffer(frames[6].buffer, dtype="<f4")
        step_offsets = np.frombuffer(frames[7].buffer, dtype="<u8")
        beam_offsets = np.frombuffer(frames[8].buffer, dtype="<u8")
        expected_logprob_tokens = int(payload.get("final_logprob_token_ids_len", -1))
        expected_logprob_values = int(payload.get("final_logprob_values_len", -1))
        expected_step_offsets = int(payload.get("final_logprob_step_offsets_len", -1))
        expected_beam_offsets = int(payload.get("final_logprob_beam_offsets_len", -1))
        if logprob_token_ids.size != expected_logprob_tokens:
            raise ValueError("BEAM_FINAL v3 logprob token count mismatch")
        if logprob_values.size != expected_logprob_values:
            raise ValueError("BEAM_FINAL v3 logprob value count mismatch")
        if logprob_token_ids.size != logprob_values.size:
            raise ValueError("BEAM_FINAL v3 logprob tokens and values are not aligned")
        if step_offsets.size != expected_step_offsets:
            raise ValueError("BEAM_FINAL v3 logprob step offset count mismatch")
        if beam_offsets.size != expected_beam_offsets:
            raise ValueError("BEAM_FINAL v3 logprob beam offset count mismatch")
        if (
            step_offsets.size == 0
            or step_offsets[0] != 0
            or step_offsets[-1] != logprob_token_ids.size
            or np.any(step_offsets[1:] < step_offsets[:-1])
        ):
            raise ValueError("BEAM_FINAL v3 logprob step offsets are invalid")
        step_count = step_offsets.size - 1
        if (
            beam_offsets.size != expected_beams + 1
            or beam_offsets[0] != 0
            or beam_offsets[-1] != step_count
            or np.any(beam_offsets[1:] < beam_offsets[:-1])
        ):
            raise ValueError("BEAM_FINAL v3 logprob beam offsets are invalid")
        # Keep the large logprob payload packed until serving constructs the
        # final FlatLogprobs objects. Materializing nested Python lists here
        # would box every token and value, only for serving to copy them again.
        payload["final_logprob_token_ids"] = logprob_token_ids
        payload["final_logprob_values"] = logprob_values
        payload["final_logprob_step_offsets"] = step_offsets
        payload["final_logprob_beam_offsets"] = beam_offsets
    return payload


def _should_collect_beamproc_apiserver_output() -> bool:
    return is_truthy_env(BEAM_PROC_COLLECT_APISERVER_OUTPUT_ENV) or is_truthy_env(
        BEAM_PROC_ENABLED_ENV
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


def ensure_beamproc_output_task(self, output_addr: str | None = None) -> None:
    """Start APIServer-side asyncio receiver for BeamProc final outputs."""
    output_addr = output_addr or os.getenv(BEAM_PROC_APISERVER_OUTPUT_ENV)
    if not output_addr:
        logger.debug(
            "BeamProc APIServer output task not started: %s is unset",
            BEAM_PROC_APISERVER_OUTPUT_ENV,
        )
        return

    resources = self.resources
    existing_task = getattr(resources, "beamproc_output_task", None)
    if existing_task is not None:
        if not existing_task.done():
            logger.debug(
                "BeamProc APIServer output task already running addr=%s task=%s",
                output_addr,
                existing_task.get_name(),
            )
            return
        logger.warning(
            "BeamProc APIServer output task was stopped; restarting addr=%s task=%s",
            output_addr,
            existing_task.get_name(),
        )
        old_socket = getattr(resources, "beamproc_output_socket", None)
        if old_socket is not None:
            with contextlib.suppress(Exception):
                old_socket.close(linger=0)

    async def process_beamproc_outputs_socket(output_socket: zmq.asyncio.Socket):
        logger.info(
            "BeamProc APIServer output task entry addr=%s task=%s",
            output_addr,
            asyncio.current_task().get_name() if asyncio.current_task() else None,
        )
        try:
            logger.info(
                "Polling BeamProc APIServer output socket in asyncio task addr=%s",
                output_addr,
            )
            while True:
                frames = await output_socket.recv_multipart(copy=False)
                message_type = frames[0].buffer if frames else b"<missing>"
                logger.debug(
                    "BeamProc APIServer output raw receive frames=%s frame_bytes=%s first_frame=%r",
                    len(frames),
                    [len(frame.buffer) for frame in frames],
                    message_type[:64],
                )
                if message_type == b"BEAM_FINAL" and len(frames) >= 2:
                    try:
                        payload = _decode_beamproc_final_frames(frames)
                    except Exception:
                        logger.exception(
                            "Failed to decode BeamProc APIServer BEAM_FINAL bytes=%s",
                            sum(len(frame.buffer) for frame in frames[1:]),
                        )
                        continue
                    payload["_beamproc_received_at"] = time.perf_counter()
                    should_collect = _should_collect_beamproc_apiserver_output()
                    final_token_ids = payload.get("final_token_ids") or []
                    logger.debug(
                        "BEAMPROC_AS_FINAL_RECV root=%s request_id=%s "
                        "beam_width=%s final_beams=%s first_final_len=%s "
                        "collected=%s has_handler=%s bytes=%s",
                        payload.get("root_request_id"),
                        payload.get("request_id"),
                        payload.get("beam_width"),
                        len(final_token_ids),
                        (
                            len(final_token_ids[0])
                            if final_token_ids and hasattr(final_token_ids[0], "__len__")
                            else 0
                        ),
                        should_collect,
                        callable(
                            getattr(
                                self,
                                "beamproc_apiserver_output_handler",
                                None,
                            )
                        ),
                        sum(len(frame.buffer) for frame in frames[1:]),
                    )
                    if should_collect:
                        handler = getattr(
                            self,
                            "beamproc_apiserver_output_handler",
                            None,
                        )
                        if callable(handler):
                            try:
                                handler(payload)
                            except Exception:
                                logger.exception(
                                    "BeamProc APIServer output handler failed root=%s "
                                    "request_id=%s",
                                    payload.get("root_request_id"),
                                    payload.get("request_id"),
                                )
                        else:
                            logger.warning(
                                "Dropping BeamProc APIServer BEAM_FINAL without handler "
                                "root=%s request_id=%s",
                                payload.get("root_request_id"),
                                payload.get("request_id"),
                            )
                    logger.debug(
                        "Received BeamProc APIServer BEAM_FINAL root=%s request_id=%s "
                        "final_beams=%s collected=%s",
                        payload.get("root_request_id"),
                        payload.get("request_id"),
                        len(payload.get("final_token_ids", ())),
                        should_collect,
                    )
                else:
                    logger.debug(
                        "Received BeamProc APIServer output type=%s frames=%s frame_bytes=%s",
                        message_type,
                        len(frames),
                        [len(frame.buffer) for frame in frames],
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("BeamProc APIServer output task failed.")
        finally:
            with contextlib.suppress(Exception):
                output_socket.close(linger=0)

    context = getattr(self, "ctx", None)
    if not isinstance(context, zmq.asyncio.Context):
        context = zmq.asyncio.Context.instance()
    output_socket = context.socket(zmq.PULL)
    output_socket.setsockopt(zmq.LINGER, 0)
    logger.info("Binding BeamProc APIServer output socket addr=%s", output_addr)
    output_socket.bind(output_addr)
    resources.beamproc_output_socket = output_socket

    logger.info(
        "Creating BeamProc APIServer output asyncio task addr=%s target=%s",
        output_addr,
        process_beamproc_outputs_socket.__name__,
    )
    output_task = asyncio.create_task(
        process_beamproc_outputs_socket(output_socket),
        name="BeamProcOutputTask",
    )
    resources.beamproc_output_task = output_task
    logger.info(
        "Started BeamProc APIServer output asyncio task addr=%s task=%s",
        output_addr,
        output_task.get_name(),
    )


def make_patched_background_resources_call(original_call):
    def patched_call(self):
        beamproc_task = getattr(self, "beamproc_output_task", None)
        beamproc_socket = getattr(self, "beamproc_output_socket", None)
        if beamproc_task is not None and not beamproc_task.done():
            with contextlib.suppress(Exception):
                beamproc_task.cancel()
        if beamproc_socket is not None:
            with contextlib.suppress(Exception):
                beamproc_socket.close(linger=0)
        return original_call(self)

    return patched_call


class SyncBeamProcReceiver:
    """Synchronous BEAM_FINAL receiver for offline (in-process) BeamProc use.

    The online APIServer path reads final replies through an asyncio task
    (``ensure_beamproc_output_task``). The offline GRLLM/LLM path is
    synchronous, so this class binds the same ``PULL`` socket on a plain
    background thread and exposes a blocking ``await_plan``.
    """

    def __init__(self, output_addr: str):
        self.output_addr = output_addr
        self._plans: dict[str, list[dict]] = {}
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="beamproc-offline-output",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVHWM, 64)
        socket.bind(self.output_addr)
        try:
            while not self._stop.is_set():
                if socket.poll(50) == 0:
                    continue
                frames = socket.recv_multipart(copy=False)
                message_type = frames[0].buffer if frames else b"<missing>"
                if message_type == b"BEAM_FINAL" and len(frames) >= 2:
                    try:
                        payload = _decode_beamproc_final_frames(frames)
                    except Exception:
                        logger.exception(
                            "Failed to decode offline BeamProc BEAM_FINAL bytes=%s",
                            sum(len(frame.buffer) for frame in frames[1:]),
                        )
                        continue
                    self._dispatch(payload)
        finally:
            with contextlib.suppress(Exception):
                socket.close(linger=0)
            context.term()

    def _dispatch(self, payload: dict) -> None:
        root_request_id = str(payload.get("root_request_id") or payload.get("request_id") or "")
        if not root_request_id:
            return
        with self._cond:
            self._plans.setdefault(root_request_id, []).append(payload)
            self._cond.notify_all()

    def await_plan(
        self,
        root_request_id: str,
        beam_width: int,
        timeout_s: float,
    ) -> dict | None:
        """Block until a final plan for ``root_request_id`` is observed."""
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self._cond:
            while True:
                plan = self._pop(root_request_id, beam_width)
                if plan is not None:
                    return plan
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def _pop(self, root_request_id: str, beam_width: int) -> dict | None:
        entries = self._plans.get(root_request_id)
        if not entries:
            return None
        for index, plan in enumerate(entries):
            if int(plan.get("beam_width") or 0) == beam_width:
                entries.pop(index)
                if not entries:
                    self._plans.pop(root_request_id, None)
                return plan
        return None

    def shutdown(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
