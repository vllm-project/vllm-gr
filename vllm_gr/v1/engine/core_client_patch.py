# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for AsyncMPClient to support beam search."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.engine.core_client import _process_utility_output
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_gr.v1.engine.types import (
    BeamSearchRequest,
    BeamSearchResult,
    PatchedEngineCoreOutputs,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.engine.core_client import AsyncMPClient


# ---------------------------------------------------------------------------
# AsyncMPClient.__init__ wrapper
# ---------------------------------------------------------------------------

def make_patched_async_mp_init(original_init):
    """Wrap AsyncMPClient.__init__ to add beam_search_futures dict and fix
    the MsgpackDecoder BEFORE original_init runs.

    This ordering is critical because original_init calls
    _ensure_output_queue_task, which captures self.decoder into a closure.
    If we replace the decoder AFTER original_init, the output-processing
    coroutine will still use the old decoder that doesn't know about
    PatchedEngineCoreOutputs, silently dropping beam search results.
    """
    from vllm.v1.serial_utils import MsgpackDecoder

    def patched_init(self, *args, **kwargs):
        # Set beam_search_futures BEFORE original_init, because it calls
        # _ensure_output_queue_task which accesses this attribute.
        self.beam_search_futures: dict[
            str, asyncio.Future[BeamSearchResult]
        ] = {}
        original_init(self, *args, **kwargs)
        # Replace the decoder with one that knows about
        # PatchedEngineCoreOutputs.  This must happen AFTER original_init
        # (which creates self.decoder) but BEFORE _ensure_output_queue_task
        # captures it.  Since original_init calls _ensure_output_queue_task
        # internally, we also need to prevent the task from starting during
        # init and restart it with the correct decoder — see below.
        self.decoder = MsgpackDecoder(PatchedEngineCoreOutputs)
        # The output queue task was already created by original_init with
        # the old decoder.  Cancel it and force re-creation so it picks up
        # the new decoder.
        resources = self.resources
        if resources.output_queue_task is not None:
            resources.output_queue_task.cancel()
            resources.output_queue_task = None
        self._ensure_output_queue_task()

    return patched_init


# ---------------------------------------------------------------------------
# AsyncMPClient._ensure_output_queue_task replacement
# ---------------------------------------------------------------------------

def patched_ensure_output_queue_task(self):
    """Replacement for AsyncMPClient._ensure_output_queue_task with beam
    search result handling."""
    resources = self.resources
    if resources.output_queue_task is not None:
        return

    decoder = self.decoder
    utility_results = self.utility_results
    beam_search_futures = self.beam_search_futures
    outputs_queue = self.outputs_queue
    output_handler: (
        Callable[[AsyncMPClient, EngineCoreOutputs], Awaitable[None]] | None
    ) = getattr(self.__class__, "process_engine_outputs", None)
    _self_ref = weakref.ref(self) if output_handler else None
    output_socket = resources.output_socket
    assert output_socket is not None

    async def process_outputs_socket():
        try:
            while True:
                frames = await output_socket.recv_multipart(copy=False)
                resources.validate_alive(frames)
                outputs: EngineCoreOutputs = decoder.decode(frames)
                if outputs.utility_output:
                    _process_utility_output(
                        outputs.utility_output, utility_results
                    )
                    continue

                if (
                    hasattr(outputs, "beam_search_result")
                    and outputs.beam_search_result is not None
                ):
                    bs_result = outputs.beam_search_result
                    logger.debug(
                        "CLIENT: received beam_search_result for %s "
                        "with %d sequences",
                        bs_result.request_id,
                        len(bs_result.sequences),
                    )
                    future = beam_search_futures.pop(
                        bs_result.request_id, None
                    )
                    logger.debug(
                        "CLIENT: future found=%s",
                        future is not None,
                    )
                    if future is not None and not future.done():
                        future.set_result(bs_result)
                    continue

                if output_handler is not None:
                    assert _self_ref is not None
                    _self = _self_ref()
                    if not _self:
                        return
                    await output_handler(_self, outputs)

                if outputs.outputs or outputs.scheduler_stats:
                    outputs_queue.put_nowait(outputs)
        except Exception as e:
            # Propagate the exception to all pending beam search futures so
            # that callers do not hang waiting for results that will never
            # arrive.
            for future in beam_search_futures.values():
                if not future.done():
                    future.set_exception(e)
            beam_search_futures.clear()
            outputs_queue.put_nowait(e)
        except asyncio.CancelledError:
            # If the output processing task is cancelled (for example,
            # because the engine died), fail all pending beam search futures
            # with an EngineDeadError.
            err = EngineDeadError()
            for future in beam_search_futures.values():
                if not future.done():
                    future.set_exception(err)
            beam_search_futures.clear()
            outputs_queue.put_nowait(err)

    resources.output_queue_task = asyncio.create_task(
        process_outputs_socket(), name="EngineCoreOutputQueueTask"
    )


# ---------------------------------------------------------------------------
# AsyncMPClient.beam_search_async  (new method)
# ---------------------------------------------------------------------------

async def beam_search_async(
    self, request: BeamSearchRequest
) -> BeamSearchResult:
    request.client_index = self.client_index
    future: asyncio.Future[BeamSearchResult] = (
        asyncio.get_running_loop().create_future()
    )
    self.beam_search_futures[request.request_id] = future
    await self._send_input(EngineCoreRequestType.BEAM_SEARCH, request)
    self._ensure_output_queue_task()
    return await future
