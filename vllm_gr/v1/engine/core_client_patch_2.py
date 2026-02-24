# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for AsyncMPClient / AsyncLLM to support
ADD_BATCH, BEAM_FORK, and BEAM_SEARCH."""

from __future__ import annotations

import asyncio
import time
import weakref
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.engine.core_client import _process_utility_output
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_gr.v1.engine.types import (
    BeamForkRequest,
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
            outputs_queue.put_nowait(e)
        except asyncio.CancelledError:
            outputs_queue.put_nowait(EngineDeadError())

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
    for request in requests:
        request.client_index = self.client_index
    await self._send_input(EngineCoreRequestType.ADD_BATCH, requests)
    self._ensure_output_queue_task()


# ---------------------------------------------------------------------------
# AsyncMPClient.beam_fork_async  (new method — BEAM_FORK path)
# ---------------------------------------------------------------------------

async def beam_fork_async(self, fork_request: BeamForkRequest) -> None:
    """Send BEAM_FORK to EngineCore."""
    fork_request.client_index = self.client_index
    await self._send_input(
        EngineCoreRequestType.BEAM_FORK, fork_request)
    self._ensure_output_queue_task()


# ---------------------------------------------------------------------------
# AsyncLLM methods for ADD_BATCH / BEAM_FORK
# ---------------------------------------------------------------------------

def prepare_request_fn(
    self,
    request_id,
    prompt,
    params,
    *,
    lora_request=None,
    tokenization_kwargs=None,
    trace_headers=None,
    priority=0,
    data_parallel_rank=None,
    prompt_text=None,
    arrival_time=None,
):
    """Process input and register output queue, but DON'T send to
    EngineCore. Returns (output_queue, engine_core_request) for later
    batch send via _add_requests_batch().

    This is intentionally synchronous to avoid event-loop overhead
    when preparing many requests in a tight loop (e.g. beam search).
    The caller is responsible for any pause-state gating before
    entering the preparation loop."""
    from vllm.pooling_params import PoolingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.output_processor import RequestOutputCollector

    if self.errored:
        from vllm.v1.engine.exceptions import EngineDeadError
        raise EngineDeadError()

    is_pooling = isinstance(params, PoolingParams)

    if (
        self.vllm_config.cache_config.kv_sharing_fast_prefill
        and not is_pooling
        and params.prompt_logprobs
    ):
        raise ValueError(
            "--kv-sharing-fast-prefill produces incorrect logprobs for "
            "prompt tokens, please disable it when the requests need "
            "prompt logprobs"
        )

    if tokenization_kwargs is None:
        tokenization_kwargs = {}
    from vllm.entrypoints.utils import _validate_truncation_size
    _validate_truncation_size(
        self.model_config.max_model_len,
        params.truncate_prompt_tokens,
        tokenization_kwargs,
    )

    # Convert Input --> Request.
    if isinstance(prompt, EngineCoreRequest):
        request = prompt
    else:
        if prompt_text is not None:
            raise ValueError(
                "should only provide prompt_text with EngineCoreRequest"
            )
        request = self.input_processor.process_inputs(
            request_id,
            prompt,
            params,
            arrival_time or time.time(),
            lora_request,
            tokenization_kwargs,
            trace_headers,
            priority,
            data_parallel_rank,
        )
        if isinstance(prompt, str):
            prompt_text = prompt
        elif isinstance(prompt, Mapping):
            prompt_text = cast(str | None, prompt.get("prompt"))

    self.input_processor.assign_request_id(request)

    self._run_output_handler()

    # Create a new output collector for the request.
    queue = RequestOutputCollector(params.output_kind, request.request_id)

    # Register the request in OutputProcessor (this process) but
    # do NOT send to EngineCore yet.
    self.output_processor.add_request(
        request, prompt_text, None, 0, queue)

    if self.log_requests:
        logger.info("Prepared request %s.", request.request_id)

    return queue, request


async def _add_requests_batch_fn(
    self,
    requests,
    use_batch_message: bool = False,
) -> None:
    """Send multiple already-prepared requests to EngineCore.

    Args:
        use_batch_message: If True, always send as ADD_BATCH (needed for
            beam cache). If False, send individual ADD messages.
    """
    await self.engine_core.add_requests_async(
        requests, force_batch=use_batch_message)


def register_beam_output_fn(
    self,
    request_id: str,
    prompt_token_ids: list[int],
    sampling_params,
    eos_token_id=None,
    lora_request=None,
    trace_headers=None,
):
    """Register output queue for a beam fork child (no ZMQ send).

    The actual request will be created by EngineCore's BEAM_FORK handler.
    We just need a queue in OutputProcessor to receive outputs."""
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.output_processor import RequestOutputCollector

    ec_request = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=eos_token_id,
        arrival_time=time.time(),
        lora_request=lora_request,
        cache_salt=None,
        data_parallel_rank=None,
    )
    # Set external_req_id directly (skip assign_request_id which
    # would mangle the ID that EngineCore already knows about).
    ec_request.external_req_id = request_id

    self._run_output_handler()
    queue = RequestOutputCollector(
        sampling_params.output_kind, request_id)
    self.output_processor.add_request(ec_request, None, None, 0, queue)
    return queue


async def beam_fork_fn(self, fork_request: BeamForkRequest) -> None:
    """Send BEAM_FORK to EngineCore."""
    await self.engine_core.beam_fork_async(fork_request)


# ---------------------------------------------------------------------------
# apply_batch_fork_patches — wire ADD_BATCH / BEAM_FORK into client classes
# ---------------------------------------------------------------------------

def apply_batch_fork_patches():
    """Monkey-patch AsyncMPClient and AsyncLLM with ADD_BATCH/BEAM_FORK
    methods. Must be called after the enum members exist (i.e. after
    _add_enum_member has been called for ADD_BATCH and BEAM_FORK in the
    parent process)."""
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import AsyncMPClient

    from vllm_gr.v1.engine.engine_core_patch_2 import _add_enum_member

    # Ensure enum members exist in parent process
    _add_enum_member("ADD_BATCH", b"\x05")
    _add_enum_member("BEAM_FORK", b"\x06")

    # Patch AsyncMPClient
    AsyncMPClient.add_requests_async = add_requests_async
    AsyncMPClient.beam_fork_async = beam_fork_async

    # Patch AsyncLLM
    AsyncLLM.prepare_request = prepare_request_fn
    AsyncLLM._add_requests_batch = _add_requests_batch_fn
    AsyncLLM.register_beam_output = register_beam_output_fn
    AsyncLLM.beam_fork = beam_fork_fn
