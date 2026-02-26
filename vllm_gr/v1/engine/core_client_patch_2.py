# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for AsyncMPClient / AsyncLLM to support
ADD_BATCH and BEAM_FORK."""

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
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.engine.core_client import AsyncMPClient


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
    priority=0
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
        trace_headers=trace_headers,
        priority=priority,
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
