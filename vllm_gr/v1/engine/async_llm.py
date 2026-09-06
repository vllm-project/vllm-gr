# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger

logger = init_logger(__name__)


async def prepare_request_fn(
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

    The 0.22 input processor validates requests against the asynchronously
    discovered supported-task set, so request preparation is asynchronous."""
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

    # Convert Input --> Request.
    if isinstance(prompt, EngineCoreRequest):
        request = prompt
    else:
        if prompt_text is not None:
            raise ValueError("should only provide prompt_text with EngineCoreRequest")
        request = self.input_processor.process_inputs(
            request_id,
            prompt,
            params,
            supported_tasks=await self.get_supported_tasks(),
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )
        from vllm.renderers.inputs.preprocess import extract_prompt_components

        prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

    self.input_processor.assign_request_id(request)

    self._run_output_handler()

    # Create a new output collector for the request.
    params = request.params
    queue = RequestOutputCollector(params.output_kind, request.request_id)

    # Register the request in OutputProcessor (this process) but
    # do NOT send to EngineCore yet.
    self.output_processor.add_request(request, prompt_text, None, 0, queue)

    if self.log_requests:
        logger.info("Prepared request %s.", request.request_id)

    return queue, request


async def _add_requests_batch_fn(
    self,
    requests,
) -> None:
    """Send prepared requests through vLLM's native ADD protocol."""
    for request in requests:
        await self.engine_core.add_request_async(request)
