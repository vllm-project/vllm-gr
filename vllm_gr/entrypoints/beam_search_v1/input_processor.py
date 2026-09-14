# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert offline and online inputs into validated native beam session requests."""

from __future__ import annotations

import time
from typing import Any

from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.config import GRConfig
from vllm_gr.config.models import (
    CONSTRAINT_TABLE_MAX_DECODE_STEPS,
    ConstraintTableBackendEnum,
)
from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchRequestError
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.entrypoints.beam_search_v1.token_utils import boundary_token_id
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    DEFAULT_CONSTRAINT_RESOURCE_ID,
    BeamBatchItem,
    BeamBatchParams,
    BeamBatchRequest,
    BeamOutputOptions,
    validate_beam_batch_request,
)


def _generate_request_id(external_request_id: str | None = None) -> str:
    """Create a unique internal request/session ID for one run."""
    prefix = external_request_id.strip() if external_request_id else "offline"
    if not prefix:
        raise BeamSearchRequestError("external request_id must not be empty")
    return f"{prefix}-gr-beam-v1-{random_uuid()}"


def _get_token_id(tokenizer: Any, token: str | None, name: str) -> int | None:
    try:
        return boundary_token_id(tokenizer, token, name)
    except ValueError as exc:
        raise BeamSearchRequestError(str(exc)) from exc


def _create_request_params(
    params: Any,
    tokenizer: Any,
) -> tuple[SamplingParams, int | None]:
    begin_token_id = _get_token_id(tokenizer, getattr(params, "begin_token", None), "begin_token")
    end_token_id = _get_token_id(tokenizer, getattr(params, "end_token", None), "end_token")
    reserved_tokens = int(begin_token_id is not None) + int(end_token_id is not None)
    decode_steps = params.max_tokens - reserved_tokens
    if decode_steps <= 0:
        raise BeamSearchRequestError(
            "max_tokens must leave at least one Decode step after begin/end tokens"
        )

    stop_token_id = end_token_id
    if stop_token_id is None:
        stop_token_id = getattr(tokenizer, "eos_token_id", None)
    stop_token_ids = () if stop_token_id is None else (int(stop_token_id),)
    if params.ignore_eos:
        stop_token_ids = ()
    sampling_params = SamplingParams(
        logprobs=params.beam_width,
        max_tokens=decode_steps,
        temperature=params.temperature,
        ignore_eos=params.ignore_eos,
        stop_token_ids=list(stop_token_ids),
        detokenize=False,
        flat_logprobs=True,
        skip_clone=True,
    )
    return sampling_params, begin_token_id


def _prepare_prompt(request: EngineCoreRequest, begin_token_id: int | None) -> None:
    # Mixed inputs also carry token IDs. Reject embeddings before changing IDs
    # so their per-position mask and embedding rows cannot become misaligned.
    if request.prompt_embeds is not None or request.prompt_token_ids is None:
        raise BeamSearchRequestError("embedding prompts are not supported by beam_search_v1")
    if begin_token_id is not None:
        request.prompt_token_ids = [*request.prompt_token_ids, begin_token_id]


def _validate_parallel_config(parallel: Any) -> None:
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        raise BeamSearchRequestError(
            "beam_search_v1 currently requires TP/PP/DP/context parallel sizes of 1"
        )


def _validate_request(request: BeamBatchRequest, vllm_config: Any) -> None:
    validate_beam_batch_request(request)
    if len(request.items) != 1:
        raise BeamSearchRequestError("beam_search_v1 currently requires exactly one prompt")

    gr_config = GRConfig.from_vllm_config(vllm_config)
    _validate_parallel_config(vllm_config.parallel_config)
    if not getattr(vllm_config.scheduler_config, "async_scheduling", True):
        raise BeamSearchRequestError("beam_search_v1 requires async_scheduling")
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise BeamSearchRequestError("beam_search_v1 does not support speculative decoding")
    model = vllm_config.model_config
    if getattr(model, "is_encoder_decoder", False) or getattr(model, "is_hybrid", False):
        raise BeamSearchRequestError("beam_search_v1 requires a decoder with uniform attention KV")
    if getattr(model, "quantization", None) is not None:
        raise BeamSearchRequestError("beam_search_v1 does not support quantized models")
    if request.items[0].request.lora_request is not None:
        raise BeamSearchRequestError("beam_search_v1 does not support LoRA")
    table = gr_config.constraint_table
    if table is None or not table.enabled or table.backend is not ConstraintTableBackendEnum.CUDA:
        raise BeamSearchRequestError(
            "beam_search_v1 requires an enabled canonical CUDA constraint table"
        )
    if request.beam_params.max_tokens > CONSTRAINT_TABLE_MAX_DECODE_STEPS:
        raise BeamSearchRequestError(
            "generation budget exceeds constraint_table_v1 depth "
            f"({CONSTRAINT_TABLE_MAX_DECODE_STEPS})"
        )
    if request.beam_params.catalog_strategy is not None:
        raise BeamSearchRequestError(
            "catalog_strategy is not applicable to the constraint_table backend"
        )
    max_width = gr_config.beam_max_width
    if max_width is None:
        raise BeamSearchRequestError(
            "beam_search_v1 requires vllm-gr beam_max_width to be configured"
        )
    if request.beam_params.beam_width > max_width:
        raise BeamSearchRequestError(
            f"beam_width {request.beam_params.beam_width} exceeds runtime limit {max_width}"
        )
    if request.beam_params.max_tokens > gr_config.beam_max_decode_steps:
        raise BeamSearchRequestError(
            f"Decode steps {request.beam_params.max_tokens} exceed runtime limit "
            f"{gr_config.beam_max_decode_steps}"
        )

    engine_request = request.items[0].request
    prompt_tokens = engine_request.prompt_token_ids
    if prompt_tokens is None:
        raise BeamSearchRequestError("embedding prompts are not supported by beam_search_v1")
    model_config = vllm_config.model_config
    total_length = len(prompt_tokens) + request.beam_params.max_tokens
    if total_length > model_config.max_model_len:
        raise BeamSearchRequestError(
            f"prompt plus Decode length {total_length} exceeds max_model_len "
            f"{model_config.max_model_len}"
        )
    scheduler_config = vllm_config.scheduler_config
    if len(prompt_tokens) > scheduler_config.max_num_batched_tokens:
        raise BeamSearchRequestError(
            f"Prefill length {len(prompt_tokens)} exceeds max_num_batched_tokens "
            f"{scheduler_config.max_num_batched_tokens}"
        )


def _create_batch_request(
    *,
    session_id: str,
    engine_request: EngineCoreRequest,
    params: Any,
    vllm_config: Any,
) -> BeamBatchRequest:
    if engine_request.request_id != session_id:
        raise BeamSearchRequestError("EngineCoreRequest.request_id must equal the GR session ID")
    sampling_params = engine_request.sampling_params
    if sampling_params is None or sampling_params.max_tokens is None:
        raise BeamSearchRequestError("beam_search_v1 requires sampling parameters with max_tokens")
    # Native processing merges generation_config EOS IDs and stores the primary
    # EOS separately. Beam metadata must describe the same effective stop set.
    stop_token_ids = set(sampling_params.stop_token_ids or ())
    if sampling_params.eos_token_id is not None:
        stop_token_ids.add(sampling_params.eos_token_id)
    batch_params = BeamBatchParams(
        beam_width=params.beam_width,
        max_tokens=sampling_params.max_tokens,
        temperature=sampling_params.temperature,
        stop_token_ids=tuple(sorted(stop_token_ids)),
        ignore_stop_tokens=sampling_params.ignore_eos,
        constraint_resource_id=DEFAULT_CONSTRAINT_RESOURCE_ID,
        length_penalty=params.length_penalty,
        catalog_strategy=getattr(params, "catalog_strategy", None),
    )
    output_options = BeamOutputOptions(
        num_return_sequences=batch_params.beam_width,
        include_stop_token=bool(getattr(params, "include_stop_str_in_output", False)),
        include_cumulative_logprob=True,
        include_parent_beam_indices=True,
    )
    request = BeamBatchRequest(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=session_id,
        items=(BeamBatchItem(item_index=0, request=engine_request),),
        beam_params=batch_params,
        output_options=output_options,
    )
    _validate_request(request, vllm_config)
    return request


class BeamSearchInputProcessor:
    """Prepare requests using the engine's native InputProcessor and limits."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def process_offline_inputs(self, llm: Any, call: OfflineBeamSearchRequest) -> BeamBatchRequest:
        if len(call.prompts) != 1:
            raise BeamSearchRequestError("beam_search_v1 currently requires one prompt")
        if call.concurrency_limit is not None and call.concurrency_limit <= 0:
            raise BeamSearchRequestError("concurrency_limit must be positive or None")
        tokenizer = llm.get_tokenizer()
        sampling_params, begin_token_id = _create_request_params(call.params, tokenizer)
        engine_inputs = llm._preprocess_cmpl(call.prompts)
        lora_requests = llm._lora_request_to_seq(call.lora_request, 1)
        request_id = _generate_request_id()
        engine_request = self.engine.input_processor.process_inputs(
            request_id,
            engine_inputs[0],
            sampling_params,
            supported_tasks=self.engine.get_supported_tasks(),
            arrival_time=time.time(),
            lora_request=lora_requests[0],
        )
        _prepare_prompt(engine_request, begin_token_id)
        self.engine.input_processor.assign_request_id(engine_request)
        session_id = engine_request.request_id
        return _create_batch_request(
            session_id=session_id,
            engine_request=engine_request,
            params=call.params,
            vllm_config=self.engine.vllm_config,
        )

    async def process_online_inputs(
        self, serving: Any, call: OnlineBeamSearchRequest
    ) -> BeamBatchRequest:
        tokenizer = serving.input_processor.tokenizer
        if tokenizer is None:
            raise BeamSearchRequestError("beam_search_v1 requires tokenizer initialization")
        sampling_params, begin_token_id = _create_request_params(call.params, tokenizer)
        request_id = _generate_request_id(call.request_id)
        engine_request = self.engine.input_processor.process_inputs(
            request_id,
            call.prompt,
            sampling_params,
            supported_tasks=await self.engine.get_supported_tasks(),
            arrival_time=time.time(),
            lora_request=call.lora_request,
            trace_headers=call.trace_headers,
            priority=call.priority,
            data_parallel_rank=call.data_parallel_rank,
        )
        _prepare_prompt(engine_request, begin_token_id)
        self.engine.input_processor.assign_request_id(engine_request)
        # Native assignment copies our generated ID into external_req_id.
        # Preserve the caller's original identity for output and abort routing.
        engine_request.external_req_id = call.request_id
        session_id = engine_request.request_id
        return _create_batch_request(
            session_id=session_id,
            engine_request=engine_request,
            params=call.params,
            vllm_config=self.engine.vllm_config,
        )
