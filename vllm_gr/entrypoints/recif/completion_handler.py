# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RECIF dispatch for vLLM's native OpenAI completion handler."""

import time
from functools import wraps
from http import HTTPStatus
from typing import Any

from fastapi import Request
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput

from vllm_gr.entrypoints.recif.serving_engine import recif_beam_search
from vllm_gr.models.recif import NUM_SID_LEVELS, RECIF_ARCHITECTURE

logger = init_logger(__name__)


def _is_recif_serving(serving: Any) -> bool:
    architectures = getattr(serving.model_config, "architectures", None) or []
    return RECIF_ARCHITECTURE in architectures


def _extract_prompt_token_ids(engine_input: Any) -> list[int]:
    if engine_input["type"] == "embeds":
        raise ValueError("RECIF does not support prompt embeddings")
    decoder_prompt = (
        engine_input if engine_input["type"] != "enc_dec" else engine_input["decoder_prompt"]
    )
    return list(decoder_prompt["prompt_token_ids"])


def _build_recif_completion_response(
    serving: OpenAIServingCompletion,
    output: RequestOutput,
    request_id: str,
    created_time: int,
    model_name: str,
    request_metadata: RequestResponseMetadata,
) -> CompletionResponse:
    """Build a RECIF completion response without repeating the prompt.

    The generic OpenAI builder attaches ``prompt_token_ids`` to every
    choice when ``return_token_ids=true``, repeating the full RECIF
    history once per result. RECIF clients only need the three generated
    level token IDs, so those are returned per choice while
    ``prompt_token_ids`` is omitted entirely.
    """

    choices: list[CompletionResponseChoice] = []
    num_prompt_tokens = len(output.prompt_token_ids or [])
    num_generated_tokens = 0
    for completion in output.outputs:
        token_ids = list(completion.token_ids)
        num_generated_tokens += len(token_ids)
        choices.append(
            CompletionResponseChoice(
                index=completion.index,
                text="",
                logprobs=None,
                finish_reason=completion.finish_reason,
                stop_reason=completion.stop_reason,
                token_ids=token_ids,
                prompt_logprobs=None,
                prompt_token_ids=None,
                routed_experts=None,
            )
        )
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    request_metadata.final_usage_info = usage
    return CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
        system_fingerprint=serving.system_fingerprint,
        kv_transfer_params=output.kv_transfer_params,
    )


async def _create_recif_completion(
    serving: OpenAIServingCompletion,
    request: CompletionRequest,
    raw_request: Request | None,
):
    if not request.use_beam_search:
        return serving.create_error_response("RECIF requires use_beam_search=true")
    if request.stream:
        return serving.create_error_response(
            "Streaming is not currently supported with RECIF beam search"
        )
    if request.max_tokens != NUM_SID_LEVELS:
        return serving.create_error_response(f"RECIF requires max_tokens={NUM_SID_LEVELS}")
    if request.echo:
        return serving.create_error_response("RECIF does not support echo=true")
    if request.logprobs is not None:
        return serving.create_error_response("RECIF does not currently support completion logprobs")
    if not request.return_token_ids:
        return serving.create_error_response(
            "RECIF requires return_token_ids=true; convert the returned token IDs in the client"
        )

    rendered = await serving.render_completion_request(request)
    if isinstance(rendered, ErrorResponse):
        return rendered
    if len(rendered) != 1:
        return serving.create_error_response(
            "RECIF currently accepts exactly one token-ID prompt per request"
        )

    request_id = f"cmpl-{serving._base_request_id(raw_request, request.request_id)}"
    request_metadata = RequestResponseMetadata(request_id=request_id)
    if raw_request is not None:
        raw_request.state.request_metadata = request_metadata

    engine_input = rendered[0]
    try:
        prompt_token_ids = _extract_prompt_token_ids(engine_input)
    except (KeyError, TypeError, ValueError) as exc:
        return serving.create_error_response(str(exc))

    lora_request = serving._maybe_get_adapters(request)
    trace_headers = (
        None if raw_request is None else await serving._get_trace_headers(raw_request.headers)
    )
    data_parallel_rank = serving._get_data_parallel_rank(raw_request)
    beam_params = request.to_beam_search_params(
        NUM_SID_LEVELS,
        serving.default_sampling_params,
    )
    serving._log_inputs(
        request_id,
        engine_input,
        params=beam_params,
        lora_request=lora_request,
    )
    try:
        output = await recif_beam_search(
            serving.engine_client,
            prompt_token_ids,
            request.n,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=request.priority,
            data_parallel_rank=data_parallel_rank,
        )
    except (TypeError, ValueError) as exc:
        return serving.create_error_response(str(exc))
    except RuntimeError as exc:
        logger.warning("RECIF beam search failed: %s", exc)
        return serving.create_error_response(
            str(exc),
            err_type="InternalServerError",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    model_name = serving.models.model_name(lora_request)
    return _build_recif_completion_response(
        serving,
        output,
        request_id,
        int(time.time()),
        model_name,
        request_metadata,
    )


def patch_recif_completion_handler() -> bool:
    """Dispatch RECIF requests inside vLLM's existing completion handler."""

    current = OpenAIServingCompletion._create_completion
    if getattr(current, "_vllm_gr_recif", False):
        return False

    @wraps(current)
    async def create_completion(
        self: OpenAIServingCompletion,
        request: CompletionRequest,
        raw_request: Request | None = None,
    ):
        if _is_recif_serving(self):
            return await _create_recif_completion(self, request, raw_request)
        return await current(self, request, raw_request)

    create_completion._vllm_gr_recif = True  # type: ignore[attr-defined]
    OpenAIServingCompletion._create_completion = create_completion
    return True
