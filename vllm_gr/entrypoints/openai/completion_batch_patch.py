# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run a prompt list as one fixed-size beam batch on the completions route.

The completions route expands a prompt list into one engine request per prompt,
so k prompts become k beam sessions the engine runs one after another, even
though the V1 engine runs the members of one logical batch in a single forward.

This module claims that case for the batch sizes the engine supports. Every
other request -- non-beam, streaming, embedded prompts, a single prompt, an
unsupported count, insufficient capacity, mismatched budgets, a LEGACY engine --
delegates to the original handler. Each such decline is logged with its reason,
because a list that did not batch is otherwise indistinguishable from one that
did: the response shape is identical either way.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import aclosing
from http import HTTPStatus
from typing import TYPE_CHECKING

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.renderers.inputs.preprocess import prompt_to_seq

from vllm_gr.config import BeamExecutionMode, GRConfig
from vllm_gr.logger import init_logger
from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES

if TYPE_CHECKING:
    from fastapi import Request
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.outputs import RequestOutput

logger = init_logger(__name__)

_COMPLETION_BATCH_PATCH_MARKER = "_vllm_gr_completion_batch_patch"

# Filled in by ``patch_completion_batch``. Kept module-level so the wrapper has
# exactly one referent: the function object the route resolved when it was
# installed, not whatever the class attribute holds later.
_original_create_completion = None


def _batch_size(serving, request: CompletionRequest) -> int | None:
    """Return the prompt count the caller asked to batch, or None to ignore.

    None is the common answer: delegating is how this patch stays inert on
    every request it does not exist to serve, so each condition below is a
    reason to run the upstream body rather than an error to report. The gates
    need nothing but the request, so they run before the renderer. A count
    above one comes back even when this route cannot form it, so the route can
    still say why it declined.
    """
    if not request.use_beam_search or request.stream:
        return None
    # Bytes prompts carry their own embeddings and are rejected deeper in as
    # unsupported; count them as unclaimed rather than as a batch.
    if request.prompt_embeds is not None:
        return None
    vllm_config = getattr(serving.engine_client, "vllm_config", None)
    # A LEGACY config means this engine was not built for the persistent V1
    # sessions this route runs on.
    if GRConfig.from_vllm_config(vllm_config).beam_execution_mode is not BeamExecutionMode.V1:
        return None
    try:
        count = len(prompt_to_seq(request.prompt))
    except (AttributeError, TypeError):
        return None
    if count <= 1:
        return None
    return count


def _internal_error_response(serving, message: str) -> ErrorResponse:
    # Reached only when the frontend contradicts itself; a server-side defect
    # must not be answered as a caller mistake.
    return serving.create_error_response(
        message, err_type="InternalServerError", status_code=HTTPStatus.INTERNAL_SERVER_ERROR
    )


def _reserved_prompt_tokens(request: CompletionRequest) -> int:
    """Tokens the engine appends to each prompt before it measures the Prefill.

    ``_prepare_prompt`` appends the begin token, so a request carrying one is a
    token longer per prompt than the renderer reports; the end token is a
    decode-side stop, not prompt length.
    """
    extra_args = getattr(request, "vllm_xargs", None) or {}
    return 1 if extra_args.get("begin_token") is not None else 0


def _effective_budgets(serving, request: CompletionRequest, engine_inputs: list) -> list[int]:
    """Resolve each input's generation budget the way the upstream body does."""
    max_model_len = serving.model_config.max_model_len
    return [
        get_max_tokens(
            max_model_len,
            request.max_tokens,
            serving._extract_prompt_len(engine_input),
            serving.default_sampling_params,
            serving.override_max_tokens,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )
        for engine_input in engine_inputs
    ]


def _output_boundary_tokens(request: CompletionRequest) -> int:
    """Public budget positions the engine produces outside the Decode steps.

    ``_create_request_params`` subtracts the begin and end tokens from
    ``max_tokens`` before the engine sees the budget, so neither position costs
    a Decode step. Both are read from the same ``vllm_xargs`` mapping the beam
    adapter uses.
    """
    extra_args = getattr(request, "vllm_xargs", None) or {}
    return int(extra_args.get("begin_token") is not None) + int(
        extra_args.get("end_token") is not None
    )


def _needs_decode_stage(request: CompletionRequest, budget: int) -> bool:
    """Whether a budget leaves work after the Prefill forward.

    The engine runs ``budget - boundary tokens - 1`` Decode steps and accepts
    zero, so an adjusted budget of one is Prefill-only and never occupies a
    Decode row.
    """
    return budget - _output_boundary_tokens(request) > 1


def _decode_batch_capacity(serving, request: CompletionRequest) -> int | None:
    """Bound NPU B*W rows by native input storage, not the Prefill budget."""
    config = serving.engine_client.vllm_config
    device = getattr(getattr(config, "device_config", None), "device", "cuda")
    if str(device).split(":")[0] != "npu":
        return None
    # The OpenAI beam adapter maps n to beam_width (None means one).
    width = request.n if request.n is not None else 1
    return config.scheduler_config.max_num_batched_tokens // width


def _batch_decline_reason(
    serving, request: CompletionRequest, engine_inputs: list, budgets: list[int]
) -> str | None:
    """Why this rendered list cannot run as one session, or None if it can.

    All conditions are checked before the first session is submitted, so a
    list that cannot batch never runs partially.
    """
    if len(set(budgets)) != 1:
        # One session runs one budget. Reducing every prompt to the smallest
        # would silently shorten the generation the caller asked for.
        return f"generation budgets differ across prompts: {budgets}"
    scheduler = serving.engine_client.vllm_config.scheduler_config
    token_budget = scheduler.max_num_batched_tokens
    scheduled_limit = getattr(scheduler, "max_num_scheduled_tokens", None)
    if scheduled_limit is not None:
        token_budget = min(token_budget, scheduled_limit)
    reserved = _reserved_prompt_tokens(request)
    total = sum(
        serving._extract_prompt_len(engine_input) + reserved for engine_input in engine_inputs
    )
    if total > token_budget:
        return f"Prefill tokens {total} exceed the scheduling budget {token_budget}"
    decode_capacity = _decode_batch_capacity(serving, request)
    if (
        _needs_decode_stage(request, budgets[0])
        and decode_capacity is not None
        and len(engine_inputs) > decode_capacity
    ):
        return (
            f"NPU Decode batch size {len(engine_inputs)} exceeds the physical input "
            f"capacity of {decode_capacity} prompts at the requested beam width"
        )
    return None


async def _create_completion_batched(
    self,
    request: CompletionRequest,
    raw_request: Request | None,
    engine_inputs: list,
    max_tokens: int,
) -> object:
    """Run one session holding every input and convert the result once."""
    batch_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
    request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"
    logger.info(
        "Beam batch session request_id=%s prompts=%d",
        request_id,
        len(engine_inputs),
    )
    created_time = int(time.time())

    request_metadata = RequestResponseMetadata(request_id=request_id)
    if raw_request:
        raw_request.state.request_metadata = request_metadata

    lora_request = self._maybe_get_adapters(request)

    trace_headers = (
        None if raw_request is None else await self._get_trace_headers(raw_request.headers)
    )
    for index, engine_input in enumerate(engine_inputs):
        self._log_inputs(
            f"{request_id}-{index}",
            engine_input,
            params=batch_params,
            lora_request=lora_request,
        )
    # One logical batch runs under the request id the caller already sees, so
    # the single-session case keeps the identifiers it had before this patch.
    generator = self.beam_search(
        prompt=engine_inputs,
        request_id=request_id,
        params=batch_params,
        lora_request=lora_request,
        trace_headers=trace_headers,
        session_id=self._get_session_id(request, raw_request),
    )
    try:
        async with aclosing(generator):
            final_res_batch: list[RequestOutput] = [res async for res in generator]
    except asyncio.CancelledError:
        # Match upstream non-streaming completions at the HTTP adapter boundary:
        # iterator cleanup precedes this error response, not a successful result.
        # Engine/client cancellation propagation remains unchanged.
        return self.create_error_response("Client disconnected")

    expected = len(engine_inputs)
    if len(final_res_batch) != expected:
        return _internal_error_response(
            self, f"beam session returned {len(final_res_batch)} outputs for {expected} prompts"
        )
    for index, final_res in enumerate(final_res_batch):
        if final_res.prompt is None:
            final_res.prompt = self._extract_prompt_text(engine_inputs[index])

    return self.request_output_to_completion_response(
        final_res_batch,
        request,
        request_id,
        created_time,
        self.models.model_name(lora_request),
        self.renderer.tokenizer,
        request_metadata,
    )


async def _decline(
    serving, request: CompletionRequest, raw_request: Request | None, count: int, reason: str
) -> object:
    """Run the original handler and record why this list was not batched."""
    logger.info(
        "Beam batch declined request_id=%s prompts=%d reason=%s",
        request.request_id,
        count,
        reason,
    )
    return await _original_create_completion(serving, request, raw_request)


async def create_completion_batch_dispatch(
    self, request: CompletionRequest, raw_request: Request | None = None
) -> object:
    """Route a completion request to the batched or the original handler."""
    count = _batch_size(self, request)
    if count is None:
        return await _original_create_completion(self, request, raw_request)
    sizes = GRConfig.from_vllm_config(self.engine_client.vllm_config).beam_online_batch_sizes
    if count not in sizes:
        return await _decline(
            self,
            request,
            raw_request,
            count,
            f"prompt count {count} is not one of the supported fixed batch sizes "
            f"{[size for size in sizes if size > 1]}",
        )

    result = await self.render_completion_request(request)
    if isinstance(result, ErrorResponse):
        return result
    engine_inputs = result
    if len(engine_inputs) != count:
        # Everything below -- one session and one output per prompt -- rests on
        # the renderer agreeing with the count taken off the request.
        return _internal_error_response(
            self, f"render produced {len(engine_inputs)} inputs for {count} prompts"
        )

    scheduler = self.engine_client.vllm_config.scheduler_config
    max_num_seqs = getattr(scheduler, "max_num_seqs", max(SUPPORTED_BEAM_BATCH_SIZES))
    if count > max_num_seqs:
        return await _decline(
            self,
            request,
            raw_request,
            count,
            f"prompt count {count} exceeds the sequence capacity {max_num_seqs}",
        )

    budgets = _effective_budgets(self, request, engine_inputs)
    reason = _batch_decline_reason(self, request, engine_inputs, budgets)
    if reason is not None:
        # Delegating re-renders. Trading that for a simpler single-session
        # route is deliberate here; the split planner is a separate change.
        return await _decline(self, request, raw_request, count, reason)

    return await _create_completion_batched(self, request, raw_request, engine_inputs, budgets[0])


def patch_completion_batch() -> None:
    """Claim ``_create_completion`` for supported beam prompt lists.

    The route calls this attribute, so replacing it is enough and the
    kv-transfer cleanup wrapped around it upstream keeps applying.
    """
    global _original_create_completion

    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

    current = OpenAIServingCompletion._create_completion
    if getattr(current, _COMPLETION_BATCH_PATCH_MARKER, False):
        return
    _original_create_completion = current
    setattr(create_completion_batch_dispatch, _COMPLETION_BATCH_PATCH_MARKER, True)
    OpenAIServingCompletion._create_completion = create_completion_batch_dispatch


def _validate_completion_batch_patch() -> str | None:
    """Verify the batch route is installed and the upstream seam still exists.

    Fail-closed on the surface the batch path calls. Instance attributes set by
    ``__init__`` (``models``, ``renderer``, ``model_config``, the sampling
    defaults) cannot be read off the class and are left to lookup.
    """
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

    if OpenAIServingCompletion._create_completion is not create_completion_batch_dispatch:
        return "OpenAIServingCompletion._create_completion is not the batch dispatch"
    missing = tuple(
        name
        for name in (
            "render_completion_request",
            "create_error_response",
            "request_output_to_completion_response",
            "_base_request_id",
            "_maybe_get_adapters",
            "_extract_prompt_len",
            "_extract_prompt_text",
            "_log_inputs",
            "_get_trace_headers",
            "_get_session_id",
            "beam_search",
        )
        if not hasattr(OpenAIServingCompletion, name)
    )
    if missing:
        return f"OpenAIServingCompletion is missing {', '.join(missing)}"
    return None
