# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import msgspec
from vllm.logger import init_logger

from vllm_gr.utils.env import is_truthy_env

logger = init_logger(__name__)


def _use_batched_beam_stream(self: Any, request: Any, reasoning_parser: Any) -> bool:
    """Return whether the simple beam response can batch its SSE choices."""
    stream_options = request.stream_options
    return bool(
        is_truthy_env("VLLM_GR_BEAM_PROC_ENABLED")
        and request.use_beam_search
        and request.stream
        and not request.logprobs
        and not request.return_token_ids
        and not request.echo
        and not request.tools
        and request.tool_choice in (None, "none")
        and reasoning_parser is None
        and not self.use_harmony
        and self.parser_cls is None
        and not self.enable_log_outputs
        and not self.enable_prompt_tokens_details
        and not self.enable_force_include_usage
        and not (stream_options is not None and stream_options.continuous_usage_stats)
    )


async def _batched_beam_stream_generator(
    self: Any,
    request: Any,
    result_generator: AsyncIterator[Any],
    request_id: str,
    model_name: str,
    conversation: Any,
    tokenizer: Any,
    request_metadata: Any,
    reasoning_parser: Any = None,
    chat_template_kwargs: Any = None,
) -> AsyncGenerator[str, None]:
    """Stream completed beams without a separate empty role event per beam."""
    from vllm.entrypoints.openai.engine.serving import GenerationError

    final_res = None
    try:
        async for res in result_generator:
            final_res = res

        if final_res is None:
            data = self.create_streaming_error_response(
                RuntimeError("No output received from the engine.")
            )
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        role = self.get_chat_request_role(request)
        created_time = int(time.time())
        for output in final_res.outputs:
            self._raise_if_error(output.finish_reason, request_id)
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [
                    {
                        "index": output.index,
                        "delta": {"role": role, "content": output.text},
                        "logprobs": None,
                        "finish_reason": output.finish_reason or "stop",
                        "stop_reason": output.stop_reason,
                    }
                ],
            }
            if self.system_fingerprint is not None:
                chunk["system_fingerprint"] = self.system_fingerprint
            yield f"data: {msgspec.json.encode(chunk).decode()}\n\n"

        prompt_tokens = len(final_res.prompt_token_ids or ())
        prompt_tokens += len(final_res.encoder_prompt_token_ids or ())
        completion_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        from vllm.entrypoints.openai.engine.protocol import UsageInfo

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        request_metadata.final_usage_info = usage
        stream_options = request.stream_options
        include_usage = bool(
            (stream_options is not None and stream_options.include_usage)
            or self.enable_force_include_usage
        )
        if include_usage:
            usage_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [],
                "usage": usage.model_dump(exclude_none=True),
            }
            if self.system_fingerprint is not None:
                usage_chunk["system_fingerprint"] = self.system_fingerprint
            yield f"data: {msgspec.json.encode(usage_chunk).decode()}\n\n"
    except asyncio.CancelledError:
        return
    except GenerationError as exc:
        yield f"data: {self._convert_generation_error_to_streaming_response(exc)}\n\n"
    except Exception as exc:
        logger.exception("Error in optimized beam stream generator.")
        data = self.create_streaming_error_response(exc)
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


def patch_beam_search() -> None:
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.engine.serving import OpenAIServing

    from vllm_gr.entrypoints.openai.serving_engine import beam_search

    OpenAIServing.beam_search = beam_search
    if not getattr(OpenAIServingChat, "_vllm_gr_batched_beam_stream", False):
        original_stream_generator = OpenAIServingChat.chat_completion_stream_generator

        def beam_stream_generator(
            self: Any, request: Any, *args: Any, **kwargs: Any
        ) -> AsyncGenerator[str, None]:
            reasoning_parser = kwargs.get("reasoning_parser")
            if reasoning_parser is None and len(args) >= 7:
                reasoning_parser = args[6]
            if _use_batched_beam_stream(self, request, reasoning_parser):
                return _batched_beam_stream_generator(self, request, *args, **kwargs)
            return original_stream_generator(self, request, *args, **kwargs)

        OpenAIServingChat.chat_completion_stream_generator = beam_stream_generator
        OpenAIServingChat._vllm_gr_batched_beam_stream = True
    logger.info(
        "vllm_gr patched OpenAIServing.beam_search with %s",
        beam_search.__module__,
    )


def patch_sampling():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    from vllm_gr.entrypoints.openai.protocol import to_beam_search_params

    ChatCompletionRequest.to_beam_search_params = to_beam_search_params


def patch_engine_core_process_entrypoint() -> None:
    """Wrap the EngineCore process entrypoint for spawn-based runtimes."""
    from vllm.v1.engine.core import EngineCoreProc

    from vllm_gr.entrypoints.openai.beam_proc_manager import ensure_beam_proc_config
    from vllm_gr.v1.engine.core_client_patch import apply_batch_fork_patches
    from vllm_gr.v1.engine.engine_core_patch import run_engine_core

    ensure_beam_proc_config()
    apply_batch_fork_patches()
    # Save the original so the wrapper can delegate without recursion
    # even if run_patch() is called again in a spawned child process.
    current_run_engine_core = EngineCoreProc.run_engine_core
    if current_run_engine_core is run_engine_core:
        return
    original_run_engine_core = getattr(
        run_engine_core,
        "_original_run_engine_core",
        current_run_engine_core,
    )
    if original_run_engine_core is run_engine_core:
        original_run_engine_core = current_run_engine_core
    run_engine_core._original_run_engine_core = original_run_engine_core
    EngineCoreProc.run_engine_core = run_engine_core


def patch_batch_and_fork() -> None:
    """Compatibility helper that installs the complete grouped-beam pipeline."""
    from vllm_gr.v1.engine.core_client_patch import apply_batch_fork_patches

    apply_batch_fork_patches()
    patch_engine_core_process_entrypoint()
