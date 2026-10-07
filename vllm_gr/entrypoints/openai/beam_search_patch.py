# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def patch_beam_search():
    from vllm.entrypoints.openai.engine.serving import OpenAIServing

    from vllm_gr.entrypoints.openai.serving_engine import beam_search, beam_search_v1

    OpenAIServing.beam_search = beam_search
    OpenAIServing.beam_search_v1 = beam_search_v1
    patch_beam_stream_fast_path()


def _server_stream_fast_path_eligible(self) -> bool:
    """True when the OpenAIServingChat instance has no custom parsers/loggers.

    Computed on first request and cached on ``self``.
    """
    cached = getattr(self, "_vllm_gr_server_stream_fast_eligible", None)
    if cached is not None:
        return bool(cached)
    eligible = (
        not getattr(self, "use_harmony", False)
        and getattr(self, "tool_parser", None) is None
        and getattr(self, "parser_cls", None) is None
        and getattr(self, "reasoning_parser_cls", None) is None
        and not (
            getattr(self, "enable_log_outputs", False) and getattr(self, "request_logger", None)
        )
    )
    self._vllm_gr_server_stream_fast_eligible = eligible
    return eligible


def _beam_stream_fast_path_eligible(self, request, reasoning_parser) -> bool:
    """True when both server and request match the plain beam-streaming shape.

    The fast path replaces the two 128-iteration Pydantic construction loops
    in ``chat_completion_stream_generator`` with template-based chunks.  Any
    optional/exotic streaming feature is delegated to the upstream generator
    to guarantee byte-for-byte JSON parity.

    Note: vllm 0.22.1 defaults ``tool_choice`` to ``True`` and
    ``include_reasoning`` to ``True`` for Qwen-style models; both are inert
    on the plain path (no tools, no tool parser, no harmony, no reasoning
    parser) so they must not gate the fast path.
    """
    stream_options = getattr(request, "stream_options", None)
    return (
        bool(getattr(request, "use_beam_search", False))
        and reasoning_parser is None
        and _server_stream_fast_path_eligible(self)
        and not request.echo
        and not request.tools
        and not getattr(request, "_grammar_from_tool_parser", False)
        and request.tool_choice not in ("auto", "required")
        and not (request.logprobs or request.top_logprobs)
        and not (request.return_token_ids or request.return_prompt_text)
        and not (
            stream_options is not None
            and bool(getattr(stream_options, "continuous_usage_stats", False))
        )
    )


def patch_beam_stream_fast_path() -> None:
    """Fast SSE serialization for the beam-search streaming path.

    ``chat_completion_stream_generator`` builds two 128-element loops of
    Pydantic ``ChatCompletionStreamResponse`` objects (one per beam, twice)
    and serializes each with ``model_dump_json``.  Measured ~16.85 ms of
    synchronous event-loop CPU per request (Phase A).  Because every chunk in
    each loop differs from its sibling only in ``index`` (role loop) or
    ``index`` and ``delta.content`` (standard content loop), the plain
    configuration can emit the exact same bytes from two per-request Pydantic
    templates (one for role chunks, one for content chunks) using plain string
    substitution.
    """
    from functools import wraps

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionResponseStreamChoice,
        ChatCompletionStreamResponse,
    )
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaMessage,
        PromptTokenUsageInfo,
        UsageInfo,
    )
    from vllm.entrypoints.openai.engine.serving import GenerationError
    from vllm.entrypoints.utils import should_include_usage

    if getattr(OpenAIServingChat, "_vllm_gr_beam_stream_fast_path", False):
        return
    _orig_generator = OpenAIServingChat.chat_completion_stream_generator

    @wraps(_orig_generator)
    async def _beam_stream_fast(
        self,
        request,
        result_generator,
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        reasoning_parser=None,
        chat_template_kwargs=None,
    ):
        if not _beam_stream_fast_path_eligible(self, request, reasoning_parser):
            async for chunk in _orig_generator(
                self,
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
                chat_template_kwargs=chat_template_kwargs,
            ):
                yield chunk
            return

        import asyncio
        import json
        import time

        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"
        role = self.get_chat_request_role(request)
        num_choices = request.n if request.n is not None else 1
        include_usage, _include_continuous = should_include_usage(
            request.stream_options, self.enable_force_include_usage
        )

        try:
            res = None
            async for item in result_generator:
                res = item
                break
            if res is None:
                raise RuntimeError("No output received from the engine.")

            num_prompt_tokens = len(res.prompt_token_ids) if res.prompt_token_ids is not None else 0

            # Role chunks: identical except for ``index``.  Derive the exact
            # upstream JSON shape from one Pydantic dump, then substitute the
            # index per chunk.  Flush the first chunk immediately so the HTTP
            # client receives the first byte while the remaining chunks are
            # prepared.
            role_choice = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role=role, content=""),
                logprobs=None,
                finish_reason=None,
            )
            role_tpl = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[role_choice],
                model=model_name,
                prompt_token_ids=None,
                prompt_text=None,
            ).model_dump_json(exclude_unset=True)
            role_prefix, role_suffix = role_tpl.split('"index":0', 1)
            for i in range(num_choices):
                yield f'data: {role_prefix}"index":{i}{role_suffix}\n\n'
                if i == 0:
                    await asyncio.sleep(0)

            # Content chunks: derive one Pydantic template for the standard
            # (finish_reason="length", stop_reason=None) beam output and
            # substitute ``index`` + JSON-encoded ``delta.content`` per beam.
            content_marker = "__VLLM_GR_CONTENT__"
            content_tpl_choice = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=content_marker),
                logprobs=None,
                finish_reason="length",
                stop_reason=None,
                token_ids=None,
            )
            content_tpl_chunk = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[content_tpl_choice],
                model=model_name,
            )
            if not include_usage and self.system_fingerprint is not None:
                content_tpl_chunk.system_fingerprint = self.system_fingerprint
            content_tpl = content_tpl_chunk.model_dump_json(exclude_unset=True)
            content_target = f'"index":0,"delta":{{"content":"{content_marker}"}}'
            content_prefix, content_suffix = content_tpl.split(content_target, 1)

            first_content = True
            for output in res.outputs:
                if output.finish_reason is not None:
                    self._raise_if_error(output.finish_reason, request_id)
                finish_reason = output.finish_reason if output.finish_reason else "stop"
                if finish_reason == "length" and output.stop_reason is None:
                    escaped_text = json.dumps(output.text or "", ensure_ascii=False)
                    yield (
                        f'data: {content_prefix}"index":{output.index},'
                        f'"delta":{{"content":{escaped_text}}}{content_suffix}\n\n'
                    )
                else:
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=output.index,
                        delta=DeltaMessage(content=output.text or ""),
                        logprobs=None,
                        finish_reason=finish_reason,
                        stop_reason=output.stop_reason,
                        token_ids=None,
                    )
                    stream_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    if (
                        not include_usage
                        and self.system_fingerprint is not None
                        and choice_data.finish_reason is not None
                    ):
                        stream_chunk.system_fingerprint = self.system_fingerprint
                    yield f"data: {stream_chunk.model_dump_json(exclude_unset=True)}\n\n"
                if first_content:
                    first_content = False
                    await asyncio.sleep(0)

            # Exhaust the beam-search generator so its ``finally`` block runs
            # the session purge (upstream's async-for does the same before the
            # usage/[DONE] tail).
            async for _item in result_generator:
                pass

            completion_tokens = sum(len(o.token_ids) for o in res.outputs)
            if include_usage:
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                if self.enable_prompt_tokens_details and res.num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=res.num_cached_tokens
                    )
                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                    system_fingerprint=self.system_fingerprint,
                )
                yield (
                    f"data: {final_usage_chunk.model_dump_json(exclude_unset=True, exclude_none=True)}\n\n"
                )

            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens,
            )
        except GenerationError as e:
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
        except Exception as e:  # noqa: BLE001
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    OpenAIServingChat.chat_completion_stream_generator = _beam_stream_fast
    OpenAIServingChat._vllm_gr_beam_stream_fast_path = True


def patch_sampling():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    from vllm_gr.entrypoints.openai.protocol import to_beam_search_params

    ChatCompletionRequest.to_beam_search_params = to_beam_search_params
    CompletionRequest.to_beam_search_params = to_beam_search_params


def patch_engine_core_process_entrypoint() -> None:
    """Wrap the EngineCore process entrypoint for spawn-based runtimes."""
    from vllm.logger import init_logger
    from vllm.v1.engine.core import EngineCoreProc

    from vllm_gr.v1.engine.engine_core_patch import run_engine_core

    init_logger(__name__).info("vllm-gr: patch_engine_core_process_entrypoint called")

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
