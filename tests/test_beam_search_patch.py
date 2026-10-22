# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_gr.entrypoints.openai.beam_search_patch import (
    _batched_beam_stream_generator,
    _use_batched_beam_stream,
)


@pytest.mark.parametrize(  # type: ignore
    ("target", "field", "value"),
    [
        ("request", "use_beam_search", False),
        ("request", "stream", False),
        ("request", "logprobs", True),
        ("request", "return_token_ids", True),
        ("request", "echo", True),
        ("request", "tools", ["tool"]),
        ("request", "tool_choice", "auto"),
        ("request", "stream_options", SimpleNamespace(continuous_usage_stats=True)),
        ("serving", "use_harmony", True),
        ("serving", "parser_cls", object()),
        ("serving", "enable_log_outputs", True),
        ("serving", "enable_prompt_tokens_details", True),
        ("serving", "enable_force_include_usage", True),
    ],
)
def test_batched_beam_stream_requires_simple_response(
    monkeypatch: pytest.MonkeyPatch, target: str, field: str, value: Any
) -> None:
    monkeypatch.setenv("VLLM_GR_BEAM_PROC_ENABLED", "1")
    serving = SimpleNamespace(
        use_harmony=False,
        parser_cls=None,
        enable_log_outputs=False,
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
    )
    request = SimpleNamespace(
        use_beam_search=True,
        stream=True,
        logprobs=False,
        return_token_ids=False,
        echo=False,
        tools=None,
        tool_choice=None,
        stream_options=None,
    )
    assert _use_batched_beam_stream(serving, request, None)
    assert not _use_batched_beam_stream(serving, request, object())
    monkeypatch.setenv("VLLM_GR_BEAM_PROC_ENABLED", "0")
    assert not _use_batched_beam_stream(serving, request, None)
    monkeypatch.setenv("VLLM_GR_BEAM_PROC_ENABLED", "1")
    setattr(request if target == "request" else serving, field, value)
    assert not _use_batched_beam_stream(serving, request, None)


@pytest.mark.asyncio  # type: ignore
@pytest.mark.parametrize("include_usage", [False, True])  # type: ignore
async def test_batched_beam_stream_preserves_choices_and_usage(include_usage: bool) -> None:
    async def results() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            prompt_token_ids=[1, 2],
            encoder_prompt_token_ids=[5],
            outputs=[
                SimpleNamespace(
                    index=index,
                    text=f"beam-{index}",
                    finish_reason="length",
                    stop_reason=None,
                    token_ids=[3, 4],
                )
                for index in range(2)
            ],
        )

    serving = SimpleNamespace(
        system_fingerprint=None,
        enable_force_include_usage=False,
        get_chat_request_role=lambda request: "assistant",
        create_streaming_error_response=lambda exc: str(exc),
        _raise_if_error=lambda finish_reason, request_id: None,
    )
    request = SimpleNamespace(stream_options=SimpleNamespace(include_usage=include_usage))
    request_metadata = SimpleNamespace(final_usage_info=None)

    chunks = [
        chunk
        async for chunk in _batched_beam_stream_generator(
            serving,
            request,
            results(),
            "request-id",
            "model",
            [],
            None,
            request_metadata,
        )
    ]

    responses = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks[:2]]
    assert [response["choices"][0]["delta"]["content"] for response in responses] == [
        "beam-0",
        "beam-1",
    ]
    for index, response in enumerate(responses):
        choice = response["choices"][0]
        assert choice["index"] == index
        assert choice["delta"]["role"] == "assistant"
        assert choice["finish_reason"] == "length"
        assert choice["stop_reason"] is None
    if include_usage:
        usage = json.loads(chunks[2].removeprefix("data: "))["usage"]
        assert usage["prompt_tokens"] == 3
        assert usage["completion_tokens"] == 4
    assert request_metadata.final_usage_info.total_tokens == 7
    assert len(chunks) == 3 + int(include_usage)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio  # type: ignore
@pytest.mark.parametrize(  # type: ignore
    "failure", ["empty", "iterator", "generation", "cancelled"]
)
async def test_batched_beam_stream_handles_errors(failure: str) -> None:
    import asyncio

    from vllm.entrypoints.openai.engine.serving import GenerationError

    async def results() -> AsyncIterator[Any]:
        if failure == "iterator":
            raise RuntimeError("engine failed")
        if failure == "cancelled":
            raise asyncio.CancelledError
        if failure == "generation":
            yield SimpleNamespace(outputs=[SimpleNamespace(finish_reason="error")])

    def raise_if_error(*args: Any) -> None:
        raise GenerationError("generation failed")

    serving = SimpleNamespace(
        get_chat_request_role=lambda request: "assistant",
        _raise_if_error=raise_if_error,
        create_streaming_error_response=lambda exc: json.dumps({"error": str(exc)}),
        _convert_generation_error_to_streaming_response=lambda exc: json.dumps(
            {"generation_error": str(exc)}
        ),
    )
    chunks = [
        chunk
        async for chunk in _batched_beam_stream_generator(
            serving, None, results(), "request-id", "model", [], None, None
        )
    ]
    if failure == "cancelled":
        assert chunks == []
    else:
        assert len(chunks) == 2
        payload = json.loads(chunks[0].removeprefix("data: "))
        assert ("generation_error" if failure == "generation" else "error") in payload
        assert chunks[-1] == "data: [DONE]\n\n"
