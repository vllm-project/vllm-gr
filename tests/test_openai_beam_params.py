# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise Beam extension parameters through the upstream HTTP routes."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from vllm.entrypoints.openai.chat_completion.api_router import router as chat_router
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.completion.api_router import router as completion_router
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.serve.engine import protocol as engine_protocol

from vllm_gr.entrypoints.openai.beam_search_patch import patch_sampling
from vllm_gr.sampling_params import BeamSearchParams


@pytest.fixture  # type: ignore[misc]
def sampling_patch() -> Iterator[None]:
    field = CompletionRequest.model_fields["vllm_xargs"]
    original_annotation = field.annotation
    original_completion = CompletionRequest.to_beam_search_params
    original_chat = ChatCompletionRequest.to_beam_search_params
    try:
        patch_sampling()
        patch_sampling()  # Startup patch installation is idempotent.
        yield
    finally:
        field.annotation = original_annotation
        CompletionRequest.model_rebuild(force=True)
        CompletionRequest.to_beam_search_params = original_completion
        ChatCompletionRequest.to_beam_search_params = original_chat


@pytest.mark.parametrize("chat", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("explicit_k", [False, True])  # type: ignore[misc]
def test_http_beam_extension_parameters_reach_handler(
    sampling_patch: None, chat: bool, explicit_k: bool
) -> None:
    _exercise_http_beam_extension_parameters(chat, explicit_k)


def _exercise_http_beam_extension_parameters(chat: bool, explicit_k: bool) -> None:
    # These are the real upstream routers, imported before the patch is applied.
    # Only model serving is recorded; JSON validation and parameter conversion
    # use the production classes and functions.
    router = chat_router if chat else completion_router
    response_type = ChatCompletionResponse if chat else CompletionResponse
    received: list[tuple[dict[str, Any], BeamSearchParams]] = []

    async def handler(
        request: Any, raw_request: Request
    ) -> ChatCompletionResponse | CompletionResponse:
        params = request.to_beam_search_params(request.max_tokens, {})
        received.append((request.vllm_xargs, params))
        return response_type(
            id="test-response",
            created=0,
            model="test-model",
            choices=[],
            usage=engine_protocol.UsageInfo(),
        )

    app = FastAPI()
    app.state.openai_serving_chat = SimpleNamespace(create_chat_completion=handler)
    app.state.openai_serving_completion = SimpleNamespace(create_completion=handler)
    app.include_router(router)
    extra: dict[str, Any] = {
        "result_width": 1,
        "custom_tag": "legacy",
        "custom_count": 7,
        "custom_scale": 0.5,
    }
    if explicit_k:
        extra.update(dynamic_beam_widths=[4, 2, 1], per_parent_beam_widths=[4, 2, 1])
    payload: dict[str, Any] = {
        "model": "test-model",
        "n": 4,
        "max_tokens": 3,
        "use_beam_search": True,
        "ignore_eos": True,
        "vllm_xargs": extra,
    }
    if chat:
        payload["messages"] = [{"role": "user", "content": "hello"}]
    else:
        payload["prompt"] = "hello"
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions" if chat else "/v1/completions", json=payload)
    assert response.status_code == 200, response.text
    assert len(received) == 1
    extra_received, params = received[0]
    assert extra_received == extra
    assert params.result_width == 1
    assert params.dynamic_beam_widths == ([4, 2, 1] if explicit_k else None)
    assert params.per_parent_beam_widths == ([4, 2, 1] if explicit_k else None)


def test_chat_http_handles_replaced_usage_info(
    sampling_patch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ascend replaces UsageInfo and rebuilds the Chat response after collection.
    replacement_usage_info = type("ReplacementUsageInfo", (engine_protocol.UsageInfo,), {})

    field = ChatCompletionResponse.model_fields["usage"]
    try:
        with monkeypatch.context() as patch:
            patch.setattr(engine_protocol, "UsageInfo", replacement_usage_info)
            patch.setitem(ChatCompletionResponse.__annotations__, "usage", replacement_usage_info)
            patch.setattr(field, "annotation", replacement_usage_info)
            ChatCompletionResponse.model_rebuild(force=True)
            for explicit_k in (False, True):
                _exercise_http_beam_extension_parameters(chat=True, explicit_k=explicit_k)
    finally:
        # Restore Pydantic's validator after monkeypatch has restored the field.
        ChatCompletionResponse.model_rebuild(force=True)
