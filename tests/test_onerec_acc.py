# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="no-untyped-def"

import pytest

from benchmarks.open_one_rec.one_rec_acc_test import (
    MockCompletionOutput,
    MockRequestOutput,
    _validate_recommendation_outputs,
    run_api_inference,
)


def test_completion_api_sends_rendered_prompt_verbatim(monkeypatch) -> None:
    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"text": "first"}, {"text": "second"}]}

    def post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return Response()

    import requests

    monkeypatch.setattr(requests, "post", post)
    outputs = run_api_inference(
        ["<|im_start|>system\nrendered"],
        "model",
        "http://127.0.0.1:8000/v1/completions",
        "token",
        {
            "num_beams": 2,
            "num_return_sequences": 2,
            "max_new_tokens": 5,
            "temperature": 0.6,
            "top_p": 0.95,
        },
        max_concurrency=1,
    )

    assert calls[0][0] == "http://127.0.0.1:8000/v1/completions"
    assert calls[0][2]["prompt"] == "<|im_start|>system\nrendered"
    assert calls[0][3] == 600.0
    assert "messages" not in calls[0][2]
    assert [choice.text for choice in outputs[0].outputs] == ["first", "second"]


def test_chat_api_keeps_message_payload_and_response_parsing(monkeypatch) -> None:
    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "reasoning-",
                            "content": "answer",
                        }
                    }
                ]
            }

    def post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return Response()

    import requests

    monkeypatch.setattr(requests, "post", post)
    outputs = run_api_inference(
        ["rendered"],
        "model",
        "http://127.0.0.1:8000/v1/chat/completions",
        "token",
        {
            "num_beams": 1,
            "num_return_sequences": 1,
            "max_new_tokens": 5,
            "temperature": 0.6,
            "top_p": 0.95,
        },
        max_concurrency=1,
    )

    assert calls[0][2]["messages"] == [{"role": "user", "content": "rendered"}]
    assert "prompt" not in calls[0][2]
    assert outputs[0].outputs[0].text == "reasoning-answer"


def test_api_transport_failure_is_not_converted_to_empty_output(monkeypatch) -> None:
    import requests

    def post(url, *, headers, json, timeout):
        raise requests.ConnectionError("connection lost")

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(RuntimeError, match="1 of 1 API requests failed"):
        run_api_inference(
            ["prompt"],
            "model",
            "http://127.0.0.1:8000/v1/completions",
            "token",
            {"num_return_sequences": 2},
            max_concurrency=1,
        )


@pytest.mark.parametrize(  # type: ignore[misc]
    "document",
    [
        {},
        {"choices": []},
        {"choices": [{"text": "only-one"}]},
        {"choices": [{"text": "first"}, {"text": None}]},
    ],
)
def test_api_malformed_or_short_response_fails(monkeypatch, document) -> None:
    class Response:
        status_code = 200
        text = "response"

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return document

    def post(url, *, headers, json, timeout):
        return Response()

    import requests

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(RuntimeError, match="1 of 1 API requests failed"):
        run_api_inference(
            ["prompt"],
            "model",
            "http://127.0.0.1:8000/v1/completions",
            "token",
            {"num_return_sequences": 2},
            max_concurrency=1,
        )


def test_recommendation_output_contract_rejects_invalid_sid() -> None:
    outputs = [
        MockRequestOutput(
            "prompt",
            [
                MockCompletionOutput("<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"),
                MockCompletionOutput(""),
            ],
        )
    ]

    with pytest.raises(RuntimeError, match="violated the OneRec output contract"):
        _validate_recommendation_outputs(outputs, expected_requests=1, beam_width=2)
