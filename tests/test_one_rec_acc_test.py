# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, NoReturn

import pytest
import requests

from benchmarks.open_one_rec.one_rec_acc_test import (
    ApiRequestError,
    _api_request_metrics,
    run_api_inference,
)


def _gen_config(num_outputs: int = 2) -> dict[str, Any]:
    return {
        "num_beams": num_outputs,
        "num_return_sequences": num_outputs,
        "max_new_tokens": 5,
        "temperature": 0.0,
    }


def _response(choices: list[dict[str, Any]]) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=200,
        text="ok",
        raise_for_status=lambda: None,
        json=lambda: {"choices": choices},
    )


def _choice(text: str, finish_reason: str = "length") -> dict[str, Any]:
    return {
        "message": {"content": text},
        "finish_reason": finish_reason,
        "logprobs": None,
    }


def test_run_api_inference_requires_every_expected_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _response([_choice("only-one")]),
    )

    with pytest.raises(ApiRequestError, match="1 choices; expected 2"):
        run_api_inference(
            ["prompt"],
            "model",
            "http://localhost/v1/chat/completions",
            "key",
            _gen_config(),
            concurrency_limit=1,
        )


def test_run_api_inference_propagates_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        raise requests.ConnectionError("connection lost")

    monkeypatch.setattr(requests, "post", fail)

    with pytest.raises(ApiRequestError, match="connection lost"):
        run_api_inference(
            ["prompt"],
            "model",
            "http://localhost/v1/chat/completions",
            "key",
            _gen_config(),
            concurrency_limit=1,
        )


def test_run_api_inference_rejects_malformed_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _response(
            [
                _choice("valid"),
                {"message": {"content": "missing-finish-reason"}},
            ]
        ),
    )

    with pytest.raises(ApiRequestError, match="has no finish_reason"):
        run_api_inference(
            ["prompt"],
            "model",
            "http://localhost/v1/chat/completions",
            "key",
            _gen_config(),
            concurrency_limit=1,
        )


def test_run_api_inference_records_success_and_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _response([_choice("first"), _choice("second", "stop")]),
    )

    outputs = run_api_inference(
        ["prompt-a", "prompt-b"],
        "model",
        "http://localhost/v1/chat/completions",
        "key",
        _gen_config(),
        concurrency_limit=2,
    )
    metrics = _api_request_metrics(outputs, attempted=2)

    assert [item.text for item in outputs[0].outputs] == ["first", "second"]
    assert outputs[0].outputs[1].finish_reason == "stop"
    assert metrics["request_attempted"] == 2
    assert metrics["request_succeeded"] == 2
    assert metrics["request_failed"] == 0
    assert metrics["request_success_rate"] == 1.0
    assert metrics["request_latency_p50_ms"] >= 0.0
    assert metrics["request_latency_p99_ms"] >= metrics["request_latency_p50_ms"]
