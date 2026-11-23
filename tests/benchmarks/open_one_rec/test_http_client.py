# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Exercise the patched OneRec benchmark HTTP client with real vLLM request types."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from tests.helpers.vllm_subprocess import run_vllm_script

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_serving_adapter_is_scoped_to_onerec() -> None:
    run_vllm_script(
        "from tests.benchmarks.open_one_rec.test_http_client import _assert_serving_adapter_scope\n"
        "_assert_serving_adapter_scope()"
    )


def _assert_serving_adapter_scope() -> None:
    from importlib import import_module

    from vllm.benchmarks.lib.endpoint_request_func import (
        ASYNC_REQUEST_FUNCS,
        RequestFuncInput,
    )

    native_handler = ASYNC_REQUEST_FUNCS["openai"]
    payloads: list[dict[str, Any]] = []
    selected_backends: list[str] = []

    class Stream:
        async def iter_any(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"text":"result"}]}\n\n'
            yield b"data: [DONE]\n\n"

    @asynccontextmanager
    async def post(*, url: str, json: dict[str, Any], headers: Any) -> AsyncIterator[Any]:
        payloads.append(json)
        yield SimpleNamespace(
            status=200,
            content=Stream(),
            json=mock.AsyncMock(return_value={"choices": [{"text": "result"}]}),
        )

    def request(**kwargs: Any) -> Any:
        return RequestFuncInput(
            api_url="http://localhost/v1/completions",
            prompt="unrendered random prompt",
            prompt_len=3,
            output_len=5,
            model="test-model",
            **kwargs,
        )

    session = SimpleNamespace(post=post)
    baseline = asyncio.run(native_handler(request(), session))
    assert baseline.success, baseline.error
    baseline_payload = payloads.pop()

    from benchmarks.open_one_rec import one_rec_patch

    # Importing the extension must preserve native and third-party handler identity.
    assert ASYNC_REQUEST_FUNCS["openai"] is native_handler
    serve = import_module("vllm.benchmarks.serve")

    async def benchmark_driver(*, endpoint_type: str, extra_body: Any = None) -> Any:
        selected_backends.append(endpoint_type)
        return await ASYNC_REQUEST_FUNCS[endpoint_type](request(extra_body=extra_body), session)

    def serve_main(args: Any) -> Any:
        assert args.backend == "openai"
        return asyncio.run(
            serve.benchmark(
                endpoint_type=args.backend,
                extra_body={"use_beam_search": True} if args.dataset_name == "onerec" else None,
            )
        )

    with (
        mock.patch.object(one_rec_patch, "original_serve_benchmark", benchmark_driver),
        mock.patch.object(one_rec_patch, "original_serve_main", side_effect=serve_main) as main,
    ):
        # Run a native workload before and after OneRec in the same process.
        for dataset in ("random", "onerec", "random"):
            args = SimpleNamespace(
                dataset_name=dataset, backend="openai", endpoint="/v1/completions"
            )
            result = serve.main(args)
            assert result.success, result.error
            payload = payloads.pop()
            if dataset == "onerec":
                assert payload["add_special_tokens"] is False
                assert payload["vllm_xargs"] == {
                    "begin_token": "<|sid_begin|>",
                    "end_token": "<|sid_end|>",
                }
                assert payload["use_beam_search"] is True
                assert payload["stream"] is False
            else:
                assert payload == baseline_payload
                assert "add_special_tokens" not in payload
                assert "vllm_xargs" not in payload
        assert selected_backends == ["openai", "onerec-openai", "openai"]

        main.side_effect = RuntimeError("benchmark failed")
        with pytest.raises(RuntimeError, match="benchmark failed"):
            serve.main(
                SimpleNamespace(dataset_name="onerec", backend="openai", endpoint="/v1/completions")
            )
        # A failed OneRec run must not affect callers of the native benchmark API.
        result = asyncio.run(serve.benchmark(endpoint_type="openai"))
        assert result.success, result.error
        assert selected_backends[-1] == "openai"
        assert payloads.pop() == baseline_payload
    assert ASYNC_REQUEST_FUNCS["openai"] is native_handler


def test_fixed_length_prompt_disables_thinking_in_http_payload() -> None:
    run_vllm_script(
        "from tests.benchmarks.open_one_rec.test_http_client import _assert_fixed_length_payload\n"
        "_assert_fixed_length_payload()"
    )


def _assert_fixed_length_payload() -> None:
    from argparse import ArgumentParser

    from vllm.benchmarks.serve import _merge_overrides

    from benchmarks.open_one_rec import one_rec_patch, open_one_rec_loader

    class Tokenizer:
        def __call__(self, prompt: str, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(input_ids=[ord(char) for char in prompt])

        def decode(self, token_ids: list[int], **kwargs: Any) -> str:
            return "".join(chr(token_id) for token_id in token_ids)

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            return "rendered prompt"

    parser = one_rec_patch.patched_add_parser(ArgumentParser())
    # rec_reason enables thinking by default.
    args = parser.parse_args(
        [
            "--dataset-name",
            "onerec",
            "--task-types",
            "rec_reason",
            "--num-prompts",
            "1",
            "--custom-input-len",
            "8",
            "--disable-thinking",
        ]
    )
    frame = mock.Mock(columns=["messages", "metadata"])
    frame.iterrows.return_value = [
        (
            0,
            {
                "messages": [{"role": "user", "content": "original prompt"}],
                "metadata": {"answer": "example"},
            },
        )
    ]
    with mock.patch.object(
        open_one_rec_loader.BaseLoader, "_load_sample_dataframe", return_value=frame
    ):
        (sample,) = one_rec_patch.patched_get_samples(args, Tokenizer())

    request = one_rec_patch.RequestFuncInput(
        api_url="http://localhost/v1/completions",
        prompt=sample.prompt,
        prompt_len=sample.prompt_len,
        output_len=sample.expected_output_len,
        model="test-model",
        chat_messages=sample.chat_messages,
        extra_body=_merge_overrides(None, sample.request_overrides),
    )
    session = mock.MagicMock()
    response = SimpleNamespace(
        status=400, reason="test response", text=mock.AsyncMock(return_value="")
    )
    session.post.return_value.__aenter__ = mock.AsyncMock(return_value=response)
    asyncio.run(one_rec_patch.patched_async_request_openai_completions(request, session))
    session.post.assert_called_once()
    payload = session.post.call_args.kwargs["json"]
    assert payload["prompt"] == "rendered"
    assert payload["add_special_tokens"] is False
    assert "messages" not in payload
    assert "chat_template_kwargs" not in payload


def test_beam_completion_passes_native_streaming_guard() -> None:
    run_vllm_script(
        "from tests.benchmarks.open_one_rec.test_http_client import _assert_beam_completion\n"
        "_assert_beam_completion()",
        "--use-beam-search",
        "--n",
        "128",
    )


def _assert_beam_completion() -> None:
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

    from benchmarks.open_one_rec import one_rec_patch

    class RenderReached(Exception):
        """Stop after the real native request guard, before model execution."""

    for extra_body in (None, {"stream": True, "stream_options": {"include_usage": True}}):
        boundary: Any = SimpleNamespace(
            create_error_response=lambda message: {"error": {"message": message}},
            render_completion_request=mock.AsyncMock(side_effect=RenderReached),
        )
        captured: list[dict[str, Any]] = []

        @asynccontextmanager
        async def post(*, url: str, json: dict[str, Any], headers: Any) -> AsyncIterator[Any]:
            captured.append(json)
            parsed = CompletionRequest.model_validate(json)
            try:
                error = await OpenAIServingCompletion._create_completion(boundary, parsed)
            except RenderReached:
                body = {
                    "choices": [{"text": "<sid-a>"}, {"text": "<sid-b>"}],
                    "usage": {"completion_tokens": 10, "prompt_tokens": 7},
                }
                yield SimpleNamespace(status=200, json=mock.AsyncMock(return_value=body))
            else:
                yield SimpleNamespace(
                    status=400,
                    reason="Bad Request",
                    text=mock.AsyncMock(return_value=str(error)),
                )

        # Import-time CLI flags above match the gate's --use-beam-search --n 128.
        request = one_rec_patch.RequestFuncInput(
            api_url="http://localhost/v1/completions",
            prompt="rendered SID prompt",
            prompt_len=7,
            output_len=5,
            model="test-model",
            extra_body=extra_body,
        )
        with mock.patch.object(one_rec_patch.time, "perf_counter", side_effect=[100.0, 102.0]):
            result = asyncio.run(
                one_rec_patch.patched_async_request_openai_completions(
                    request, SimpleNamespace(post=post)
                )
            )
        assert result.success, result.error
        boundary.render_completion_request.assert_awaited_once()
        (payload,) = captured
        assert payload["use_beam_search"] is True
        assert payload["n"] == 128
        assert payload["max_tokens"] == 5
        assert payload["stream"] is False
        assert "stream_options" not in payload
        assert result.generated_text == "<sid-a><sid-b>"
        assert result.output_tokens == 10
        assert result.prompt_len == 7
        assert result.ttft == result.latency == 2.0
        assert result.itl == []


def test_beam_completion_reports_response_errors() -> None:
    run_vllm_script(
        "from tests.benchmarks.open_one_rec.test_http_client import _assert_beam_response_errors\n"
        "_assert_beam_response_errors()"
    )


def _assert_beam_response_errors() -> None:
    from benchmarks.open_one_rec import one_rec_patch

    cases = [
        (200, {"choices": []}, "response contains no text output"),
        (200, {"choices": [{"text": ""}]}, "response contains no text output"),
        (200, {"choices": [{"text": None}]}, "invalid completion text"),
        (200, {"error": {"message": "engine failed"}}, "engine failed"),
        (400, {"error": {"message": "invalid Beam width"}}, "invalid Beam width"),
        (502, "upstream unavailable", "upstream unavailable"),
    ]
    for status, body, expected in cases:
        request = one_rec_patch.RequestFuncInput(
            api_url="http://localhost/v1/completions",
            prompt="question",
            prompt_len=1,
            output_len=5,
            model="test-model",
            extra_body={"use_beam_search": True},
        )
        response = SimpleNamespace(
            status=status,
            reason="Bad Request" if status == 400 else "Bad Gateway",
            json=mock.AsyncMock(return_value=body),
            text=mock.AsyncMock(return_value=body if isinstance(body, str) else json.dumps(body)),
        )
        session = mock.MagicMock()
        session.post.return_value.__aenter__ = mock.AsyncMock(return_value=response)
        result = asyncio.run(
            one_rec_patch.patched_async_request_openai_completions(request, session)
        )
        assert not result.success
        assert expected in result.error, result.error
        if status != 200:
            assert result.error.startswith(f"HTTP {status} "), result.error


@pytest.mark.parametrize(  # type: ignore[misc]
    ("line_ending", "split_after"),
    [
        pytest.param(b"\n", b"\n\n", id="complete-events"),
        pytest.param(b"\n", b"\n\ndata: ", id="split-done-prefix"),
        pytest.param(b"\n", b"hello ", id="split-content-space"),
        pytest.param(b"\r\n", b"}]}", id="crlf-after-json"),
        pytest.param(b"\r\n", b"\r", id="split-crlf"),
        pytest.param(b"\r", b"\r", id="split-cr-separator"),
    ],
)
def test_streaming_preserves_response_across_network_chunks(
    line_ending: bytes, split_after: bytes
) -> None:
    separator = line_ending * 2
    stream = b'data: {"choices": [{"text": "hello world"}]}' + separator
    stream += b"data: [DONE]" + separator
    offset = stream.index(split_after) + len(split_after)
    run_vllm_script(
        "import sys\n"
        "from tests.benchmarks.open_one_rec.test_http_client import _assert_stream\n"
        "_assert_stream([bytes.fromhex(arg) for arg in sys.argv[1:]], 'hello world', '')",
        stream[:offset].hex(),
        stream[offset:].hex(),
    )


def test_streaming_completion() -> None:
    run_vllm_script(
        "from tests.benchmarks.open_one_rec.test_http_client import _assert_stream_completion\n"
        "_assert_stream_completion()"
    )


def _assert_stream_completion() -> None:
    text = b'data: {"choices":[{"text":"hello"}]}\n\n'
    done = b"data: [DONE]\n\n"
    error = b'data: {"error":{"message":"engine failed"}}\n\n'
    # None raises a connection reset on the next read.
    cases: list[tuple[str, list[bytes | None], str, str]] = [
        ("eof before done", [text], "hello", "response ended without [DONE]"),
        ("reset before done", [text, None], "hello", "ConnectionResetError: connection reset"),
        ("reset after done", [text, done, None], "hello", ""),
        ("same chunk error after done", [text, done + error], "hello", ""),
        ("error before done", [text, error, done], "hello", "engine failed"),
        (
            "first error before reset",
            [text, error, b'data: {"error":"later"}\n\n', None],
            "hello",
            "engine failed",
        ),
        ("no space after colon", [text, b"data:[DONE]\n\n"], "hello", ""),
        ("comment before data", [text, b": ping\ndata: [DONE]\n\n"], "hello", ""),
        (
            "multiline data and split UTF-8",
            [
                b'data: {"choices":\r',
                b'\ndata:[{"text":"hello \xc3',
                b'\xa9"}]}\n\n',
                done,
            ],
            "hello \u00e9",
            "",
        ),
    ]
    for name, chunks, expected_text, expected_error in cases:
        try:
            _assert_stream(chunks, expected_text, expected_error)
        except AssertionError as exc:
            exc.add_note(name)
            raise


def _assert_stream(chunks: Sequence[bytes | None], expected_text: str, expected_error: str) -> None:
    from benchmarks.open_one_rec import one_rec_patch

    class Stream:
        async def iter_any(self) -> AsyncIterator[bytes]:
            for chunk in chunks:
                if chunk is None:
                    raise ConnectionResetError("connection reset")
                yield chunk

    response = SimpleNamespace(status=200, content=Stream())
    session = mock.MagicMock()
    session.post.return_value.__aenter__ = mock.AsyncMock(return_value=response)
    request = one_rec_patch.RequestFuncInput(
        api_url="http://localhost/v1/completions",
        prompt="question",
        prompt_len=1,
        output_len=5,
        model="test-model",
    )
    result = asyncio.run(one_rec_patch.patched_async_request_openai_completions(request, session))
    assert result.generated_text == expected_text, repr(result.generated_text)
    assert result.success == (not expected_error), result.error
    error_lines = result.error.splitlines()
    assert (error_lines[-1] if error_lines else "") == expected_error, result.error
