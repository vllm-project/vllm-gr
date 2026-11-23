# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Exercise the patched OneRec benchmark HTTP client with real vLLM request types."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from tests.helpers.vllm_subprocess import run_vllm_script

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


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
        api_url="http://localhost/v1/chat/completions",
        prompt=sample.prompt,
        prompt_len=sample.prompt_len,
        output_len=sample.expected_output_len,
        model="test-model",
        chat_messages=sample.chat_messages,
        extra_body=_merge_overrides(None, sample.request_overrides),
    )
    session = mock.MagicMock()
    response = SimpleNamespace(status=400, reason="test response")
    session.post.return_value.__aenter__ = mock.AsyncMock(return_value=response)
    asyncio.run(one_rec_patch.patched_async_request_openai_chat_completions(request, session))
    session.post.assert_called_once()
    payload = session.post.call_args.kwargs["json"]
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "rendered"}]}
    ]
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize(  # type: ignore[misc]
    ("line_ending", "split_after"),
    [
        pytest.param(b"\n", b"\n\n", id="complete-events"),
        pytest.param(b"\n", b"\n\ndata: ", id="split-done-prefix"),
        pytest.param(b"\n", b"hello ", id="split-content-space"),
        pytest.param(b"\r\n", b"}}]}", id="crlf-after-json"),
        pytest.param(b"\r\n", b"\r", id="split-crlf"),
        pytest.param(b"\r", b"\r", id="split-cr-separator"),
    ],
)
def test_streaming_preserves_response_across_network_chunks(
    line_ending: bytes, split_after: bytes
) -> None:
    separator = line_ending * 2
    stream = b'data: {"choices": [{"delta": {"content": "hello world"}}]}' + separator
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
    text = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
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
                b'\ndata:[{"delta":{"content":"hello \xc3',
                b'\xa9"}}]}\n\n',
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
        api_url="http://localhost/v1/chat/completions",
        prompt="question",
        prompt_len=1,
        output_len=5,
        model="test-model",
    )
    result = asyncio.run(
        one_rec_patch.patched_async_request_openai_chat_completions(request, session)
    )
    assert result.generated_text == expected_text, repr(result.generated_text)
    assert result.success == (not expected_error), result.error
    error_lines = result.error.splitlines()
    assert (error_lines[-1] if error_lines else "") == expected_error, result.error
