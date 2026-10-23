# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin route selection, decline reporting, and single-session execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

import vllm_gr.entrypoints.openai.completion_batch_patch as patch_module
from vllm_gr.config import BeamExecutionMode
from vllm_gr.entrypoints.openai.completion_batch_patch import (
    _batch_size,
    _create_completion_batched,
    create_completion_batch_dispatch,
    patch_completion_batch,
)

_DECLINED = "Beam batch declined"

_EQUAL_LENGTHS = dict.fromkeys(range(4), 10)


class _ModeStub:
    """Stand-in for ``GRConfig`` that reports whatever mode a test sets."""

    beam_execution_mode = BeamExecutionMode.V1
    beam_online_batch_sizes = (1, 2, 4)

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> _ModeStub:
        del vllm_config
        return cls()


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patch_module, "GRConfig", _ModeStub)
    monkeypatch.setattr(_ModeStub, "beam_execution_mode", BeamExecutionMode.V1)


class _BatchSelf:
    """The slice of ``OpenAIServingCompletion`` the batch path borrows."""

    def __init__(
        self,
        engine_inputs: list[int],
        prompt_lengths: dict[int, int],
        max_num_batched_tokens: int = 4096,
        max_num_seqs: int = 8,
    ) -> None:
        self.engine_inputs = engine_inputs
        self.prompt_lengths = prompt_lengths
        self.render_calls = 0
        self.beam_calls: list[dict[str, Any]] = []
        self.response_calls: list[list[Any]] = []
        self.logged: list[str] = []
        self.engine_client = SimpleNamespace(
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                )
            )
        )
        self.model_config = SimpleNamespace(max_model_len=4096)
        self.default_sampling_params: dict[str, Any] = {}
        self.override_max_tokens = None
        self.models = SimpleNamespace(model_name=lambda lora: "a-model")
        self.renderer = SimpleNamespace(tokenizer=None)

    async def render_completion_request(self, request: Any) -> list[int]:
        del request
        self.render_calls += 1
        return self.engine_inputs

    def _base_request_id(self, raw_request: Any, default: Any) -> str:
        del raw_request
        return str(default)

    def _maybe_get_adapters(self, request: Any) -> None:
        del request

    def _extract_prompt_len(self, engine_input: int) -> int:
        return self.prompt_lengths[engine_input]

    def _extract_prompt_text(self, engine_input: int) -> str:
        return f"text-{engine_input}"

    def _log_inputs(self, request_id: str, engine_input: Any, **kwargs: Any) -> None:
        del engine_input, kwargs
        self.logged.append(request_id)

    def beam_search(self, **kwargs: Any) -> Any:
        self.beam_calls.append(kwargs)
        outputs = [SimpleNamespace(prompt=None) for _ in kwargs["prompt"]]

        async def generator() -> Any:
            for output in outputs:
                yield output

        return generator()

    def request_output_to_completion_response(self, final_res_batch: list[Any], *args: Any) -> Any:
        self.response_calls.append(final_res_batch)
        return f"response-for-{len(final_res_batch)}"

    def create_error_response(self, message: Any, **kwargs: Any) -> Any:
        self.error = (message, kwargs)
        return "error-response"


@pytest.fixture  # type: ignore[misc]
def make_serving(monkeypatch: pytest.MonkeyPatch) -> Callable[..., _BatchSelf]:
    """Stub per-prompt budgets; test the real formula separately below."""

    def _make(**kwargs: Any) -> _BatchSelf:
        serving = _BatchSelf(
            kwargs.pop("engine_inputs", [0, 1, 2, 3]),
            kwargs.pop("prompt_lengths", dict(_EQUAL_LENGTHS)),
            kwargs.pop("max_num_batched_tokens", 4096),
            kwargs.pop("max_num_seqs", 8),
        )
        for name, value in kwargs.items():
            setattr(serving, name, value)
        monkeypatch.setattr(
            patch_module,
            "get_max_tokens",
            lambda _len, _max, input_length, _default, _override, **_: 1000 - input_length,
        )
        return serving

    return _make


@pytest.fixture  # type: ignore[misc]
def serving(make_serving: Callable[..., _BatchSelf]) -> _BatchSelf:
    return make_serving()


@pytest.fixture  # type: ignore[misc]
def make_request() -> Callable[..., Any]:
    """Build a completion request whose beam parameters echo their budget."""

    def _make(**overrides: Any) -> Any:
        fields: dict[str, Any] = {
            "prompt": ["a", "b", "c", "d"],
            "use_beam_search": True,
            "stream": False,
            "prompt_embeds": None,
            "max_tokens": 5,
            "truncate_prompt_tokens": None,
            "request_id": "request-0",
            "to_beam_search_params": lambda tokens, _defaults: {"max_tokens": tokens},
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    return _make


@pytest.fixture  # type: ignore[misc]
def logged(monkeypatch: pytest.MonkeyPatch) -> Mock:
    info = Mock()
    monkeypatch.setattr(patch_module.logger, "info", info)
    return info


@pytest.fixture  # type: ignore[misc]
def delegate(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture the upstream handler and record what reaches it."""
    calls: list[Any] = []

    async def original(self: Any, request: Any, raw_request: Any = None) -> str:
        del self, raw_request
        calls.append(request)
        return "original"

    monkeypatch.setattr(patch_module, "_original_create_completion", original)
    return calls


def _declines(info: Mock) -> list[tuple[str, int, str]]:
    """The ``(request id, prompt count, reason)`` triples logged for declines."""
    return [
        # (format string, request id, count, reason)
        (call.args[1], call.args[2], call.args[3])
        for call in info.call_args_list
        if call.args and isinstance(call.args[0], str) and call.args[0].startswith(_DECLINED)
    ]


@pytest.mark.parametrize(  # type: ignore[misc]
    "overrides",
    [
        {"use_beam_search": False},
        {"stream": True},
        {"prompt_embeds": b"embedding"},
        {"prompt": ["only-one"]},
        {"prompt": [1, 2, 3]},
        {"prompt": []},
    ],
)
def test_everything_else_is_left_to_the_original_handler(
    serving: _BatchSelf, make_request: Callable[..., Any], overrides: dict[str, Any]
) -> None:
    assert _batch_size(serving, make_request(**overrides)) is None


@pytest.mark.parametrize("count", [2, 4, 3, 5, 6, 9])  # type: ignore[misc]
def test_a_prompt_list_is_counted_before_capacity_is_consulted(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any], count: int
) -> None:
    """Count unsupported lists too, so the decline log can identify them."""
    capacity = make_serving(max_num_seqs=2)
    assert _batch_size(capacity, make_request(prompt=["a"] * count)) == count
    assert _batch_size(capacity, make_request(prompt=[[1, 2]] * count)) == count


def test_an_engine_without_gr_beam_is_left_alone(
    serving: _BatchSelf, make_request: Callable[..., Any]
) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(_ModeStub, "beam_execution_mode", None)
        assert _batch_size(serving, make_request()) is None


_DECLINE_CASES = [
    ("count 3", {}, {"prompt": ["a"] * 3}, 0, "not one of the supported fixed batch sizes"),
    ("count 5", {}, {"prompt": ["a"] * 5}, 0, "not one of the supported fixed batch sizes"),
    ("count 6", {}, {"prompt": ["a"] * 6}, 0, "not one of the supported fixed batch sizes"),
    ("count 9", {}, {"prompt": ["a"] * 9}, 0, "not one of the supported fixed batch sizes"),
    ("capacity 2", {"max_num_seqs": 2}, {}, 1, "exceeds the sequence capacity 2"),
    (
        "unequal budgets",
        {"prompt_lengths": {0: 10, 1: 20, 2: 30, 3: 40}},
        {},
        1,
        "generation budgets differ across prompts",
    ),
    ("budget 39", {"max_num_batched_tokens": 39}, {}, 1, "exceed the scheduling budget 39"),
    (
        "begin token",
        {"max_num_batched_tokens": 40},
        {"vllm_xargs": {"begin_token": "<|sid_begin|>"}},
        1,
        "exceed the scheduling budget 40",
    ),
]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("case", "serving_kwargs", "request_kwargs", "renders", "reason"),
    _DECLINE_CASES,
    ids=[case for case, *_ in _DECLINE_CASES],
)
def test_a_list_that_cannot_batch_runs_the_original_handler(
    make_serving: Callable[..., _BatchSelf],
    make_request: Callable[..., Any],
    delegate: list[Any],
    logged: Mock,
    case: str,
    serving_kwargs: dict[str, Any],
    request_kwargs: dict[str, Any],
    renders: int,
    reason: str,
) -> None:
    serving = make_serving(**serving_kwargs)
    request = make_request(**request_kwargs)
    assert asyncio.run(create_completion_batch_dispatch(serving, request)) == "original"
    assert delegate == [request]
    assert serving.beam_calls == []
    assert serving.render_calls == renders
    ((request_id, logged_count, logged_reason),) = _declines(logged)
    assert request_id == "request-0"
    assert logged_count == len(request.prompt)
    assert reason in logged_reason


def test_a_batch_renders_and_submits_once_at_capacity(
    make_serving: Callable[..., _BatchSelf],
    make_request: Callable[..., Any],
    delegate: list[Any],
    logged: Mock,
) -> None:
    serving = make_serving(max_num_batched_tokens=40)
    to_params = Mock(return_value={"max_tokens": 990})
    request = make_request(to_beam_search_params=to_params)
    assert _batch_size(serving, request) == 4
    assert asyncio.run(create_completion_batch_dispatch(serving, request)) == ("response-for-4")
    assert delegate == []
    assert _declines(logged) == []
    assert serving.render_calls == 1
    assert [call["prompt"] for call in serving.beam_calls] == [[0, 1, 2, 3]]
    to_params.assert_called_once_with(990, serving.default_sampling_params)
    assert serving.beam_calls[0]["params"] == {"max_tokens": 990}
    assert serving.logged == [f"cmpl-request-0-{i}" for i in range(4)]
    assert len(serving.response_calls) == 1
    assert [res.prompt for res in serving.response_calls[0]] == [f"text-{i}" for i in range(4)]
    assert logged.call_args.args == (
        "Beam batch session request_id=%s prompts=%d",
        "cmpl-request-0",
        4,
    )


@pytest.mark.parametrize("sizes", [None, [1, 3, 7]], ids=["default", "custom"])  # type: ignore[misc]
@pytest.mark.parametrize("count", [2, 3, 4, 7, 8])  # type: ignore[misc]
def test_batch_policy_selects_fixed_http_sizes(
    make_serving: Callable[..., _BatchSelf],
    make_request: Callable[..., Any],
    delegate: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    sizes: list[int] | None,
) -> None:
    from vllm_gr.config import GRConfig

    monkeypatch.setattr(patch_module, "GRConfig", GRConfig)
    serving = make_serving(
        engine_inputs=list(range(count)), prompt_lengths=dict.fromkeys(range(count), 10)
    )
    serving.engine_client.vllm_config.additional_config = {
        "vllm_gr_beam_execution_mode": "v1",
    }
    if sizes is not None:
        serving.engine_client.vllm_config.additional_config["vllm_gr_beam_online_batch_sizes"] = (
            sizes
        )
    request = make_request(prompt=["a"] * count)
    result = asyncio.run(create_completion_batch_dispatch(serving, request))
    if count in (sizes if sizes is not None else [1, 2, 4]):
        assert result == f"response-for-{count}"
        assert [call["prompt"] for call in serving.beam_calls] == [list(range(count))]
        assert not delegate
    else:
        assert result == "original" and delegate == [request]
        assert not serving.beam_calls and serving.render_calls == 0


def test_different_lengths_can_share_the_real_generation_budget(
    make_serving: Callable[..., _BatchSelf],
    make_request: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the real formula the budget is the request's, not the length's."""
    from vllm.entrypoints.serve.utils.api_utils import get_max_tokens

    serving = make_serving(prompt_lengths={0: 10, 1: 20, 2: 30, 3: 40})
    monkeypatch.setattr(patch_module, "get_max_tokens", get_max_tokens)

    asyncio.run(create_completion_batch_dispatch(serving, make_request()))

    assert len(serving.beam_calls) == 1
    assert serving.beam_calls[0]["params"]["max_tokens"] == 5


def test_a_render_that_disagrees_with_the_count_is_a_server_fault(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any]
) -> None:
    """Nothing downstream holds if the renderer and the request disagree."""
    serving = make_serving(engine_inputs=[0, 1], prompt_lengths={0: 10, 1: 20})

    result = asyncio.run(create_completion_batch_dispatch(serving, make_request()))

    assert result == "error-response"
    assert serving.error[1]["err_type"] == "InternalServerError"
    assert serving.beam_calls == []


def test_a_short_result_batch_is_a_server_fault(
    serving: _BatchSelf, make_request: Callable[..., Any]
) -> None:
    """A result that does not match the request is a defect, not a caller error."""

    def short(**kwargs: Any) -> Any:
        async def generator() -> Any:
            yield SimpleNamespace(prompt=None)

        return generator()

    serving.beam_search = short  # type: ignore[method-assign]

    result = asyncio.run(create_completion_batch_dispatch(serving, make_request()))

    assert result == "error-response"
    assert serving.error[1]["err_type"] == "InternalServerError"


def test_a_cancelled_client_is_answered_rather_than_raised(
    make_serving: Callable[..., _BatchSelf], make_request: Callable[..., Any]
) -> None:
    serving = make_serving(engine_inputs=[0], prompt_lengths={0: 10})
    request = make_request(prompt=["a"])

    def cancelled(**kwargs: Any) -> Any:
        async def generator() -> Any:
            raise asyncio.CancelledError
            yield  # pragma: no cover - unreachable, makes this an async generator

        return generator()

    serving.beam_search = cancelled  # type: ignore[method-assign]

    result = asyncio.run(_create_completion_batched(serving, request, None, [0], 5))

    assert result == "error-response"
    assert serving.error[0] == "Client disconnected"


def test_real_response_conversion_preserves_indices_and_usage(
    make_serving: Callable[..., _BatchSelf], monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.outputs import CompletionOutput, RequestOutput

    from vllm_gr.entrypoints.openai.protocol import to_beam_search_params

    monkeypatch.setattr(
        CompletionRequest, "to_beam_search_params", to_beam_search_params, raising=False
    )
    serving = make_serving(
        enable_prompt_tokens_details=False,
        system_fingerprint=None,
        _raise_if_error=lambda *_: None,
    )
    serving.request_output_to_completion_response = MethodType(  # type: ignore[method-assign]
        OpenAIServingCompletion.request_output_to_completion_response,
        serving,
    )

    def beam_search(**kwargs: Any) -> Any:
        serving.beam_calls.append(kwargs)

        async def generator() -> Any:
            for index in kwargs["prompt"]:
                yield RequestOutput(
                    request_id=kwargs["request_id"],
                    prompt=None,
                    prompt_token_ids=[index] * 10,
                    prompt_logprobs=None,
                    finished=True,
                    outputs=[
                        CompletionOutput(
                            index=beam,
                            text=f"{index}:{beam}",
                            token_ids=[index, beam],
                            cumulative_logprob=0.0,
                            logprobs=None,
                            finish_reason="length",
                        )
                        for beam in range(2)
                    ],
                )

        return generator()

    serving.beam_search = beam_search  # type: ignore[method-assign]
    request = CompletionRequest(
        model="a-model",
        prompt=["prompt"] * 4,
        n=2,
        max_tokens=5,
        use_beam_search=True,
        request_id="request-0",
        return_token_ids=True,
    )

    result = asyncio.run(create_completion_batch_dispatch(serving, request))

    assert result.id == "cmpl-request-0"
    assert [c.index for c in result.choices] == list(range(8))
    assert [c.text for c in result.choices] == [f"{i}:{b}" for i in range(4) for b in range(2)]
    assert result.usage.prompt_tokens == 40
    assert result.usage.completion_tokens == 16
    assert result.usage.total_tokens == 56
    assert [len(call["prompt"]) for call in serving.beam_calls] == [4]


@pytest.fixture  # type: ignore[misc]
def restore_route() -> Any:
    """Installing the patch touches the class itself, so put it back after."""
    before = OpenAIServingCompletion._create_completion
    original = patch_module._original_create_completion
    try:
        yield
    finally:
        OpenAIServingCompletion._create_completion = before
        patch_module._original_create_completion = original


def test_the_patch_installs_verifies_and_is_idempotent(restore_route: Any) -> None:
    patch_completion_batch()

    assert OpenAIServingCompletion._create_completion is create_completion_batch_dispatch
    assert patch_module._validate_completion_batch_patch() is None
    original = patch_module._original_create_completion
    assert callable(original)
    patch_completion_batch()

    assert patch_module._original_create_completion is original


def test_restore_route_preserves_preinstalled_fallback(
    monkeypatch: pytest.MonkeyPatch,
    serving: _BatchSelf,
    make_request: Callable[..., Any],
    delegate: list[Any],
) -> None:
    original = patch_module._original_create_completion
    monkeypatch.setattr(OpenAIServingCompletion, "_create_completion", original)
    patch_completion_batch()
    cleanup = restore_route.__wrapped__()
    next(cleanup)
    patch_module._original_create_completion = None
    with pytest.raises(StopIteration):
        next(cleanup)
    patch_completion_batch()
    assert patch_module._original_create_completion is original
    request = make_request(prompt="single")
    assert asyncio.run(OpenAIServingCompletion._create_completion(serving, request)) == "original"
    assert delegate == [request]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("device", "tokens", "width", "budget", "boundaries", "batched"),
    [
        # Adjusted budget (public budget minus boundary tokens) above one
        # needs Decode rows; at or below one the request is Prefill-only.
        ("npu", 512, 256, 5, ("begin",), False),
        ("npu:0", 1023, 256, 5, ("begin",), False),
        ("npu", 1024, 256, 5, ("begin",), True),
        ("npu", 512, 128, 5, ("begin",), True),
        ("npu", 512, 256, 3, ("begin",), False),
        ("npu", 512, 256, 4, ("begin", "end"), False),
        ("npu", 512, 256, 2, ("begin",), True),
        ("npu", 512, 256, 3, ("begin", "end"), True),
        ("cuda", 512, 256, 5, ("begin",), True),
    ],
)
def test_decode_capacity_is_checked_before_batch_submission(
    monkeypatch: pytest.MonkeyPatch,
    make_serving: Callable[..., _BatchSelf],
    make_request: Callable[..., Any],
    delegate: list[Any],
    logged: Mock,
    device: str,
    tokens: int,
    width: int,
    budget: int,
    boundaries: tuple[str, ...],
    batched: bool,
) -> None:
    serving = make_serving(
        max_num_batched_tokens=tokens, prompt_lengths=dict.fromkeys(range(4), 16)
    )
    serving.engine_client.vllm_config.device_config = SimpleNamespace(device=device)
    monkeypatch.setattr(patch_module, "get_max_tokens", lambda *_args, **_kwargs: budget)
    xargs = {"begin_token": 1} if "begin" in boundaries else {}
    if "end" in boundaries:
        xargs["end_token"] = 2
    request = make_request(n=width, max_tokens=budget, vllm_xargs=xargs)
    result = asyncio.run(create_completion_batch_dispatch(serving, request))
    if batched:
        assert result == "response-for-4"
        assert len(serving.beam_calls) == 1
        assert not delegate
    else:
        assert result == "original"
        assert not serving.beam_calls
        assert delegate == [request]
        assert "NPU Decode" in _declines(logged)[0][2]
