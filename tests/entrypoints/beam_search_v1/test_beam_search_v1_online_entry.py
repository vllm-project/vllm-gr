# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""What the online V1 entry point supplies, refuses, and preserves itself.

The V1 engine accepts exactly one value for ``ignore_eos``: a beam that stopped
early would make the terminal step data-dependent, so such a request is refused
rather than served. Requiring every client to restate that would leave the
interface unusable with an ordinary request, so the online entry point supplies
it. The same boundary owns the two interface-level answers a client cannot be
expected to discover: encoder-decoder input has no V1 implementation at all,
and the frontend heap has to be frozen once serving has started. These tests pin
all three on the object that actually reaches the engine, without a device.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import vllm_gr.config.execution as execution
import vllm_gr.entrypoints.beam_search_v1 as beam_search_v1_package
import vllm_gr.entrypoints.openai.serving_engine as serving_engine
from vllm_gr.entrypoints.openai.serving_engine import beam_search_v1
from vllm_gr.sampling_params import BeamSearchParams

BEAM_WIDTH = 128
MAX_TOKENS = 3


async def _rank(client: object) -> int:
    del client
    return 0


def _run(
    prompt: Any,
    client_params: BeamSearchParams,
    calls: list[Any] | None = None,
    *,
    session_id: str | None = None,
    trie_ids: list[int] | None = None,
) -> list[str]:
    """Drive one online V1 request; return the GC freezes it asked for.

    ``calls`` receives whatever was handed to the engine, so a caller that
    expects no submission can show the engine was never reached.
    """
    if calls is None:
        calls = []
    freezes: list[str] = []

    async def fake_run(client: object, call: Any) -> list[Any]:
        del client
        calls.append(call)
        return []

    async def drain() -> None:
        serving = SimpleNamespace(engine_client=SimpleNamespace(vllm_config=object()))
        async for _ in beam_search_v1(
            serving,
            prompt=prompt,
            request_id="request-id",
            params=client_params,
            session_id=session_id,
            trie_ids=trie_ids,
        ):
            pass

    with pytest.MonkeyPatch.context() as monkeypatch:
        # Startup validation is covered by the dispatch tests.
        monkeypatch.setattr(execution, "require_beam_config", lambda *a, **k: None)
        monkeypatch.setattr(serving_engine, "_next_data_parallel_rank", _rank)
        monkeypatch.setattr(
            serving_engine, "create_online_beam_search_v1_client", lambda self: object()
        )
        monkeypatch.setattr(beam_search_v1_package, "run_beam_search_v1_async", fake_run)
        # The real freeze collects and freezes this process's heap for the rest
        # of the run. That is a serving-boundary effect the entry point asks
        # for, not something an entry-point test should trigger.
        monkeypatch.setattr(
            serving_engine, "schedule_frontend_gc_freeze", lambda: freezes.append("freeze")
        )
        asyncio.run(drain())

    return freezes


def _sent_to_engine(client_params: BeamSearchParams) -> Any:
    """Run one online V1 request and return the call the engine would receive."""
    calls: list[Any] = []
    _run("prompt", client_params, calls)

    assert calls, "the entry point never reached the engine"
    return calls[0]


def _client_params(*, ignore_eos: bool) -> BeamSearchParams:
    return BeamSearchParams(beam_width=BEAM_WIDTH, max_tokens=MAX_TOKENS, ignore_eos=ignore_eos)


def test_the_engine_receives_ignore_eos_true_when_the_client_asked_for_it() -> None:
    call = _sent_to_engine(_client_params(ignore_eos=True))

    assert call.params.ignore_eos is True


def test_the_engine_receives_ignore_eos_true_when_the_client_asked_for_false() -> None:
    call = _sent_to_engine(_client_params(ignore_eos=False))

    assert call.params.ignore_eos is True


def test_the_callers_own_params_are_left_alone() -> None:
    params = _client_params(ignore_eos=False)

    _sent_to_engine(params)

    assert params.ignore_eos is False


def test_only_ignore_eos_is_rewritten() -> None:
    call = _sent_to_engine(_client_params(ignore_eos=False))

    assert call.params.beam_width == BEAM_WIDTH
    assert call.params.max_tokens == MAX_TOKENS
    assert call.request_id == "request-id"
    assert call.prompts == ("prompt",)


def test_caller_session_hint_is_preserved_separately_from_request_id() -> None:
    calls: list[Any] = []
    _run("prompt", _client_params(ignore_eos=True), calls, session_id="caller-session")
    assert calls[0].session_id == "caller-session"
    assert calls[0].request_id == "request-id"


def test_trie_ids_travel_separately_from_search_params() -> None:
    calls: list[Any] = []
    _run(["first", "second"], _client_params(ignore_eos=False), calls, trie_ids=[1, 0])
    assert calls[0].trie_ids == [1, 0]
    assert calls[0].prompts == ("first", "second")


def test_the_frontend_heap_is_frozen_once_the_engine_has_answered() -> None:
    """The driver's own GC pauses otherwise land inside a request's wall time.

    Freezing the frontend heap is what keeps them out of the response path, and
    the entry point is the only place that knows a beam request has begun.
    """
    freezes = _run("prompt", _client_params(ignore_eos=True))

    assert freezes == ["freeze"]


def test_encoder_decoder_input_is_refused_before_the_engine_is_reached() -> None:
    """Speech-to-text arrives here as an encoder-decoder prompt.

    A session runs the single token stream its decoder owns, and the engine
    refuses the multimodal features such a request carries. The answer has to
    come from this boundary; otherwise the caller inherits a schema failure
    from a layer the interface never described, and reads it as a server fault.
    """
    prompt = {
        "type": "enc_dec",
        "encoder_prompt": {"prompt_token_ids": [1]},
        "decoder_prompt": {"prompt_token_ids": [2]},
    }
    calls: list[Any] = []

    with pytest.raises(NotImplementedError, match="encoder-decoder"):
        _run(prompt, _client_params(ignore_eos=True), calls)

    assert calls == []


pytestmark = pytest.mark.cpu_test
