# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The upstream Beam name must never reach the retired GR frontend loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.exceptions import VLLMValidationError

from vllm_gr.entrypoints.gr import GRLLM
from vllm_gr.entrypoints.openai import serving_engine
from vllm_gr.entrypoints.openai.beam_search_patch import (
    beam_search_dispatch,
    patch_beam_search,
)
from vllm_gr.patch import _validate_openai_beam_search_patch
from vllm_gr.sampling_params import BeamSearchParams


def _recorder(name: str, calls: list[tuple[str, dict[str, Any]]]) -> Callable[..., Any]:
    async def impl(self: object, **kwargs: Any) -> AsyncGenerator[str, None]:
        calls.append((name, kwargs))
        yield name

    return impl


def _dispatch(
    additional: dict[str, Any] | None,
    monkeypatch: Any,
    params: Any = "params",
    session_id: str | None = None,
    *,
    trie_ids: list[int] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(serving_engine, "beam_search_v1", _recorder("v1", calls))
    monkeypatch.setattr(BeamSearchOnlineMixin, "beam_search", _recorder("native", calls))
    config = None if additional is None else SimpleNamespace(additional_config=additional)
    serving = SimpleNamespace(engine_client=SimpleNamespace(vllm_config=config))

    async def drain() -> None:
        outputs = [
            output
            async for output in beam_search_dispatch(
                serving,
                "prompt",
                "request-id",
                params,
                "lora",
                {"trace": "id"},
                session_id=session_id,
                trie_ids=trie_ids,
            )
        ]
        assert len(outputs) == 1

    asyncio.run(drain())
    return calls


def test_v1_dispatch_preserves_upstream_arguments(monkeypatch: Any) -> None:
    assert _dispatch({"vllm_gr_beam_execution_mode": "v1"}, monkeypatch) == [
        (
            "v1",
            {
                "prompt": "prompt",
                "request_id": "request-id",
                "params": "params",
                "lora_request": "lora",
                "trace_headers": {"trace": "id"},
                "session_id": None,
            },
        )
    ]


def test_non_gr_engine_uses_upstream_beam(monkeypatch: Any) -> None:
    params = BeamSearchParams(beam_width=4, max_tokens=3)
    for additional in (None, {}, {"unrelated": True}):
        assert _dispatch(additional, monkeypatch, params)[0][0] == "native"


@pytest.mark.parametrize("additional", [None, {}, {"unrelated": True}])  # type: ignore[misc]
def test_native_fallback_rejects_explicit_per_parent_widths(
    additional: dict[str, Any] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = BeamSearchParams(beam_width=4, max_tokens=3, per_parent_beam_widths=[4, 2, 1])
    with pytest.raises(VLLMValidationError, match="requires GR beam_search_v1"):
        _dispatch(additional, monkeypatch, params)


def test_v1_dispatch_preserves_explicit_per_parent_widths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = BeamSearchParams(beam_width=4, max_tokens=3, per_parent_beam_widths=[4, 2, 1])
    calls = _dispatch({"vllm_gr_beam_execution_mode": "v1"}, monkeypatch, params)
    assert calls[0][0] == "v1"
    assert calls[0][1]["params"] is params


def test_dispatch_preserves_native_session_hint_for_both_engines(monkeypatch: Any) -> None:
    for config in (None, {"vllm_gr_beam_execution_mode": "v1"}):
        calls = _dispatch(config, monkeypatch, session_id="caller-session")
        assert calls[0][1]["session_id"] == "caller-session"


def test_trie_ids_are_forwarded_only_to_gr(monkeypatch: Any) -> None:
    calls = _dispatch({"vllm_gr_beam_execution_mode": "v1"}, monkeypatch, trie_ids=[1, 0])
    assert calls[0][1]["trie_ids"] == [1, 0]
    with pytest.raises(ValueError, match="trie_ids require GR"):
        _dispatch(None, monkeypatch, trie_ids=[1])


def test_completion_fallback_binds_ids_before_iteration(monkeypatch: Any) -> None:
    from vllm_gr.entrypoints.openai import completion_batch_patch as completion
    from vllm_gr.entrypoints.openai.beam_search_patch import _COMPLETION_TRIE_IDS

    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(serving_engine, "beam_search_v1", _recorder("v1", calls))
    serving = SimpleNamespace(
        engine_client=SimpleNamespace(
            vllm_config=SimpleNamespace(additional_config={"vllm_gr_beam_execution_mode": "v1"})
        )
    )

    async def upstream(self: Any, request: Any, raw_request: Any) -> list[Any]:
        generators = []
        for i in range(3):
            await asyncio.sleep(0)  # Interleave two requests during splitting.
            generators.append(beam_search_dispatch(self, i, request.request_id, "params"))
        return generators

    monkeypatch.setattr(completion, "_original_create_completion", upstream)

    async def run() -> None:
        requests = [
            SimpleNamespace(request_id=name, vllm_xargs={"trie_ids": ids})
            for name, ids in (("a", [0, 1, 2]), ("b", [2, 0, 1]))
        ]
        batches = await asyncio.gather(
            *[completion._delegate_completion(serving, request, None) for request in requests]
        )
        assert _COMPLETION_TRIE_IDS.get() is None
        for generators in batches:
            for generator in reversed(generators):
                assert [output async for output in generator] == ["v1"]
        assert [
            output async for output in beam_search_dispatch(serving, "default", "c", "params")
        ] == ["v1"]

    asyncio.run(run())
    assert [
        (kwargs["request_id"], kwargs["prompt"], kwargs.get("trie_ids")) for _, kwargs in calls
    ] == [
        ("a", 2, [2]),
        ("a", 1, [1]),
        ("a", 0, [0]),
        ("b", 2, [1]),
        ("b", 1, [0]),
        ("b", 0, [2]),
        ("c", "default", None),
    ]


def test_completion_schema_accepts_trie_id_list() -> None:
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    from vllm_gr.entrypoints.openai.beam_search_patch import patch_sampling

    patch_sampling()
    request = CompletionRequest.model_validate_json(
        '{"model":"test","prompt":["a","b"],"use_beam_search":true,"vllm_xargs":{"trie_ids":[1,0]}}'
    )
    assert request.vllm_xargs["trie_ids"] == [1, 0]


def test_legacy_gr_configuration_is_rejected(monkeypatch: Any) -> None:
    for additional in (
        {"vllm_gr_beam_execution_mode": "legacy"},
        {"vllm_gr_beam_max_width": 128},
        {"vllm_gr_catalog_path": "/unused/catalog.json"},
    ):
        with pytest.raises(ValueError, match="removed.*execution_mode='v1'"):
            _dispatch(additional, monkeypatch)


def test_patch_is_idempotent_and_keeps_native_mixin(monkeypatch: Any) -> None:
    native = BeamSearchOnlineMixin.beam_search
    patch_beam_search()
    patch_beam_search()
    assert GenerateBaseServing.beam_search is beam_search_dispatch
    assert GenerateBaseServing.beam_search_v1 is serving_engine.beam_search_v1
    assert BeamSearchOnlineMixin.beam_search is native
    assert _validate_openai_beam_search_patch() is None
    assert _dispatch(None, monkeypatch)[0][0] == "native"


def test_offline_old_name_cannot_fall_back_to_upstream() -> None:
    # No engine/tokenizer is needed to reject an old caller, including legacy
    # kwargs which must not be silently translated into a V1 request.
    llm = object.__new__(GRLLM)
    with pytest.raises(NotImplementedError, match=r"removed.*beam_search_v1"):
        llm.beam_search([], object(), output_mode="compact")
