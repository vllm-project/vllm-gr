# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The upstream Beam name must never reach the retired GR frontend loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.entrypoints.openai.engine.serving import OpenAIServing
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
    additional: dict[str, Any] | None, monkeypatch: Any, params: Any = "params"
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
                serving, "prompt", "request-id", params, "lora", {"trace": "id"}
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
    assert OpenAIServing.beam_search is beam_search_dispatch
    assert OpenAIServing.beam_search_v1 is serving_engine.beam_search_v1
    assert BeamSearchOnlineMixin.beam_search is native
    assert _validate_openai_beam_search_patch() is None
    assert _dispatch(None, monkeypatch)[0][0] == "native"


def test_offline_old_name_cannot_fall_back_to_upstream() -> None:
    # No engine/tokenizer is needed to reject an old caller, including legacy
    # kwargs which must not be silently translated into a V1 request.
    llm = object.__new__(GRLLM)
    with pytest.raises(NotImplementedError, match=r"removed.*beam_search_v1"):
        llm.beam_search([], object(), output_mode="compact")
