# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm_gr.entrypoints.openai import beam_proc_manager, serving_engine, serving_models


def _initialize_models(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool, failure_stage: str | None = None
) -> list[str]:
    calls: list[str] = []

    def record(stage: str) -> None:
        calls.append(stage)
        if stage == failure_stage:
            raise RuntimeError(f"{stage} failed")

    tokenizer = object()
    model_config = object()
    engine_client = SimpleNamespace(vllm_config=object())

    def original_init(self: Any, *_args: Any, **_kwargs: Any) -> None:
        self.input_processor = SimpleNamespace(tokenizer=tokenizer)
        self.model_config = model_config

    catalog = MagicMock()
    catalog.load.side_effect = lambda _path: calls.append("catalog-load")
    monkeypatch.setattr(serving_models, "_original_init", original_init)
    monkeypatch.setattr(serving_models, "Catalog", MagicMock(return_value=catalog))
    monkeypatch.setattr(
        serving_models.GRConfig,
        "from_vllm_config",
        MagicMock(return_value=SimpleNamespace(catalog_path="/catalog.json")),
    )
    monkeypatch.setattr(beam_proc_manager, "beam_proc_enabled", lambda: enabled)
    monkeypatch.setattr(
        beam_proc_manager,
        "prepare_beam_proc_catalog",
        MagicMock(side_effect=lambda *_args, **_kwargs: record("prepare")),
    )
    monkeypatch.setattr(
        beam_proc_manager,
        "ensure_beam_proc_started",
        MagicMock(side_effect=lambda: record("start")),
    )

    monkeypatch.setattr(
        serving_engine,
        "_ensure_beamproc_apiserver_output",
        MagicMock(side_effect=lambda *_args: record("receiver")),
    )
    serving_models.init_gr(SimpleNamespace(), engine_client, [])
    return calls


def test_enabled_beam_proc_starts_after_catalog_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _initialize_models(monkeypatch, enabled=True) == [
        "catalog-load",
        "prepare",
        "start",
        "receiver",
    ]


def test_disabled_beam_proc_is_not_prepared_or_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _initialize_models(monkeypatch, enabled=False) == ["catalog-load"]


@pytest.mark.parametrize("failure_stage", ["prepare", "start", "receiver"])  # type: ignore
def test_beamproc_failure_prevents_model_initialization(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        _initialize_models(monkeypatch, enabled=True, failure_stage=failure_stage)
