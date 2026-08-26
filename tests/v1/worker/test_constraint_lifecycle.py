# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.config import ConstraintTableConfig, GRConfig
from vllm_gr.config.parser import gr_config_to_mapping
from vllm_gr.v1.worker.model_runner_patch import (
    _runner_vocab_size,
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)

_ARTIFACT_DIGEST = "sha256:" + "1" * 64
_TOKENIZER_DIGEST = "sha256:" + "2" * 64


class _Backend:
    def __init__(self, *, device_memory_bytes: int = 32) -> None:
        self.info = SimpleNamespace(device_memory_bytes=device_memory_bytes)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _vllm_config(
    *,
    enabled: bool = True,
    beam_max_width: int | None = 4,
    max_top_k: int = 4,
) -> SimpleNamespace:
    config = GRConfig(
        beam_max_width=beam_max_width,
        beam_max_decode_steps=3,
        constraint_table=ConstraintTableConfig(
            max_top_k=max_top_k,
            enabled=enabled,
            path="/models/constraint-table.safetensors" if enabled else None,
            artifact_digest=_ARTIFACT_DIGEST if enabled else None,
            tokenizer_digest=_TOKENIZER_DIGEST if enabled else None,
        ),
    )
    return SimpleNamespace(
        additional_config={"vllm_gr": gr_config_to_mapping(config)},
    )


def _runner(
    *,
    enabled: bool = True,
    beam_max_width: int | None = 4,
    max_top_k: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        constraint_backend=None,
        vllm_config=_vllm_config(
            enabled=enabled,
            beam_max_width=beam_max_width,
            max_top_k=max_top_k,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            get_vocab_size=lambda: 32000,
        ),
        device=torch.device("cpu"),
        model_memory_usage=0,
    )


def test_runner_vocab_size_prefers_upstream_api() -> None:
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            vocab_size=8,
            get_vocab_size=lambda: 32000,
        )
    )
    assert _runner_vocab_size(runner) == 32000


def test_weights_then_readiness_then_publication_and_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    backend = _Backend(device_memory_bytes=37)
    events: list[str] = []

    def original(self: SimpleNamespace) -> str:
        assert self.constraint_backend is None
        events.append("weights")
        self.model_memory_usage = 100
        return "loaded"

    def load_backend(**kwargs: Any) -> _Backend:
        events.append("readiness")
        assert runner.constraint_backend is None
        assert kwargs["max_rows"] == 4
        assert kwargs["vocab_size"] == 32000
        assert kwargs["dtype"] is torch.float16
        return backend

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        load_backend,
    )

    wrapped = make_load_model_with_constraint_backend(original)
    assert wrapped(runner) == "loaded"
    assert events == ["weights", "readiness"]
    assert runner.constraint_backend is backend
    assert runner.model_memory_usage == 137


def test_loader_failure_never_publishes_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()

    def original(self: SimpleNamespace) -> None:
        self.model_memory_usage = 100

    def fail_readiness(**kwargs: Any) -> None:
        del kwargs
        assert runner.constraint_backend is None
        raise RuntimeError("readiness failed")

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        fail_readiness,
    )

    with pytest.raises(RuntimeError, match="readiness failed"):
        make_load_model_with_constraint_backend(original)(runner)
    assert runner.constraint_backend is None


def test_model_load_failure_closes_old_backend() -> None:
    runner = _runner()
    old_backend = _Backend()
    runner.constraint_backend = old_backend

    def fail_weights(self: SimpleNamespace) -> None:
        assert self.constraint_backend is None
        raise RuntimeError("weights failed")

    with pytest.raises(RuntimeError, match="weights failed"):
        make_load_model_with_constraint_backend(fail_weights)(runner)
    assert runner.constraint_backend is None
    assert old_backend.close_calls == 1


def test_complete_reload_and_shutdown_close_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    created: list[_Backend] = []

    def original_load(self: SimpleNamespace) -> None:
        self.model_memory_usage = 10

    def load_backend(**kwargs: Any) -> _Backend:
        del kwargs
        backend = _Backend()
        created.append(backend)
        return backend

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        load_backend,
    )
    load = make_load_model_with_constraint_backend(original_load)
    load(runner)
    load(runner)

    assert len(created) == 2
    assert created[0].close_calls == 1
    assert runner.constraint_backend is created[1]

    native_shutdown_calls: list[int] = []

    def native_shutdown(self: SimpleNamespace) -> None:
        del self
        native_shutdown_calls.append(1)

    shutdown = make_shutdown_with_constraint_backend(native_shutdown)
    shutdown(runner)
    shutdown(runner)
    assert created[1].close_calls == 1
    assert runner.constraint_backend is None
    assert len(native_shutdown_calls) == 2


def test_shutdown_is_installed_when_runner_has_no_native_method() -> None:
    runner = _runner()
    backend = _Backend()
    runner.constraint_backend = backend

    shutdown = make_shutdown_with_constraint_backend()
    shutdown(runner)
    shutdown(runner)

    assert backend.close_calls == 1
    assert runner.constraint_backend is None


def test_disabled_configuration_never_calls_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(enabled=False)

    def forbidden(**kwargs: Any) -> None:
        raise AssertionError(f"disabled configuration called loader: {kwargs}")

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        forbidden,
    )

    def original(self: SimpleNamespace) -> str:
        self.model_memory_usage = 9
        return "loaded"

    assert make_load_model_with_constraint_backend(original)(runner) == "loaded"
    assert runner.constraint_backend is None
    assert runner.model_memory_usage == 9


@pytest.mark.parametrize(  # type: ignore[misc]
    ("beam_max_width", "max_top_k", "message"),
    [
        (None, 4, "beam_max_width"),
        (5, 4, "max_top_k"),
    ],
)
def test_canonical_capacity_validation_happens_after_weights(
    beam_max_width: int | None,
    max_top_k: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(
        beam_max_width=beam_max_width,
        max_top_k=max_top_k,
    )
    events: list[str] = []

    def original(self: SimpleNamespace) -> None:
        del self
        events.append("weights")

    def forbidden(**kwargs: Any) -> None:
        raise AssertionError(f"invalid capacity called loader: {kwargs}")

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        forbidden,
    )

    with pytest.raises(ValueError, match=message):
        make_load_model_with_constraint_backend(original)(runner)
    assert events == ["weights"]
    assert runner.constraint_backend is None
