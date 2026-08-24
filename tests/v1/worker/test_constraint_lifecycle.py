# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.config import BeamConfig, ConstraintTableConfig, GRConfig
from vllm_gr.config.parser import gr_config_to_mapping
from vllm_gr.constraints.backend import ConstraintOutput
from vllm_gr.v1.worker.constraint_hook import try_sample_constrained
from vllm_gr.v1.worker.model_runner_patch import (
    _runner_vocab_size,
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)

DIGEST = "sha256:" + "1" * 64
TOKENIZER_DIGEST = "sha256:" + "2" * 64


def _vllm_config(*, enabled: bool = True) -> SimpleNamespace:
    config = GRConfig(
        beam=BeamConfig(max_width=4, max_decode_steps=3),
        constraint_table=ConstraintTableConfig(
            enabled=enabled,
            path="/models/constraint_table.safetensors" if enabled else None,
            artifact_digest=DIGEST if enabled else None,
            tokenizer_digest=TOKENIZER_DIGEST if enabled else None,
            max_top_k=4,
        ),
    )
    return SimpleNamespace(
        additional_config={"vllm_gr": gr_config_to_mapping(config)},
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )


class FakeBackend:
    def __init__(self, identity: int = 0, events: list[str] | None = None) -> None:
        self.identity = identity
        self.events = events
        self.close_calls = 0
        self.info = SimpleNamespace(device_memory_bytes=32)

    def close(self) -> None:
        self.close_calls += 1

    def select(self, logits: torch.Tensor, metadata: object) -> ConstraintOutput:
        if self.events is not None:
            self.events.append("select")
        token_ids = torch.tensor([[1, 2]], dtype=torch.int32, device=logits.device)
        logprobs = torch.tensor([[-0.1, -0.2]], dtype=torch.float32, device=logits.device)
        return ConstraintOutput(
            token_ids=token_ids,
            logprobs=logprobs,
            valid_mask=torch.ones_like(token_ids, dtype=torch.bool),
        )


def _runner(*, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        constraint_backend=None,
        vllm_config=_vllm_config(enabled=enabled),
        model_config=SimpleNamespace(vocab_size=8, dtype=torch.float16),
        device=torch.device("cpu"),
        model_memory_usage=0,
    )


def test_vocab_size_uses_vllm_model_config_api() -> None:
    runner = SimpleNamespace(model_config=SimpleNamespace(get_vocab_size=lambda: 32000))
    assert _runner_vocab_size(runner) == 32000


def test_weights_backend_profile_order_and_memory_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    backend = FakeBackend(events=events)
    runner = _runner()

    def original(self) -> str:
        events.append("weights")
        self.model_memory_usage = 100
        return "loaded"

    def load_backend(**kwargs):
        events.append("backend")
        assert runner.constraint_backend is None
        return backend

    monkeypatch.setattr("vllm_gr.constraints.loader.load_constraint_backend", load_backend)
    wrapped = make_load_model_with_constraint_backend(original)
    assert wrapped(runner) == "loaded"
    events.append("profile")

    assert events == ["weights", "backend", "profile"]
    assert runner.constraint_backend is backend
    assert runner.model_memory_usage == 132


def test_backend_failure_leaves_none_and_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()

    def original(self) -> None:
        self.model_memory_usage = 100

    def fail(**kwargs):
        raise RuntimeError("ready check failed")

    monkeypatch.setattr("vllm_gr.constraints.loader.load_constraint_backend", fail)
    wrapped = make_load_model_with_constraint_backend(original)
    with pytest.raises(RuntimeError, match="ready check failed"):
        wrapped(runner)
    assert runner.constraint_backend is None


def test_model_load_failure_closes_old_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    old = FakeBackend()
    runner = _runner()
    runner.constraint_backend = old

    def original(self) -> None:
        raise RuntimeError("weights failed")

    wrapped = make_load_model_with_constraint_backend(original)
    with pytest.raises(RuntimeError, match="weights failed"):
        wrapped(runner)
    assert old.close_calls == 1
    assert runner.constraint_backend is None


def test_two_dp_workers_receive_distinct_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeBackend] = []

    def load_backend(**kwargs):
        backend = FakeBackend(identity=len(created))
        created.append(backend)
        return backend

    monkeypatch.setattr("vllm_gr.constraints.loader.load_constraint_backend", load_backend)

    def original(self) -> None:
        self.model_memory_usage = 0

    wrapped = make_load_model_with_constraint_backend(original)
    worker0 = _runner()
    worker1 = _runner()
    wrapped(worker0)
    wrapped(worker1)
    assert worker0.constraint_backend is created[0]
    assert worker1.constraint_backend is created[1]
    assert worker0.constraint_backend is not worker1.constraint_backend


def test_reload_and_shutdown_close_are_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeBackend] = []

    def load_backend(**kwargs):
        backend = FakeBackend(identity=len(created))
        created.append(backend)
        return backend

    monkeypatch.setattr("vllm_gr.constraints.loader.load_constraint_backend", load_backend)

    def original_load(self) -> None:
        self.model_memory_usage = 0

    runner = _runner()
    load = make_load_model_with_constraint_backend(original_load)
    load(runner)
    load(runner)
    assert created[0].close_calls == 1
    assert runner.constraint_backend is created[1]

    shutdown_calls: list[int] = []

    def original_shutdown(self) -> None:
        shutdown_calls.append(1)

    shutdown = make_shutdown_with_constraint_backend(original_shutdown)
    shutdown(runner)
    shutdown(runner)
    assert created[1].close_calls == 1
    assert len(shutdown_calls) == 2


def test_shutdown_method_is_installed_for_runner_without_native_shutdown() -> None:
    runner = _runner()
    backend = FakeBackend()
    runner.constraint_backend = backend

    shutdown = make_shutdown_with_constraint_backend()
    shutdown(runner)
    shutdown(runner)

    assert backend.close_calls == 1
    assert runner.constraint_backend is None


def _sampling_runner(backend: FakeBackend | None) -> SimpleNamespace:
    return SimpleNamespace(
        constraint_backend=backend,
        vllm_config=_vllm_config(),
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            sampling_metadata=SimpleNamespace(temperature=torch.ones(1, dtype=torch.float32))
        ),
        _current_beam_data={
            "req": {
                "is_constrained": True,
                "constraint_step": 0,
                "beam_width": 2,
                "beam_suffix": [],
                "max_decode_steps": 3,
            }
        },
        _current_scheduler_output=SimpleNamespace(_gr_request_row_configs=[("req", 0, 1, [])]),
    )


def test_first_sample_consumes_preloaded_backend_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    backend = FakeBackend(events=events)

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling attempted a persistent resource load")

    monkeypatch.setattr("vllm_gr.constraints.loader.load_constraint_backend", forbidden)
    monkeypatch.setattr("vllm_gr.constraints.artifact.load_constraint_table_artifact", forbidden)
    monkeypatch.setattr("vllm_gr.constraints.npu_backend.resolve_npu_constraint_op", forbidden)

    runner = _sampling_runner(backend)
    runner.input_batch.update_async_output_token_ids = lambda: events.append("async")
    output = try_sample_constrained(
        runner,
        torch.zeros((1, 8)),
    )
    assert events == ["async", "select"]
    assert output is not None
    assert output.logprobs_tensors.logprob_token_ids.tolist() == [[1, 2]]


def test_missing_backend_never_enters_unconstrained_sampling() -> None:
    with pytest.raises(RuntimeError, match="constraint_backend is None"):
        try_sample_constrained(
            _sampling_runner(None),
            torch.zeros((1, 8)),
        )


def test_constrained_sampling_rejects_speculative_metadata() -> None:
    events: list[str] = []
    backend = FakeBackend(events=events)
    with pytest.raises(RuntimeError, match="does not support speculative"):
        try_sample_constrained(
            _sampling_runner(backend),
            torch.zeros((1, 8)),
            spec_decode_metadata=object(),
        )
    assert events == []


def test_disabled_config_does_not_call_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner(enabled=False)

    def forbidden(**kwargs):
        raise AssertionError("disabled configuration called loader")

    monkeypatch.setattr("vllm_gr.constraints.loader.load_constraint_backend", forbidden)

    def original(self) -> None:
        self.model_memory_usage = 9

    make_load_model_with_constraint_backend(original)(runner)
    assert runner.constraint_backend is None
    assert runner.model_memory_usage == 9
