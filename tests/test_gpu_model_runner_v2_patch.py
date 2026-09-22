# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for ModelRunner V2 GPU wrappers."""

import logging
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from vllm_gr.v1.worker import gpu_model_runner_v2_patch as v2
from vllm_gr.v1.worker import model_runner_common as common

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_worker_runners_returns_v2_class() -> None:
    """V2 discovery yields the V2 GPUModelRunner (prepare_inputs, no _prepare_inputs)."""
    runners = v2.discover_worker_runners()
    assert runners
    name, cls = runners[0]
    assert name == "GPUModelRunnerV2"
    assert hasattr(cls, "prepare_inputs")
    assert not hasattr(cls, "_prepare_inputs")
    assert hasattr(cls, "sample")
    assert hasattr(cls, "sample_tokens")


def test_discover_worker_runners_empty_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = ModuleType("vllm.v1.worker.gpu.model_runner")

    def raise_import(name: str) -> None:
        raise ImportError(f"no {name}")

    # ModuleType.__getattr__ is read-only for mypy; stub it at the instance
    # level so any missing attribute on the fake module raises ImportError.
    fake.__getattr__ = raise_import  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.model_runner", fake)
    assert v2.discover_worker_runners() == ()


# ---------------------------------------------------------------------------
# is_worker_runner_patched V2 dispatch
# ---------------------------------------------------------------------------


def _mark(fn: Any, marker: str) -> Any:
    setattr(fn, marker, True)
    return fn


def test_is_worker_runner_v2_patched_requires_full_method_set() -> None:
    def stub() -> None:
        return None

    class Runner:
        execute_model: Any
        prepare_inputs: Any
        sample: Any
        sample_tokens: Any
        load_model: Any
        shutdown: Any

    Runner.execute_model = _mark(stub, common._WORKER_EXECUTE_PATCH_MARKER)
    Runner.prepare_inputs = _mark(stub, common._WORKER_V2_PREPARE_PATCH_MARKER)
    Runner.sample = stub
    Runner.sample_tokens = _mark(stub, common._WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER)
    Runner.load_model = _mark(stub, common._CONSTRAINT_LOAD_PATCH_MARKER)
    Runner.shutdown = _mark(stub, common._CONSTRAINT_SHUTDOWN_PATCH_MARKER)

    assert common.is_worker_runner_v2_patched(Runner)
    assert common.is_worker_runner_patched(Runner)


def test_is_worker_runner_v2_patched_missing_method_fails() -> None:
    def stub() -> None:
        return None

    class Runner:
        execute_model: Any
        prepare_inputs: Any
        sample: Any
        sample_tokens: Any
        load_model: Any
        shutdown: Any

    Runner.execute_model = _mark(stub, common._WORKER_EXECUTE_PATCH_MARKER)
    Runner.prepare_inputs = _mark(stub, common._WORKER_V2_PREPARE_PATCH_MARKER)
    # sample_tokens / load / shutdown missing; native sample is intentionally
    # not part of the beam_search_v1-only V2 patch contract.
    Runner.sample = stub
    Runner.sample_tokens = stub
    Runner.load_model = stub
    Runner.shutdown = stub

    assert not common.is_worker_runner_v2_patched(Runner)
    assert not common.is_worker_runner_patched(Runner)


# ---------------------------------------------------------------------------
# _apply_v2_runner_patches — marker stamping + idempotency
# ---------------------------------------------------------------------------


def test_apply_v2_runner_patches_stamps_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Runner:
        def execute_model(self, scheduler_output: Any, *a: Any, **k: Any) -> None:
            return None

        def prepare_inputs(self, scheduler_output: Any, batch_desc: Any) -> Any:
            return None

        def sample(self, hidden_states: Any, input_batch: Any, grammar_output: Any = None) -> Any:
            return None

        def sample_tokens(self, grammar_output: Any = None) -> Any:
            return None

        def load_model(self, *a: Any, **k: Any) -> None:
            return None

        def shutdown(self) -> None:
            return None

    native_sample = Runner.sample
    monkeypatch.setattr(v2, "discover_worker_runners", lambda: (("GPUModelRunnerV2", Runner),))
    v2._apply_v2_runner_patches()

    # The slim V2 install set: execute_model, sample_tokens, load, shutdown.
    # Native sample plus prepare_inputs / finish_requests / add_requests are
    # NOT patched under V2.
    assert getattr(Runner.execute_model, common._WORKER_EXECUTE_PATCH_MARKER, False)
    assert Runner.sample is native_sample
    assert getattr(Runner.sample_tokens, common._WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, False)
    assert getattr(Runner.load_model, common._CONSTRAINT_LOAD_PATCH_MARKER, False)
    assert getattr(Runner.shutdown, common._CONSTRAINT_SHUTDOWN_PATCH_MARKER, False)
    assert not getattr(Runner.prepare_inputs, common._WORKER_V2_PREPARE_PATCH_MARKER, False)
    assert common.is_worker_runner_v2_patched(Runner)

    # Idempotent: a second apply leaves every wrapper unchanged.
    execute, sample, sample_tokens, load, shutdown = (
        Runner.execute_model,
        Runner.sample,
        Runner.sample_tokens,
        Runner.load_model,
        Runner.shutdown,
    )
    v2._apply_v2_runner_patches()
    assert (Runner.execute_model, Runner.sample, Runner.sample_tokens) == (
        execute,
        sample,
        sample_tokens,
    )
    assert (Runner.load_model, Runner.shutdown) == (load, shutdown)


# ---------------------------------------------------------------------------
# prepare_inputs remap
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# P1: cached remap constant tensors
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# sample wrapper — constraint bypass / native sampler / decision / collapse
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# sample_tokens attach
# ---------------------------------------------------------------------------


def test_sample_tokens_v2_plain_fallback_is_pure_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a pending stage runner, sample_tokens_v2 is a pure native
    pass-through (no legacy beam-result attach — LEGACY frames are gone)."""
    out = SimpleNamespace()
    calls: list[Any] = []

    def native(self: Any, grammar_output: Any) -> Any:
        calls.append(grammar_output)
        return out

    wrapped = v2.make_patched_sample_tokens_v2(native)
    assert getattr(wrapped, common._WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, False)
    assert v2.make_patched_sample_tokens_v2(wrapped) is wrapped

    runner = SimpleNamespace()  # no _gr_gpu_stage_runner
    assert wrapped(runner, None) is out
    assert calls == [None]


def test_sample_tokens_v2_stage_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending stage runner takes over sample_tokens entirely."""
    sampled = SimpleNamespace()
    stage_runner = SimpleNamespace(pending=object(), sample=lambda g: sampled)

    def native(self: Any, grammar_output: Any) -> Any:
        raise AssertionError("native sample_tokens must not run with a pending stage")

    wrapped = v2.make_patched_sample_tokens_v2(native)
    runner = SimpleNamespace(_gr_gpu_stage_runner=stage_runner)
    assert wrapped(runner, None) is sampled


# ---------------------------------------------------------------------------
# configure_model_runner_default env behavior
# ---------------------------------------------------------------------------


def test_configure_model_runner_default_warns_on_explicit_v2(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from vllm_gr._env_check import configure_model_runner_default

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with caplog.at_level(logging.WARNING):
        configure_model_runner_default()
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert "experimental" in caplog.text


def test_configure_model_runner_default_normalizes_truthy_spellings(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from vllm_gr._env_check import configure_model_runner_default

    # vLLM parses the flag with int(); non-numeric truthy spellings must be
    # normalized to "1" so they don't crash at vllm.envs import time.
    for value in ("true", "yes", "on"):
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)
        with caplog.at_level(logging.WARNING):
            configure_model_runner_default()
        assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_configure_model_runner_default_normalizes_falsy_spellings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr._env_check import configure_model_runner_default

    for value in ("false", "no", "off"):
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)
        configure_model_runner_default()
        assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_configure_model_runner_default_defaults_to_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr._env_check import configure_model_runner_default

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    configure_model_runner_default()
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"


# ---------------------------------------------------------------------------
# M3 graph-replay contracts (regression guards for the two V2-only fixes)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# P0: beam steady-state state-machine bypass
# ---------------------------------------------------------------------------


class _SmRecorder:
    """Native-chain recorder: counts finish/add invocations.

    The methods use the native (unbound) signature: the V2 wrappers call
    ``_orig(runner, scheduler_output)``.
    """

    def __init__(self) -> None:
        self.finish_calls = 0
        self.add_calls = 0
        self.removed: list[str] = []
        self.registered: list[str] = []

    def finish_requests(self, runner: Any, scheduler_output: Any) -> None:
        self.finish_calls += 1
        self.removed.extend(scheduler_output.finished_req_ids)

    def add_requests(self, runner: Any, scheduler_output: Any) -> None:
        self.add_calls += 1
        self.registered.extend([r.req_id for r in scheduler_output.scheduled_new_reqs])


def test_apply_v2_runner_patches_does_not_patch_state_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """finish_requests / add_requests are NOT patched under V2 — the stage path
    drives the native state machine directly via consume_native_state."""

    class Runner:
        def execute_model(self, scheduler_output: Any, *a: Any, **k: Any) -> None:
            return None

        def sample(self, hidden_states: Any, input_batch: Any, grammar_output: Any = None) -> Any:
            return None

        def sample_tokens(self, grammar_output: Any = None) -> Any:
            return None

        def finish_requests(self, scheduler_output: Any) -> None:
            return None

        def add_requests(self, scheduler_output: Any) -> None:
            return None

        def load_model(self, *a: Any, **k: Any) -> None:
            return None

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(v2, "discover_worker_runners", lambda: (("GPUModelRunnerV2", Runner),))
    v2._apply_v2_runner_patches()

    assert not getattr(
        Runner.finish_requests, common._WORKER_V2_FINISH_REQUESTS_PATCH_MARKER, False
    )
    assert not getattr(Runner.add_requests, common._WORKER_V2_ADD_REQUESTS_PATCH_MARKER, False)
