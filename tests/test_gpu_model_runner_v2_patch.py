# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for ModelRunner V2 GPU wrappers."""

import logging
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, call

import pytest

from vllm_gr.v1.worker import gpu_model_runner_v2_patch as v2
from vllm_gr.v1.worker import model_runner_common as common


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

        def finish_requests(self, scheduler_output: Any) -> None:
            return None

        def add_requests(self, scheduler_output: Any) -> None:
            return None

    native_methods = {name: getattr(Runner, name) for name in ("sample", "finish_requests")}
    monkeypatch.setattr(v2, "discover_worker_runners", lambda: (("GPUModelRunnerV2", Runner),))
    v2._apply_v2_runner_patches()

    # Input preparation/registration have guarded GR specializations; native
    # sampling and request retirement remain unchanged.
    assert getattr(Runner.execute_model, common._WORKER_EXECUTE_PATCH_MARKER, False)
    assert all(getattr(Runner, name) is method for name, method in native_methods.items())
    assert getattr(Runner.sample_tokens, common._WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, False)
    assert getattr(Runner.load_model, common._CONSTRAINT_LOAD_PATCH_MARKER, False)
    assert getattr(Runner.shutdown, common._CONSTRAINT_SHUTDOWN_PATCH_MARKER, False)
    assert getattr(Runner.prepare_inputs, common._WORKER_V2_PREPARE_PATCH_MARKER, False)
    assert getattr(Runner.add_requests, common._WORKER_V2_ADD_REQUESTS_PATCH_MARKER, False)
    assert common.is_worker_runner_v2_patched(Runner)
    assert common.is_worker_runner_patched(Runner)

    # Idempotent: a second apply leaves every wrapper unchanged.
    execute, sample, sample_tokens, prepare, add, load, shutdown = (
        Runner.execute_model,
        Runner.sample,
        Runner.sample_tokens,
        Runner.prepare_inputs,
        Runner.add_requests,
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
    assert Runner.prepare_inputs is prepare
    assert Runner.add_requests is add


def test_sample_tokens_v2_plain_fallback_is_pure_native() -> None:
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


def test_sample_tokens_v2_stage_bypass() -> None:
    """A pending stage runner takes over sample_tokens entirely."""
    sampled = SimpleNamespace()
    stage_runner = SimpleNamespace(pending=object(), sample=lambda g: sampled)

    def native(self: Any, grammar_output: Any) -> Any:
        raise AssertionError("native sample_tokens must not run with a pending stage")

    wrapped = v2.make_patched_sample_tokens_v2(native)
    runner = SimpleNamespace(_gr_gpu_stage_runner=stage_runner)
    assert wrapped(runner, None) is sampled


def test_configure_model_runner_default_normalizes_opt_in(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from vllm_gr._env_check import configure_model_runner_default

    for value in (None, "0", "false", "no", "off", "1", "true", "yes", "on"):
        caplog.clear()
        if value is None:
            monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
        else:
            monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)
        enabled = value in ("1", "true", "yes", "on")
        with caplog.at_level(logging.WARNING):
            configure_model_runner_default()
        assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == str(int(enabled))
        assert ("experimental" in caplog.text) is enabled


@pytest.mark.parametrize(  # type: ignore[misc]
    "beam_enabled,failure", [(False, None), (True, None), (False, "native"), (True, "decode")]
)
def test_capture_order_and_failure_propagation(
    monkeypatch: pytest.MonkeyPatch, beam_enabled: bool, failure: str | None
) -> None:
    from vllm_gr.v1.worker import gpu_model_runner_v2_graph as graphs
    from vllm_gr.v1.worker import gpu_prefill_graph_runner_v2 as prefill
    from vllm_gr.v1.worker import gr_startup
    from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import V2StageExecutionHooks

    worker = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(beam_max_width=4 if beam_enabled else None)
    )
    events = Mock()
    events.native.return_value = 17
    monkeypatch.setattr(prefill, "install_prefill_graph_manager_v2", events.plan)
    monkeypatch.setattr(gr_startup, "capture_async_forward", events.decode)
    monkeypatch.setattr(common, "freeze_gc_after_graph_capture", events.freeze)
    expected = [call.native(worker)]
    if beam_enabled:
        expected.insert(0, call.plan(worker))
        expected.append(
            call.decode(worker, hooks_type=V2StageExecutionHooks, allow_eager_fallback=True)
        )

    def native(runner: Any) -> int:
        return cast(int, events.native(runner))

    wrapped = graphs.make_patched_capture_model_v2(native)
    if failure:
        error = RuntimeError("operation not supported when stream is capturing")
        getattr(events, failure).side_effect = error
        with pytest.raises(RuntimeError) as raised:
            wrapped(worker)
        assert raised.value is error
    else:
        assert wrapped(worker) == 17
        expected.append(call.freeze(worker))
    assert events.mock_calls == expected
    if beam_enabled:
        assert not worker._gr_async_decode_graph_capture_failed


def test_stage_hooks_require_native_prefill_output_and_scheduler_kv() -> None:
    from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import V2StageExecutionHooks

    hooks = V2StageExecutionHooks(SimpleNamespace())
    with pytest.raises(RuntimeError, match="hidden states"):
        hooks.prefill_logits(lambda worker, scheduled: object(), object())
    with pytest.raises(RuntimeError, match="kv_blocks"):
        hooks.decode_block_ids("session")
