# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
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

        def prepare_inputs(
            self, scheduler_output: Any, batch_req_state: Any, batch_desc: Any
        ) -> Any:
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

        def initialize_kv_cache(self, config: Any, *, is_profiling: bool = False) -> None:
            return None

        def profile_cudagraph_memory(self) -> int:
            return 0

        def capture_model(self, *, profile_only: bool = False) -> int:
            return 0

    native_methods = {name: getattr(Runner, name) for name in ("sample", "finish_requests")}
    constraint_load = Mock(wraps=v2.make_load_model_with_constraint_backend)
    monkeypatch.setattr(v2, "make_load_model_with_constraint_backend", constraint_load)
    monkeypatch.setattr(v2, "discover_worker_runners", lambda: (("GPUModelRunnerV2", Runner),))
    v2._apply_v2_runner_patches()
    assert constraint_load.call_args.kwargs["prepare_backend"] is v2._prepare_constraint_backend

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
    from vllm_gr.v1.worker import gpu_model_runner_v2_graph as graphs

    graph_methods = {
        "initialize_kv_cache": graphs._WORKER_V2_KV_PATCH_MARKER,
        "profile_cudagraph_memory": graphs._WORKER_V2_PROFILE_PATCH_MARKER,
        "capture_model": graphs._WORKER_V2_GRAPH_PATCH_MARKER,
    }
    captured_methods = {name: getattr(Runner, name) for name in graph_methods}
    for name, marker in graph_methods.items():
        assert getattr(captured_methods[name], marker, False)

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
    assert all(getattr(Runner, n) is method for n, method in captured_methods.items())


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

    worker = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(beam_max_width=4 if beam_enabled else None),
        cudagraph_manager=None,
    )
    events = Mock()
    events.native.return_value = 17
    monkeypatch.setattr(prefill, "install_prefill_graph_manager_v2", events.plan)
    monkeypatch.setattr(graphs, "capture_multibatch_forward", events.decode)
    monkeypatch.setattr(graphs.torch.accelerator, "get_memory_info", lambda: (4096, 8192))
    monkeypatch.setattr(common, "freeze_gc_after_graph_capture", events.freeze)
    expected = [call.native(worker)]
    if beam_enabled:
        expected.insert(0, call.plan(worker))
        expected.append(call.decode(worker))

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
    from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import CudaStageHooksV2

    hooks = CudaStageHooksV2(SimpleNamespace())
    with pytest.raises(RuntimeError, match="hidden states"):
        hooks.prefill_logits(lambda worker, scheduled: object(), object())
    with pytest.raises(RuntimeError, match="kv_blocks"):
        hooks.decode_block_ids("session")


@pytest.mark.parametrize("capturing_enabled", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("gr_plan", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("profile_only", [False, True])  # type: ignore[misc]
def test_profile_capture_includes_decode_without_freezing(
    monkeypatch: pytest.MonkeyPatch, capturing_enabled: bool, gr_plan: bool, profile_only: bool
) -> None:
    from vllm.compilation import monitor

    from vllm_gr.v1.worker import gpu_model_runner_v2_graph as graphs
    from vllm_gr.v1.worker import gpu_prefill_graph_runner_v2 as prefill

    manager = (
        prefill.GRPrefillCudaGraphManager.__new__(prefill.GRPrefillCudaGraphManager)
        if gr_plan
        else SimpleNamespace()
    )
    manager._max_full_descs_to_capture = 2
    samples: list[int] = []
    manager._capture_mem_samples = samples
    worker = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(beam_max_width=128),
        cudagraph_manager=manager,
    )
    native = Mock(return_value=4096)
    setattr(native, graphs._WORKER_V2_GRAPH_PATCH_MARKER, False)
    install = Mock()
    decode = Mock()
    freeze = Mock()
    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", capturing_enabled)
    decode.side_effect = lambda worker: monitor.set_cudagraph_capturing_enabled(False)
    monkeypatch.setattr(prefill, "install_prefill_graph_manager_v2", install)
    monkeypatch.setattr(graphs, "capture_multibatch_forward", decode)
    monkeypatch.setattr(common, "freeze_gc_after_graph_capture", freeze)
    memory = iter((8192, 7168))
    monkeypatch.setattr(graphs.torch.accelerator, "get_memory_info", lambda: (next(memory), 16384))
    assert graphs.make_patched_capture_model_v2(native)(worker, profile_only=profile_only) == 5120
    native.assert_called_once_with(worker, profile_only=profile_only)
    install.assert_called_once_with(worker)
    decode.assert_called_once_with(worker)
    if profile_only:
        freeze.assert_not_called()
    else:
        freeze.assert_called_once_with(worker)
    assert monitor.cudagraph_capturing_enabled == (capturing_enabled if profile_only else False)
    measure_all = profile_only and gr_plan
    assert manager._max_full_descs_to_capture == (None if measure_all else 2)
    assert manager._capture_mem_samples is (None if measure_all else samples)
    assert samples == []  # The native profiler must not extrapolate a full measurement.


@pytest.mark.parametrize("failure", [False, True])  # type: ignore[misc]
def test_native_profiler_uses_gr_plan_and_cleans_temporary_decode(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    from contextlib import nullcontext

    from vllm.config import CUDAGraphMode as Mode
    from vllm.v1.worker.gpu import cudagraph_utils as native

    from vllm_gr.v1.worker import gpu_model_runner_v2_graph as graphs
    from vllm_gr.v1.worker import gpu_prefill_graph_runner_v2 as prefill

    mib = 1 << 20
    manager = SimpleNamespace(
        needs_capture=lambda: True,
        use_breakable_cg=False,
        _capture_descs={Mode.FULL: list(range(7))},
    )
    worker = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(beam_max_width=128),
        compilation_config=SimpleNamespace(cudagraph_mode=Mode.FULL),
        vllm_config=object(),
    )
    graph = Mock()
    events: list[str] = []

    def install(runner: Any) -> None:
        assert runner.cudagraph_manager is not manager
        runner.cudagraph_manager = manager

    def init(runner: Any, config: Any, *, is_profiling: bool) -> None:
        assert is_profiling
        runner.cudagraph_manager = object()

    def capture(*, profile_only: bool) -> int:
        assert profile_only
        assert manager._max_full_descs_to_capture == 2
        assert manager.pool == native.current_platform.get_global_graph_pool()
        manager._capture_mem_samples.extend([10 * mib, 2 * mib])
        worker._beam_decode_graph = graph
        worker._gr_gpu_stage_runner = object()
        worker._gr_input_buffers = object()
        worker._beam_context = object()
        events.append("capture")
        if failure:
            raise RuntimeError("capture failure")
        return 32 * mib  # 12 sampled FULL + 20 PW/Decode/fixed state.

    def teardown(runner: Any) -> None:
        events.append("native teardown")
        runner.cudagraph_manager = None

    worker.capture_model = capture
    monkeypatch.setattr(prefill, "install_prefill_graph_manager_v2", install)
    monkeypatch.setattr(native, "set_current_vllm_config", lambda config: nullcontext())
    monkeypatch.setattr(
        native,
        "_init_minimal_kv_cache_for_profiling",
        lambda r: graphs.make_patched_initialize_kv_cache_v2(init)(r, object(), is_profiling=True),
    )
    monkeypatch.setattr(native, "_teardown_profiling_state", teardown)
    monkeypatch.setattr(native.current_platform, "graph_pool_handle", lambda: (123, 456))
    monkeypatch.setattr(graphs.torch.accelerator, "empty_cache", lambda: None)
    pool = type(native.current_platform)._global_graph_pool
    profile = graphs.make_patched_profile_cudagraph_memory_v2(native.profile_cudagraph_memory)
    if failure:
        with pytest.raises(RuntimeError, match="capture failure"):
            profile(worker)
    else:
        # Extrapolation must use the GR plan's seven FULL graphs.
        assert profile(worker) == (20 + 10 + 6 * 2) * mib
    assert events == ["capture", "native teardown"]
    assert type(native.current_platform)._global_graph_pool == pool
    assert worker.cudagraph_manager is None and worker._beam_decode_graph is None
    assert not any(
        hasattr(worker, n)
        for n in (
            "_gr_gpu_stage_runner",
            "_gr_input_buffers",
            "_beam_context",
        )
    )
    graph.uninstall.assert_called_once()
    graph.clear.assert_called_once()


def test_profile_rejects_bound_runtime_and_preserves_non_beam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker import gpu_model_runner_v2_graph as graphs

    worker = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(beam_max_width=128),
        _beam_decode_graph=object(),
    )
    original = Mock(return_value=42)
    setattr(original, graphs._WORKER_V2_PROFILE_PATCH_MARKER, False)
    profile = graphs.make_patched_profile_cudagraph_memory_v2(original)
    with pytest.raises(RuntimeError, match="unbound GR runtime"):
        profile(worker)
    original.assert_not_called()
    worker._vllm_gr_config.beam_max_width = None
    assert profile(worker) == 42
    original.assert_called_once_with(worker)


def test_prepare_inputs_preserves_native_batch_state_for_ordinary_requests() -> None:
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    worker, state, result = object(), object(), object()
    scheduled = SimpleNamespace(num_scheduled_tokens={"ordinary": 2})
    desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, 2, 1)
    native = Mock(return_value=result)
    prepared = v2.make_patched_prepare_inputs_v2(native)(worker, scheduled, state, desc)
    assert prepared is result
    native.assert_called_once_with(worker, scheduled, state, desc)


pytestmark = pytest.mark.cpu_test
