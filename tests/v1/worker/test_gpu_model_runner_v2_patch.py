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

from tests.helpers.gr_config import update_worker_config, worker_config
from vllm_gr.config import GRConfig
from vllm_gr.config.models import DeploymentOptions
from vllm_gr.v1.worker import gr_gc as _gc_runtime
from vllm_gr.v1.worker import model_runner_contracts as _patch_contracts
from vllm_gr.v1.worker.cuda.v2 import gpu_model_runner_v2_patch as v2
from vllm_gr.v1.worker.cuda.v2 import gpu_model_runner_v2_resources as resources
from vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_resources import get_v2_runtime


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
    assert constraint_load.call_args.kwargs["prepare_backend"] is v2._warmup_constraint_backend

    # Input preparation/registration have guarded GR specializations; native
    # sampling and request retirement remain unchanged.
    assert getattr(Runner.execute_model, _patch_contracts._WORKER_EXECUTE_PATCH_MARKER, False)
    assert all(getattr(Runner, name) is method for name, method in native_methods.items())
    assert getattr(
        Runner.sample_tokens, _patch_contracts._WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, False
    )
    assert getattr(Runner.load_model, _patch_contracts._CONSTRAINT_LOAD_PATCH_MARKER, False)
    assert getattr(Runner.shutdown, _patch_contracts._CONSTRAINT_SHUTDOWN_PATCH_MARKER, False)
    assert getattr(Runner.prepare_inputs, _patch_contracts._WORKER_V2_PREPARE_PATCH_MARKER, False)
    assert getattr(
        Runner.add_requests, _patch_contracts._WORKER_V2_ADD_REQUESTS_PATCH_MARKER, False
    )
    assert _patch_contracts.is_worker_runner_v2_patched(Runner)
    assert _patch_contracts.is_worker_runner_patched(Runner)

    graph_methods = {
        "initialize_kv_cache": v2._WORKER_V2_KV_PATCH_MARKER,
        "profile_cudagraph_memory": v2._WORKER_V2_PROFILE_PATCH_MARKER,
        "capture_model": v2._WORKER_V2_GRAPH_PATCH_MARKER,
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
    assert getattr(wrapped, _patch_contracts._WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, False)
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


def test_configure_model_runner_default_normalizes_overrides(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from vllm_gr.engine.startup import configure_model_runner_default

    monkeypatch.setattr(
        "vllm_gr.platforms.identity.current_platform_identity",
        lambda: SimpleNamespace(platform_enum="CUDA", device_type="cuda"),
    )
    for value in (None, "", "0", "false", "no", "off", "1", "true", "yes", "on", " TRUE "):
        caplog.clear()
        if value is None:
            monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
        else:
            monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)
        enabled = value in ("1", "true", "yes", "on", " TRUE ")
        with caplog.at_level(logging.WARNING):
            configure_model_runner_default()
        expected = None if value is None else str(int(enabled))
        assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == expected
        assert "experimental" not in caplog.text

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "invalid")
    configure_model_runner_default()
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "invalid"


@pytest.mark.parametrize(  # type: ignore[misc]
    "beam_enabled,failure", [(False, None), (True, None), (False, "native"), (True, "decode")]
)
def test_capture_order_and_failure_propagation(
    monkeypatch: pytest.MonkeyPatch, beam_enabled: bool, failure: str | None
) -> None:
    from vllm_gr.v1.worker.cuda.v2 import gpu_prefill_graph_runner_v2 as prefill

    worker = SimpleNamespace(
        vllm_config=worker_config(GRConfig(beam_max_width=4 if beam_enabled else None)),
        cudagraph_manager=None,
    )
    events = Mock()
    events.native.return_value = 17
    monkeypatch.setattr(prefill, "install_prefill_graph_manager_v2", events.plan)
    monkeypatch.setattr(resources, "capture_multibatch_forward", events.decode)
    monkeypatch.setattr(v2.torch.accelerator, "get_memory_info", lambda: (4096, 8192))
    monkeypatch.setattr(_gc_runtime, "freeze_gc_after_graph_capture", events.freeze)
    expected = [call.native(worker)]
    if beam_enabled:
        expected.insert(0, call.plan(worker))
        expected.append(call.decode(worker))

    def native(runner: Any) -> int:
        return cast(int, events.native(runner))

    wrapped = v2.make_patched_capture_model_v2(native)
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
        assert not get_v2_runtime(worker).capture_failed


@pytest.mark.l1  # type: ignore[misc]
def test_repeated_capture_preserves_published_decode_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.attention.backends import beam_prefix_calibrate
    from vllm_gr.v1.worker import gr_startup
    from vllm_gr.v1.worker.cuda.v2 import gpu_prefill_graph_runner_v2 as prefill
    from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2

    worker = SimpleNamespace(
        device="cuda",
        vllm_config=worker_config(
            GRConfig(beam_max_width=4, deployment=DeploymentOptions(beam_batch_sizes=(1,))),
            SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=1)),
        ),
        cudagraph_manager=object(),
    )
    reason = "Decode capture refused"

    def refuse(runner: Any, batch: int, **kwargs: Any) -> str:
        assert runner is worker
        runner._gr_gpu_stage_runner = SimpleNamespace(closed=True, hooks=CudaStageHooksV2(worker))
        return reason

    capture = Mock(side_effect=refuse)
    monkeypatch.setattr(gr_startup, "_capture_enabled", Mock(return_value=True))
    monkeypatch.setattr(gr_startup, "_capture_batch", capture)
    monkeypatch.setattr(beam_prefix_calibrate, "ensure_prefix_calibration", Mock())
    monkeypatch.setattr(prefill, "install_prefill_graph_manager_v2", Mock())
    monkeypatch.setattr(_gc_runtime, "freeze_gc_after_graph_capture", Mock())
    monkeypatch.setattr(v2.torch.accelerator, "get_memory_info", lambda: (8192, 16384))

    def native(runner: Any) -> int:
        return 17

    wrapped = v2.make_patched_capture_model_v2(native)
    for _ in range(2):
        assert wrapped(worker) == 17
        runtime = get_v2_runtime(worker)
        assert runtime.startup_failures == {1: reason}
        assert runtime.capture_failed
        assert runtime.capture_error == reason

    capture.assert_called_once_with(
        worker, 1, hooks_type=CudaStageHooksV2, allow_eager_fallback=True
    )


def test_stage_hooks_require_native_prefill_output_and_scheduler_kv() -> None:
    from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2

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

    from vllm_gr.v1.worker.cuda.v2 import gpu_prefill_graph_runner_v2 as prefill

    manager = (
        prefill.GRPrefillCudaGraphManager.__new__(prefill.GRPrefillCudaGraphManager)
        if gr_plan
        else SimpleNamespace()
    )
    manager._max_full_descs_to_capture = 2
    samples: list[int] = []
    manager._capture_mem_samples = samples
    worker = SimpleNamespace(
        vllm_config=worker_config(GRConfig(beam_max_width=128)),
        cudagraph_manager=manager,
    )
    native = Mock(return_value=4096)
    setattr(native, v2._WORKER_V2_GRAPH_PATCH_MARKER, False)
    install = Mock()
    decode = Mock()
    freeze = Mock()
    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", capturing_enabled)
    decode.side_effect = lambda worker: monitor.set_cudagraph_capturing_enabled(False)
    monkeypatch.setattr(prefill, "install_prefill_graph_manager_v2", install)
    monkeypatch.setattr(resources, "capture_multibatch_forward", decode)
    monkeypatch.setattr(_gc_runtime, "freeze_gc_after_graph_capture", freeze)
    memory = iter((8192, 7168))
    monkeypatch.setattr(v2.torch.accelerator, "get_memory_info", lambda: (next(memory), 16384))
    assert v2.make_patched_capture_model_v2(native)(worker, profile_only=profile_only) == 5120
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


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("capturing_enabled", [False, True])  # type: ignore[misc]
def test_profile_decode_failure_restores_native_capture_state(
    monkeypatch: pytest.MonkeyPatch, capturing_enabled: bool
) -> None:
    from vllm.compilation import monitor

    from vllm_gr.v1.worker.cuda.v2 import gpu_prefill_graph_runner_v2 as prefill

    worker = SimpleNamespace(
        vllm_config=worker_config(GRConfig(beam_max_width=128)),
        cudagraph_manager=object(),
    )
    error = RuntimeError("Decode capture failed")
    native = Mock(return_value=4096)
    setattr(native, v2._WORKER_V2_GRAPH_PATCH_MARKER, False)
    install, freeze = Mock(), Mock()

    def capture(runner: Any) -> bool:
        assert runner is worker
        monitor.set_cudagraph_capturing_enabled(False)
        raise error

    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", capturing_enabled)
    monkeypatch.setattr(prefill, "install_prefill_graph_manager_v2", install)
    monkeypatch.setattr(resources, "capture_multibatch_forward", capture)
    monkeypatch.setattr(v2.torch.accelerator, "get_memory_info", lambda: (8192, 16384))
    monkeypatch.setattr(_gc_runtime, "freeze_gc_after_graph_capture", freeze)

    with pytest.raises(RuntimeError) as caught:
        v2.make_patched_capture_model_v2(native)(worker, profile_only=True)
    assert caught.value is error
    assert monitor.cudagraph_capturing_enabled is capturing_enabled
    native.assert_called_once_with(worker, profile_only=True)
    install.assert_called_once_with(worker)
    freeze.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])  # type: ignore[misc]
def test_native_profiler_uses_gr_plan_and_cleans_temporary_decode(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    from contextlib import nullcontext

    from vllm.config import CUDAGraphMode as Mode
    from vllm.v1.worker.gpu import cudagraph_utils as native

    from vllm_gr.v1.worker.cuda.v2 import gpu_prefill_graph_runner_v2 as prefill

    mib = 1 << 20
    manager = SimpleNamespace(
        needs_capture=lambda: True,
        use_breakable_cg=False,
        _capture_descs={Mode.FULL: list(range(7))},
    )
    worker = SimpleNamespace(
        vllm_config=worker_config(GRConfig(beam_max_width=128)),
        compilation_config=SimpleNamespace(cudagraph_mode=Mode.FULL),
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
        from threading import Lock

        worker._gr_gpu_stage_runner = SimpleNamespace(
            closed=True,
            pending=None,
            control_busy=[],
            control_lock=Lock(),
            terminal_cleanup_pending=False,
        )
        get_v2_runtime(worker).inputs = object()
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
        lambda r: v2.make_patched_initialize_kv_cache_v2(init)(r, object(), is_profiling=True),
    )
    monkeypatch.setattr(native, "_teardown_profiling_state", teardown)
    monkeypatch.setattr(native.current_platform, "graph_pool_handle", lambda: (123, 456))
    monkeypatch.setattr(v2.torch.accelerator, "empty_cache", lambda: None)
    pool = type(native.current_platform)._global_graph_pool
    profile = v2.make_patched_profile_cudagraph_memory_v2(native.profile_cudagraph_memory)
    if failure:
        with pytest.raises(RuntimeError, match="capture failure"):
            profile(worker)
    else:
        # Extrapolation must use the GR plan's seven FULL v2.
        assert profile(worker) == (20 + 10 + 6 * 2) * mib
    assert events == ["capture", "native teardown"]
    assert type(native.current_platform)._global_graph_pool == pool
    assert worker.cudagraph_manager is None and worker._beam_decode_graph is None
    assert not any(
        hasattr(worker, n)
        for n in (
            "_gr_gpu_stage_runner",
            "_gr_v2_runtime",
            "_beam_context",
        )
    )
    graph.uninstall.assert_called_once()
    graph.clear.assert_called_once()


def test_profile_rejects_bound_runtime_and_preserves_non_beam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = SimpleNamespace(
        vllm_config=worker_config(GRConfig(beam_max_width=128)),
        _beam_decode_graph=object(),
    )
    original = Mock(return_value=42)
    setattr(original, v2._WORKER_V2_PROFILE_PATCH_MARKER, False)
    profile = v2.make_patched_profile_cudagraph_memory_v2(original)
    with pytest.raises(RuntimeError, match="unbound GR runtime"):
        profile(worker)
    original.assert_not_called()
    update_worker_config(worker, beam_max_width=None)
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
