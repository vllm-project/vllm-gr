# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Core contracts for active-platform ModelRunner patch dispatch."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from vllm_gr.config import SUPPORTED_BEAM_BATCH_SIZES, BeamExecutionMode
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.npu import NPU_BACKEND
from vllm_gr.v1.worker import gpu_model_runner_patch, model_runner_patch


def _use_backend(monkeypatch: pytest.MonkeyPatch, provider: Any) -> Any:
    """Pin the activated backend so patch dispatch has a provider to consult.

    Module selection reads the activated provider rather than the native
    platform, so a test supplies that decision directly instead of faking
    ``vllm.platforms.current_platform``.
    """
    monkeypatch.setattr(model_runner_patch, "_active_provider", lambda: provider)
    return provider


@pytest.mark.l1  # type: ignore[misc]
def test_active_cuda_worker_hot_update_utility_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    class Worker:
        model_runner: Any

        def reload_weights(self, weights_path: str) -> str:
            return weights_path

    module = ModuleType("vllm.v1.worker.gpu_worker")
    setattr(module, "Worker", Worker)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        model_runner_patch,
        "_active_platform_module_name",
        lambda: model_runner_patch._GPU_PATCH_MODULE,
    )
    native_reload = Worker.reload_weights
    model_runner_patch._install_active_worker_resource_hook()
    reload_trie = getattr(Worker, "reload_trie")
    model_runner_patch._install_active_worker_resource_hook()
    assert getattr(Worker, "reload_trie") is reload_trie
    assert Worker.reload_weights is native_reload

    worker = Worker()
    backend = Mock()
    worker.model_runner = SimpleNamespace(constraint_backend=backend)
    assert getattr(worker, "reload_trie")("/tries/v2", "v2") == {"version": "v2", "ready": True}
    backend.reload.assert_called_once_with("/tries/v2", "v2")
    assert worker.reload_weights("/weights/v2") == "/weights/v2"


def test_dispatcher_selects_one_platform_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test exercises the V1 platform routing; pin the V2 flag off so an
    # ambient VLLM_USE_V2_MODEL_RUNNER=1 (e.g. a V2 CI shard) cannot reroute
    # _active_runner_module_name to the V2 module.
    monkeypatch.setattr(model_runner_patch, "_use_v2_model_runner", lambda: False)

    _use_backend(monkeypatch, NPU_BACKEND)
    assert model_runner_patch._active_platform_module_name() == (
        model_runner_patch._NPU_PATCH_MODULE
    )
    _use_backend(monkeypatch, CUDA_BACKEND)
    assert model_runner_patch._active_platform_module_name() == (
        model_runner_patch._GPU_PATCH_MODULE
    )

    selected = object()
    imported: list[str] = []

    def import_selected(name: str) -> object:
        imported.append(name)
        return selected

    _use_backend(monkeypatch, NPU_BACKEND)
    monkeypatch.setattr(
        model_runner_patch,
        "import_module",
        import_selected,
    )

    assert model_runner_patch._get_active_platform_module() is selected
    assert imported == [model_runner_patch._NPU_PATCH_MODULE]


@pytest.mark.parametrize("with_bookkeeping", [False, True])  # type: ignore[misc]
def test_dispatcher_installs_required_and_optional_hooks_once(
    monkeypatch: pytest.MonkeyPatch,
    with_bookkeeping: bool,
) -> None:
    # V1-platform install test: pin the V2 flag off (see
    # test_dispatcher_selects_one_platform_module).
    monkeypatch.setattr(model_runner_patch, "_use_v2_model_runner", lambda: False)
    native_result = object()

    class Runner:
        def execute_model(self, scheduler_output: Any) -> None:
            return None

        def _prepare_inputs(
            self,
            scheduler_output: Any,
            num_scheduled_tokens: Any,
        ) -> tuple[object, object]:
            return object(), object()

        def _bookkeeping_sync(
            self,
            scheduler_output: Any,
            *args: Any,
            **kwargs: Any,
        ) -> object:
            return native_result

    platform_module = SimpleNamespace(
        discover_worker_runners=lambda: (("GPUModelRunner", Runner),),
        make_patched_prepare_inputs=gpu_model_runner_patch.make_patched_prepare_inputs,
    )
    if with_bookkeeping:
        from vllm_gr.v1.worker import npu_model_runner_patch

        platform_module.make_patched_bookkeeping_sync = (
            npu_model_runner_patch.make_patched_bookkeeping_sync
        )
    monkeypatch.setattr(
        model_runner_patch,
        "_get_active_platform_module",
        lambda: platform_module,
    )
    _use_backend(monkeypatch, CUDA_BACKEND)

    bookkeeping = Runner._bookkeeping_sync
    model_runner_patch.apply_worker_patches()
    first = (Runner.execute_model, Runner._prepare_inputs, Runner._bookkeeping_sync)
    if with_bookkeeping:
        assert Runner._bookkeeping_sync is not bookkeeping
        assert getattr(Runner._bookkeeping_sync, "__wrapped__") is bookkeeping
    else:
        assert Runner._bookkeeping_sync is bookkeeping
    # Ordinary inference still delegates to the native method on either path.
    assert Runner()._bookkeeping_sync(SimpleNamespace()) is native_result
    assert model_runner_patch.is_worker_runner_patched(Runner)
    assert getattr(Runner, "_patched_for_beam_pos_ids") is True
    # No _sample on this Runner -> the Constraint Tables hook must not be added.
    from vllm_gr.v1.worker.model_runner_common import _WORKER_SAMPLE_PATCH_MARKER

    assert not hasattr(getattr(Runner, "_sample", None), _WORKER_SAMPLE_PATCH_MARKER)

    model_runner_patch.apply_worker_patches()
    assert (Runner.execute_model, Runner._prepare_inputs, Runner._bookkeeping_sync) == first


@pytest.mark.parametrize(  # type: ignore[misc]
    ("platform_module_name", "expected_max_batch"),
    [
        (model_runner_patch._GPU_PATCH_MODULE, 1),
        (model_runner_patch._NPU_PATCH_MODULE, max(SUPPORTED_BEAM_BATCH_SIZES)),
    ],
)
def test_dispatcher_preserves_native_hooks_and_platform_constraint_capacity(
    monkeypatch: pytest.MonkeyPatch,
    platform_module_name: str,
    expected_max_batch: int,
) -> None:
    # V1-platform install test: pin the V2 flag off (see
    # test_dispatcher_selects_one_platform_module).
    monkeypatch.setattr(model_runner_patch, "_use_v2_model_runner", lambda: False)
    _use_backend(
        monkeypatch,
        NPU_BACKEND
        if platform_module_name == model_runner_patch._NPU_PATCH_MODULE
        else CUDA_BACKEND,
    )
    from vllm_gr.v1.worker.model_runner_common import _WORKER_SAMPLE_PATCH_MARKER
    from vllm_gr.v1.worker.model_runner_patch import (
        _CONSTRAINT_LOAD_PATCH_MARKER,
        _CONSTRAINT_SHUTDOWN_PATCH_MARKER,
    )

    class Runner:
        def execute_model(self, scheduler_output: Any) -> None:
            return None

        def _prepare_inputs(
            self,
            scheduler_output: Any,
            num_scheduled_tokens: Any,
        ) -> tuple[object, object]:
            return object(), object()

        def _bookkeeping_sync(
            self,
            scheduler_output: Any,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            return None

        def _sample(self, logits: Any, spec_decode_metadata: Any = None) -> Any:
            return object()

        def load_model(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

        def reload_weights(self) -> None:
            return None

    platform_module = SimpleNamespace(
        discover_worker_runners=lambda: (("GPUModelRunner", Runner),),
        make_patched_prepare_inputs=gpu_model_runner_patch.make_patched_prepare_inputs,
    )
    platform_hooks: list[str] = []
    if platform_module_name == model_runner_patch._NPU_PATCH_MODULE:

        def install_platform_load(original: Any) -> Any:
            platform_hooks.append("load")
            return original

        def install_platform_shutdown(original: Any) -> Any:
            platform_hooks.append("shutdown")
            return original

        platform_module.make_patched_load_model = install_platform_load
        platform_module.make_patched_shutdown = install_platform_shutdown
    monkeypatch.setattr(
        model_runner_patch,
        "_get_active_platform_module",
        lambda: platform_module,
    )

    load_wrapper = model_runner_patch.make_load_model_with_constraint_backend
    max_batches: list[int] = []

    def install_load(original: Any, *, max_batch: int = 1) -> Any:
        max_batches.append(max_batch)
        return load_wrapper(original, max_batch=max_batch)

    monkeypatch.setattr(model_runner_patch, "make_load_model_with_constraint_backend", install_load)

    reload_weights = Runner.reload_weights
    sample = Runner._sample
    bookkeeping = Runner._bookkeeping_sync
    model_runner_patch.apply_worker_patches()

    assert Runner._sample is sample
    assert Runner._bookkeeping_sync is bookkeeping
    assert not getattr(Runner._sample, _WORKER_SAMPLE_PATCH_MARKER, False)
    assert getattr(Runner.load_model, _CONSTRAINT_LOAD_PATCH_MARKER, False)
    assert max_batches == [expected_max_batch]
    assert getattr(Runner.shutdown, _CONSTRAINT_SHUTDOWN_PATCH_MARKER, False)
    # In-place weight reloads do not rebuild the tokenizer/vocab-bound Backend.
    assert Runner.reload_weights is reload_weights
    assert model_runner_patch.is_worker_runner_patched(Runner)
    expected_hooks = ["load", "shutdown"] if expected_max_batch > 1 else []
    assert platform_hooks == expected_hooks
    model_runner_patch.apply_worker_patches()
    assert platform_hooks == expected_hooks


def test_npu_module_exports_sample_tokens_alias() -> None:
    """The dispatcher looks up ``make_patched_sample_tokens`` by exact name.

    The NPU module defines the implementation with an ``_npu`` suffix; the
    module-level alias must exist or the hook is silently skipped and the
    worker beam decision never reaches the ModelRunnerOutput under async
    scheduling (regression: bw128_1k_t10 trace had 0 patched_sample_tokens
    frames while the frontend kept its per-step RPC loop).
    """
    from vllm_gr.v1.worker import npu_model_runner_patch

    assert npu_model_runner_patch.make_patched_sample_tokens is (
        npu_model_runner_patch.make_patched_sample_tokens_npu
    )


def test_dispatcher_installs_sample_tokens_hook_when_exposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # V1-platform install test: pin the V2 flag off (see
    # test_dispatcher_selects_one_platform_module).
    monkeypatch.setattr(model_runner_patch, "_use_v2_model_runner", lambda: False)
    from vllm_gr.v1.worker.model_runner_common import _WORKER_SAMPLE_TOKENS_PATCH_MARKER

    class Runner:
        def execute_model(self, scheduler_output: Any) -> None:
            return None

        def _prepare_inputs(
            self,
            scheduler_output: Any,
            num_scheduled_tokens: Any,
        ) -> tuple[object, object]:
            return object(), object()

        def _bookkeeping_sync(
            self,
            scheduler_output: Any,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            return None

        def sample_tokens(self, grammar_output: Any = None) -> Any:
            return object()

    def fake_sample_tokens_patch(orig: Any) -> Any:
        def patched(self: Any, grammar_output: Any = None) -> Any:
            return orig(self, grammar_output)

        setattr(patched, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, True)
        return patched

    platform_module = SimpleNamespace(
        discover_worker_runners=lambda: (("NPUModelRunner", Runner),),
        make_patched_prepare_inputs=gpu_model_runner_patch.make_patched_prepare_inputs,
        make_patched_sample_tokens=fake_sample_tokens_patch,
    )
    monkeypatch.setattr(
        model_runner_patch,
        "_get_active_platform_module",
        lambda: platform_module,
    )
    _use_backend(monkeypatch, NPU_BACKEND)

    model_runner_patch.apply_worker_patches()

    assert getattr(Runner.sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, False)


# ---------------------------------------------------------------------------
# V2 dispatch: _active_runner_module_name + apply_worker_patches branching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    "device_type,use_v2",
    [("cuda", False), ("cuda", True), ("npu", True), ("cpu", True), ("hip", True)],
)
def test_dispatcher_routes_platform_before_v2_flag(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    use_v2: bool,
) -> None:
    """Only a backend with a V2 runner consults the V2 flag.

    The flag is never read for a backend that has no V2 patch module, so an
    ambient VLLM_USE_V2_MODEL_RUNNER cannot reroute an unsupported backend to
    another backend's module.
    """
    provider = CUDA_BACKEND if device_type == "cuda" else NPU_BACKEND

    def v2_flag() -> bool:
        assert provider.supports_v2_model_runner(), (
            "a backend without a V2 runner must not consult the V2 flag"
        )
        return use_v2

    monkeypatch.setattr(model_runner_patch, "_use_v2_model_runner", v2_flag)
    _use_backend(monkeypatch, provider)
    expected = model_runner_patch._GPU_PATCH_MODULE
    if device_type != "cuda":
        expected = model_runner_patch._NPU_PATCH_MODULE
    elif use_v2:
        expected = model_runner_patch._V2_GPU_PATCH_MODULE
    assert model_runner_patch._active_runner_module_name() == expected

    target = (("selected", object),)
    imported: list[str] = []

    def import_selected(name: str) -> Any:
        imported.append(name)
        return SimpleNamespace(discover_worker_runners=lambda: target)

    monkeypatch.setattr(model_runner_patch, "import_module", import_selected)
    assert model_runner_patch.discover_active_worker_runners() == target
    assert imported == [expected]


@pytest.mark.parametrize(  # type: ignore[misc]
    "device_type,expected",
    [("cuda", ["staging"]), ("npu", [])],
)
def test_numpy_token_staging_installs_for_cuda_runners_only(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    expected: list[str],
) -> None:
    """V1 and V2 share StagedWriteTensor, so staging is runner-agnostic."""
    from vllm_gr.v1.worker import numpy_staging

    calls: list[str] = []
    monkeypatch.setattr(numpy_staging, "install_numpy_staging", lambda: calls.append("staging"))
    monkeypatch.setattr(
        model_runner_patch,
        "_active_platform_module_name",
        lambda: (
            model_runner_patch._GPU_PATCH_MODULE
            if device_type == "cuda"
            else model_runner_patch._NPU_PATCH_MODULE
        ),
    )
    model_runner_patch._install_numpy_token_staging()
    assert calls == expected


def test_apply_worker_patches_dispatches_to_v2_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the V2 flag the V1 body must not run: the V1 installer does
    unguarded _prepare_inputs access the V2 runner does not expose."""
    called: list[str] = []

    import vllm_gr.v1.worker.gpu_model_runner_v2_patch as v2_patch_module

    _use_backend(monkeypatch, CUDA_BACKEND)
    monkeypatch.setattr(model_runner_patch, "_use_v2_model_runner", lambda: True)
    monkeypatch.setattr(model_runner_patch, "_install_active_worker_resource_hook", lambda: None)
    monkeypatch.setattr(
        model_runner_patch, "_install_numpy_token_staging", lambda: called.append("staging")
    )
    # The platform path must not be touched under V2.
    monkeypatch.setattr(
        model_runner_patch,
        "_get_active_platform_module",
        lambda: (_ for _ in ()).throw(AssertionError("V1 installer must not run")),
    )
    monkeypatch.setattr(v2_patch_module, "_apply_v2_runner_patches", lambda: called.append("v2"))

    model_runner_patch.apply_worker_patches()
    assert called == ["staging", "v2"]


@pytest.mark.parametrize("mode", [None, BeamExecutionMode.V1])  # type: ignore[misc]
def test_npu_fallback_installs_v1_and_keeps_capture_protocol_selection(
    monkeypatch: pytest.MonkeyPatch, mode: BeamExecutionMode | None
) -> None:
    import builtins

    from vllm_gr.v1.worker import gr_startup

    _use_backend(monkeypatch, NPU_BACKEND)
    monkeypatch.setattr(model_runner_patch, "_use_v2_model_runner", lambda: True)
    monkeypatch.setattr(model_runner_patch, "_install_active_worker_resource_hook", lambda: None)
    events: list[str] = []

    class Runner:
        _vllm_gr_config = SimpleNamespace(beam_execution_mode=mode)

        def execute_model(self, *args: Any) -> None:
            pass

        def _prepare_inputs(self, *args: Any) -> None:
            pass

        def _bookkeeping_sync(self, *args: Any) -> None:
            pass

        def capture_model(self) -> int:
            events.append("native")
            return 17

    def import_npu(name: str) -> Any:
        assert name == model_runner_patch._NPU_PATCH_MODULE
        return SimpleNamespace(
            discover_worker_runners=lambda: (("NPUModelRunner", Runner),),
            make_patched_prepare_inputs=gpu_model_runner_patch.make_patched_prepare_inputs,
            capture_beam_decode_graph=lambda _: events.append("legacy"),
        )

    monkeypatch.setattr(model_runner_patch, "import_module", import_npu)
    monkeypatch.setattr(
        gr_startup,
        "capture_async_forward",
        lambda _, *, hooks_type: events.append("stages")
        if hooks_type is None
        else pytest.fail("NPU received CUDA hooks"),
    )
    monkeypatch.setattr(model_runner_patch, "freeze_gc_after_graph_capture", lambda _: None)

    original_import = builtins.__import__

    def without_cuda_hooks(name: str, *args: Any, **kwargs: Any) -> Any:
        assert not name.startswith("vllm_gr.v1.worker.gpu_beam_stage_runner")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cuda_hooks)
    model_runner_patch.apply_worker_patches()
    assert model_runner_patch.is_worker_runner_patched(Runner)
    assert Runner().capture_model() == 17
    assert events == ["native", "stages" if mode is BeamExecutionMode.V1 else "legacy"]


pytestmark = pytest.mark.cpu_test
