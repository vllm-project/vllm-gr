# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for active-platform ModelRunner patch dispatch."""

from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from vllm_gr.v1.worker import gpu_model_runner_patch, model_runner_patch


def test_dispatcher_selects_one_platform_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fake vllm.platforms so the test runs without a real vLLM install.
    fake_platforms = ModuleType("vllm.platforms")
    setattr(fake_platforms, "current_platform", SimpleNamespace(device_type="npu"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)

    assert model_runner_patch._active_platform_module_name() == (
        model_runner_patch._NPU_PATCH_MODULE
    )
    getattr(fake_platforms, "current_platform").device_type = "cuda"
    assert model_runner_patch._active_platform_module_name() == (
        model_runner_patch._GPU_PATCH_MODULE
    )

    selected = object()
    imported: list[str] = []

    def import_selected(name: str) -> object:
        imported.append(name)
        return selected

    monkeypatch.setattr(
        model_runner_patch,
        "_active_platform_module_name",
        lambda: model_runner_patch._NPU_PATCH_MODULE,
    )
    monkeypatch.setattr(
        model_runner_patch,
        "import_module",
        import_selected,
    )

    assert model_runner_patch._get_active_platform_module() is selected
    assert imported == [model_runner_patch._NPU_PATCH_MODULE]


def test_missing_optional_diagnostic_installer_does_not_block_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_runner_patch,
        "_get_active_platform_module",
        lambda: SimpleNamespace(),
    )
    assert model_runner_patch.install_active_worker_diagnostic_hook(type("Worker", (), {})) == 0


@pytest.mark.parametrize("has_input_prep_hook", [False, True])
def test_dispatcher_installs_three_wrappers_once(
    monkeypatch: pytest.MonkeyPatch,
    has_input_prep_hook: bool,
) -> None:
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

        @contextmanager
        def synchronize_input_prep(self):
            yield

    platform_module = SimpleNamespace(
        discover_worker_runners=lambda: (("GPUModelRunner", Runner),),
        make_patched_prepare_inputs=gpu_model_runner_patch.make_patched_prepare_inputs,
        make_patched_bookkeeping_sync=gpu_model_runner_patch.make_patched_bookkeeping_sync,
        make_patched_sample=gpu_model_runner_patch.make_patched_sample,
    )
    original_input_prep = Runner.synchronize_input_prep
    if has_input_prep_hook:
        platform_module.make_patched_synchronize_input_prep = (
            gpu_model_runner_patch.make_patched_synchronize_input_prep
        )
    monkeypatch.setattr(
        model_runner_patch,
        "_get_active_platform_module",
        lambda: platform_module,
    )

    model_runner_patch.apply_worker_patches()
    first = (Runner.execute_model, Runner._prepare_inputs, Runner._bookkeeping_sync)
    first_input_prep = Runner.synchronize_input_prep
    assert (first_input_prep is not original_input_prep) == has_input_prep_hook
    assert model_runner_patch.is_worker_runner_patched(Runner)
    assert getattr(Runner, "_patched_for_beam_pos_ids") is True
    # No _sample on this Runner -> the Constraint Tables hook must not be added.
    from vllm_gr.v1.worker.model_runner_common import _WORKER_SAMPLE_PATCH_MARKER

    assert not hasattr(getattr(Runner, "_sample", None), _WORKER_SAMPLE_PATCH_MARKER)

    model_runner_patch.apply_worker_patches()
    assert (Runner.execute_model, Runner._prepare_inputs, Runner._bookkeeping_sync) == first
    assert Runner.synchronize_input_prep is first_input_prep


@pytest.mark.parametrize("has_copy_hook", [False, True])
@pytest.mark.parametrize("has_copy_method", [False, True])
def test_dispatcher_installs_optional_input_copy_hook_on_selected_subclass_only(
    monkeypatch: pytest.MonkeyPatch, has_copy_hook: bool, has_copy_method: bool
) -> None:
    calls = []
    result = object()

    class Base:
        pass

    def original_copy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return result

    if has_copy_method:
        Base._prepare_input_ids = original_copy

    class Runner(Base):
        def execute_model(self, scheduler_output):
            return None

        def _prepare_inputs(self, scheduler_output, counts):
            return None, None, 1

        def _bookkeeping_sync(self, scheduler_output, *args, **kwargs):
            return None

    installed = []

    def patch_copy(original):
        installed.append(original)

        def copy(self, *args, **kwargs):
            return original(self, *args, **kwargs)

        return copy

    platform = SimpleNamespace(
        discover_worker_runners=lambda: (("SelectedV1", Runner),),
        make_patched_prepare_inputs=gpu_model_runner_patch.make_patched_prepare_inputs,
        make_patched_bookkeeping_sync=gpu_model_runner_patch.make_patched_bookkeeping_sync,
    )
    if has_copy_hook:
        platform.make_patched_prepare_input_ids = patch_copy
    monkeypatch.setattr(model_runner_patch, "_get_active_platform_module", lambda: platform)
    monkeypatch.setattr(model_runner_patch, "_install_active_worker_resource_hook", lambda: None)
    model_runner_patch.apply_worker_patches()
    first = getattr(Runner, "_prepare_input_ids", None)
    model_runner_patch.apply_worker_patches()
    assert getattr(Runner, "_prepare_input_ids", None) is first
    assert installed == ([original_copy] if has_copy_hook and has_copy_method else [])
    assert ("_prepare_input_ids" in Runner.__dict__) == (has_copy_hook and has_copy_method)
    if has_copy_method:
        assert Base._prepare_input_ids is original_copy
        assert Runner()._prepare_input_ids("output", rows=8) is result
        assert calls == [(("output",), {"rows": 8})]
    else:
        assert not hasattr(Base, "_prepare_input_ids")
        assert not hasattr(Runner, "_prepare_input_ids")


def test_dispatcher_installs_sample_hook_when_runner_exposes_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        make_patched_bookkeeping_sync=gpu_model_runner_patch.make_patched_bookkeeping_sync,
        make_patched_sample=gpu_model_runner_patch.make_patched_sample,
    )
    monkeypatch.setattr(
        model_runner_patch,
        "_get_active_platform_module",
        lambda: platform_module,
    )

    reload_weights = Runner.reload_weights
    model_runner_patch.apply_worker_patches()

    assert getattr(Runner._sample, _WORKER_SAMPLE_PATCH_MARKER, False)
    assert getattr(Runner.load_model, _CONSTRAINT_LOAD_PATCH_MARKER, False)
    assert getattr(Runner.shutdown, _CONSTRAINT_SHUTDOWN_PATCH_MARKER, False)
    # In-place weight reloads do not rebuild the tokenizer/vocab-bound Backend.
    assert Runner.reload_weights is reload_weights
    assert model_runner_patch.is_worker_runner_patched(Runner)


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
        make_patched_bookkeeping_sync=gpu_model_runner_patch.make_patched_bookkeeping_sync,
        make_patched_sample=gpu_model_runner_patch.make_patched_sample,
        make_patched_sample_tokens=fake_sample_tokens_patch,
    )
    monkeypatch.setattr(
        model_runner_patch,
        "_get_active_platform_module",
        lambda: platform_module,
    )

    model_runner_patch.apply_worker_patches()

    assert getattr(Runner.sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, False)


def test_runner_init_selects_gr_backend_once_and_not_after_failed_init() -> None:
    calls: list[str] = []

    def original(self: Any, fail: bool = False) -> None:
        calls.append("init")
        if fail:
            raise RuntimeError("init failed")

    backend = object()

    def factory(runner: Any) -> object:
        calls.append("factory")
        return backend

    wrapped = model_runner_patch.make_init_with_gr_device_backend(original, factory)
    runner = SimpleNamespace()
    wrapped(runner)
    wrapped(runner)
    assert runner._gr_device_backend is backend
    assert calls == ["init", "factory", "init"]

    failed = SimpleNamespace()
    with pytest.raises(RuntimeError, match="init failed"):
        wrapped(failed, True)
    assert not hasattr(failed, "_gr_device_backend_initialized")


def test_gr_state_named_worker_hooks_install_on_any_backend() -> None:
    class Worker:
        pass

    assert model_runner_patch.install_gr_state_worker_hooks(Worker) == 1
    assert model_runner_patch.install_gr_state_worker_hooks(Worker) == 0
    worker = Worker()
    worker.model_runner = SimpleNamespace(_gr_model_runner_states={})
    assert worker.get_gr_model_runner_state_stats() == {
        "created": 0,
        "freed": 0,
        "active": 0,
    }


def test_shutdown_runs_native_even_when_gr_release_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker import gr_model_runner_state

    events: list[str] = []

    def fail_release(runner: Any) -> None:
        events.append("release")
        raise RuntimeError("release failed")

    monkeypatch.setattr(
        gr_model_runner_state,
        "release_gr_model_runner_states",
        fail_release,
    )

    def native(runner: Any) -> None:
        events.append("native")

    runner = SimpleNamespace(constraint_backend=None)
    wrapped = model_runner_patch.make_shutdown_with_constraint_backend(native)
    with pytest.raises(RuntimeError, match="release failed"):
        wrapped(runner)
    assert events == ["release", "native"]
