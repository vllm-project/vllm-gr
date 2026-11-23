# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Core contracts for active-platform ModelRunner patch dispatch."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tests.helpers.gr_config import worker_config
from vllm_gr.config import SUPPORTED_BEAM_BATCH_SIZES, GRConfig
from vllm_gr.platforms.ascend import provider as npu
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.v1.worker import model_runner_patch
from vllm_gr.v1.worker.cuda import runner_installer as cuda
from vllm_gr.v1.worker.cuda.v1 import gpu_model_runner_patch


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
    native_reload = Worker.reload_weights
    cuda.install_worker_utilities()
    reload_trie = getattr(Worker, "reload_trie")
    cuda.install_worker_utilities()
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
    target = (("ExternalRunner", object),)
    hooks = SimpleNamespace(discover_worker_runners=lambda: target)
    monkeypatch.setattr(model_runner_patch, "get_runner_hooks", lambda: hooks)
    assert model_runner_patch.discover_active_worker_runners() == target


def test_dispatcher_installs_required_and_optional_hooks_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        model_runner_patch,
        "get_runner_hooks",
        lambda: platform_module,
    )

    bookkeeping = Runner._bookkeeping_sync
    model_runner_patch.apply_worker_patches()
    first = (Runner.execute_model, Runner._prepare_inputs, Runner._bookkeeping_sync)
    assert Runner._bookkeeping_sync is bookkeeping
    # Ordinary inference still delegates to the native method on either path.
    assert Runner()._bookkeeping_sync(SimpleNamespace()) is native_result
    assert model_runner_patch.is_worker_runner_patched(Runner)
    assert getattr(Runner, "_patched_for_beam_pos_ids") is True
    # Beam stages do not install a native sampler override.
    assert not hasattr(getattr(Runner, "_sample", None), "_vllm_gr_beam_sample_patch")

    model_runner_patch.apply_worker_patches()
    assert (Runner.execute_model, Runner._prepare_inputs, Runner._bookkeeping_sync) == first


@pytest.mark.parametrize(  # type: ignore[misc]
    ("device_type", "expected_max_batch"),
    [
        ("cuda", 1),
        ("npu", max(SUPPORTED_BEAM_BATCH_SIZES)),
    ],
)
def test_dispatcher_preserves_native_hooks_and_platform_constraint_capacity(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    expected_max_batch: int,
) -> None:
    from vllm_gr.v1.worker.model_runner_patch import (
        _CONSTRAINT_LOAD_PATCH_MARKER,
        _CONSTRAINT_SHUTDOWN_PATCH_MARKER,
    )
    from vllm_gr.v1.worker.npu import npu_model_runner_patch

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

    # Exercise the platform installer's real capacity setting. The reduced V1
    # hook surface keeps the test focused on native fallback and load ownership.
    runner_hooks = gpu_model_runner_patch if device_type == "cuda" else npu_model_runner_patch
    platform_module = SimpleNamespace(
        discover_worker_runners=lambda: (("TestRunner", Runner),),
        make_patched_prepare_inputs=gpu_model_runner_patch.make_patched_prepare_inputs,
    )
    platform_hooks: list[str] = []
    if device_type == "npu":

        def install_platform_load(original: Any) -> Any:
            platform_hooks.append("load")
            return original

        def install_platform_shutdown(original: Any) -> Any:
            platform_hooks.append("shutdown")
            return original

        platform_module.make_patched_load_model = install_platform_load
        platform_module.make_patched_shutdown = install_platform_shutdown

    install_v1 = model_runner_patch.install_v1_worker_hooks

    def install_test_runner(hooks: Any, **kwargs: Any) -> None:
        assert hooks is runner_hooks
        install_v1(platform_module, **kwargs)

    monkeypatch.setattr(model_runner_patch, "install_v1_worker_hooks", install_test_runner)
    monkeypatch.setattr(model_runner_patch, "get_runner_hooks", lambda: runner_hooks)
    monkeypatch.setattr(cuda, "install_worker_hooks", lambda: None)

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
    assert not getattr(Runner._sample, "_vllm_gr_beam_sample_patch", False)
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
    from vllm_gr.v1.worker.npu import npu_model_runner_patch

    assert npu_model_runner_patch.make_patched_sample_tokens is (
        npu_model_runner_patch.make_patched_sample_tokens_npu
    )


def test_dispatcher_installs_sample_tokens_hook_when_exposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker.model_runner_contracts import (
        _WORKER_SAMPLE_TOKENS_PATCH_MARKER,
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
        "get_runner_hooks",
        lambda: platform_module,
    )

    model_runner_patch.apply_worker_patches()

    assert getattr(Runner.sample_tokens, _WORKER_SAMPLE_TOKENS_PATCH_MARKER, False)


@pytest.mark.parametrize(  # type: ignore[misc]
    "device_type,use_v2",
    [("cuda", False), ("cuda", True), ("npu", True)],
)
def test_dispatcher_routes_platform_before_v2_flag(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    use_v2: bool,
) -> None:
    """CUDA defers runner discovery until the Worker knows its configuration."""
    envs = ModuleType("vllm.envs")
    if device_type == "cuda":
        setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", use_v2)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    provider = CUDA_BACKEND if device_type == "cuda" else npu.NPU_BACKEND
    from vllm_gr.v1.worker.cuda import runner_installer as runner

    target = (("selected", object),)
    monkeypatch.setattr(runner, "discover_worker_runners", lambda: target)
    imported: list[str] = []

    def import_selected(name: str) -> Any:
        imported.append(name)
        return SimpleNamespace(discover_worker_runners=lambda: target)

    monkeypatch.setattr(cuda if device_type == "cuda" else npu, "import_module", import_selected)
    monkeypatch.setattr(model_runner_patch, "get_runner_hooks", provider.get_model_runner_hooks)
    assert model_runner_patch.discover_active_worker_runners() == target
    assert imported == (
        [] if device_type == "cuda" else ["vllm_gr.v1.worker.npu.npu_model_runner_patch"]
    )


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
    from vllm_gr.v1.worker.cuda import numpy_staging
    from vllm_gr.v1.worker.npu import npu_model_runner_patch

    runner_hooks = gpu_model_runner_patch if device_type == "cuda" else npu_model_runner_patch
    calls: list[str] = []
    monkeypatch.setattr(model_runner_patch, "get_runner_hooks", lambda: runner_hooks)
    monkeypatch.setattr(runner_hooks, "discover_worker_runners", lambda: ())
    monkeypatch.setattr(cuda, "install_worker_utilities", lambda: None)
    monkeypatch.setattr(numpy_staging, "install_numpy_staging", lambda: calls.append("staging"))
    model_runner_patch.apply_worker_patches()
    assert calls == expected


def test_apply_worker_patches_dispatches_to_v2_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the V2 flag the V1 body must not run: the V1 installer does
    unguarded _prepare_inputs access the V2 runner does not expose."""
    called: list[str] = []

    import vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_patch as v2_patch_module

    monkeypatch.setattr(model_runner_patch, "get_runner_hooks", lambda: v2_patch_module)
    monkeypatch.setattr(cuda, "install_worker_hooks", lambda: called.append("staging"))
    monkeypatch.setattr(v2_patch_module, "_apply_v2_runner_patches", lambda: called.append("v2"))

    model_runner_patch.apply_worker_patches()
    assert called == ["staging", "v2"]


@pytest.mark.parametrize("mode", [None, 4])  # type: ignore[misc]
def test_npu_fallback_installs_v1_and_keeps_capture_protocol_selection(
    monkeypatch: pytest.MonkeyPatch, mode: int | None
) -> None:
    import builtins

    from vllm_gr.v1.worker import gr_startup

    events: list[str] = []

    class Runner:
        vllm_config = worker_config(GRConfig(beam_max_width=mode))

        def execute_model(self, *args: Any) -> None:
            pass

        def _prepare_inputs(self, *args: Any) -> None:
            pass

        def _update_states(self, *args: Any) -> None:
            pass

        def _bookkeeping_sync(self, *args: Any) -> None:
            pass

        def capture_model(self) -> int:
            events.append("native")
            return 17

    from vllm_gr.v1.worker.npu import npu_model_runner_patch

    monkeypatch.setattr(model_runner_patch, "get_runner_hooks", lambda: npu_model_runner_patch)
    monkeypatch.setattr(npu_model_runner_patch, "capture_prefill_graphs", lambda _: None)
    monkeypatch.setattr(
        npu_model_runner_patch, "discover_worker_runners", lambda: (("NPU", Runner),)
    )
    monkeypatch.setattr(
        npu_model_runner_patch, "capture_beam_decode_graph", lambda _: events.append("legacy")
    )
    monkeypatch.setattr(
        gr_startup,
        "capture_async_forward",
        lambda _, *, hooks_type: (
            events.append("stages")
            if hooks_type is None
            else pytest.fail("NPU received CUDA hooks")
        ),
    )
    monkeypatch.setattr(model_runner_patch, "freeze_gc_after_graph_capture", lambda _: None)

    original_import = builtins.__import__

    def without_cuda_hooks(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name != "vllm_gr.v1.worker.cuda" and not name.startswith("vllm_gr.v1.worker.cuda.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cuda_hooks)
    model_runner_patch.apply_worker_patches()
    assert model_runner_patch.is_worker_runner_patched(Runner)
    assert Runner().capture_model() == 17
    assert events == ["native", "stages" if mode is not None else "legacy"]


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("use_v2", [False, True])  # type: ignore[misc]
def test_public_installer_owns_numpy_staging_once(
    monkeypatch: pytest.MonkeyPatch, use_v2: bool
) -> None:
    from unittest.mock import Mock

    from vllm_gr.v1.worker import gr_async_output
    from vllm_gr.v1.worker import model_runner_patch as installer
    from vllm_gr.v1.worker.cuda import numpy_staging
    from vllm_gr.v1.worker.cuda.v2 import gpu_model_runner_v2_patch as v2

    install = Mock()
    discover = Mock(return_value=())
    monkeypatch.setattr(numpy_staging, "install_numpy_staging", install)
    monkeypatch.setattr(v2, "discover_worker_runners", discover)
    monkeypatch.setattr(gr_async_output, "install_async_output_guard", lambda: False)
    monkeypatch.setattr(cuda, "install_worker_utilities", lambda: None)
    monkeypatch.setattr(gpu_model_runner_patch, "discover_worker_runners", lambda: ())
    monkeypatch.setattr(
        installer, "get_runner_hooks", lambda: v2 if use_v2 else gpu_model_runner_patch
    )
    installer.apply_worker_patches()
    install.assert_called_once_with()
    assert discover.call_count == int(use_v2)
