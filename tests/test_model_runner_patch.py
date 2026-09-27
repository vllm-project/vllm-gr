# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for active-platform ModelRunner patch dispatch."""

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


def test_dispatcher_installs_three_wrappers_once(
    monkeypatch: pytest.MonkeyPatch,
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

    model_runner_patch.apply_worker_patches()
    first = (Runner.execute_model, Runner._prepare_inputs, Runner._bookkeeping_sync)
    assert model_runner_patch.is_worker_runner_patched(Runner)
    assert getattr(Runner, "_patched_for_beam_pos_ids") is True
    # No _sample on this Runner -> the Constraint Tables hook must not be added.
    from vllm_gr.v1.worker.model_runner_common import _WORKER_SAMPLE_PATCH_MARKER

    assert not hasattr(getattr(Runner, "_sample", None), _WORKER_SAMPLE_PATCH_MARKER)

    model_runner_patch.apply_worker_patches()
    assert (Runner.execute_model, Runner._prepare_inputs, Runner._bookkeeping_sync) == first


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


def test_load_model_promotes_server_wide_catalog_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy CLI (constraint_table + triples catalog) publishes a backend at load."""
    from types import SimpleNamespace

    import torch

    from vllm_gr.config import ConstraintBackend, ConstraintTableBackendEnum

    fake_gr_config = SimpleNamespace(
        constraint_backend=ConstraintBackend.CONSTRAINT_TABLE,
        catalog_path="triples.json",
        beam_max_width=128,
        constraint_table=None,
    )
    monkeypatch.setattr(
        "vllm_gr.config.GRConfig.from_vllm_config",
        lambda vllm_config: fake_gr_config,
    )
    monkeypatch.setattr(
        "vllm.tokenizers.get_tokenizer",
        lambda name, trust_remote_code=False: object(),
    )
    fake_resource = SimpleNamespace(digest="promoted-digest")
    monkeypatch.setattr(
        "vllm_gr.constraints.startup.compile_constraint_table_resource",
        lambda catalog_path, tokenizer, vocab_size: fake_resource,
    )
    from vllm_gr.constraints.resource_manager import ConstraintResourceManager

    fake_manager = ConstraintResourceManager()
    monkeypatch.setattr(
        "vllm_gr.constraints.worker_inject.get_worker_manager",
        lambda: fake_manager,
    )
    monkeypatch.setattr(
        "vllm_gr.constraints.artifact_compiler.constraint_table_from_resource",
        lambda resource: object(),
    )
    fake_backend = SimpleNamespace(
        info=SimpleNamespace(
            backend=ConstraintTableBackendEnum.CUDA,
            device_memory_bytes=4096,
        ),
        vocab_slice=(151669, 24576),
    )
    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_cuda_backend_from_table",
        lambda table, *, device, dtype, max_top_k, max_rows: fake_backend,
    )

    class FakeRunner:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(tensor_parallel_size=1)
            )
            self.device = torch.device("cuda")
            self.model_config = SimpleNamespace(
                tokenizer="test/model",
                trust_remote_code=False,
                dtype=torch.bfloat16,
                get_vocab_size=lambda: 176256,
            )
            self.model = SimpleNamespace(modules=lambda: [])
            self.model_memory_usage = 100
            self.constraint_backend = None
            self._gr_vocab_slice_processors = ()

        def load_model(self, *args: Any, **kwargs: Any) -> str:
            return "loaded"

    original = FakeRunner.load_model
    wrapped = model_runner_patch.make_load_model_with_constraint_backend(original)
    runner = FakeRunner()
    result = wrapped(runner)

    assert result == "loaded"
    assert runner.constraint_backend is fake_backend
    assert runner.model_memory_usage == 100 + 4096
    # The compiled resource is registered in the worker digest cache so the
    # frontend's first-request broadcast reuses the same digest (no-op).
    assert fake_manager.current_digest == "promoted-digest"


def test_load_model_promotion_skipped_for_trie_and_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trie backend / no catalog / non-CUDA device keep the legacy path."""
    from types import SimpleNamespace

    import torch

    from vllm_gr.config import ConstraintBackend

    def make_config(backend: Any, catalog_path: Any) -> SimpleNamespace:
        return SimpleNamespace(
            constraint_backend=backend,
            catalog_path=catalog_path,
            beam_max_width=128,
            constraint_table=None,
        )

    monkeypatch.setattr(
        "vllm_gr.constraints.startup.compile_constraint_table_resource",
        lambda *a, **k: pytest.fail("promotion must not compile"),
    )

    class FakeRunner:
        def __init__(self, device: torch.device) -> None:
            self.vllm_config = SimpleNamespace()
            self.device = device
            self.model_config = SimpleNamespace(
                tokenizer="test/model",
                trust_remote_code=False,
                dtype=torch.bfloat16,
                get_vocab_size=lambda: 176256,
            )
            self.model = SimpleNamespace(modules=lambda: [])
            self.constraint_backend = None

        def load_model(self, *args: Any, **kwargs: Any) -> str:
            return "loaded"

    cases = [
        (ConstraintBackend.TRIE, "triples.json", torch.device("cuda")),
        (ConstraintBackend.CONSTRAINT_TABLE, None, torch.device("cuda")),
        (ConstraintBackend.CONSTRAINT_TABLE, "triples.json", torch.device("cpu")),
    ]
    for backend, catalog_path, device in cases:
        monkeypatch.setattr(
            "vllm_gr.config.GRConfig.from_vllm_config",
            lambda vllm_config, cfg=make_config(backend, catalog_path): cfg,
        )
        wrapped = model_runner_patch.make_load_model_with_constraint_backend(FakeRunner.load_model)
        runner = FakeRunner(device)
        assert wrapped(runner) == "loaded"
        assert runner.constraint_backend is None


def test_load_model_promotion_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A catalog compile failure fails startup loudly, never degrades."""
    from types import SimpleNamespace

    import torch

    from vllm_gr.config import ConstraintBackend

    fake_gr_config = SimpleNamespace(
        constraint_backend=ConstraintBackend.CONSTRAINT_TABLE,
        catalog_path="triples.json",
        beam_max_width=128,
        constraint_table=None,
    )
    monkeypatch.setattr(
        "vllm_gr.config.GRConfig.from_vllm_config",
        lambda vllm_config: fake_gr_config,
    )
    monkeypatch.setattr(
        "vllm.tokenizers.get_tokenizer",
        lambda name, trust_remote_code=False: object(),
    )

    def failing_compile(*args: Any, **kwargs: Any) -> None:
        raise ValueError("bad triples")

    monkeypatch.setattr(
        "vllm_gr.constraints.startup.compile_constraint_table_resource",
        failing_compile,
    )

    class FakeRunner:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace()
            self.device = torch.device("cuda")
            self.model_config = SimpleNamespace(
                tokenizer="test/model",
                trust_remote_code=False,
                dtype=torch.bfloat16,
                get_vocab_size=lambda: 176256,
            )
            self.constraint_backend = None

        def load_model(self, *args: Any, **kwargs: Any) -> str:
            return "loaded"

    wrapped = model_runner_patch.make_load_model_with_constraint_backend(FakeRunner.load_model)
    with pytest.raises(ValueError, match="bad triples"):
        wrapped(FakeRunner())
