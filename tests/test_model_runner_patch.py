# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for active-platform ModelRunner patch dispatch."""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_gr.v1.worker import gpu_model_runner_patch, model_runner_patch


def test_dispatcher_selects_one_platform_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm import platforms

    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(device_type="npu"),
    )
    assert model_runner_patch._active_platform_module_name() == (
        model_runner_patch._NPU_PATCH_MODULE
    )
    platforms.current_platform.device_type = "cuda"
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

    model_runner_patch.apply_worker_patches()
    assert (Runner.execute_model, Runner._prepare_inputs, Runner._bookkeeping_sync) == first
