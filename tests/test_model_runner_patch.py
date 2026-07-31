# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Provider-driven ModelRunner patch installation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from vllm_gr.platforms.errors import BackendNotActiveError
from vllm_gr.platforms.registry import _reset_backend_registry_for_tests
from vllm_gr.v1.worker import gpu_model_runner_patch, model_runner_patch


def setup_function() -> None:
    _reset_backend_registry_for_tests()


def teardown_function() -> None:
    _reset_backend_registry_for_tests()


@dataclass
class FakeBackend:
    hooks: Any
    device_type: str = "fake"
    hook_reads: int = 0

    def get_model_runner_hooks(self) -> Any:
        self.hook_reads += 1
        return self.hooks


def _runner_class() -> type[Any]:
    class Runner:
        def execute_model(self, scheduler_output: Any) -> str:
            return "executed"

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
        ) -> str:
            return "bookkept"

    return Runner


def _gpu_hooks(runner_class: type[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        discover_worker_runners=lambda: (("GPUModelRunner", runner_class),),
        make_patched_prepare_inputs=gpu_model_runner_patch.make_patched_prepare_inputs,
        make_patched_bookkeeping_sync=gpu_model_runner_patch.make_patched_bookkeeping_sync,
    )


def test_discovery_uses_active_backend_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr import platforms

    runner_class = _runner_class()
    backend = FakeBackend(_gpu_hooks(runner_class))
    get_active = mock.Mock(return_value=backend)
    monkeypatch.setattr(platforms, "get_active_backend", get_active)

    assert model_runner_patch.discover_active_worker_runners() == (
        ("GPUModelRunner", runner_class),
    )
    get_active.assert_called_once_with()
    assert backend.hook_reads == 1


def test_apply_resolves_backend_and_hooks_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr import platforms

    hooks = SimpleNamespace(discover_worker_runners=lambda: ())
    backend = FakeBackend(hooks)
    get_active = mock.Mock(return_value=backend)
    monkeypatch.setattr(platforms, "get_active_backend", get_active)

    model_runner_patch.apply_worker_patches()

    get_active.assert_called_once_with()
    assert backend.hook_reads == 1


def test_apply_installs_three_wrappers_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr import platforms

    runner_class = _runner_class()
    backend = FakeBackend(_gpu_hooks(runner_class), device_type="cuda")
    monkeypatch.setattr(platforms, "get_active_backend", lambda: backend)

    model_runner_patch.apply_worker_patches()
    first = (
        runner_class.execute_model,
        runner_class._prepare_inputs,
        runner_class._bookkeeping_sync,
    )
    assert model_runner_patch.is_worker_runner_patched(runner_class)
    assert runner_class._patched_for_beam_pos_ids is True

    model_runner_patch.apply_worker_patches()

    assert (
        runner_class.execute_model,
        runner_class._prepare_inputs,
        runner_class._bookkeeping_sync,
    ) == first


def test_apply_fails_without_active_backend() -> None:
    with pytest.raises(BackendNotActiveError):
        model_runner_patch.apply_worker_patches()


def test_active_backend_is_not_read_by_wrapped_runtime_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr import platforms

    runner_class = _runner_class()
    backend = FakeBackend(_gpu_hooks(runner_class), device_type="cuda")
    monkeypatch.setattr(platforms, "get_active_backend", lambda: backend)
    model_runner_patch.apply_worker_patches()

    def unexpected_registry_read() -> Any:
        raise AssertionError("runtime wrapper read the backend registry")

    monkeypatch.setattr(platforms, "get_active_backend", unexpected_registry_read)

    runner = runner_class()
    scheduler_output = SimpleNamespace(beam_data={})
    assert runner.execute_model(scheduler_output) == "executed"
    assert len(runner._prepare_inputs(scheduler_output, {})) == 2
    assert runner._bookkeeping_sync(scheduler_output) == "bookkept"
