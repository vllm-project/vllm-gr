# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native entry-point dispatch, completion ownership, and generation cleanup."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.platforms import binding
from vllm_gr.platforms.binding import (
    BackendBinding,
    bind_after_model_load,
    close_backend_binding,
)
from vllm_gr.platforms.plan import BackendPlan, BackendRequest, PlanData
from vllm_gr.platforms.services import LegacyPlanningProvider
from vllm_gr.v1.worker.runner_lifecycle import (
    bind_native_runner,
    has_runner_lifecycle,
    install_runner_lifecycle,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def _plan(implementation: str) -> BackendPlan:
    empty = PlanData({})
    request = BackendRequest(empty, empty, empty, empty)
    return BackendPlan("test", 2, request, PlanData({"runner": implementation}))


@pytest.mark.parametrize("prepare", ["_prepare_inputs", "prepare_inputs"])  # type: ignore[misc]
def test_native_calls_use_the_owning_adapter_and_keep_late_completion(prepare: str) -> None:
    events: list[tuple[str, str]] = []

    class Runner:
        _gr_backend_binding: BackendBinding

        def __init__(self, name: str) -> None:
            self.name = name
            self.receipt: Any = None

        def execute_model(self, receipt: Any) -> None:
            events.append((self.name, "execute"))
            self.receipt = receipt

        def sample_tokens(self, *, fail: bool = False) -> Any:
            events.append((self.name, "sample"))
            if fail:
                raise ValueError("sample failed")
            receipt, self.receipt = self.receipt, None
            return receipt

    def prepare_inputs(runner: Runner, *, fail: bool = False) -> int:
        events.append((runner.name, "prepare"))
        if fail:
            raise ValueError("prepare failed")
        return 7

    setattr(Runner, prepare, prepare_inputs)
    install_runner_lifecycle(Runner, "native", prepare=prepare)
    entry = Runner.execute_model
    install_runner_lifecycle(Runner, "native", prepare=prepare)
    assert Runner.execute_model is entry
    assert has_runner_lifecycle(Runner, "native")
    first, second = Runner("first"), Runner("second")
    receipt = object()
    first.execute_model(receipt)
    assert first.sample_tokens() is receipt  # unplanned native fallback

    plan = _plan("native")
    for runner in (first, second):
        adapter = bind_native_runner(plan, runner)
        runner._gr_backend_binding = BackendBinding(plan, adapter, None, None)
    probe = Mock(wraps=first._gr_backend_binding.runner_adapter.execute_model)
    first._gr_backend_binding.runner_adapter.execute_model = probe
    events.clear()
    assert getattr(first, prepare)() == 7
    first.execute_model(receipt)
    probe.assert_called_once_with(receipt)
    with pytest.raises(ValueError, match="prepare failed"):
        getattr(second, prepare)(fail=True)
    with pytest.raises(ValueError, match="sample failed"):
        first.sample_tokens(fail=True)
    assert first.receipt is receipt and second.receipt is None
    assert first.sample_tokens() is receipt
    assert events == [
        ("first", "prepare"),
        ("first", "execute"),
        ("second", "prepare"),
        ("first", "sample"),
        ("first", "sample"),
    ]
    old = first._gr_backend_binding.runner_adapter
    close_backend_binding(first)
    close_backend_binding(first)
    with pytest.raises(RuntimeError, match="closed"):
        old.execute_model(receipt)
    assert second._gr_backend_binding.runner_adapter.plan == plan
    with pytest.raises(RuntimeError, match="admitted backend plan"):
        bind_native_runner(_plan("other"), first)


def test_legacy_provider_delegates_to_the_same_installed_module() -> None:
    class Runner:
        def prepare_inputs(self) -> int:
            return 0

        def execute_model(self) -> None:
            raise RuntimeError("native execution failed")

    module = ModuleType("example.runner")
    install_runner_lifecycle(Runner, module.__name__, prepare="prepare_inputs")
    provider = SimpleNamespace(
        backend_id="test", api_version=2, get_model_runner_hooks=lambda: module
    )
    legacy = LegacyPlanningProvider(provider)
    runner, plan = Runner(), _plan("legacy-provider-hooks")
    adapter = legacy.create_model_runner_adapter(plan=plan, native_runner=runner)
    setattr(runner, "_gr_backend_binding", BackendBinding(plan, adapter, None, None))
    assert runner.prepare_inputs() == 0
    with pytest.raises(RuntimeError, match="native execution failed"):
        runner.execute_model()
    provider.get_model_runner_hooks = lambda: ModuleType("other.runner")
    with pytest.raises(RuntimeError, match="differ from the installed"):
        legacy.create_model_runner_adapter(plan=plan, native_runner=runner)


@pytest.mark.parametrize("cleanup_fails", [False, True])  # type: ignore[misc]
def test_reload_preserves_live_generation_and_releases_failed_replacement(
    monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    events: list[str] = []
    plan = _plan("native")
    previous = Mock()
    runner = SimpleNamespace(
        _gr_backend_binding=BackendBinding(plan, previous, None, None), pending_receipt=object()
    )
    native_error = RuntimeError("pending native receipt")

    def release_runtime(self: Any) -> None:
        if self.pending_receipt is not None:
            raise native_error
        events.append("release runtime")
        if cleanup_fails and self._gr_backend_binding is None:
            raise RuntimeError("native cleanup failed")

    def native_load(self: Any) -> None:
        pytest.fail("an undrained receipt must prevent native model replacement")

    with pytest.raises(RuntimeError) as rejected:
        bind_after_model_load(native_load, close_runtime=release_runtime)(runner)
    assert rejected.value is native_error
    previous.close.assert_not_called()
    assert events == []
    runner.pending_receipt = None

    def replace(self: Any) -> None:
        close_backend_binding(self)
        events.append("native load")

    def fail_binding(self: Any) -> None:
        events.append("bind")
        raise ValueError("factory failed")

    def release_table(self: Any) -> None:
        events.append("release table")

    cleanup_module = ModuleType("vllm_gr.v1.worker.model_runner_patch")
    setattr(cleanup_module, "close_constraint_backend", release_table)
    monkeypatch.setitem(sys.modules, cleanup_module.__name__, cleanup_module)
    monkeypatch.setattr(binding, "bind_backend_services", fail_binding)
    with pytest.raises(ValueError, match="factory failed"):
        bind_after_model_load(replace, close_runtime=release_runtime)(runner)
    previous.close.assert_called_once_with()
    assert runner._gr_backend_binding is None
    assert events == ["release runtime", "native load", "bind", "release runtime", "release table"]


def test_first_load_cleans_partial_resources_before_a_binding_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = SimpleNamespace(native_resource=None)
    native_resource = Mock()
    failure = ValueError("partial native load")

    def allocate_then_fail(self: Any) -> None:
        self.native_resource = native_resource
        raise failure

    def release_runtime(self: Any) -> None:
        if self.native_resource is not None:
            self.native_resource.close()
            self.native_resource = None

    close_table = Mock()
    cleanup_module = ModuleType("vllm_gr.v1.worker.model_runner_patch")
    setattr(cleanup_module, "close_constraint_backend", close_table)
    monkeypatch.setitem(sys.modules, cleanup_module.__name__, cleanup_module)
    with pytest.raises(ValueError) as raised:
        bind_after_model_load(allocate_then_fail, close_runtime=release_runtime)(runner)
    assert raised.value is failure
    native_resource.close.assert_called_once_with()
    close_table.assert_called_once_with(runner)
    assert runner.native_resource is None
