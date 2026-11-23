# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native entry-point dispatch, completion ownership, and generation cleanup."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.config import GRConfig
from vllm_gr.config.backend_plan import BackendPlan, BackendRequest, PlanData
from vllm_gr.platforms import loader
from vllm_gr.platforms.planning import admit_backend_plan
from vllm_gr.platforms.services import LegacyPlanningProvider
from vllm_gr.v1.engine.worker_readiness import worker_backend_readiness
from vllm_gr.v1.worker import runner_binding as binding
from vllm_gr.v1.worker.runner_binding import (
    BackendBinding,
    bind_after_model_load,
    close_backend_binding,
)
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


def test_full_load_rebinds_the_same_plan_only_once_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Mock(
        backend_id="test",
        api_version=2,
        plan_api_version=1,
        adapter_revision="1",
        vendor_distribution=None,
    )
    provider.resolve.side_effect = lambda request: BackendPlan(
        "test", 2, request, PlanData({"runner": "native"})
    )
    old, new = Mock(implementation_id="native"), Mock(implementation_id="native")
    provider.create_model_runner_adapter.side_effect = [old, new]
    monkeypatch.setattr(loader, "activate_current_backend", lambda: provider)
    config = SimpleNamespace(additional_config={})
    GRConfig(beam_max_width=16).write_to_vllm_config(config)
    plan = admit_backend_plan(config)
    assert plan is not None
    runner = SimpleNamespace(vllm_config=config)
    worker = SimpleNamespace(model_runner=runner)

    def load_weights(self: Any, model: object) -> object:
        assert getattr(self, "_gr_backend_binding", None) is None
        self.model = model
        return model

    load = bind_after_model_load(load_weights)
    for adapter in (old, new):
        model = object()
        assert load(runner, model) is model
        assert runner._gr_backend_binding.runner_adapter is adapter
        assert worker_backend_readiness(worker)["fingerprint"] == plan.fingerprint
        assert binding.bind_backend_services(runner) is runner._gr_backend_binding
        adapter.close.assert_not_called()
    old.close.assert_called_once_with()
    assert provider.create_model_runner_adapter.call_count == 2


@pytest.fixture  # type: ignore[misc]
def autofit_runner(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, BackendPlan]:
    cleanup_module = ModuleType("vllm_gr.v1.worker.model_runner_patch")
    setattr(cleanup_module, "close_constraint_backend", Mock())
    monkeypatch.setitem(sys.modules, cleanup_module.__name__, cleanup_module)
    provider = Mock(
        backend_id="test",
        api_version=2,
        plan_api_version=1,
        adapter_revision="1",
        vendor_distribution=None,
    )
    provider.resolve.side_effect = lambda request: BackendPlan(
        "test",
        2,
        request,
        PlanData(
            {
                "runner": "native",
                "admitted_max_model_len": request.native.to_dict()["model_config"]["max_model_len"],
            }
        ),
    )
    monkeypatch.setattr(loader, "activate_current_backend", lambda: provider)
    config = SimpleNamespace(
        additional_config={}, model_config=SimpleNamespace(max_model_len=4096, dtype="bfloat16")
    )
    GRConfig(beam_max_width=16).write_to_vllm_config(config)
    plan = admit_backend_plan(config)
    assert plan is not None
    return SimpleNamespace(vllm_config=config), provider, plan


def test_first_load_binds_admitted_plan_after_native_autofit(
    autofit_runner: tuple[Any, Any, BackendPlan],
) -> None:
    runner, provider, plan = autofit_runner
    adapter = Mock(implementation_id="native")
    provider.create_model_runner_adapter.return_value = adapter

    def load_weights(self: Any, model: object) -> object:
        assert getattr(self, "_gr_backend_binding", None) is None
        self.vllm_config.model_config.max_model_len = 2048
        self.model = model
        return model

    observed_lengths = []
    provider.validate_worker.side_effect = lambda **kwargs: observed_lengths.append(
        kwargs["native_runner"].vllm_config.model_config.max_model_len
    )
    model = object()
    assert bind_after_model_load(load_weights)(runner, model) is model
    assert runner._gr_backend_binding.plan == plan
    assert runner._gr_backend_binding.runner_adapter is adapter
    assert runner.vllm_config.model_config.max_model_len == 2048
    assert runner.vllm_config.additional_config["vllm_gr_backend_plan"] == plan.to_wire()
    assert observed_lengths == [2048]
    provider.create_model_runner_adapter.assert_called_once_with(plan=plan, native_runner=runner)
    assert worker_backend_readiness(SimpleNamespace(model_runner=runner))["fingerprint"] == (
        plan.fingerprint
    )


def test_full_reload_rebinds_admitted_plan_after_native_autofit(
    autofit_runner: tuple[Any, Any, BackendPlan],
) -> None:
    runner, provider, plan = autofit_runner
    old, new = Mock(implementation_id="native"), Mock(implementation_id="native")
    provider.create_model_runner_adapter.side_effect = [old, new]

    def load_weights(self: Any, model: object) -> object:
        assert getattr(self, "_gr_backend_binding", None) is None
        self.model = model
        return model

    load = bind_after_model_load(load_weights)
    first = object()
    assert load(runner, first) is first
    previous = runner._gr_backend_binding
    runner.vllm_config.model_config.max_model_len = 2048
    assert worker_backend_readiness(SimpleNamespace(model_runner=runner)) == previous.report()
    second = object()
    assert load(runner, second) is second
    assert runner._gr_backend_binding is not previous
    assert runner._gr_backend_binding.plan == plan
    assert runner._gr_backend_binding.runner_adapter is new
    assert runner.vllm_config.model_config.max_model_len == 2048
    assert runner.vllm_config.additional_config["vllm_gr_backend_plan"] == plan.to_wire()
    assert provider.create_model_runner_adapter.call_count == 2
    assert provider.validate_worker.call_count == 2
    assert all(call.kwargs["plan"] == plan for call in provider.validate_worker.call_args_list)
    old.close.assert_called_once_with()
    new.close.assert_not_called()
    assert worker_backend_readiness(SimpleNamespace(model_runner=runner))["fingerprint"] == (
        plan.fingerprint
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "drift", ["length-increase", "length-zero", "length-bool", "dtype", "beam", "runner", "adapter"]
)
def test_autofit_does_not_hide_worker_plan_drift(
    autofit_runner: tuple[Any, Any, BackendPlan], drift: str
) -> None:
    runner, provider, _ = autofit_runner

    def load_weights(self: Any) -> None:
        self.vllm_config.model_config.max_model_len = 2048
        if drift.startswith("length-"):
            self.vllm_config.model_config.max_model_len = {
                "length-increase": 8192,
                "length-zero": 0,
                "length-bool": True,
            }[drift]
        elif drift == "dtype":
            self.vllm_config.model_config.dtype = "float32"
        elif drift == "beam":
            GRConfig(beam_max_width=32).write_to_vllm_config(self.vllm_config)
        elif drift == "runner":
            self.vllm_config.use_v2_model_runner = True
        else:
            provider.adapter_revision = "2"

    with pytest.raises(RuntimeError, match="Worker backend plan mismatch"):
        bind_after_model_load(load_weights)(runner)
    provider.validate_worker.assert_not_called()
    provider.create_model_runner_adapter.assert_not_called()
    assert getattr(runner, "_gr_backend_binding", None) is None


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
        assert self._gr_backend_binding is None
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
