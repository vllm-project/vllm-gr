# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host-only plan, binding and startup gate tests; no inference runtime."""

from __future__ import annotations

import json
import pickle
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_gr.config import GRConfig
from vllm_gr.config.parser import parse_gr_config
from vllm_gr.platforms import loader
from vllm_gr.platforms.ascend import NPU_BACKEND
from vllm_gr.platforms.binding import (
    bind_backend_services,
    close_backend_binding,
    validate_worker_readiness,
    verify_engine_backend_readiness,
    worker_backend_readiness,
)
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.plan import (
    BACKEND_PLAN_KEY,
    BackendPlan,
    BackendRequest,
    PlanData,
    ServiceRequirements,
    read_backend_plan,
)
from vllm_gr.platforms.planning import (
    admit_backend_plan,
    normalize_backend_request,
    resolve_backend_plan,
)
from vllm_gr.platforms.services import planning_provider, validate_planning_api

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


class Service:
    implementation_id = "test-service-v1"

    def __init__(self, plan: BackendPlan, owner: Any) -> None:
        self.plan = plan
        self.owner = owner
        self.closed = False

    def is_available(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


class Provider:
    backend_id = "test-backend"
    api_version = 2
    plan_api_version = 1
    adapter_revision = "test-adapter-v1"

    def __init__(self) -> None:
        self.created: list[Service] = []
        self.unavailable = False

    def resolve(self, request: BackendRequest) -> BackendPlan:
        return BackendPlan(
            self.backend_id,
            self.api_version,
            request,
            PlanData(
                {
                    "requires_attention": True,
                    "constraint": "legacy",
                    "custom-layout": request.options.to_dict().get("layout", "native"),
                }
            ),
        )

    def validate_worker(self, *, plan: BackendPlan, native_runner: Any) -> None:
        if self.unavailable:
            raise RuntimeError("required kernel missing")

    def create_model_runner_adapter(self, *, plan: BackendPlan, native_runner: Any) -> Service:
        service = Service(plan, native_runner)
        self.created.append(service)
        return service

    def create_attention_executor(self, *, plan: BackendPlan, device: Any) -> Service:
        service = Service(plan, device)
        self.created.append(service)
        return service

    def create_constrained_topk_execution(self, *, plan: BackendPlan, device: Any) -> Service:
        service = Service(plan, device)
        self.created.append(service)
        return service


def config(width: int = 16) -> Any:
    native = SimpleNamespace(additional_config={}, parallel_config=SimpleNamespace(world_size=1))
    GRConfig(beam_max_width=width).write_to_vllm_config(native)
    return native


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Provider:
    selected = Provider()
    monkeypatch.setattr(loader, "activate_current_backend", lambda: selected)
    return selected


def test_canonical_encoding_is_deeply_immutable_and_order_independent() -> None:
    source = {"options": {"tiles": [32, 64]}, "name": "external"}
    value = PlanData(source)
    source["options"]["tiles"].append(128)
    decoded = value.to_dict()
    decoded["options"]["tiles"].append(256)
    assert value == PlanData({"name": "external", "options": {"tiles": [32, 64]}})
    assert pickle.loads(pickle.dumps(value)) == value
    with pytest.raises(FrozenInstanceError):
        value.canonical_json = "{}"  # type: ignore[misc]


@pytest.mark.parametrize(
    "value", [object(), {1: "key"}, {"set": {1, 2}}, {"n": float("nan")}, {"n": float("inf")}]
)
def test_rejects_nonportable_values(value: Any) -> None:
    with pytest.raises(ValueError):
        PlanData(value)


def test_wire_round_trip_and_repeated_admission_preserve_config(provider: Provider) -> None:
    native = config()
    original = parse_gr_config(native)
    plan = admit_backend_plan(native)
    assert plan is not None
    assert plan == admit_backend_plan(native)
    assert parse_gr_config(native) == original
    assert BackendPlan.from_wire(json.loads(json.dumps(plan.to_wire()))) == plan
    assert pickle.loads(pickle.dumps(plan)) == plan
    assert len(plan.fingerprint) == 64


def test_service_requirements_extend_existing_wire_plans() -> None:
    empty = PlanData({})
    legacy = BackendPlan(
        "fixture", 2, BackendRequest(empty, empty, empty, empty), PlanData({"stage": "old"})
    )
    assert legacy.fingerprint == "3e1fe9147d21e927146995e78cffd1a2559e9fa936dfa13eb23c7b3a86af71d5"
    assert "requirements" not in legacy.to_dict()
    assert BackendPlan.from_wire(legacy.to_wire()) == legacy
    plan = replace(legacy, requirements=ServiceRequirements(stage="external.stage"))
    wire = plan.to_wire()
    assert wire["plan"]["requirements"] == {"stage": "external.stage"}
    assert BackendPlan.from_wire(wire) == plan
    wire["plan"]["requirements"]["stage"] = "other.stage"
    with pytest.raises(ValueError, match="fingerprint"):
        BackendPlan.from_wire(wire)
    for invalid in ({"stage": " "}, {"unknown": "id"}, {"attention": None}):
        with pytest.raises(ValueError):
            ServiceRequirements.from_dict(invalid)


def test_explicit_service_ids_bind_and_reject_matching_but_wrong_worker_reports(
    provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        provider,
        "resolve",
        lambda request: BackendPlan(
            provider.backend_id,
            provider.api_version,
            request,
            PlanData({}),
            requirements=ServiceRequirements(
                attention=Service.implementation_id, constraint=Service.implementation_id
            ),
        ),
    )
    native = config()
    plan = admit_backend_plan(native)
    runner = SimpleNamespace(vllm_config=native, device="test:0")
    report = worker_backend_readiness(SimpleNamespace(model_runner=runner))
    assert (
        report["services"]["attention"]
        == report["services"]["constraint"]
        == Service.implementation_id
    )
    validate_worker_readiness(plan, [report, report], expected_workers=2)
    report["services"]["attention"] = "other.attention"
    with pytest.raises(RuntimeError, match="attention implementation mismatch"):
        validate_worker_readiness(plan, [report, report], expected_workers=2)
    close_backend_binding(runner)
    assert len(provider.created) == 3 and all(service.closed for service in provider.created)


@pytest.mark.parametrize("service", ["attention", "stage", "graph"])
def test_unmet_pinned_requirement_rolls_back_binding(
    provider: Provider, monkeypatch: pytest.MonkeyPatch, service: str
) -> None:
    monkeypatch.setattr(
        provider,
        "resolve",
        lambda request: BackendPlan(
            provider.backend_id,
            provider.api_version,
            request,
            PlanData({}),
            requirements=ServiceRequirements.from_dict({service: "missing.implementation"}),
        ),
    )
    native = config()
    admit_backend_plan(native)
    runner = SimpleNamespace(vllm_config=native)
    with pytest.raises(RuntimeError, match=service):
        bind_backend_services(runner)
    assert getattr(runner, "_gr_backend_binding", None) is None
    assert provider.created and all(service.closed for service in provider.created)


def test_canonical_table_reports_its_identity_and_keeps_its_native_owner(
    provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        provider,
        "resolve",
        lambda request: BackendPlan(
            provider.backend_id,
            provider.api_version,
            request,
            PlanData({"constraint": "canonical"}),
        ),
    )
    native = config()
    plan = admit_backend_plan(native)
    table = Service(plan, native)
    runner = SimpleNamespace(vllm_config=native, constraint_backend=table)
    report = worker_backend_readiness(SimpleNamespace(model_runner=runner))
    assert report["services"]["constraint"] == table.implementation_id
    close_backend_binding(runner)
    assert not table.closed and runner.constraint_backend is table


@pytest.mark.parametrize(
    "field,value", [("schema_version", 2), ("schema_version", True), ("plan_api_version", 2)]
)
def test_wire_rejects_unsupported_schema(provider: Provider, field: str, value: Any) -> None:
    plan = admit_backend_plan(config())
    wire = plan.to_wire()
    wire["plan"][field] = value
    with pytest.raises(ValueError):
        BackendPlan.from_wire(wire)


def test_fingerprint_covers_choices_and_compatibility_but_not_local_device(
    provider: Provider,
) -> None:
    first = config()
    second = config()
    first.device_config = SimpleNamespace(device="test:0")
    second.device_config = SimpleNamespace(device="test:7")
    plan = admit_backend_plan(first)
    assert plan == admit_backend_plan(second)
    changed = replace(
        plan, request=replace(plan.request, compatibility=PlanData({"adapter": "v2"}))
    )
    assert changed.fingerprint != plan.fingerprint
    wire = plan.to_wire()
    wire["plan"]["choices"]["custom-layout"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        BackendPlan.from_wire(wire)


def test_two_engines_bind_their_own_plan_runner_and_device(provider: Provider) -> None:
    first = SimpleNamespace(vllm_config=config(16), device="test:0")
    second = SimpleNamespace(vllm_config=config(32), device="test:1")
    first_plan = admit_backend_plan(first.vllm_config)
    second_plan = admit_backend_plan(second.vllm_config)
    a = bind_backend_services(first)
    b = bind_backend_services(second)
    assert a.plan == first_plan and b.plan == second_plan and a.plan != b.plan
    assert a.runner_adapter.owner is first and b.runner_adapter.owner is second
    assert a.attention_executor.owner == first.device
    assert b.constraint_execution.owner == second.device
    assert all(service.plan == first_plan for service in provider.created[:3])
    assert not hasattr(provider, "current_plan")
    assert bind_backend_services(first) is a
    close_backend_binding(first)
    close_backend_binding(first)
    assert all(service.closed for service in provider.created[:3])
    assert not any(service.closed for service in provider.created[3:])


def test_worker_configuration_drift_fails_before_any_factory(provider: Provider) -> None:
    native = config()
    admit_backend_plan(native)
    native.additional_config["vllm_gr_beam_max_width"] = 32
    native._vllm_gr_parsed_config = None
    with pytest.raises(RuntimeError, match="plan mismatch"):
        bind_backend_services(SimpleNamespace(vllm_config=native, device="test:0"))
    assert provider.created == []


def test_readiness_preserves_binding_after_native_kv_cache_autofit(provider: Provider) -> None:
    native = config()
    native.model_config = SimpleNamespace(max_model_len=4096)
    plan = admit_backend_plan(native)
    runner = SimpleNamespace(vllm_config=native)
    binding = bind_backend_services(runner)
    native.model_config.max_model_len = 2048
    assert worker_backend_readiness(SimpleNamespace(model_runner=runner)) == binding.report()
    assert binding.plan == plan
    assert len(provider.created) == 3


def test_worker_capability_failure_is_not_a_fallback(provider: Provider) -> None:
    native = config()
    admit_backend_plan(native)
    provider.unavailable = True
    with pytest.raises(RuntimeError, match="kernel missing"):
        worker_backend_readiness(SimpleNamespace(model_runner=SimpleNamespace(vllm_config=native)))
    assert provider.created == []


def test_partial_factory_failure_releases_already_created_services(
    provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = config()
    admit_backend_plan(native)
    monkeypatch.setattr(provider, "create_constrained_topk_execution", lambda **_: None)
    runner = SimpleNamespace(vllm_config=native, device="test:0")
    with pytest.raises(RuntimeError, match="constraint execution"):
        bind_backend_services(runner)
    assert all(service.closed for service in provider.created)
    assert getattr(runner, "_gr_backend_binding", None) is None


@pytest.mark.parametrize(
    "fault", ["missing", "count", "not-ready", "fingerprint", "implementation"]
)
def test_readiness_requires_every_worker_to_agree(provider: Provider, fault: str) -> None:
    native = config()
    plan = admit_backend_plan(native)
    report = worker_backend_readiness(
        SimpleNamespace(model_runner=SimpleNamespace(vllm_config=native))
    )
    reports = [report, json.loads(json.dumps(report))]
    validate_worker_readiness(plan, reports, expected_workers=2)
    if fault == "missing":
        reports = []
    elif fault == "count":
        reports.pop()
    elif fault == "not-ready":
        reports[1]["ready"] = False
    elif fault == "fingerprint":
        reports[1]["fingerprint"] = "different"
    else:
        reports[1]["services"]["attention"] = "another-implementation"
    with pytest.raises(RuntimeError):
        validate_worker_readiness(plan, reports, expected_workers=2)


def test_failed_engine_gate_shuts_down_executor(provider: Provider) -> None:
    native = config()
    admit_backend_plan(native)
    shutdown = []
    engine = SimpleNamespace(
        vllm_config=native,
        model_executor=SimpleNamespace(
            collective_rpc=lambda method: [], shutdown=lambda: shutdown.append(True)
        ),
    )
    with pytest.raises(RuntimeError, match="worker reports"):
        verify_engine_backend_readiness(engine)
    assert shutdown == [True]
    assert not hasattr(engine, "_gr_backend_fingerprint")


@pytest.mark.parametrize("failure", [None, "local", "remote", "plan-drift"])
def test_external_launcher_readiness_gathers_each_rank(
    provider: Provider, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    import sys
    from unittest.mock import Mock

    native = config()
    native.parallel_config = SimpleNamespace(
        world_size=2, distributed_executor_backend="external_launcher"
    )
    plan = admit_backend_plan(native)
    runner = SimpleNamespace(vllm_config=native)
    good = worker_backend_readiness(SimpleNamespace(model_runner=runner))
    remote = dict(good)
    if failure == "remote":
        remote = {"ready": False, "error": "remote worker failed"}
    elif failure == "plan-drift":
        remote["fingerprint"] = "different"
    group = object()
    received = []

    def gather(reports: list, local: dict, **kwargs: Any) -> None:
        assert kwargs["group"] is group
        received.append(local)
        reports[:] = [local, remote]

    dist = SimpleNamespace(get_world_size=lambda group: 2, all_gather_object=gather)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=dist))
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        SimpleNamespace(get_world_group=lambda: SimpleNamespace(cpu_group=group)),
    )
    rpc = Mock(
        return_value=[good],
        side_effect=RuntimeError("local worker failed") if failure == "local" else None,
    )
    executor = SimpleNamespace(collective_rpc=rpc, shutdown=Mock())
    engine = SimpleNamespace(vllm_config=native, model_executor=executor)
    if failure is None:
        verify_engine_backend_readiness(engine)
        assert engine._gr_backend_fingerprint == plan.fingerprint
        executor.shutdown.assert_not_called()
    else:
        with pytest.raises(
            RuntimeError, match="local worker failed|remote worker failed|fingerprint mismatch"
        ):
            verify_engine_backend_readiness(engine)
        executor.shutdown.assert_called_once_with()
    rpc.assert_called_once_with(worker_backend_readiness)
    assert len(received) == 1
    assert received[0]["ready"] is (failure != "local")


@pytest.mark.parametrize("service", ["runner", "attention", "constraint"])
def test_readiness_rejects_missing_required_service(provider: Provider, service: str) -> None:
    native = config()
    plan = admit_backend_plan(native)
    report = worker_backend_readiness(
        SimpleNamespace(model_runner=SimpleNamespace(vllm_config=native))
    )
    report["services"][service] = None
    with pytest.raises(RuntimeError, match=f"no bound {service}"):
        validate_worker_readiness(plan, [report], expected_workers=1)


def test_builtin_plans_are_pure_and_preserve_runner_and_decode_policy() -> None:
    native = config()
    request = normalize_backend_request(native, CUDA_BACKEND)
    assert CUDA_BACKEND.resolve(request) == CUDA_BACKEND.resolve(request)
    assert CUDA_BACKEND.resolve(request).choices.to_dict()["stage"] == "cuda.v1"
    native.use_v2_model_runner = True
    request = normalize_backend_request(native, CUDA_BACKEND)
    assert CUDA_BACKEND.resolve(request).choices.to_dict()["stage"] == "cuda.v2"
    npu_request = normalize_backend_request(native, NPU_BACKEND)
    choices = NPU_BACKEND.resolve(npu_request).choices.to_dict()
    assert choices["stage"] == "ascend.v1"
    assert choices["decode_attention"] == "auto"
    with pytest.raises(ValueError, match="xllm"):
        CUDA_BACKEND.resolve(
            replace(
                request,
                config=PlanData({**request.config.to_dict(), "npu_decode_execution": "xllm"}),
            )
        )


def test_legacy_provider_never_infers_a_planning_extension(provider: Provider) -> None:
    class Legacy(Provider):
        plan_api_version = None

    old = Legacy()
    request = normalize_backend_request(config(), old)
    adapted = planning_provider(old)
    assert (
        adapted.resolve(request).choices.to_dict()["compatibility"] == "provider-api-v2-delegation"
    )
    with pytest.raises(ValueError, match="typed backend options"):
        adapted.resolve(replace(request, options=PlanData({"tiles": [32]})))


def test_bad_planning_version_and_missing_methods_fail_registration() -> None:
    for version in (True, 2, "1"):
        with pytest.raises(ValueError, match="plan_api_version"):
            validate_planning_api(SimpleNamespace(plan_api_version=version))
    with pytest.raises(ValueError, match="missing methods"):
        validate_planning_api(SimpleNamespace(plan_api_version=1))


def test_resolver_cannot_change_request_identity(
    provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = normalize_backend_request(config(), provider)
    plan = provider.resolve(request)
    monkeypatch.setattr(provider, "resolve", lambda _: replace(plan, backend_id="wrong"))
    with pytest.raises(ValueError, match="provider identity"):
        resolve_backend_plan(provider, request)


def test_unknown_reserved_key_still_fails_and_invalid_plan_is_not_ignored() -> None:
    native = config()
    native.additional_config[BACKEND_PLAN_KEY] = {"invalid": True}
    assert parse_gr_config(native).beam_max_width == 16
    with pytest.raises(ValueError, match="envelope"):
        read_backend_plan(native)
    native.additional_config["vllm_gr_backend_typo"] = 1
    with pytest.raises(ValueError, match="unknown"):
        parse_gr_config(native)


def test_disabling_gr_removes_a_previous_plan(provider: Provider) -> None:
    native = config()
    assert admit_backend_plan(native) is not None
    GRConfig().write_to_vllm_config(native)
    assert admit_backend_plan(native) is None
    assert read_backend_plan(native) is None


def test_ordinary_native_config_does_not_activate_a_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected() -> None:
        raise AssertionError("Native engine must not activate a GR backend")

    monkeypatch.setattr(loader, "activate_current_backend", unexpected)
    assert admit_backend_plan(SimpleNamespace(additional_config={})) is None


def test_binding_adopts_constraint_execution_preloaded_with_the_same_plan(
    provider: Provider,
) -> None:
    native = config()
    plan = admit_backend_plan(native)
    runner = SimpleNamespace(vllm_config=native, device="test:0")
    preloaded = Service(plan, runner.device)
    runner._gr_constraint_execution = preloaded
    runner._gr_constraint_plan_fingerprint = plan.fingerprint
    binding = bind_backend_services(runner)
    assert binding.constraint_execution is preloaded
    assert len(provider.created) == 2
    close_backend_binding(runner)
    assert preloaded.closed
    assert runner._gr_constraint_execution is None


def test_failed_binding_detaches_unavailable_preloaded_constraint(provider: Provider) -> None:
    native = config()
    plan = admit_backend_plan(native)
    runner = SimpleNamespace(vllm_config=native, device="test:0")

    class Unavailable(Service):
        def is_available(self) -> bool:
            return False

    preloaded = Unavailable(plan, runner.device)
    runner._gr_constraint_execution = preloaded
    runner._gr_constraint_plan_fingerprint = plan.fingerprint
    with pytest.raises(RuntimeError, match="constraint execution is unavailable"):
        bind_backend_services(runner)
    assert preloaded.closed
    assert runner._gr_constraint_execution is None
    assert runner._gr_constraint_plan_fingerprint is None
    assert bind_backend_services(runner).constraint_execution is not preloaded


def test_bound_constraint_honors_explicit_torch_oracle(
    provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    native = config()
    plan = admit_backend_plan(native)
    oracle = Service(plan, "test:0")
    monkeypatch.setenv("VLLM_GR_CONSTRAINT_EXECUTION", "torch")
    monkeypatch.setitem(
        sys.modules,
        "vllm_gr.constraints.execution",
        SimpleNamespace(TorchConstrainedTopKExecution=lambda *, device: oracle),
    )
    binding = bind_backend_services(SimpleNamespace(vllm_config=native, device="test:0"))
    assert binding.constraint_execution is oracle
    assert len(provider.created) == 2
