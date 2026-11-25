# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""EngineArgs integration; requires the normal Torch/vLLM test runtime."""

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from tests.platforms.test_backend_plan import Provider
from vllm_gr.config import GRConfig
from vllm_gr.config.backend_plan import (
    BACKEND_PLAN_KEY,
    BackendPlan,
    BackendRequest,
    PlanData,
    ServiceRequirements,
    read_backend_plan,
)
from vllm_gr.engine import arg_utils_gr
from vllm_gr.platforms import config as platform_config
from vllm_gr.platforms import loader
from vllm_gr.platforms.config import BackendConfigService
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.legacy import legacy_config_service

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_engine_args_round_trip_keeps_the_admitted_backend_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *args: native)
    monkeypatch.setattr(
        arg_utils_gr, "engine_config_service", lambda: legacy_config_service("cuda")
    )
    monkeypatch.setattr(loader, "activate_current_backend", lambda: CUDA_BACKEND)
    args = SimpleNamespace(
        additional_config={
            "vllm_gr": {
                "gr": {
                    "beam_width": [32] * 3,
                    "candidate_width": [32] * 3,
                    "constraint_mode": "none",
                }
            }
        }
    )
    first = arg_utils_gr.create_engine_config(args)
    plan = read_backend_plan(first)
    gr = GRConfig.from_vllm_config(first)
    assert plan is not None
    native.additional_config = {}
    assert arg_utils_gr.create_engine_config(args) is first
    assert read_backend_plan(first) == plan
    assert GRConfig.from_vllm_config(first) == gr
    assert plan.request.config.to_dict()["beam_max_width"] == 32


def test_backend_plan_changes_in_native_compilation_hash() -> None:
    from vllm.config import VllmConfig
    from vllm.config.kernel import IrOpPriorityConfig

    config = VllmConfig()
    config.kernel_config.ir_op_priority = IrOpPriorityConfig(
        rms_norm=["native"], fused_add_rms_norm=["native"], gelu_and_mul_sparse=["native"]
    )
    request = BackendRequest(
        config=PlanData({}),
        native=PlanData({"use_v2_model_runner": False}),
        compatibility=PlanData({"adapter_revision": "1"}),
        options=PlanData({}),
    )
    plan = BackendPlan(
        backend_id="example",
        provider_api_version=2,
        request=request,
        choices=PlanData({}),
        requirements=ServiceRequirements(graph="example.graph.v1"),
    )

    def compilation_hash(admitted: BackendPlan) -> str:
        config.additional_config = {BACKEND_PLAN_KEY: admitted.to_wire()}
        result = config.compute_hash()
        assert isinstance(result, str)
        return result

    original = compilation_hash(plan)
    assert (
        compilation_hash(BackendPlan.from_wire(json.loads(json.dumps(plan.to_wire())))) == original
    )
    for changed in (
        replace(plan, request=replace(request, native=PlanData({"use_v2_model_runner": True}))),
        replace(plan, requirements=ServiceRequirements(graph="example.graph.v2")),
        replace(plan, request=replace(request, compatibility=PlanData({"adapter_revision": "2"}))),
    ):
        assert compilation_hash(changed) != original


def test_external_policy_defaults_reach_engine_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Policy(BackendConfigService):  # type: ignore[misc]
        default_worker_decision = True

        def apply_native_defaults(self, config: SimpleNamespace) -> None:
            config.cache_config.block_size = 64
            events.append("defaults")

        def validate_engine(self, config: SimpleNamespace, gr: GRConfig) -> None:
            assert config.cache_config.block_size == 64
            assert gr.beam_worker_decision is True
            events.append("validate")

        def finalize_native_config(self, config: SimpleNamespace, gr: GRConfig) -> None:
            assert config is native
            events.append("finalize")

    provider = Provider()
    monkeypatch.setattr(provider, "create_config_service", Policy, raising=False)
    monkeypatch.setattr(platform_config, "get_active_backend", lambda: provider)
    monkeypatch.setattr(loader, "activate_current_backend", lambda: provider)
    native = SimpleNamespace(
        additional_config={},
        device_config=SimpleNamespace(device="cuda:0"),
        cache_config=SimpleNamespace(block_size=8),
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *args: native)
    args = SimpleNamespace(
        additional_config={
            "vllm_gr": {
                "gr": {
                    "beam_width": [32] * 3,
                    "candidate_width": [32] * 3,
                    "constraint_mode": "none",
                }
            }
        }
    )
    plan = read_backend_plan(arg_utils_gr.create_engine_config(args))
    assert events == ["defaults", "validate", "finalize"]
    assert plan is not None and plan.backend_id == provider.backend_id
    assert plan.request.native.to_dict()["cache_config"]["block_size"] == 64
    assert plan.request.config.to_dict()["beam_worker_decision"] is True


def test_frontend_settings_are_isolated_from_snapshot_plan_and_graph_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.config import VllmConfig
    from vllm.config.kernel import IrOpPriorityConfig

    from vllm_gr.config.parser import gr_config_to_mapping, resolve_startup_config
    from vllm_gr.platforms.planning import admit_backend_plan

    monkeypatch.setattr(loader, "activate_current_backend", lambda: CUDA_BACKEND)

    def build(host: str, port: int | None, width: int) -> tuple[dict[str, Any], str, str]:
        config, bind_host, bind_port = resolve_startup_config(
            {"gr": {"admin_port": 7999}},
            {
                "admin_host": host,
                "admin_port": port,
                "beam_width": [width],
                "candidate_width": [width],
            },
        )
        assert (bind_host, bind_port) == (host, port)
        native = VllmConfig()
        native.kernel_config.ir_op_priority = IrOpPriorityConfig(
            rms_norm=["native"], fused_add_rms_norm=["native"], gelu_and_mul_sparse=["native"]
        )
        config.write_to_vllm_config(native)
        plan = admit_backend_plan(native)
        assert plan is not None
        return gr_config_to_mapping(config), plan.fingerprint, native.compute_hash()

    original = build("localhost", None, 4)
    assert original == build("127.0.0.1", 9100, 4)
    assert original[-1] != build("127.0.0.1", 9100, 8)[-1]
    for field, value in (("admin_port", True), ("admin_port", 65536), ("admin_host", "")):
        with pytest.raises(ValueError, match=field):
            resolve_startup_config({}, {field: value})


def test_reusing_engine_args_after_failure_preserves_input_and_native_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.config import gr_config_to_mapping
    from vllm_gr.platforms import planning

    attrs = {
        "attention_backend": None,
        "additional_config": {"other": {"value": 3}, "vllm_gr": gr_config_to_mapping(GRConfig())},
    }
    args = SimpleNamespace(**attrs)
    original = json.loads(json.dumps(args.additional_config))
    attempts: list[Any] = []

    def create(current: Any, *unused: Any) -> SimpleNamespace:
        assert current.additional_config == {"other": {"value": 3}}
        current.additional_config["native_key"] = len(attempts)
        attempts.append(current)
        if len(attempts) == 1:
            raise RuntimeError("native startup failed")
        return SimpleNamespace(additional_config=current.additional_config)

    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", create)
    monkeypatch.setattr(arg_utils_gr, "engine_config_service", BackendConfigService)
    monkeypatch.setattr(planning, "admit_backend_plan", lambda config: None)
    with pytest.raises(RuntimeError, match="startup failed"):
        arg_utils_gr.create_engine_config(args)
    first = arg_utils_gr.create_engine_config(args)
    second = arg_utils_gr.create_engine_config(args)
    assert first.additional_config == {**original, "native_key": 1}
    assert second.additional_config == {**original, "native_key": 2}
    assert args.additional_config == original
