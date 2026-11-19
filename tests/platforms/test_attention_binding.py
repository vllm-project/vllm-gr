# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host conformance for runner-owned attention and cold native selector queries."""

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from tests.helpers.paths import REPO_ROOT
from vllm_gr.platforms import loader
from vllm_gr.platforms.ascend.attention import AscendAttention
from vllm_gr.platforms.attention import (
    AttentionExecutor,
    admission_for_config,
    bind_attention,
    for_config,
    loading_attention,
    validate_beam_attention,
    validate_selected_attention,
)
from vllm_gr.platforms.binding import _close_services
from vllm_gr.platforms.plan import (
    BACKEND_PLAN_KEY,
    BackendPlan,
    BackendRequest,
    PlanData,
    ServiceRequirements,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize(
    "path",
    [
        "platforms/attention.py",
        "v1/attention/selector_patch.py",
        "v1/attention/backends/beam_attn.py",
        "v1/attention/backends/beam_attn_metadata.py",
        "v1/attention/backends/beam_attn_metadata_tensors.py",
    ],
)
def test_shared_attention_dispatch_uses_the_selected_provider(path: str) -> None:
    tree = ast.parse((REPO_ROOT / "vllm_gr" / path).read_text(encoding="utf-8"))
    hardware = {"ascend", "cuda", "gpu", "npu", "rocm", "xpu", "triton"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value.lower() not in hardware, (path, node.lineno)
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"is_cuda", "is_cuda_alike", "is_npu", "is_rocm", "is_xpu"}, (
                path,
                node.lineno,
            )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                assert hardware.isdisjoint(name.lower().replace(".", "_").split("_")), (
                    path,
                    node.lineno,
                )


class Executor(AttentionExecutor):
    def __init__(self, device: Any) -> None:
        super().__init__(SimpleNamespace())
        self.device = device
        self.closes = 0

    def close(self) -> None:
        self.closes += 1
        super().close()


class Provider:
    backend_id = "demo"
    api_version = 2
    plan_api_version = 1
    adapter_revision = "test"

    def __init__(self) -> None:
        self.created: list[Executor] = []

    def attention_admission(self, *, plan: BackendPlan) -> object:
        return self

    def create_attention_executor(self, *, plan: BackendPlan, device: Any) -> Executor:
        assert device is not None
        service = Executor(device)
        self.created.append(service)
        return service

    def resolve(self, *args: Any) -> None:
        pass

    def validate_worker(self, *args: Any) -> None:
        pass

    def create_model_runner_adapter(self, *args: Any) -> None:
        pass

    def create_constrained_topk_execution(self, *args: Any) -> None:
        pass


def setup_provider(monkeypatch: pytest.MonkeyPatch) -> Any:
    provider = Provider()
    request = BackendRequest(PlanData({}), PlanData({}), PlanData({}), PlanData({}))
    plan = BackendPlan("demo", 2, request, PlanData({"requires_attention": True}))
    config = SimpleNamespace(
        additional_config={BACKEND_PLAN_KEY: plan.to_wire()},
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(loader, "activate_current_backend", lambda: provider)
    return provider, plan, config


def test_planned_attention_requirement_binds_layers_without_a_choice_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, plan, config = setup_provider(monkeypatch)
    plan = replace(plan, choices=PlanData({}), requirements=ServiceRequirements(attention="demo"))
    config.additional_config[BACKEND_PLAN_KEY] = plan.to_wire()
    runner = SimpleNamespace(vllm_config=config, device="test")
    with loading_attention(runner):
        layer_executor = for_config()
        assert bind_attention(runner, provider, plan) == (layer_executor, layer_executor)
    assert provider.created == [layer_executor]


def test_early_selector_and_two_owners_sharing_native_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, plan, config = setup_provider(monkeypatch)
    assert for_config(config) is provider
    assert provider.created == []
    layers = config.compilation_config.static_forward_context
    owners = []
    for name in ("first", "second"):
        runner = SimpleNamespace(vllm_config=config, device=name)
        with loading_attention(runner):
            executor = for_config(config)
            assert for_config() is executor
            layers[name] = SimpleNamespace(impl=SimpleNamespace(_execution=executor))
            assert bind_attention(runner, provider, plan) == (executor, executor)
        assert for_config(config, layer_names=[name]) is executor
        owners.append(executor)
    assert owners == provider.created
    assert owners[0] is not owners[1]
    with pytest.raises(RuntimeError, match="multiple runner bindings"):
        for_config(config, layer_names=list(layers))
    _close_services([owners[0]])
    assert [owner.closes for owner in owners] == [1, 0]


def test_failed_model_load_releases_only_provisional_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, plan, config = setup_provider(monkeypatch)
    runner = SimpleNamespace(vllm_config=config, device="test")
    with pytest.raises(ValueError, match="weights"), loading_attention(runner):
        failed = for_config()
        raise ValueError("weights")
    assert failed.closes == 1
    assert not hasattr(runner, "_gr_loading_attention")
    with loading_attention(runner):
        recovered = for_config()
        assert bind_attention(runner, provider, plan) == (recovered, recovered)
    assert recovered is not failed
    assert recovered.closes == 0


def test_primitive_provider_selector_never_creates_a_device_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _, config = setup_provider(monkeypatch)
    primitives = object()
    monkeypatch.setattr(provider, "attention_admission", None)
    monkeypatch.setattr(provider, "get_attention_execution", lambda: primitives, raising=False)
    assert for_config(config).execution_module() is primitives
    assert provider.created == []


def test_explicit_ascend_decode_and_physical_layout_are_independent() -> None:
    from vllm_gr.platforms.cuda.attention import CudaAttention

    cascade = AscendAttention("cascade", kv_layout="LBHNC")
    fused = AscendAttention("xllm", kv_layout="LBNHC")
    assert cascade.decode_mode() == "cascade"
    assert fused.decode_mode() == "xllm"
    assert cascade.kv_cache_shape(4, 16, 2, 64) == (2, 4, 2, 16, 64)
    assert fused.kv_cache_shape(4, 16, 2, 64) == (2, 4, 16, 2, 64)
    assert CudaAttention().supports_sliding_window()
    assert not cascade.supports_sliding_window()


def test_invalidate_releases_only_registered_runner_geometry() -> None:
    class Builder:
        def __init__(self) -> None:
            self.cached = True

        def invalidate_geometry(self) -> Any:
            self.cached = False

    a, b = AttentionExecutor(), AttentionExecutor()
    first, second = Builder(), Builder()
    a.track_builder(first)
    b.track_builder(second)
    a.close()
    a.close()
    assert not first.cached
    assert second.cached


@pytest.mark.parametrize(("available", "mode"), [(True, "xllm"), (False, "cascade")])
def test_auto_decode_uses_the_worker_capability_once(monkeypatch, available, mode):
    import sys
    from types import ModuleType

    package = ModuleType("vllm_gr.ops.npu")
    package.beam_ops = SimpleNamespace(is_available=lambda: available)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    executor = AscendAttention("auto")
    package.beam_ops.is_available = lambda: not available
    assert executor.decode_mode() == mode


def test_selector_validation_during_model_load_never_allocates_execution(monkeypatch):
    from vllm_gr.platforms.attention import selector_admission

    provider, plan, config = setup_provider(monkeypatch)
    runner = SimpleNamespace(vllm_config=config, device="test")
    with loading_attention(runner):
        assert admission_for_config() is provider
        with selector_admission():
            assert admission_for_config() is provider
        assert provider.created == []
        layer_executor = for_config()
        assert bind_attention(runner, provider, plan) == (layer_executor, layer_executor)
    assert provider.created == [layer_executor]


def test_default_suffix_writer_is_checked_only_for_incremental_beam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, plain, _ = setup_provider(monkeypatch)
    request = BackendRequest(
        PlanData({"beam_execution_mode": "v1"}), PlanData({}), PlanData({}), PlanData({})
    )
    beam = BackendPlan("demo", 2, request, PlanData({}))
    primitives = SimpleNamespace(
        is_available=lambda: True,
        get_flash_attn_version=lambda: None,
        supports_sinks=lambda: False,
        cascade_attention=lambda **kwargs: None,
        reshape_and_cache=lambda *args: None,
    )
    validate_beam_attention(primitives, plain)
    with pytest.raises(RuntimeError, match="cache_unshared_kv"):
        validate_beam_attention(primitives, beam)

    class CustomSuffixWriter(AttentionExecutor):
        def write_suffix_kv(
            self, pool: Any, key: Any, value: Any, session: Any, metadata: Any
        ) -> None:
            pool[session.slot] = (key, value)

    validate_beam_attention(CustomSuffixWriter(primitives), beam)
    for mode in (True, "shared"):
        prefill = BackendPlan("demo", 2, plain.request, PlanData({"prefill_graph": mode}))
        with pytest.raises(RuntimeError, match="prefill_attention"):
            validate_beam_attention(primitives, prefill)
    managed = BackendPlan("demo", 2, plain.request, PlanData({"prefill_graph": "provider"}))
    validate_beam_attention(primitives, managed)


def test_concrete_attention_impl_validates_its_own_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, plan, config = setup_provider(monkeypatch)

    class NativeImpl:
        @staticmethod
        def validate_gr_attention_executor(executor: Any, plan: BackendPlan) -> None:
            if not callable(getattr(executor, "vendor_forward", None)):
                raise RuntimeError("vendor_forward unavailable")

    config.compilation_config.static_forward_context["layer"] = SimpleNamespace(
        attn_backend=object(), impl=NativeImpl()
    )
    runner = SimpleNamespace(vllm_config=config)
    with pytest.raises(RuntimeError, match="vendor_forward"):
        validate_selected_attention(runner, object(), plan)
    validate_selected_attention(runner, SimpleNamespace(vendor_forward=lambda: 1), plan)


def test_dense_prefill_environment_only_belongs_to_cuda(monkeypatch):
    from vllm_gr.config import GRConfig
    from vllm_gr.config.models import DeploymentOptions
    from vllm_gr.platforms.ascend import NPU_BACKEND
    from vllm_gr.platforms.planning import normalize_backend_request

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "unused")
    config = SimpleNamespace(additional_config={})
    GRConfig(deployment=DeploymentOptions(dense_prefill="fa2")).write_to_vllm_config(config)
    for provider in (NPU_BACKEND, Provider()):
        request = normalize_backend_request(config, provider)
        assert "dense_prefill_batch_invariant" not in request.native.to_dict()


def test_cuda_dense_prefill_and_graph_replay_use_the_admitted_mode(monkeypatch):
    from vllm_gr.config import GRAttentionBackend, GRConfig
    from vllm_gr.config.models import DeploymentOptions
    from vllm_gr.platforms.cuda import CUDA_BACKEND
    from vllm_gr.platforms.planning import normalize_backend_request
    from vllm_gr.v1.attention.backends.beam_dense_prefill import (
        requires_fresh_prefill_replay,
    )

    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    owners = []
    for setting in ("cudnn", "off", "fa2"):
        config = SimpleNamespace(additional_config={})
        GRConfig(
            beam_max_width=4,
            attention_backend=GRAttentionBackend.CUSTOM,
            deployment=DeploymentOptions(dense_prefill=setting),
        ).write_to_vllm_config(config)
        request = normalize_backend_request(config, CUDA_BACKEND)
        plan = CUDA_BACKEND.resolve(request)
        config.additional_config[BACKEND_PLAN_KEY] = plan.to_wire()
        owners.append((SimpleNamespace(vllm_config=config), plan))

    assert len({plan.fingerprint for _, plan in owners}) == 3
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    for (runner, plan), mode in zip(owners, ("cudnn", None, "fa2"), strict=True):
        assert (
            CUDA_BACKEND.resolve(
                normalize_backend_request(runner.vllm_config, CUDA_BACKEND, admitted=plan)
            )
            == plan
        )
        assert requires_fresh_prefill_replay(runner) is (mode is not None)
        executor = CUDA_BACKEND.attention_admission(plan=plan)
        runner._gr_backend_binding = SimpleNamespace(attention_executor=executor)
        assert executor.dense_prefill_mode(runner.vllm_config) == mode
        assert requires_fresh_prefill_replay(runner) is (mode is not None)


@pytest.mark.parametrize("guard", ["prefix", "chunked", "connector", "invariant", "native"])
def test_cuda_dense_prefill_plan_disables_ineligible_paths(monkeypatch, guard):
    from vllm_gr.config import GRAttentionBackend, GRConfig
    from vllm_gr.config.models import DeploymentOptions
    from vllm_gr.platforms.cuda import CUDA_BACKEND
    from vllm_gr.platforms.planning import normalize_backend_request

    config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=object() if guard == "connector" else None,
        cache_config=SimpleNamespace(enable_prefix_caching=guard == "prefix"),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=guard == "chunked"),
    )
    backend = None if guard == "native" else GRAttentionBackend.CUSTOM
    GRConfig(
        beam_max_width=4,
        attention_backend=backend,
        deployment=DeploymentOptions(dense_prefill="fa2"),
    ).write_to_vllm_config(config)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1" if guard == "invariant" else "0")
    plan = CUDA_BACKEND.resolve(normalize_backend_request(config, CUDA_BACKEND))
    executor = CUDA_BACKEND.attention_admission(plan=plan)
    assert executor.dense_prefill_mode(config) is None
    assert not executor.requires_fresh_prefill
