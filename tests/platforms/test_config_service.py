# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from types import SimpleNamespace

import pytest

from tests.platforms.test_backend_plan import Provider, config
from vllm_gr.platforms import loader
from vllm_gr.platforms.config import (
    BackendConfigService,
    engine_config_service,
    request_config_service,
)
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.planning import admit_backend_plan, normalize_backend_request

pytestmark = [pytest.mark.l1, pytest.mark.cpu_test]


def test_admission_uses_active_external_provider_on_shared_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Policy(BackendConfigService):
        default_worker_decision = True

    provider = Provider()
    monkeypatch.setattr(provider, "create_config_service", Policy, raising=False)
    native = config()
    native.device_config = SimpleNamespace(device="cuda:0")
    monkeypatch.setattr(loader, "activate_current_backend", lambda: provider)
    # Invalid CUDA-only input must not affect an unrelated provider sharing placement.
    monkeypatch.setenv("VLLM_GR_DENSE_PREFILL", "fa2")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "invalid")
    assert isinstance(engine_config_service(), Policy)
    plan = admit_backend_plan(native)
    assert plan is not None
    assert "dense_prefill_request" not in plan.request.native.to_dict()
    assert isinstance(request_config_service(native), Policy)
    monkeypatch.setattr(loader, "activate_current_backend", lambda: CUDA_BACKEND)
    with pytest.raises(ValueError, match="differs from the active provider"):
        request_config_service(native)


def test_worker_reuses_admitted_cuda_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    native = config()
    monkeypatch.setattr(loader, "activate_current_backend", lambda: CUDA_BACKEND)
    monkeypatch.setenv("VLLM_GR_DENSE_PREFILL", "fa2")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    plan = admit_backend_plan(native)
    assert plan is not None
    monkeypatch.setenv("VLLM_GR_DENSE_PREFILL", "off")
    request = normalize_backend_request(native, CUDA_BACKEND, admitted=plan)
    assert request == plan.request


def test_bsnd_graph_options_belong_to_the_ascend_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.platforms.ascend import NPU_BACKEND
    from vllm_gr.platforms.ascend.graph import AscendGraph

    key = "VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS"
    monkeypatch.setenv(key, "128,1")
    native = config()
    for provider in (CUDA_BACKEND, Provider()):
        assert (
            key not in normalize_backend_request(native, provider).native.to_dict()["graph_options"]
        )
    monkeypatch.setattr(loader, "activate_current_backend", lambda: NPU_BACKEND)
    plan = admit_backend_plan(native)
    assert plan is not None
    assert plan.choices.to_dict()["graph_options"][key] == "128,1"
    monkeypatch.setenv(key, "256,2")
    assert normalize_backend_request(native, NPU_BACKEND, admitted=plan) == plan.request
    runner = SimpleNamespace(vllm_config=native)
    assert AscendGraph(runner, plan).options[key] == "128,1"
    assert AscendGraph(runner, None).options[key] == "256,2"
