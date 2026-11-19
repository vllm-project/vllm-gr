# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from types import SimpleNamespace
from typing import Any

import pytest

from tests.platforms.test_backend_plan import Provider, config
from vllm_gr.config import gr_config_from_mapping
from vllm_gr.platforms import loader
from vllm_gr.platforms.config import (
    BackendConfigService,
    engine_config_service,
    request_config_service,
)
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.planning import admit_backend_plan, normalize_backend_request

pytestmark = [pytest.mark.l1, pytest.mark.cpu_test]


def configured_native(**settings: Any) -> Any:
    native = config()
    gr_config_from_mapping(
        {"gr": {"beam_width": [16], "candidate_width": [16], **settings}}
    ).write_to_vllm_config(native)
    return native


def test_admission_uses_active_external_provider_on_shared_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Policy(BackendConfigService):
        default_worker_decision = True

    provider = Provider()
    monkeypatch.setattr(provider, "create_config_service", Policy, raising=False)
    native = configured_native(dense_prefill="fa2")
    native.device_config = SimpleNamespace(device="cuda:0")
    monkeypatch.setattr(loader, "activate_current_backend", lambda: provider)
    # Invalid CUDA-only input must not affect an unrelated provider sharing placement.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "invalid")
    assert isinstance(engine_config_service(), Policy)
    plan = admit_backend_plan(native)
    assert plan is not None
    assert "dense_prefill_batch_invariant" not in plan.request.native.to_dict()
    assert isinstance(request_config_service(native), Policy)
    monkeypatch.setattr(loader, "activate_current_backend", lambda: CUDA_BACKEND)
    with pytest.raises(ValueError, match="differs from the active provider"):
        request_config_service(native)


def test_worker_reuses_admitted_cuda_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    native = configured_native(dense_prefill="fa2")
    monkeypatch.setattr(loader, "activate_current_backend", lambda: CUDA_BACKEND)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    plan = admit_backend_plan(native)
    assert plan is not None
    assert plan.request.native.to_dict()["dense_prefill_batch_invariant"] is True
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    request = normalize_backend_request(native, CUDA_BACKEND, admitted=plan)
    assert request == plan.request


def test_ascend_graph_specs_follow_each_engine_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.platforms.ascend import NPU_BACKEND

    monkeypatch.setattr(loader, "activate_current_backend", lambda: NPU_BACKEND)
    plans = []
    for spec in ("128,1", "256,2"):
        native = configured_native(npu_bsnd_capture_specs=spec)
        plan = admit_backend_plan(native)
        assert plan is not None
        assert normalize_backend_request(native, NPU_BACKEND, admitted=plan) == plan.request
        plans.append(plan)
    assert plans[0].fingerprint != plans[1].fingerprint
    assert [
        plan.request.config.to_dict()["deployment"]["npu_bsnd_capture_specs"] for plan in plans
    ] == ["128,1", "256,2"]
