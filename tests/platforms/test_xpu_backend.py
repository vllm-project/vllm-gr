# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""XPU provider admission and independent attention/Beam numerical contracts."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_gr.config import GRAttentionBackend, GRConfig
from vllm_gr.config.models import DeploymentOptions
from vllm_gr.platforms.identity import PlatformIdentity
from vllm_gr.platforms.xpu.provider import XPU_BACKEND, XPUConfigService


@pytest.mark.l1
@pytest.mark.cpu_test
def test_xpu_identity_is_not_cuda_placement() -> None:
    identity = PlatformIdentity("XPU", "xpu", "xpu", "vllm.platforms.xpu.XPUPlatform", ())
    assert XPU_BACKEND.matches(identity)
    assert not XPU_BACKEND.matches(PlatformIdentity("CUDA", "cuda", "cuda", "native", ()))
    assert not XPU_BACKEND.matches(PlatformIdentity("OOT", "xpu", "xpu", "external", ()))


@pytest.mark.l1
@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "field,value",
    [
        ("enforce_eager", False),
        ("cache_dtype", "fp8"),
        ("beam_graph_enabled", True),
        ("attention_backend", None),
        ("prefill_graph_enabled", True),
    ],
)
def test_xpu_admission_rejects_unsupported_execution(field: str, value: object) -> None:
    native = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    settings = {
        "beam_max_width": 4,
        "beam_worker_decision": True,
        "attention_backend": GRAttentionBackend.CUSTOM,
        "beam_graph_enabled": False,
        "deployment": DeploymentOptions(prefill_graph_enabled=False),
    }
    if field == "enforce_eager":
        native.model_config.enforce_eager = value
    elif field == "cache_dtype":
        native.cache_config.cache_dtype = value
    elif field == "prefill_graph_enabled":
        settings["deployment"] = DeploymentOptions(prefill_graph_enabled=True)
    else:
        settings[field] = value
    with pytest.raises(ValueError, match="XPU Beam"):
        XPUConfigService().validate_engine(native, GRConfig(**settings))


@pytest.mark.l1
@pytest.mark.cpu_test
def test_xpu_admission_rejects_native_v1_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    native = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )
    gr = GRConfig(
        beam_max_width=4,
        beam_worker_decision=True,
        attention_backend=GRAttentionBackend.CUSTOM,
        beam_graph_enabled=False,
        deployment=DeploymentOptions(prefill_graph_enabled=False),
    )
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=1"):
        XPUConfigService().validate_engine(native, gr)


@pytest.mark.l2
@pytest.mark.cpu_test
def test_xpu_prefix_suffix_attention_matches_dense_gqa() -> None:
    from vllm_gr.v1.attention.backends.beam_attn_xpu import beam_attention

    generator = torch.Generator().manual_seed(7)
    query = torch.randn(3, 4, 8, generator=generator)
    prefix_k = torch.randn(5, 2, 8, generator=generator)
    prefix_v = torch.randn(5, 2, 8, generator=generator)
    suffix_k = torch.randn(3, 2, 2, 8, generator=generator)
    suffix_v = torch.randn(3, 2, 2, 8, generator=generator)
    actual = beam_attention(query, prefix_k, prefix_v, suffix_k, suffix_v, 0.3)
    expected = []
    for beam in range(3):
        heads = []
        for head in range(4):
            keys = torch.cat((prefix_k[:, head // 2], suffix_k[beam, :, head // 2]))
            values = torch.cat((prefix_v[:, head // 2], suffix_v[beam, :, head // 2]))
            probabilities = (keys @ query[beam, head] * 0.3).softmax(dim=0)
            heads.append(probabilities @ values)
        expected.append(torch.stack(heads))
    torch.testing.assert_close(actual, torch.stack(expected), atol=1e-5, rtol=1e-5)


@pytest.mark.l1
@pytest.mark.cpu_test
def test_xpu_selection_normalizes_full_vocab_and_recombines_parents() -> None:
    from vllm_gr.v1.worker.xpu.runner import select_candidates

    logits = torch.tensor([[5.0, 4.0, 3.0], [2.0, 2.0, 2.0]])
    scores = torch.tensor([-3.0, -0.1])
    tokens, parents, actual = select_candidates(logits, scores, 3, 3, 0.0)
    assert parents.tolist() == [1, 1, 1]
    assert tokens.tolist() == [0, 1, 2]
    torch.testing.assert_close(actual, torch.full((3,), -0.1 - torch.log(torch.tensor(3.0))))
    # Uniform ties follow stable token order; temperature zero does not divide
    # logits by zero or turn Beam scoring into greedy-only sampling.
    tokens, parents, _ = select_candidates(torch.zeros(1, 4), torch.zeros(1), 3, 3, 0.0)
    assert tokens.tolist() == [0, 1, 2]
    assert parents.tolist() == [0, 0, 0]


@pytest.mark.l1
@pytest.mark.cpu_test
def test_xpu_selector_uses_provider_and_preserves_native_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.selector as native

    from vllm_gr.platforms import registry
    from vllm_gr.v1.attention import selector_patch

    selected = object()
    monkeypatch.setattr(
        registry,
        "get_active_backend",
        lambda: SimpleNamespace(get_attention_backend=lambda: selected),
    )
    calls = []

    def original(backend: Any, attn_selector_config: Any, num_heads: int | None = None) -> str:
        calls.append((backend, attn_selector_config, num_heads))
        return "native"

    monkeypatch.setattr(native, "_cached_get_attn_backend", original)
    selector_patch.apply_custom_attention_backend_patch()
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert native._cached_get_attn_backend(AttentionBackendEnum.CUSTOM, None) is selected
    assert native._cached_get_attn_backend("native", None, num_heads=4) == "native"
    assert calls == [("native", None, 4)]


@pytest.mark.l1
@pytest.mark.cpu_test
def test_xpu_prefill_adapts_native_staging_without_mutating_gr_arrays() -> None:
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    from tests.helpers.beam_requests import make_batch_request
    from vllm_gr.v1.worker.xpu.runner import prepare_native_prefill

    @dataclass
    class Scheduled:
        scheduled_new_reqs: list[Any]

    source = make_batch_request().items[0].request
    native = Request.from_engine_core_request(source, None)
    data = NewRequestData.from_request(native, ([0],), source.prompt_token_ids)
    original = Scheduled([data])
    prepared = prepare_native_prefill(original)
    assert prepared.scheduled_new_reqs[0].prefill_token_ids == [1, 2, 3]
    assert prepared.scheduled_new_reqs[0].block_ids == data.block_ids
    prepared.scheduled_new_reqs[0].prefill_token_ids.append(99)
    assert original.scheduled_new_reqs[0] is data
    np.testing.assert_array_equal(data.prefill_token_ids, [1, 2, 3])
    np.testing.assert_array_equal(source.prompt_token_ids, [1, 2, 3])


@pytest.mark.l2
@pytest.mark.cpu_test
def test_xpu_owner_reorders_duplicate_parents_and_releases_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.helpers.beam_requests import make_batch_request
    from vllm_gr.v1.beam.dispatch import GR_WORKER_RESULT_ATTR, GRDeviceStateRef
    from vllm_gr.v1.worker.xpu.runner import XPUBeamStageRunner

    @dataclass
    class Scheduled:
        finished_req_ids: set[str]

    finished: list[set[str]] = []
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        finish_requests=lambda output: finished.append(output.finished_req_ids),
        execute_model_state=object(),
    )
    owner = XPUBeamStageRunner.__new__(XPUBeamStageRunner)
    owner.worker = worker
    owner.request = make_batch_request()
    owner.width = 4
    owner.sequence = torch.tensor([[10], [20], [30], [40]])
    owner.parents = torch.zeros(4, 1, dtype=torch.long)
    owner.scores = torch.tensor([-20.0, -20.0, 0.0, -20.0])
    owner.closed = False
    owner.execution_error = None
    suffix = torch.arange(8).reshape(4, 1, 1, 2).float()
    prefix = torch.ones(2, 1, 2)
    owner.layer_kv = {"layer": (prefix, prefix, suffix.clone(), suffix.clone())}
    metadata = SimpleNamespace(
        beam_params=owner.request.beam_params,
        session_id=owner.request.batch_id,
        request_ids=(owner.request.batch_id,),
        stage_index=1,
        dispatch_id=2,
        output_device_state=GRDeviceStateRef(owner.request.batch_id, 1),
        is_last_stage=False,
        owner_generation=7,
    )
    owner.pending = (metadata, torch.zeros(4, 4))
    scheduled = Scheduled(set())
    output = owner.sample(scheduled)
    assert owner.sequence.tolist() == [[30, 0], [30, 1], [30, 2], [30, 3]]
    assert owner.parents[:, -1].tolist() == [2, 2, 2, 2]
    for cache in owner.layer_kv["layer"][2:]:
        torch.testing.assert_close(cache, suffix[2:3].expand_as(cache))
    assert owner.layer_kv["layer"][0] is prefix
    assert not getattr(output, GR_WORKER_RESULT_ATTR)[owner.request.batch_id].finished

    monkeypatch.setattr(torch.xpu, "synchronize", lambda *_: None)
    metadata.is_last_stage = True
    owner.pending = (metadata, torch.zeros(4, 4))
    final = owner.sample(scheduled)
    result = getattr(final, GR_WORKER_RESULT_ATTR)[owner.request.batch_id]
    assert result.finished and result.resources_released
    assert result.terminal_result.output.items[0].sequences[0].token_ids[:2] == (30, 0)
    assert finished == [{owner.request.batch_id}]
    assert owner.closed and owner.pending is None and not owner.layer_kv
    assert owner.sequence.numel() == owner.parents.numel() == owner.scores.numel() == 0
    assert worker.execute_model_state is None
