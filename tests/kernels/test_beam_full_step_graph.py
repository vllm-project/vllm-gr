# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State preservation and real CUDA candidate selection through a full step."""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.test_gpu_beam_stage_runner import (
    dispatch,
    metadata,
    prepare_binding,
    runtime,
)
from vllm_gr.v1.worker.beam_full_step_graph import FullBeamStepGraph

pytestmark = [
    pytest.mark.gpu_test,
    pytest.mark.l2,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]


def test_capture_preserves_state_and_replay_reads_updated_inputs() -> None:
    value = torch.ones(4, device="cuda")
    step = torch.zeros(1, device="cuda")

    def operation() -> None:
        value.add_(step + 2)
        step.add_(1)

    graph = FullBeamStepGraph(operation, (value, step))
    assert torch.equal(value, torch.ones_like(value))
    assert step.item() == 0
    graph.replay()
    assert torch.equal(value, torch.full_like(value, 3))
    assert step.item() == 1
    value.fill_(10)
    graph.replay()
    assert torch.equal(value, torch.full_like(value, 13))
    assert step.item() == 2


def test_warmup_failure_restores_admitted_state() -> None:
    value = torch.ones(4, device="cuda")

    def fail() -> None:
        value.add_(2)
        raise ValueError("warmup failed")

    with pytest.raises(ValueError, match="warmup failed"):
        FullBeamStepGraph(fail, (value,))
    assert torch.equal(value, torch.ones_like(value))


def test_private_pools_allow_alternating_replay_order() -> None:
    value = torch.ones(4, device="cuda")

    def add() -> None:
        intermediate = value + 2
        value.copy_(intermediate)

    def multiply() -> None:
        intermediate = value * 3
        value.copy_(intermediate)

    first = FullBeamStepGraph(add, (value,))
    second = FullBeamStepGraph(multiply, (value,))
    assert first.graph.pool() != second.graph.pool()
    second.replay()
    first.replay()
    second.replay()
    assert torch.equal(value, torch.full_like(value, 15))


def test_warmup_and_capture_use_inference_mode() -> None:
    value = torch.ones(4, device="cuda")
    modes: list[bool] = []

    def operation() -> None:
        modes.append(torch.is_inference_mode_enabled())
        value.add_(1)

    graph = FullBeamStepGraph(operation, (value,))
    assert modes == [True, True, True]
    graph.replay()
    assert torch.equal(value, torch.full_like(value, 2))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_full_step_matches_eager_candidate_state_and_kv(dtype: torch.dtype) -> None:
    owners = [runtime(dtype), runtime(dtype)]
    generator = torch.Generator(device="cuda").manual_seed(42)
    prefill = torch.randn((1, 6), generator=generator, dtype=dtype, device="cuda")
    logits = torch.randn((4, 6), generator=generator, dtype=dtype, device="cuda")
    for index, owner in enumerate(owners):
        owner.admit(dispatch(metadata(0)))
        owner.logits = prefill
        owner.sample().get_output()
        owner.worker.model_config.enforce_eager = False
        owner.worker.model = SimpleNamespace()
        owner.enable_full_step_graph = index == 1
        owner._forward_hidden = lambda _capture: logits
        owner._decode_logits = lambda hidden: hidden
        for pool in owner.beam.session.context.pools:
            pool.unshared_key.fill_(2)
            pool.unshared_value.fill_(3)
        execution = owner.admit(dispatch(metadata(1)))
        entry = execution.entries[0]
        prepare_binding(owner, entry)
        owner.logits = logits
        assert owner._try_full_step(entry) is (index == 1)
        if index == 1:
            captured = next(iter(owner._full_step_graphs.values()))
            assert captured.logits is owner.logits
            assert any(binding is owner.metadata for binding in captured.bindings)
        owner.sample().get_output()
    for name, value in vars(owners[0].beam.state).items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, getattr(owners[1].beam.state, name)), name
    for first, second in zip(
        owners[0].beam.session.context.pools, owners[1].beam.session.context.pools
    ):
        assert torch.equal(first.unshared_key, second.unshared_key)
        assert torch.equal(first.unshared_value, second.unshared_value)


def test_dynamic_width_does_not_launch_fixed_width_full_graph() -> None:
    owner = runtime()
    owner.enable_full_step_graph = True
    owner.dynamic = SimpleNamespace()
    entry = owner.admit(dispatch(metadata(0))).entries[0]
    assert not owner._try_full_step(entry)
    assert not owner._full_step_graphs
