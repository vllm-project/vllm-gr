# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""A warmed graph must survive B changes without replaying stale inputs."""

import gc
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
import torch

from tests.kernels.beam_stage_test_utils import (
    admission_worker,
    dispatch,
    install_decode_model,
    metadata,
)
from vllm_gr.v1.engine.gr_async_scheduler import GR_WORKER_RESULT_ATTR
from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeGraph
from vllm_gr.v1.worker.gpu_beam_stage_runner import get_stage_runner
from vllm_gr.v1.worker.gpu_beam_stage_runner_v1 import CudaStageHooksV1

pytestmark = [
    pytest.mark.gpu_test,
    pytest.mark.l2,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]


def make_worker(dtype: torch.dtype, graph: bool, constrained: bool) -> tuple[Any, torch.Tensor]:
    worker = admission_worker(dtype=dtype, max_rows=16)
    worker.model_config.enforce_eager = not graph
    worker._vllm_gr_config.beam_graph_enabled = graph
    worker._vllm_gr_config.beam_adapter_cache_enabled = True
    if not constrained:
        worker.constraint_backend = None
        worker._vllm_gr_config.constrained_generation = False
    base = torch.tensor([0.1, 0.7, -0.3, 0.2, 0.5, -0.9], device="cuda", dtype=dtype)
    weights = torch.arange(6, device="cuda", dtype=dtype) * 0.031
    install_decode_model(
        worker,
        lambda ids, positions: base + ids[:, None] * weights + positions[:, None] * weights * 0.01,
    )
    return worker, base


def run(
    worker: Any,
    base: torch.Tensor,
    batch: int,
    iteration: int,
    widths: tuple[int, ...] | None,
    factors: tuple[int, ...] | None = None,
) -> tuple[Any, Any]:
    for stage in range(3):
        m = metadata(stage)
        lengths = tuple(5 + iteration % 7 + i for i in range(batch))
        m = replace(
            m,
            request_ids=tuple(f"session-0:{i}" for i in range(batch)),
            kv_blocks=tuple((i + 1,) for i in range(batch)),
            num_prompt_tokens=sum(lengths),
            num_prompt_tokens_by_request=lengths,
            beam_params=msgspec.structs.replace(
                m.beam_params, dynamic_beam_widths=widths, per_parent_beam_widths=factors
            ),
        )
        scheduled = dispatch(m)
        scheduled.num_scheduled_tokens = dict(
            zip(m.request_ids, lengths if stage == 0 else (1,) * batch)
        )
        owner = get_stage_runner(worker, scheduled, hooks_type=CudaStageHooksV1)

        def prefill(w: Any, scheduled: Any) -> None:
            w.execute_model_state = SimpleNamespace(logits=base.repeat(batch, 1))

        owner.execute(scheduled, prefill)
        result = getattr(owner.sample().get_output(), GR_WORKER_RESULT_ATTR)[m.session_id]
    assert result.resources_released and owner.closed
    assert not worker._beam_sessions
    assert worker._beam_context.registry.num_active_sessions == 0
    return owner, result.terminal_result.output


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
@pytest.mark.parametrize("widths", [None, (2, 3, 4)])  # type: ignore[misc]
@pytest.mark.parametrize("constrained", [False, True])  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_batch_switch_reuses_graphs_and_updates_inputs(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    widths: tuple[int, ...] | None,
    constrained: bool,
) -> None:
    gc.collect()
    worker, base = make_worker(dtype, True, constrained)
    reference, reference_base = make_worker(dtype, False, constrained)
    captures: list[Any] = []
    original = BeamDecodeGraph.capture

    def capture(self: BeamDecodeGraph, *args: Any, **kwargs: Any) -> None:
        if not self.ready:
            captures.append(self.signature)
        original(self, *args, **kwargs)

    monkeypatch.setattr(BeamDecodeGraph, "capture", capture)
    owners: dict[int, Any] = {}
    for iteration, batch in enumerate((1, 2, 3, 4, 1, 4, 2, 3, 1, 2, 4, 3)):
        before = len(captures)
        owner, output = run(worker, base, batch, iteration, widths)
        _, expected = run(reference, reference_base, batch, iteration, widths)
        assert output == expected
        if batch in owners:
            assert owner is owners[batch]
            assert len(captures) == before
        else:
            owners[batch] = owner
            assert len(captures) - before == (1 if widths is None else 2)
        assert len(worker._gr_gpu_stage_runner_cache) <= 4
    assert set(worker._gr_gpu_stage_runner_cache) == {1, 2, 3, 4}


@torch.inference_mode()  # type: ignore[misc]
def test_changed_schedule_discards_old_cache_and_recaptures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gc.collect()
    worker, base = make_worker(torch.float16, True, True)
    first, _ = run(worker, base, 1, 0, (2, 3, 4))
    run(worker, base, 2, 1, (2, 3, 4))
    assert len(worker._gr_gpu_stage_runner_cache) == 2
    owner, _ = run(worker, base, 1, 2, (3, 2, 4))
    assert owner is not first
    assert set(worker._gr_gpu_stage_runner_cache) == {1}
    assert all(sig.batch_size == 1 for sig in worker._beam_decode_graph._bindings)


@torch.inference_mode()  # type: ignore[misc]
def test_candidate_width_change_keeps_model_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    gc.collect()
    worker, base = make_worker(torch.float16, True, True)
    reference, reference_base = make_worker(torch.float16, False, True)
    captures: list[Any] = []
    original = BeamDecodeGraph.capture

    def capture(self: BeamDecodeGraph, *args: Any, **kwargs: Any) -> None:
        if not self.ready:
            captures.append(self.signature)
        original(self, *args, **kwargs)

    monkeypatch.setattr(BeamDecodeGraph, "capture", capture)
    owners: dict[int, Any] = {}
    for iteration, (batch, factors) in enumerate(((1, (2, 2, 2)), (2, (2, 3, 3)), (1, (2, 3, 3)))):
        owner, output = run(worker, base, batch, iteration, (2, 3, 4), factors)
        _, expected = run(reference, reference_base, batch, iteration, (2, 3, 4), factors)
        assert output == expected
        if batch in owners:
            assert owner is owners[batch]
            assert len(worker._beam_decode_graph._bindings) == 4
        owners[batch] = owner
        assert len(captures) == 2 * len(owners)


@torch.inference_mode()  # type: ignore[misc]
def test_cancelled_adapter_can_be_reused_after_another_batch() -> None:
    gc.collect()
    worker, base = make_worker(torch.float16, True, True)
    m = metadata(0)
    m = replace(
        m, beam_params=msgspec.structs.replace(m.beam_params, dynamic_beam_widths=(2, 3, 4))
    )
    scheduled = dispatch(m)
    owner = get_stage_runner(worker, scheduled, hooks_type=CudaStageHooksV1)

    def prefill(w: Any, scheduled: Any) -> None:
        w.execute_model_state = SimpleNamespace(logits=base[None, :])

    owner.execute(scheduled, prefill)
    owner.sample().get_output()
    owner.cancel(owner.binding)
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)
    run(worker, base, 2, 1, (2, 3, 4))
    restored, output = run(worker, base, 1, 2, (2, 3, 4))
    assert restored is owner
    reference, reference_base = make_worker(torch.float16, False, True)
    _, expected = run(reference, reference_base, 1, 2, (2, 3, 4))
    assert output == expected


@torch.inference_mode()  # type: ignore[misc]
def test_resource_change_is_rejected_before_selecting_cached_batch() -> None:
    gc.collect()
    worker, base = make_worker(torch.float16, True, True)
    first, _ = run(worker, base, 1, 0, (2, 3, 4))
    current, _ = run(worker, base, 2, 1, (2, 3, 4))
    current.beam.workspace.selected_token_ids = current.beam.workspace.selected_token_ids.clone()
    with pytest.raises(ValueError, match="resource addresses or layouts changed"):
        run(worker, base, 1, 2, (2, 3, 4))
    assert current.closed and first.closed
    assert not worker._beam_sessions
    assert set(worker._gr_gpu_stage_runner_cache) == {1, 2}


@torch.inference_mode()  # type: ignore[misc]
def test_incompatible_cached_adapter_invalidates_its_old_graphs() -> None:
    gc.collect()
    worker, base = make_worker(torch.float16, True, True)
    first, _ = run(worker, base, 1, 0, (2, 3, 4))
    run(worker, base, 2, 1, (2, 3, 4))
    first.beam.runners[0]._scratch_owner = object()
    restored, output = run(worker, base, 1, 2, (2, 3, 4))
    assert restored is not first
    assert set(worker._gr_gpu_stage_runner_cache) == {1}
    reference, reference_base = make_worker(torch.float16, False, True)
    _, expected = run(reference, reference_base, 1, 2, (2, 3, 4))
    assert output == expected


@pytest.mark.parametrize("explicit_false", [False, True])  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_cache_disabled_keeps_latest_adapter_and_cuda_graphs(
    monkeypatch: pytest.MonkeyPatch, explicit_false: bool
) -> None:
    worker, base = make_worker(torch.float16, True, True)
    if explicit_false:
        worker._vllm_gr_config.beam_adapter_cache_enabled = False
    else:
        del worker._vllm_gr_config.beam_adapter_cache_enabled
    reference, reference_base = make_worker(torch.float16, False, True)
    captures: list[Any] = []
    original = BeamDecodeGraph.capture

    def capture(self: BeamDecodeGraph, *args: Any, **kwargs: Any) -> None:
        if not self.ready:
            captures.append(self.signature)
        original(self, *args, **kwargs)

    monkeypatch.setattr(BeamDecodeGraph, "capture", capture)
    previous = None
    for i, (batch, expected_captures) in enumerate(((1, 2), (1, 2), (2, 4), (1, 6))):
        owner, output = run(worker, base, batch, i, (2, 3, 4))
        _, expected = run(reference, reference_base, batch, i, (2, 3, 4))
        assert output == expected
        assert len(captures) == expected_captures
        assert not getattr(worker, "_gr_gpu_stage_runner_cache", {})
        assert owner.graph is not None and owner.graph.ready
        if i == 1:
            assert owner is previous  # Same-B reuse is independent of the new cache.
        previous = owner
