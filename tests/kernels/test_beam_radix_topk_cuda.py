# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exact request-level Top-K comparisons against a stable CPU FP32 oracle."""

# mypy: disallow-untyped-decorators=False

from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
import torch

from tests.kernels.beam_runner_test_utils import make_request_runner as _runtime
from tests.kernels.test_beam_search_runner_cuda import _write_kv
from vllm_gr.ops.cuda.beam_radix_topk import (
    allocate_beam_radix_scratch,
    beam_radix_topk_out,
)
from vllm_gr.ops.cuda.beam_selection import (
    _hierarchical_topk_out,
    allocate_beam_selection_scratch,
)
from vllm_gr.v1.beam.attention_context import BeamAttentionContext
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.beam.session import BeamSearchSession
from vllm_gr.v1.worker.cuda import beam_search_runner
from vllm_gr.v1.worker.cuda.beam_search_runner import (
    BeamBatchSearchRunner,
    BeamCandidateBatch,
    BeamSearchRunner,
)
from vllm_gr.v1.worker.cuda.beam_state import BeamSearchState
from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import BeamPostprocessWorkspace

pytestmark = [
    pytest.mark.l2,
    pytest.mark.gpu_test,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]


def _fp32_reference(
    parent: torch.Tensor,
    local: torch.Tensor,
    valid: torch.Tensor | None,
    batch: int,
    width: int,
    candidate_width: int,
    winners: int,
    *,
    mask_invalid_indices: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = parent.cpu().float().reshape(batch, width, 1) + local.cpu().float().reshape(
        batch, width, candidate_width
    )
    scores = scores.reshape(batch, -1)
    available = (
        torch.ones_like(scores, dtype=torch.bool)
        if valid is None
        else valid.cpu().reshape(batch, -1)
    )
    scores = scores.masked_fill(scores.isnan() | ~available, -float("inf"))
    scores = torch.where(scores == 0.0, 0.0, scores)
    expected_scores = torch.full((batch, winners), -float("inf"), dtype=torch.float32)
    expected_indices = torch.full((batch, winners), -1, dtype=torch.int32)
    for request in range(batch):
        # The input order breaks score ties by ascending request-local flat index.
        indices = torch.arange(scores.shape[1])
        if mask_invalid_indices:
            indices = indices[available[request]]
        row_scores = scores[request, indices]
        order = torch.argsort(row_scores, descending=True, stable=True)[:winners]
        count = order.numel()
        expected_scores[request, :count] = row_scores[order]
        expected_indices[request, :count] = indices[order].to(torch.int32)
    return expected_scores, expected_indices


def _assert_result(
    result: tuple[torch.Tensor, torch.Tensor, int],
    reference: tuple[torch.Tensor, torch.Tensor],
) -> None:
    scores, indices, stride = result
    batch, winners = reference[0].shape
    assert scores.dtype == torch.float32 and indices.dtype == torch.int32
    assert scores.stride(0) == indices.stride(0) == stride
    actual_scores = scores[:batch, :winners].cpu().contiguous().view(torch.int32)
    assert torch.equal(indices[:batch, :winners].cpu(), reference[1])
    assert torch.equal(actual_scores, reference[0].view(torch.int32))


@pytest.mark.parametrize(
    "width,candidate_width,winners",
    [
        (128, 128, 128),
        (256, 256, 256),
        (512, 256, 512),
        (1024, 256, 1024),
        (131, 11, 127),
        (7, 3, 25),
    ],
)
def test_radix_matches_current_merge_and_cpu_fp32(
    width: int, candidate_width: int, winners: int
) -> None:
    batch = 2
    generator = torch.Generator().manual_seed(20261115)
    shape = (batch * width, candidate_width)
    parent = torch.randn(batch * width, generator=generator).cuda()
    local = torch.randn(shape, generator=generator).cuda()
    valid = (torch.rand(shape, generator=generator) > 0.1).cuda()
    args = (parent, local, valid, batch, width, candidate_width, winners)
    reference = _fp32_reference(*args, mask_invalid_indices=True)
    baseline = allocate_beam_selection_scratch(
        batch, width, candidate_width, winners, parent.device
    )
    radix = allocate_beam_radix_scratch(batch, width, candidate_width, winners, parent.device)
    _assert_result(_hierarchical_topk_out(*args, baseline, mask_invalid_indices=True), reference)
    _assert_result(beam_radix_topk_out(*args, radix, mask_invalid_indices=True), reference)


@pytest.mark.parametrize("mask_invalid_indices", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_radix_ties_nonfinite_and_invalid(mask_invalid_indices: bool, masked: bool) -> None:
    batch, width, candidate_width, winners = 3, 129, 3, 97
    parent = torch.full((batch * width,), -0.0, device="cuda")
    local = torch.zeros((batch, width * candidate_width), device="cuda")
    # More tied scores than the output can hold, with signed zero on both sides.
    local[0, ::2] = -0.0
    local[0, 320] = float("inf")
    local[0, 380] = 1.0
    # A sparse row mixes finite, -inf and NaN candidates.
    local[1, 1] = float("nan")
    local[1, 385] = -float("inf")
    local[1, 386] = -1
    valid = torch.zeros_like(local, dtype=torch.bool)
    valid[0, 2::3] = True
    valid[1, [1, 385, 386]] = True

    flat_local = local.reshape(-1, candidate_width)
    flat_valid = valid.reshape_as(flat_local) if masked else None
    args = (parent, flat_local, flat_valid, batch, width, candidate_width, winners)
    scratch = allocate_beam_radix_scratch(batch, width, candidate_width, winners, parent.device)
    reference = _fp32_reference(*args, mask_invalid_indices=mask_invalid_indices)
    _assert_result(
        beam_radix_topk_out(*args, scratch, mask_invalid_indices=mask_invalid_indices),
        reference,
    )


def test_radix_graph_replay_resets_workspace_and_reads_new_candidates() -> None:
    batch, width, candidate_width, winners = 2, 128, 64, 128
    generator = torch.Generator().manual_seed(20261115)
    shape = (batch * width, candidate_width)
    parent = torch.randn(batch * width, generator=generator).cuda()
    local = torch.randn(shape, generator=generator).cuda()
    valid = torch.ones_like(local, dtype=torch.bool)
    scratch = allocate_beam_radix_scratch(batch, width, candidate_width, winners, parent.device)
    args = (parent, local, valid, batch, width, candidate_width, winners)
    beam_radix_topk_out(*args, scratch, mask_invalid_indices=True)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = beam_radix_topk_out(*args, scratch, mask_invalid_indices=True)
    graph.replay()
    _assert_result(captured, _fp32_reference(*args, mask_invalid_indices=True))

    valid.zero_()
    valid[width:, ::3] = True
    parent.fill_(1.0)
    local.zero_()
    local[-1, 0] = 2.0
    graph.replay()
    _assert_result(captured, _fp32_reference(*args, mask_invalid_indices=True))


def _next_request(runner: BeamSearchRunner) -> BeamSearchRunner:
    runner.end_session()
    context = runner.session.context
    width = runner.state.beam_width
    slot = context.registry.allocate("next", width, 0)
    assert slot == runner.session.slot
    session = BeamSearchSession("next", slot, context)
    session.begin_session(width, 0, decision_index=0)
    state = BeamSearchState(
        context.decision,
        beam_width=width,
        max_new_tokens=3,
        candidate_width=runner.workspace.key.top_k,
    )
    return BeamSearchRunner(session, state, runner.workspace)


def _assert_same_state(actual: BeamSearchState, expected: BeamSearchState) -> None:
    for left, right in zip(
        (*actual.owned_tensors(), actual.sequence, actual.beam_scores),
        (*expected.owned_tensors(), expected.sequence, expected.beam_scores),
        strict=True,
    ):
        if left.dtype == torch.float32:
            left, right = left.view(torch.int32), right.view(torch.int32)
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.l2
def test_radix_capture_refusal_uses_eager_fp32_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width = 4
    baseline = _runtime(width=width, steps=3)
    candidate = _runtime(width=width, steps=3)
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=width,
        top_k=width,
        max_decode_steps=3,
        device=candidate.state.device,
        logits_dtype=torch.float16,
        enable_radix_topk=True,
    )
    fallback = BeamSearchRunner(candidate.session, candidate.state, workspace)
    with monkeypatch.context() as capture:
        capture.setattr(
            torch.cuda,
            "graph",
            Mock(side_effect=RuntimeError("operation not supported when stream is capturing")),
        )
        fallback.capture_radix_selection()
    assert workspace.radix_graphs == {}
    _assert_same_state(fallback.state, baseline.state)

    tokens = torch.arange(width, device="cuda", dtype=torch.int32)[None, :]
    scores = -tokens.float() / 8
    for runner in (baseline, fallback):
        runner.initialize_from_prefill(tokens, scores, torch.ones_like(tokens, dtype=torch.bool))
        _write_kv(runner)
    candidates = torch.arange(width * width, device="cuda", dtype=torch.int32).view(width, width)
    local_scores = -(candidates.float() % 7) / 4
    valid = torch.ones_like(candidates, dtype=torch.bool)
    reference = _fp32_reference(
        fallback.state.beam_scores.flatten(),
        local_scores,
        valid,
        1,
        width,
        width,
        width,
        mask_invalid_indices=True,
    )
    baseline.advance(candidates, local_scores, valid)
    with patch.object(
        beam_search_runner, "beam_radix_topk_out", wraps=beam_radix_topk_out
    ) as eager_radix:
        fallback.advance(candidates, local_scores, valid)
    eager_radix.assert_called_once()
    _assert_same_state(fallback.state, baseline.state)
    scratch = workspace.radix_scratch
    assert scratch is not None
    _assert_result((scratch.scores, scratch.indices, scratch.scores.shape[1]), reference)
    baseline.end_session()
    fallback.end_session()


@pytest.mark.parametrize(("width", "candidates"), [(512, 128), (1024, 256)])
def test_startup_radix_graph_replays_current_candidates_across_requests(
    width: int, candidates: int
) -> None:
    eager = _runtime(width=width, candidate_width=candidates, steps=4)
    capture_input = _runtime(width=width, candidate_width=candidates, steps=4)
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=width,
        top_k=candidates,
        max_decode_steps=4,
        device=capture_input.state.device,
        logits_dtype=torch.float16,
        enable_radix_topk=True,
    )
    captured = BeamSearchRunner(capture_input.session, capture_input.state, workspace)
    captured.capture_radix_selection()
    graph = workspace.radix_graphs[0]
    assert graph.scratch is workspace.radix_scratch
    _assert_same_state(captured.state, eager.state)

    columns = torch.arange(candidates, device="cuda", dtype=torch.int32)[None, :]
    rows = torch.arange(width, device="cuda", dtype=torch.int32)[:, None]
    flat = rows * candidates + columns
    prefill_scores = -columns.to(torch.float32) / candidates
    prefill_valid = torch.ones_like(columns, dtype=torch.bool)
    graph_scores = workspace.topk_scores[0]

    for request in range(2):
        if request:
            eager = _next_request(eager)
            captured = _next_request(captured)
        for runner in (eager, captured):
            runner.reset()
            runner.initialize_from_prefill(columns + request * width, prefill_scores, prefill_valid)
            for pool in runner.session.pools:
                pool.unshared_key.zero_()
                pool.unshared_value.zero_()

        for step in range(eager.state.max_new_tokens - 1):
            tokens = flat + (request * 4 + step + 1) * width * candidates
            scores = -((flat * (step + 1) + request) % 113).to(torch.float32) / 16
            valid = (columns + rows * 3 + step + request) % candidates < 4
            _write_kv(eager)
            eager.advance(tokens, scores, valid)
            _write_kv(captured)
            foreign_scores = request == 0 and step == 1
            if foreign_scores:
                # A fresh allocation must not replay using stale workspace scores.
                graph_scores.fill_(-100)
                replay_scores = scores
            else:
                graph_scores.copy_(scores)
                replay_scores = graph_scores

            with patch.object(
                beam_search_runner, "beam_radix_topk_out", wraps=beam_radix_topk_out
            ) as eager_radix:
                captured.advance(tokens, replay_scores, valid)
            assert eager_radix.call_count == int(foreign_scores)
            assert captured.workspace.radix_graphs[0] is graph
            _assert_same_state(captured.state, eager.state)
            assert captured.state.error_code.item() == 0
            assert captured.state.num_generated_tokens.item() == step + 2
            for actual, expected in zip(captured.session.pools, eager.session.pools, strict=True):
                torch.testing.assert_close(
                    actual.unshared_key, expected.unshared_key, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual.unshared_value, expected.unshared_value, rtol=0, atol=0
                )

    captured.end_session()
    eager.end_session()


def _batch_runner(
    layout: BeamBatchExecutionView, *, enable_radix_topk: bool
) -> BeamBatchSearchRunner:
    width, candidates = layout.beam_width, layout.candidate_top_k
    device, dtype = torch.device("cuda"), torch.float16
    context = BeamAttentionContext(layout.batch_size, width, 1, 3, 8, 2, device, dtype)
    state = BeamSearchState(
        context.decision, beam_width=width, max_new_tokens=3, candidate_width=candidates
    )
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=layout.batch_size,
        beam_width=width,
        top_k=candidates,
        max_decode_steps=3,
        device=device,
        logits_dtype=dtype,
        enable_radix_topk=enable_radix_topk,
    )
    runners = []
    for index, request_id in enumerate(layout.request_ids):
        prompt_tokens = layout.num_prompt_tokens_by_request[index]
        slot = context.registry.allocate(request_id, width, prompt_tokens)
        session = BeamSearchSession(request_id, slot, context)
        session.begin_session(width, prompt_tokens, decision_index=index)
        state.reset(index)
        runners.append(
            BeamSearchRunner(session, state, workspace, state_index=index, workspace_index=index)
        )
    return BeamBatchSearchRunner(tuple(runners), layout)


def test_batched_decode_shares_radix_scratch_between_replay_and_eager() -> None:
    width, candidates = 9, 7
    prefill = BeamBatchExecutionView(
        session_id="batch",
        dispatch_id=0,
        stage_index=0,
        kind="prefill",
        decode_step=None,
        owner_generation=0,
        request_ids=("first", "second"),
        num_prompt_tokens_by_request=(1, 1),
        beam_width=width,
        produces_output=True,
        candidate_width=candidates,
    )
    baseline = _batch_runner(prefill, enable_radix_topk=False)
    radix = _batch_runner(prefill, enable_radix_topk=True)
    radix.runners[0].capture_radix_selection()
    workspace = radix.workspace
    assert baseline.workspace.radix_scratch is None
    assert set(workspace.radix_graphs) == {0}
    assert workspace.radix_graphs[0].scratch is workspace.radix_scratch

    tokens = torch.arange(2 * candidates, device="cuda", dtype=torch.int32).view(2, candidates)
    scores = -tokens.to(torch.float32) / 16
    initial = BeamCandidateBatch(prefill, tokens, scores, torch.ones_like(tokens, dtype=torch.bool))
    baseline.initialize_from_prefill(initial)
    radix.initialize_from_prefill(initial)
    _assert_same_state(baseline.state, radix.state)

    flat = torch.arange(2 * width * candidates, device="cuda", dtype=torch.int32)
    flat = flat.view(2 * width, candidates)
    graph_scores = workspace.topk_scores.view(2 * width, candidates)
    for step in range(2):
        decode = replace(
            prefill, kind="decode", stage_index=step + 1, dispatch_id=step + 1, decode_step=step
        )
        tokens = flat + (step + 1) * flat.numel()
        scores = -((flat * 3 + step) % 23).to(torch.float32) / 16
        valid = (flat + step) % 3 != 0
        graph_scores.copy_(scores)
        for batch in (baseline, radix):
            for runner in batch.runners:
                _write_kv(runner)
        baseline.advance(BeamCandidateBatch(decode, tokens, scores, valid))
        with patch.object(
            beam_search_runner, "beam_radix_topk_out", wraps=beam_radix_topk_out
        ) as eager_radix:
            radix.advance(BeamCandidateBatch(decode, tokens, graph_scores, valid))
        assert eager_radix.call_count == 1
        _assert_same_state(baseline.state, radix.state)
        assert not torch.any(radix.state.error_code)

    radix.end_session()
    baseline.end_session()
