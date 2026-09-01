# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.config import ConstraintBackend, GRConfig
from vllm_gr.ops.cuda.prefill_beam_result import (
    STATUS_NO_VALID_CANDIDATE,
    STATUS_OK,
    prefill_beam_result_out,
)
from vllm_gr.v1.beam.beam_decision import (
    BeamPostprocessWorkspace,
    BeamPostprocessWorkspacePool,
    _encode_cuda_beam_result,
    maybe_run_worker_beam_decision,
    unpack_beam_result_buffer,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")

# pytest parametrization decorators are untyped under strict mypy.
# mypy: disable-error-code="misc"


def _run(
    counts: list[int],
    beam_width: int,
    *,
    invalid_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(counts)
    token_ids = torch.arange(
        100,
        100 + batch_size * beam_width,
        dtype=torch.int32,
        device="cuda",
    ).view(batch_size, beam_width)
    scores = torch.linspace(
        -0.0,
        -7.0,
        batch_size * beam_width,
        dtype=torch.float32,
        device="cuda",
    ).view(batch_size, beam_width)
    candidate_counts = torch.tensor(counts, dtype=torch.int32, device="cuda")
    status = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    payload = torch.empty(3 * batch_size * beam_width, dtype=torch.int32, device="cuda")
    prefill_beam_result_out(
        token_ids,
        scores,
        candidate_counts,
        status,
        payload,
        invalid_token_id,
    )
    return token_ids, scores, candidate_counts, status, payload


@pytest.mark.parametrize("beam_width", [1, 64, 128, 256])
@pytest.mark.parametrize("count_kind", ["zero", "one", "partial", "full", "overfull"])
def test_prefill_beam_result_candidate_count_semantics(
    beam_width: int,
    count_kind: str,
) -> None:
    counts = {
        "zero": 0,
        "one": 1,
        "partial": max(1, beam_width // 2),
        "full": beam_width,
        "overfull": beam_width + 7,
    }
    count = counts[count_kind]
    token_ids, scores, _, status, payload = _run([count], beam_width, invalid_token_id=17)
    active = min(max(count, 0), beam_width)

    host_status = status.cpu()
    host_payload = payload.cpu()
    out_tokens = host_payload[:beam_width]
    out_parents = host_payload[beam_width : 2 * beam_width]
    out_scores = host_payload[2 * beam_width :].view(torch.float32)

    assert int(host_status[0]) == (STATUS_OK if count > 0 else STATUS_NO_VALID_CANDIDATE)
    torch.testing.assert_close(out_tokens[:active], token_ids.cpu()[0, :active])
    assert torch.equal(
        out_tokens[active:],
        torch.full((beam_width - active,), 17, dtype=torch.int32),
    )
    assert torch.equal(out_parents, torch.zeros(beam_width, dtype=torch.int32))
    assert torch.equal(
        out_scores[:active].view(torch.int32), scores.cpu()[0, :active].view(torch.int32)
    )
    assert torch.isneginf(out_scores[active:]).all()


def test_prefill_beam_result_mixed_batch_reports_exact_bad_row() -> None:
    _, _, _, status, payload = _run([64, 0], 64, invalid_token_id=23)
    assert status.cpu().tolist() == [STATUS_OK, STATUS_NO_VALID_CANDIDATE]
    host_payload = payload.cpu()
    tokens = host_payload[:128].view(2, 64)
    parents = host_payload[128:256].view(2, 64)
    scores = host_payload[256:].view(torch.float32).view(2, 64)
    assert torch.equal(tokens[1], torch.full((64,), 23, dtype=torch.int32))
    assert torch.isneginf(scores[1]).all()
    assert torch.equal(parents, torch.zeros_like(parents))


def test_prefill_beam_result_rejects_scores_on_another_device() -> None:
    token_ids = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    scores = torch.zeros((1, 4), dtype=torch.float32, device="cpu")
    counts = torch.ones((1,), dtype=torch.int32, device="cuda")
    status = torch.empty((1,), dtype=torch.int32, device="cuda")
    payload = torch.empty((12,), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="scores.device"):
        prefill_beam_result_out(token_ids, scores, counts, status, payload, 0)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_prefill_beam_result_preserves_existing_wire_bytes(batch_size: int) -> None:
    beam_width = 64
    token_ids, scores, _, status, payload = _run(
        [beam_width] * batch_size,
        beam_width,
    )
    assert status.cpu().tolist() == [STATUS_OK] * batch_size
    parents = torch.zeros_like(token_ids)
    encoded = _encode_cuda_beam_result(
        {
            "out_token_ids": token_ids,
            "out_token_index": parents,
            "out_log_probs": scores,
        },
        is_last_step=False,
    )
    assert encoded is not None
    expected, _ = encoded
    assert payload.cpu().numpy().tobytes() == expected["buffer"]


def _prefill_runtime(candidate_count: int) -> tuple[SimpleNamespace, SimpleNamespace]:
    beam_width = 64
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=beam_width,
        top_k=beam_width,
        max_decode_steps=3,
        device="cuda",
        logits_dtype=torch.float16,
    )
    pool = BeamPostprocessWorkspacePool()
    pool.add(workspace)
    tokens = workspace.candidate_rows(
        workspace.topk_token_ids,
        1,
        compact=True,
    )
    scores = workspace.candidate_rows(
        workspace.topk_scores,
        1,
        compact=True,
    )
    wire_scores = workspace.candidate_rows(
        workspace.wire_topk_scores,
        1,
        compact=True,
    )
    tokens.copy_(torch.arange(beam_width, dtype=torch.int32, device="cuda").view(1, -1))
    scores.copy_(torch.linspace(-0.0, -6.0, beam_width, device="cuda").view(1, -1))
    wire_scores.copy_(scores)
    workspace.topk_valid_counts.view(-1)[0] = candidate_count

    config = GRConfig(
        catalog_path="fixture.json",
        constraint_backend=ConstraintBackend.CONSTRAINT_TABLE,
        beam_max_width=beam_width,
        beam_worker_decision=True,
    )
    runner = SimpleNamespace(
        device=torch.device("cuda"),
        model_config=SimpleNamespace(dtype=torch.float16),
        _vllm_gr_config=config,
        _beam_postprocess_workspaces=pool,
        _current_beam_info={
            "beam_data": {
                "request": {
                    "is_constraint_prefill": True,
                    "beam_width": beam_width,
                }
            }
        },
        _beam_decision_result=None,
        _beam_decision_tensors=None,
    )
    sampler_output = SimpleNamespace(
        logprobs_tensors=SimpleNamespace(logprobs=wire_scores),
        _gr_constraint_token_ids=tokens,
        _gr_constraint_logprobs=scores,
        _gr_constraint_valid_counts=workspace.topk_valid_counts.view(-1)[:1],
        _gr_deferred_prefill_status=True,
        _gr_beam_postprocess_workspace=workspace,
    )
    return runner, sampler_output


def test_worker_prefill_fast_path_publishes_existing_payload_protocol() -> None:
    runner, sampler_output = _prefill_runtime(64)
    maybe_run_worker_beam_decision(runner, sampler_output)

    assert runner._beam_result_host_ready is True
    decoded = unpack_beam_result_buffer(runner._beam_decision_result)
    assert decoded is not None
    token_ids, scores, parents = decoded
    assert token_ids == list(range(64))
    assert parents == [0] * 64
    torch.testing.assert_close(
        torch.tensor(scores),
        torch.linspace(-0.0, -6.0, 64),
        rtol=0,
        atol=0,
    )


def test_worker_prefill_host_views_survive_workspace_reuse() -> None:
    runner, sampler_output = _prefill_runtime(64)
    maybe_run_worker_beam_decision(runner, sampler_output)
    decision = runner._beam_decision_tensors
    expected_ids = decision["_host_token_ids"].clone()
    expected_scores = decision["_host_log_probs"].clone()
    expected_ranks = decision["_host_selected_token_ranks"].clone()
    workspace = sampler_output._gr_beam_postprocess_workspace

    assert decision["_host_token_ids"].untyped_storage().data_ptr() != (
        workspace.host_prefill_staging.untyped_storage().data_ptr()
    )
    workspace.host_prefill_staging.fill_(12345)

    assert torch.equal(decision["_host_token_ids"], expected_ids)
    assert torch.equal(decision["_host_log_probs"], expected_scores)
    assert torch.equal(decision["_host_selected_token_ranks"], expected_ranks)


def test_worker_prefill_fast_path_fails_before_publish_with_bad_rows() -> None:
    runner, sampler_output = _prefill_runtime(0)
    with pytest.raises(RuntimeError, match=r"batch rows \[0\]"):
        maybe_run_worker_beam_decision(runner, sampler_output)
    assert runner._beam_decision_result is None


def test_prefill_beam_result_1000_calls_keep_device_memory_stable() -> None:
    beam_width = 128
    token_ids = torch.arange(beam_width, dtype=torch.int32, device="cuda").view(1, -1)
    scores = torch.linspace(-1.0, -8.0, beam_width, device="cuda").view(1, -1)
    counts = torch.tensor([beam_width], dtype=torch.int32, device="cuda")
    status = torch.empty(1, dtype=torch.int32, device="cuda")
    payload = torch.empty(3 * beam_width, dtype=torch.int32, device="cuda")

    prefill_beam_result_out(token_ids, scores, counts, status, payload, 0)
    torch.accelerator.synchronize()
    allocated_before = torch.accelerator.memory_allocated()
    reserved_before = torch.accelerator.memory_reserved()
    addresses = (status.data_ptr(), payload.data_ptr())

    for _ in range(1000):
        prefill_beam_result_out(token_ids, scores, counts, status, payload, 0)
    torch.accelerator.synchronize()

    assert (status.data_ptr(), payload.data_ptr()) == addresses
    assert torch.accelerator.memory_allocated() == allocated_before
    assert torch.accelerator.memory_reserved() == reserved_before
