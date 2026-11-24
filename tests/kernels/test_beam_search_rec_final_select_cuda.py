# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA precision guards for deterministic OneRec final Beam selection."""

from __future__ import annotations

import pytest
import torch

from vllm_gr.ops.gpu.beam_selection import (
    allocate_beam_selection_scratch,
    beam_search_rec_final_select_cuda,
    beam_search_rec_final_select_cuda_out,
)
from vllm_gr.ops.onerec_decode import (
    _beam_search_rec_final_select_torch,
    beam_search_rec_final_select,
)

# pytest parametrization decorators are untyped under strict mypy.
# mypy: disable-error-code="misc"


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _inputs(
    batch_size: int,
    beam_width: int,
    candidate_width: int,
    current_step: int,
    *,
    max_steps: int = 3,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260827)
    cumulative = torch.randn(
        (batch_size * beam_width, 1),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    top_tokens = torch.arange(
        batch_size * beam_width * candidate_width,
        dtype=torch.int32,
        device="cuda",
    ).reshape(batch_size * beam_width, candidate_width)
    top_logprobs = torch.randn(
        (batch_size * beam_width, candidate_width),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    sequence = torch.arange(
        batch_size * beam_width * max_steps,
        dtype=torch.int32,
        device="cuda",
    ).reshape(batch_size, beam_width, max_steps)
    assert current_step < max_steps
    return cumulative, top_tokens, top_logprobs, sequence


@pytest.mark.parametrize(
    "beam_width,candidate_width,result_width,current_step",
    [
        (1, 1, 1, 0),
        (2, 4, 1, 1),
        (4, 2, 2, 2),
        (8, 32, 8, 1),
        (32, 8, 4, 2),
        (64, 64, 32, 2),
        (128, 32, 128, 2),
        (256, 256, 256, 2),
    ],
)
def test_final_select_cuda_matches_fixed_extent_torch(
    beam_width: int,
    candidate_width: int,
    result_width: int,
    current_step: int,
) -> None:
    inputs = _inputs(1, beam_width, candidate_width, current_step)
    expected = _beam_search_rec_final_select_torch(*inputs, current_step, result_width)
    actual = beam_search_rec_final_select_cuda(*inputs, current_step, result_width)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert actual[-1].shape == (1, result_width, 3)


def test_final_select_cuda_multi_request_global_parent() -> None:
    batch_size, beam_width, candidate_width, result_width = 2, 8, 16, 4
    inputs = _inputs(batch_size, beam_width, candidate_width, 2)
    actual = beam_search_rec_final_select_cuda(*inputs, 2, result_width)
    expected = _beam_search_rec_final_select_torch(*inputs, 2, result_width)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert torch.all(actual[1][result_width:] >= beam_width)


def test_final_select_cuda_exact_tie_preserves_score_order() -> None:
    beam_width, candidate_width, result_width = 4, 8, 4
    cumulative, top_tokens, top_logprobs, sequence = _inputs(1, beam_width, candidate_width, 2)
    cumulative.zero_()
    top_logprobs.zero_()
    token, parent, score, _ = beam_search_rec_final_select_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        2,
        result_width,
    )
    assert torch.equal(token[:, 0], torch.arange(result_width, dtype=torch.int32, device="cuda"))
    assert torch.equal(parent[:, 0], torch.zeros(result_width, dtype=torch.int32, device="cuda"))
    assert torch.equal(score, torch.zeros_like(score))


def test_final_select_cuda_explicit_candidate_validity() -> None:
    beam_width, candidate_width, result_width = 8, 8, 8
    cumulative, top_tokens, top_logprobs, sequence = _inputs(1, beam_width, candidate_width, 2)
    valid_mask = torch.zeros_like(top_tokens, dtype=torch.bool)
    valid_mask[0, :3] = True
    masked_logprobs = top_logprobs.masked_fill(~valid_mask, -float("inf"))
    expected = _beam_search_rec_final_select_torch(
        cumulative,
        top_tokens,
        masked_logprobs,
        sequence,
        2,
        result_width,
    )
    actual = beam_search_rec_final_select_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        2,
        result_width,
        valid_mask,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert torch.equal(actual[1], torch.zeros_like(actual[1]))
    assert torch.isneginf(actual[2][3:]).all()


def test_final_select_cuda_out_variant_graph_replay_uses_fixed_addresses() -> None:
    batch_size, beam_width, candidate_width, result_width = 2, 8, 16, 4
    inputs = _inputs(batch_size, beam_width, candidate_width, 2)
    scratch = allocate_beam_selection_scratch(
        batch_size,
        beam_width,
        candidate_width,
        result_width,
        torch.device("cuda"),
    )
    outputs = (
        torch.empty((batch_size * result_width, 1), dtype=torch.int32, device="cuda"),
        torch.empty((batch_size * result_width, 1), dtype=torch.int32, device="cuda"),
        torch.empty((batch_size * result_width, 1), dtype=torch.float32, device="cuda"),
        torch.empty((batch_size, result_width, 3), dtype=torch.int32, device="cuda"),
    )
    scratch_tensors = (
        scratch.local_scores,
        scratch.local_flat_indices,
        scratch.merge_scores,
        scratch.merge_flat_indices,
    )
    addresses = tuple(tensor.data_ptr() for tensor in (*outputs, *scratch_tensors))

    def run() -> None:
        beam_search_rec_final_select_cuda_out(
            *inputs,
            2,
            result_width,
            *outputs,
            *scratch_tensors,
        )

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    inputs[2].copy_(inputs[2].flip(1))
    expected = _beam_search_rec_final_select_torch(*inputs, 2, result_width)
    graph.replay()
    torch.accelerator.synchronize()
    assert tuple(tensor.data_ptr() for tensor in (*outputs, *scratch_tensors)) == addresses
    for actual_tensor, expected_tensor in zip(outputs, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_public_final_select_dispatches_to_cuda() -> None:
    inputs = _inputs(1, 4, 8, 2)
    expected = beam_search_rec_final_select_cuda(*inputs, 2, 4)
    results = (
        beam_search_rec_final_select(*inputs, 2, 4),
        torch.ops.vllm_gr.beam_search_rec_final_select(*inputs, 2, 4),
    )
    for actual in results:
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_final_select_cuda_fail_fast_result_width_and_dtype() -> None:
    inputs = _inputs(1, 4, 8, 2)
    with pytest.raises(ValueError, match="result_width"):
        beam_search_rec_final_select_cuda(*inputs, 2, 5)
    with pytest.raises(TypeError, match="top_tokens"):
        beam_search_rec_final_select_cuda(inputs[0], inputs[1].long(), *inputs[2:], 2, 4)
    long_sequence_inputs = _inputs(1, 4, 8, 3, max_steps=4)
    with pytest.raises(ValueError, match="current_step"):
        beam_search_rec_final_select_cuda(*long_sequence_inputs, 3, 4)
