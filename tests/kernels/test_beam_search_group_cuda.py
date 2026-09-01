# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA precision guards for deterministic OneRec BeamSearchGroup."""

from __future__ import annotations

import pytest
import torch

from vllm_gr.ops.cuda.beam_selection import (
    allocate_beam_selection_scratch,
    beam_search_group_cuda,
    beam_search_group_cuda_out,
)
from vllm_gr.ops.onerec_decode import _beam_search_group_torch, beam_search_group
from vllm_gr.v1.attention.backends.beam_kernels import (
    select_unshared_kv,
    select_unshared_kv_fused,
)

# pytest parametrization decorators are untyped under strict mypy.
# mypy: disable-error-code="misc"


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _inputs(
    batch_size: int,
    beam_width: int,
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
        batch_size * beam_width * beam_width,
        dtype=torch.int32,
        device="cuda",
    ).reshape(batch_size * beam_width, beam_width)
    top_logprobs = torch.randn(
        (batch_size * beam_width, beam_width),
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


@pytest.mark.parametrize("beam_width", [1, 2, 4, 8, 32, 64, 128, 256])
@pytest.mark.parametrize("current_step", [1, 2])
def test_group_cuda_matches_fixed_extent_torch(
    beam_width: int,
    current_step: int,
) -> None:
    inputs = _inputs(1, beam_width, current_step)
    expected = _beam_search_group_torch(*inputs, current_step, beam_width)
    actual = beam_search_group_cuda(*inputs, current_step, beam_width)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert actual[-1].shape == (1, beam_width, 3)


def test_group_cuda_multi_request_global_parent_and_prefix() -> None:
    batch_size, beam_width, current_step = 2, 8, 1
    inputs = _inputs(batch_size, beam_width, current_step)
    actual = beam_search_group_cuda(*inputs, current_step, beam_width)
    expected = _beam_search_group_torch(*inputs, current_step, beam_width)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert torch.all(actual[1][1] >= beam_width)
    assert actual[3][0, -1] == beam_width
    assert actual[3][1, -1] == 2 * beam_width


def test_group_cuda_step_zero_bootstraps_first_parent() -> None:
    batch_size, beam_width = 2, 8
    cumulative, top_tokens, top_logprobs, sequence = _inputs(batch_size, beam_width, 0)
    token, parent, score, prefix, next_sequence = beam_search_group_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        0,
        beam_width,
    )
    request = torch.arange(batch_size, dtype=torch.int32, device="cuda")[:, None]
    candidate_tokens = top_tokens.reshape(batch_size, beam_width, beam_width)[:, 0]
    candidate_scores = (
        cumulative.reshape(batch_size, beam_width, 1)[:, 0]
        + top_logprobs.reshape(batch_size, beam_width, beam_width)[:, 0]
    )
    assert torch.equal(token, candidate_tokens)
    assert torch.equal(parent, (request * beam_width).expand_as(parent))
    torch.testing.assert_close(score, candidate_scores, rtol=0, atol=0)
    assert torch.equal(prefix, (request * beam_width + beam_width).expand_as(prefix))
    assert torch.equal(next_sequence[:, :, 0], token)
    assert torch.count_nonzero(next_sequence[:, :, 1:]) == 0


def test_group_cuda_step_zero_accepts_compact_rectopk_rows() -> None:
    batch_size, beam_width = 2, 8
    cumulative, top_tokens, top_logprobs, sequence = _inputs(batch_size, beam_width, 0)
    expected = beam_search_group_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        0,
        beam_width,
    )
    compact_cumulative = cumulative.reshape(batch_size, beam_width, 1)[:, 0].contiguous()
    compact_tokens = top_tokens.reshape(batch_size, beam_width, beam_width)[:, 0].contiguous()
    compact_logprobs = top_logprobs.reshape(batch_size, beam_width, beam_width)[:, 0].contiguous()
    compact_valid = torch.ones_like(compact_tokens, dtype=torch.bool)

    actual = beam_search_group_cuda(
        compact_cumulative,
        compact_tokens,
        compact_logprobs,
        sequence,
        0,
        beam_width,
        compact_valid,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_group_cuda_exact_tie_uses_flat_index_then_stable_parent_grouping() -> None:
    batch_size, beam_width, current_step = 1, 8, 1
    cumulative, top_tokens, top_logprobs, sequence = _inputs(batch_size, beam_width, current_step)
    cumulative.zero_()
    top_logprobs.zero_()
    token, parent, score, prefix, _ = beam_search_group_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        beam_width,
    )
    assert torch.equal(token[0], torch.arange(beam_width, dtype=torch.int32, device="cuda"))
    assert torch.equal(parent[0], torch.zeros(beam_width, dtype=torch.int32, device="cuda"))
    assert torch.equal(score, torch.zeros_like(score))
    assert torch.equal(
        prefix[0], torch.full((beam_width,), beam_width, dtype=torch.int32, device="cuda")
    )


def test_group_cuda_explicit_candidate_validity() -> None:
    batch_size, beam_width, current_step = 1, 8, 1
    cumulative, top_tokens, top_logprobs, sequence = _inputs(batch_size, beam_width, current_step)
    valid_mask = torch.zeros_like(top_tokens, dtype=torch.bool)
    valid_mask[0, :3] = True
    masked_logprobs = top_logprobs.masked_fill(~valid_mask, -float("inf"))
    expected = _beam_search_group_torch(
        cumulative,
        top_tokens,
        masked_logprobs,
        sequence,
        current_step,
        beam_width,
    )
    actual = beam_search_group_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        beam_width,
        valid_mask,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert torch.equal(actual[1], torch.zeros_like(actual[1]))
    assert torch.isneginf(actual[2][:, 3:]).all()
    assert actual[3][0, -1] == beam_width


def test_group_cuda_out_variant_graph_replay_uses_fixed_addresses() -> None:
    batch_size, beam_width, current_step = 2, 8, 2
    inputs = _inputs(batch_size, beam_width, current_step)
    scratch = allocate_beam_selection_scratch(
        batch_size,
        beam_width,
        beam_width,
        beam_width,
        torch.device("cuda"),
    )
    outputs = (
        torch.empty((batch_size, beam_width), dtype=torch.int32, device="cuda"),
        torch.empty((batch_size, beam_width), dtype=torch.int32, device="cuda"),
        torch.empty((batch_size, beam_width), dtype=torch.float32, device="cuda"),
        torch.empty((batch_size, beam_width), dtype=torch.int32, device="cuda"),
        torch.empty((batch_size, beam_width, 3), dtype=torch.int32, device="cuda"),
    )
    scratch_tensors = (
        scratch.local_scores,
        scratch.local_flat_indices,
        scratch.merge_scores,
        scratch.merge_flat_indices,
    )
    addresses = tuple(tensor.data_ptr() for tensor in (*outputs, *scratch_tensors))

    def run() -> None:
        beam_search_group_cuda_out(
            *inputs,
            current_step,
            beam_width,
            *outputs,
            scratch.local_scores,
            scratch.local_flat_indices,
            scratch.merge_scores,
            scratch.merge_flat_indices,
        )

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    inputs[2].copy_(inputs[2].flip(1))
    expected = _beam_search_group_torch(*inputs, current_step, beam_width)
    graph.replay()
    torch.accelerator.synchronize()
    assert tuple(tensor.data_ptr() for tensor in (*outputs, *scratch_tensors)) == addresses
    for actual_tensor, expected_tensor in zip(outputs, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_public_group_dispatches_to_cuda() -> None:
    inputs = _inputs(1, 4, 1)
    expected = beam_search_group_cuda(*inputs, 1, 4)
    results = (
        beam_search_group(*inputs, 1, 4),
        torch.ops.vllm_gr.beam_search_group(*inputs, 1, 4),
    )
    for actual in results:
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_group_parent_indices_drive_existing_select_unshared_kv_fused() -> None:
    beam_width, max_steps, head_dim = 4, 3, 8
    cumulative, top_tokens, top_logprobs, sequence = _inputs(1, beam_width, 1)
    cumulative.zero_()
    top_logprobs.fill_(-100.0)
    top_logprobs[2, :2] = torch.tensor([4.0, 3.0], device="cuda")
    top_logprobs[0, 0] = 2.0
    top_logprobs[3, 0] = 1.0
    _, parent, _, _, _ = beam_search_group_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        1,
        beam_width,
    )
    assert torch.equal(
        parent[0],
        torch.tensor([0, 2, 2, 3], dtype=torch.int32, device="cuda"),
    )

    shape = (1, beam_width, 1, max_steps, head_dim)
    key = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(shape).half().cuda()
    value = key + 1000
    expected_key = key.clone()
    expected_value = value.clone()
    reference_aux_key = torch.empty(
        (beam_width, 1, max_steps, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    reference_aux_value = torch.empty_like(reference_aux_key)
    select_unshared_kv(
        parent.reshape(-1),
        0,
        [expected_key],
        [expected_value],
        1,
        aux_key_bufs=[reference_aux_key],
        aux_value_bufs=[reference_aux_value],
    )

    aux_key = torch.empty(
        (1, beam_width, 1, max_steps, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    aux_value = torch.empty_like(aux_key)
    assert select_unshared_kv_fused(
        parent.reshape(-1),
        0,
        [key],
        [value],
        1,
        aux_key=aux_key,
        aux_value=aux_value,
    )
    assert torch.equal(key, expected_key)
    assert torch.equal(value, expected_value)


def test_group_cuda_fail_fast_fixed_width_and_dtype() -> None:
    inputs = _inputs(1, 4, 1)
    with pytest.raises(ValueError, match="top_k == beam_width"):
        beam_search_group_cuda(*inputs, 1, 2)
    with pytest.raises(TypeError, match="cumulative_logprobs"):
        beam_search_group_cuda(inputs[0].half(), *inputs[1:], 1, 4)
