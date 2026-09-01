# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests derived from the corresponding xllm-ops NPU tests."""

from __future__ import annotations

from functools import partial

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_gr.ops.onerec_decode import (
    _beam_search_group_torch,
    _beam_search_rec_final_select_torch,
    _select_unshared_kv_torch,
)


def _prefix_from_parents(parents: torch.Tensor, beam_size: int) -> torch.Tensor:
    return torch.stack([torch.bincount(row, minlength=beam_size).cumsum(0) for row in parents]).to(
        torch.int32
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_select_unshared_kv_torch_full_cache_and_untouched_blocks(
    dtype: torch.dtype,
) -> None:
    beam_size, layers = 6, 2
    num_blocks, heads, steps, dim = 5, 2, 3, 4
    shape = (num_blocks, beam_size, heads, steps, dim)
    base = torch.arange(torch.tensor(shape).prod()).reshape(shape).to(dtype)
    keys = [base.clone() + layer * 1000 for layer in range(layers)]
    values = [base.clone() + layer * 2000 + 17 for layer in range(layers)]
    original_keys = [tensor.clone() for tensor in keys]
    original_values = [tensor.clone() for tensor in values]
    blocks = torch.tensor([3, 1], dtype=torch.int32)
    # identity + right move + left move + one parent feeding many targets
    parents = torch.tensor([[0, 0, 2, 4, 4, 4], [1, 1, 1, 3, 5, 5]])
    prefix = _prefix_from_parents(parents, beam_size).reshape(-1)
    beam_index = parents.to(torch.int32).reshape(-1)

    _select_unshared_kv_torch(beam_index, keys, values, blocks, prefix, 1, beam_size, layers)

    for layer in range(layers):
        expected_key = original_keys[layer].clone()
        expected_value = original_values[layer].clone()
        for request, block in enumerate(blocks.tolist()):
            expected_key[block] = original_keys[layer][block, parents[request]]
            expected_value[block] = original_values[layer][block, parents[request]]
        assert torch.equal(keys[layer], expected_key)
        assert torch.equal(values[layer], expected_value)
        # decode_step=1 still copies the complete cache tail.
        assert torch.equal(keys[layer][:, :, :, 1:], expected_key[:, :, :, 1:])


def _final_inputs(
    batch: int, beam: int, candidates: int, step: int, device: str
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(2026)
    log_probs = torch.randn((batch * beam, 1), dtype=torch.float32, device=device)
    top_tokens = torch.arange(batch * beam * candidates, dtype=torch.int32, device=device).reshape(
        batch * beam, candidates
    )
    top_probs = torch.randn((batch * beam, candidates), dtype=torch.float32, device=device)
    sequence = torch.arange(batch * beam * (step + 1), dtype=torch.int32, device=device).reshape(
        batch, beam, step + 1
    )
    return log_probs, top_tokens, top_probs, sequence


@pytest.mark.parametrize(
    "batch,beam,candidates,result_width,step",
    [
        (1, 8, 32, 32, 1),
        (1, 16, 128, 64, 2),
        (2, 16, 64, 64, 2),
        (1, 8, 32, 1, 1),
        (1, 8, 32, 8 * 32, 1),
    ],
)  # type: ignore[misc]
def test_final_select_torch_shapes_and_lineage(
    batch: int, beam: int, candidates: int, result_width: int, step: int
) -> None:
    inputs = _final_inputs(batch, beam, candidates, step, "cpu")
    token, parent, score, sequence = _beam_search_rec_final_select_torch(
        *inputs, step, result_width
    )
    assert token.shape == (batch * result_width, 1)
    assert parent.shape == (batch * result_width, 1)
    assert score.shape == (batch * result_width, 1)
    assert sequence.shape == (batch, result_width, step + 1)
    assert token.dtype == parent.dtype == sequence.dtype == torch.int32
    assert score.dtype == torch.float32
    assert torch.equal(sequence[:, :, step].reshape(-1, 1), token)


def test_final_select_torch_ties_use_lower_flat_index() -> None:
    batch, beam, candidates, width, step = 1, 3, 4, 8, 1
    log_probs = torch.zeros((beam, 1), dtype=torch.float32)
    top_probs = torch.zeros((beam, candidates), dtype=torch.float32)
    top_tokens = torch.arange(beam * candidates, dtype=torch.int32).reshape(beam, candidates)
    sequence = torch.arange(beam * 2, dtype=torch.int32).reshape(batch, beam, 2)
    token, parent, score, _ = _beam_search_rec_final_select_torch(
        log_probs, top_tokens, top_probs, sequence, step, width
    )
    assert torch.equal(token[:, 0], torch.arange(width, dtype=torch.int32))
    assert torch.equal(parent[:, 0], torch.arange(width) // candidates)
    assert torch.equal(score, torch.zeros_like(score))


def _group_inputs(batch: int, beam: int, step: int, device: str) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(10)
    log_probs = torch.randn((batch * beam, 1), dtype=torch.float32, device=device)
    top_tokens = torch.arange(batch * beam * beam, dtype=torch.int32, device=device).reshape(
        batch * beam, beam
    )
    top_probs = torch.randn((batch * beam, beam), dtype=torch.float32, device=device)
    sequence = torch.arange(batch * beam * (step + 1), dtype=torch.int32, device=device).reshape(
        batch, beam, step + 1
    )
    return log_probs, top_tokens, top_probs, sequence


@pytest.mark.parametrize(
    "batch,beam,top_k,step",
    [(1, 8, 4, 1), (1, 8, 8, 2), (2, 8, 16, 1), (2, 9, 7, 2)],
)  # type: ignore[misc]
def test_group_torch_prefix_grouping_and_sequence(
    batch: int, beam: int, top_k: int, step: int
) -> None:
    inputs = _group_inputs(batch, beam, step, "cpu")
    token, parent, score, prefix, sequence = _beam_search_group_torch(*inputs, step, top_k)
    assert token.shape == parent.shape == score.shape == (batch, top_k)
    assert prefix.shape == (batch, beam)
    assert sequence.shape == (batch, top_k, step + 1)
    assert torch.equal(sequence[:, :, step], token)
    expected_last = torch.arange(batch, dtype=torch.int32) * beam + top_k
    assert torch.equal(prefix[:, -1], expected_last)
    local_parent = parent - torch.arange(batch, dtype=torch.int32)[:, None] * beam
    assert torch.all(local_parent[:, 1:] >= local_parent[:, :-1])


def test_group_torch_stable_parent_grouping() -> None:
    beam, top_k, step = 4, 4, 1
    log_probs = torch.zeros((beam, 1), dtype=torch.float32)
    top_probs = torch.full((beam, beam), -10.0, dtype=torch.float32)
    # TopK parent order before grouping is [3, 1, 1, 1].
    top_probs[3, 0] = 4.0
    top_probs[1, 0:3] = torch.tensor([3.0, 2.0, 1.0])
    top_tokens = torch.arange(beam * beam, dtype=torch.int32).reshape(beam, beam)
    sequence = torch.arange(beam * 2, dtype=torch.int32).reshape(1, beam, 2)
    token, parent, _, prefix, _ = _beam_search_group_torch(
        log_probs, top_tokens, top_probs, sequence, step, top_k
    )
    assert torch.equal(parent[0], torch.tensor([1, 1, 1, 3], dtype=torch.int32))
    assert torch.equal(token[0], torch.tensor([4, 5, 6, 12], dtype=torch.int32))
    assert torch.equal(prefix[0], torch.tensor([0, 3, 3, 4], dtype=torch.int32))


def test_group_torch_step0_matches_npu_observable_initialization() -> None:
    batch, beam, top_k = 2, 8, 8
    inputs = _group_inputs(batch, beam, 0, "cpu")
    token, parent, score, prefix, sequence = _beam_search_group_torch(*inputs, 0, top_k)
    # Legacy kernel-initialization contract (verified on device): only the
    # flat sequence rows [0, batch) receive their own row's first candidate.
    expected = torch.zeros_like(sequence)
    expected.reshape(-1, 1)[:batch, 0] = inputs[1][:batch, 0]
    assert torch.equal(sequence, expected)
    assert torch.count_nonzero(token) == 0
    assert torch.count_nonzero(parent) == 0
    assert torch.count_nonzero(score) == 0
    assert torch.count_nonzero(prefix) == 0


def test_group_torch_preserves_fixed_sequence_extent() -> None:
    batch, beam, step, max_steps = 1, 4, 1, 3
    log_probs = torch.randn((batch * beam, 1), dtype=torch.float32)
    top_tokens = torch.arange(batch * beam * beam, dtype=torch.int32).reshape(beam, beam)
    top_probs = torch.randn((batch * beam, beam), dtype=torch.float32)
    sequence = torch.arange(batch * beam * max_steps, dtype=torch.int32).reshape(
        batch, beam, max_steps
    )
    token, _, _, _, next_sequence = _beam_search_group_torch(
        log_probs,
        top_tokens,
        top_probs,
        sequence,
        step,
        beam,
    )
    assert next_sequence.shape == (batch, beam, max_steps)
    assert torch.equal(next_sequence[:, :, step], token)
    assert torch.count_nonzero(next_sequence[:, :, step + 1 :]) == 0


def test_group_prefix_can_feed_select_directly() -> None:
    batch, beam, top_k, step = 2, 4, 4, 1
    group_inputs = _group_inputs(batch, beam, step, "cpu")
    _, parent, _, prefix, _ = _beam_search_group_torch(*group_inputs, step, top_k)
    shape = (batch, beam, 1, 2, 2)
    key = torch.arange(torch.tensor(shape).prod()).reshape(shape).to(torch.float16)
    value = key + 100
    expected_key = key.clone()
    expected_value = value.clone()
    blocks = torch.arange(batch, dtype=torch.int32)
    local_parent = (parent - torch.arange(batch, dtype=torch.int32)[:, None] * beam).to(torch.int64)
    for request in range(batch):
        expected_key[request] = key[request, local_parent[request]]
        expected_value[request] = value[request, local_parent[request]]
    _select_unshared_kv_torch(
        parent.reshape(-1), [key], [value], blocks, prefix.reshape(-1), 1, beam, 1
    )
    assert torch.equal(key, expected_key)
    assert torch.equal(value, expected_value)


def test_registered_schemas_match_public_contract() -> None:
    select_schema = str(torch.ops.vllm_gr.select_unshared_kv.default._schema)
    assert "Tensor(a!)[] x_key_block" in select_schema
    assert "Tensor(b!)[] x_value_block" in select_schema
    assert select_schema.endswith("-> ()")
    assert torch.ops.vllm_gr.beam_search_group.default._schema.arguments[-1].name == ("top_k")


def test_registered_beam_ops_have_meta_implementations() -> None:
    batch, beam, candidates, max_steps = 2, 4, 8, 3
    cumulative = torch.empty((batch * beam, 1), dtype=torch.float32, device="meta")
    group_tokens = torch.empty((batch * beam, beam), dtype=torch.int32, device="meta")
    group_probs = torch.empty((batch * beam, beam), dtype=torch.float32, device="meta")
    sequence = torch.empty((batch, beam, max_steps), dtype=torch.int32, device="meta")
    group = torch.ops.vllm_gr.beam_search_group(
        cumulative,
        group_tokens,
        group_probs,
        sequence,
        1,
        beam,
    )
    assert [tuple(tensor.shape) for tensor in group] == [
        (batch, beam),
        (batch, beam),
        (batch, beam),
        (batch, beam),
        (batch, beam, max_steps),
    ]

    final_tokens = torch.empty((batch * beam, candidates), dtype=torch.int32, device="meta")
    final_probs = torch.empty((batch * beam, candidates), dtype=torch.float32, device="meta")
    final = torch.ops.vllm_gr.beam_search_rec_final_select(
        cumulative,
        final_tokens,
        final_probs,
        sequence,
        2,
        2,
    )
    assert [tuple(tensor.shape) for tensor in final] == [
        (batch * 2, 1),
        (batch * 2, 1),
        (batch * 2, 1),
        (batch, 2, max_steps),
    ]


def test_registered_beam_ops_support_fake_tensor_propagation() -> None:
    batch, beam, candidates, max_steps = 2, 4, 8, 3
    with FakeTensorMode():
        cumulative = torch.empty((batch * beam, 1), dtype=torch.float32)
        group_tokens = torch.empty((batch * beam, beam), dtype=torch.int32)
        group_probs = torch.empty((batch * beam, beam), dtype=torch.float32)
        sequence = torch.empty((batch, beam, max_steps), dtype=torch.int32)
        group = torch.ops.vllm_gr.beam_search_group(
            cumulative,
            group_tokens,
            group_probs,
            sequence,
            1,
            beam,
        )
        final = torch.ops.vllm_gr.beam_search_rec_final_select(
            cumulative,
            torch.empty((batch * beam, candidates), dtype=torch.int32),
            torch.empty((batch * beam, candidates), dtype=torch.float32),
            sequence,
            2,
            2,
        )
    assert [tuple(tensor.shape) for tensor in group] == [
        (batch, beam),
        (batch, beam),
        (batch, beam),
        (batch, beam),
        (batch, beam, max_steps),
    ]
    assert [tuple(tensor.shape) for tensor in final] == [
        (batch * 2, 1),
        (batch * 2, 1),
        (batch * 2, 1),
        (batch, 2, max_steps),
    ]


def test_registered_group_is_visible_to_torch_compile() -> None:
    batch, beam, max_steps = 1, 4, 3
    inputs = (
        torch.randn((batch * beam, 1), dtype=torch.float32),
        torch.arange(batch * beam * beam, dtype=torch.int32).reshape(batch * beam, beam),
        torch.randn((batch * beam, beam), dtype=torch.float32),
        torch.arange(batch * beam * max_steps, dtype=torch.int32).reshape(batch, beam, max_steps),
    )
    compiled = torch.compile(
        torch.ops.vllm_gr.beam_search_group.default,
        backend="eager",
        fullgraph=True,
    )
    expected = _beam_search_group_torch(*inputs, 1, beam)
    actual = compiled(*inputs, 1, beam)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_registered_final_select_is_visible_to_torch_compile() -> None:
    batch, beam, candidates, max_steps, result_width = 1, 4, 8, 3, 2
    inputs = (
        torch.randn((batch * beam, 1), dtype=torch.float32),
        torch.arange(batch * beam * candidates, dtype=torch.int32).reshape(
            batch * beam, candidates
        ),
        torch.randn((batch * beam, candidates), dtype=torch.float32),
        torch.arange(batch * beam * max_steps, dtype=torch.int32).reshape(batch, beam, max_steps),
    )
    compiled = torch.compile(
        torch.ops.vllm_gr.beam_search_rec_final_select.default,
        backend="eager",
        fullgraph=True,
    )
    expected = _beam_search_rec_final_select_torch(*inputs, 2, result_width)
    actual = compiled(*inputs, 2, result_width)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.mark.skipif(  # type: ignore[misc]
    not torch.cuda.is_available(), reason="CUDA is required"
)
@pytest.mark.parametrize("case", ["final", "group"])  # type: ignore[misc]
def test_topk_torch_cuda_graph(case: str) -> None:
    if case == "final":
        inputs = _final_inputs(2, 8, 16, 2, "cuda")
        torch_result = _beam_search_rec_final_select_torch(*inputs, 2, 16)
        replay = partial(_beam_search_rec_final_select_torch, *inputs, 2, 16)
    else:
        inputs = _group_inputs(2, 8, 2, "cuda")
        torch_result = _beam_search_group_torch(*inputs, 2, 8)
        replay = partial(_beam_search_group_torch, *inputs, 2, 8)

    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_result = replay()
    graph.replay()
    torch.accelerator.synchronize()
    for expected, actual in zip(torch_result, graph_result):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
