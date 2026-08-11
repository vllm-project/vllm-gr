# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for Triton OneRec decode operators."""

from __future__ import annotations

from functools import partial

import pytest
import torch

from vllm_gr.ops.cache_unshared_kv import cache_unshared_kv_torch
from vllm_gr.ops.onerec_decode import (
    _beam_search_group_torch,
    _beam_search_rec_final_select_torch,
    _select_unshared_kv_torch,
)

cache_triton = pytest.importorskip("vllm_gr.ops.cache_unshared_kv_triton")
decode_triton = pytest.importorskip("vllm_gr.ops.onerec_decode_triton")
cache_unshared_kv_triton = cache_triton.cache_unshared_kv_triton
select_unshared_kv_triton = decode_triton.select_unshared_kv_triton
beam_search_rec_final_select_triton = decode_triton.beam_search_rec_final_select_triton
beam_search_group_triton = decode_triton.beam_search_group_triton
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _strided_zeros(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    storage_shape = (*shape[:-1], shape[-1] * 2)
    return torch.zeros(storage_shape, device="cuda", dtype=dtype)[..., ::2]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])  # type: ignore[misc]
@pytest.mark.parametrize("strided", [False, True])  # type: ignore[misc]
def test_cache_unshared_kv_triton_matches_torch(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    strided: bool,
) -> None:
    batch_size, beam_size, heads, steps, dim, blocks = 3, 5, 2, 3, 17, 7
    shape = (blocks, beam_size, heads, steps, dim)
    make_cache = (
        _strided_zeros
        if strided
        else lambda shape, dtype: torch.zeros(shape, device="cuda", dtype=dtype)
    )
    actual_key = make_cache(shape, dtype)
    actual_value = make_cache(shape, dtype)
    expected_key = actual_key.clone()
    expected_value = actual_value.clone()
    current_shape = (batch_size * beam_size, heads, dim)
    curr_key = torch.randn(current_shape, device="cuda", dtype=dtype)
    curr_value = torch.randn_like(curr_key)
    block_table = torch.tensor([5, 1, 3], device="cuda", dtype=index_dtype)
    decode_step = torch.tensor([2], device="cuda", dtype=index_dtype)
    key_ptr = actual_key.data_ptr()
    value_ptr = actual_value.data_ptr()

    cache_unshared_kv_torch(
        expected_key,
        expected_value,
        curr_key,
        curr_value,
        block_table,
        decode_step,
    )
    returned_key, returned_value = cache_unshared_kv_triton(
        actual_key,
        actual_value,
        curr_key,
        curr_value,
        block_table,
        decode_step,
    )

    assert returned_key is actual_key
    assert returned_value is actual_value
    assert actual_key.data_ptr() == key_ptr
    assert actual_value.data_ptr() == value_ptr
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def test_cache_unshared_kv_triton_cuda_graph_uses_device_step() -> None:
    shape = (2, 4, 2, 3, 16)
    key = torch.zeros(shape, device="cuda", dtype=torch.float16)
    value = torch.zeros_like(key)
    curr_key = torch.randn((4, 2, 16), device="cuda", dtype=key.dtype)
    curr_value = torch.randn_like(curr_key)
    block_table = torch.tensor([1], device="cuda", dtype=torch.int32)
    decode_step = torch.tensor([1], device="cuda", dtype=torch.int32)
    replay = partial(
        cache_unshared_kv_triton,
        key,
        value,
        curr_key,
        curr_value,
        block_table,
        decode_step,
    )

    replay()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned_key, returned_value = replay()

    decode_step.fill_(3)
    graph.replay()
    torch.accelerator.synchronize()

    assert returned_key is key
    assert returned_value is value
    torch.testing.assert_close(key[1, :, :, 2, :], curr_key, rtol=0, atol=0)
    torch.testing.assert_close(value[1, :, :, 2, :], curr_value, rtol=0, atol=0)


def _prefix_from_parents(parents: torch.Tensor, beam_size: int) -> torch.Tensor:
    return torch.stack([torch.bincount(row, minlength=beam_size).cumsum(0) for row in parents]).to(
        torch.int32
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
@pytest.mark.parametrize("global_prefix", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("strided", [False, True])  # type: ignore[misc]
def test_select_unshared_kv_triton_matches_torch(
    dtype: torch.dtype,
    global_prefix: bool,
    strided: bool,
) -> None:
    batch_size, beam_size, layers = 2, 6, 2
    blocks, heads, steps, dim = 5, 2, 3, 17
    shape = (blocks, beam_size, heads, steps, dim)
    make_cache = (
        _strided_zeros
        if strided
        else lambda shape, dtype: torch.zeros(shape, device="cuda", dtype=dtype)
    )
    actual_keys = [make_cache(shape, dtype) for _ in range(layers)]
    actual_values = [make_cache(shape, dtype) for _ in range(layers)]
    for layer in range(layers):
        actual_keys[layer].copy_(torch.randn_like(actual_keys[layer]))
        actual_values[layer].copy_(torch.randn_like(actual_values[layer]))
    expected_keys = [key.clone() for key in actual_keys]
    expected_values = [value.clone() for value in actual_values]
    parents = torch.tensor(
        [[0, 0, 2, 4, 4, 4], [1, 1, 1, 3, 5, 5]],
        device="cuda",
        dtype=torch.int64,
    )
    prefix = _prefix_from_parents(parents, beam_size).to(device="cuda")
    if global_prefix:
        prefix += torch.arange(batch_size, device="cuda", dtype=torch.int32)[:, None] * beam_size
    beam_index = parents.to(torch.int32).reshape(-1)
    block_table = torch.tensor([3, 1], device="cuda", dtype=torch.int32)

    _select_unshared_kv_torch(
        beam_index,
        expected_keys,
        expected_values,
        block_table,
        prefix.reshape(-1),
        1,
        beam_size,
        layers,
    )
    result = select_unshared_kv_triton(
        beam_index,
        actual_keys,
        actual_values,
        block_table,
        prefix.reshape(-1),
        1,
        beam_size,
        layers,
    )

    assert result is None
    for actual, expected in zip(actual_keys, expected_keys):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(actual_values, expected_values):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_select_unshared_kv_triton_cuda_graph() -> None:
    beam_size, layers = 4, 1
    shape = (2, beam_size, 2, 3, 16)
    key = torch.randn(shape, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    original_key = key.clone()
    original_value = value.clone()
    parents = torch.tensor([[0, 0, 2, 2]], device="cuda", dtype=torch.int64)
    prefix = _prefix_from_parents(parents, beam_size).to(device="cuda").reshape(-1)
    beam_index = parents.to(torch.int32).reshape(-1)
    block_table = torch.tensor([1], device="cuda", dtype=torch.int32)
    replay = partial(
        select_unshared_kv_triton,
        beam_index,
        [key],
        [value],
        block_table,
        prefix,
        1,
        beam_size,
        layers,
    )

    replay()
    key.copy_(original_key)
    value.copy_(original_value)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay()
    graph.replay()
    torch.accelerator.synchronize()

    expected_key = original_key.clone()
    expected_value = original_value.clone()
    _select_unshared_kv_torch(
        beam_index,
        [expected_key],
        [expected_value],
        block_table,
        prefix,
        1,
        beam_size,
        layers,
    )
    torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value, expected_value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "batch,beam,candidates,result_width,step",
    [
        (1, 8, 32, 32, 1),
        (2, 16, 64, 64, 2),
        (1, 128, 128, 128, 2),
        (1, 8, 32, 1, 1),
        (1, 8, 32, 8 * 32, 1),
    ],
)  # type: ignore[misc]
def test_final_select_triton_matches_torch(
    batch: int,
    beam: int,
    candidates: int,
    result_width: int,
    step: int,
) -> None:
    torch.manual_seed(2026)
    log_probs = torch.randn((batch * beam, 1), device="cuda", dtype=torch.float32)
    top_tokens = torch.arange(batch * beam * candidates, device="cuda", dtype=torch.int32).reshape(
        batch * beam, candidates
    )
    top_probs = torch.randn((batch * beam, candidates), device="cuda", dtype=torch.float32)
    sequence = torch.arange(batch * beam * (step + 2), device="cuda", dtype=torch.int32).reshape(
        batch, beam, step + 2
    )

    expected = _beam_search_rec_final_select_torch(
        log_probs, top_tokens, top_probs, sequence, step, result_width
    )
    actual = beam_search_rec_final_select_triton(
        log_probs, top_tokens, top_probs, sequence, step, result_width
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_final_select_triton_stable_ties_and_cuda_graph() -> None:
    batch, beam, candidates, width, step = 1, 3, 4, 8, 1
    log_probs = torch.zeros((beam, 1), device="cuda", dtype=torch.float32)
    top_probs = torch.zeros((beam, candidates), device="cuda", dtype=torch.float32)
    top_tokens = torch.arange(beam * candidates, device="cuda", dtype=torch.int32).reshape(
        beam, candidates
    )
    sequence = torch.arange(beam * 3, device="cuda", dtype=torch.int32).reshape(batch, beam, 3)
    replay = partial(
        beam_search_rec_final_select_triton,
        log_probs,
        top_tokens,
        top_probs,
        sequence,
        step,
        width,
    )

    expected = _beam_search_rec_final_select_torch(
        log_probs, top_tokens, top_probs, sequence, step, width
    )
    replay()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = replay()
    graph.replay()
    torch.accelerator.synchronize()

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize(
    "batch,beam,top_k,step",
    [
        (1, 8, 4, 1),
        (1, 8, 8, 2),
        (2, 8, 16, 1),
        (1, 128, 128, 2),
        (2, 9, 7, 2),
        (1, 4, 20, 1),
        (2, 8, 8, 0),
    ],
)  # type: ignore[misc]
def test_group_triton_matches_torch(
    batch: int,
    beam: int,
    top_k: int,
    step: int,
) -> None:
    torch.manual_seed(10)
    log_probs = torch.randn((batch * beam, 1), device="cuda", dtype=torch.float32)
    top_tokens = torch.arange(batch * beam * beam, device="cuda", dtype=torch.int32).reshape(
        batch * beam, beam
    )
    top_probs = torch.randn((batch * beam, beam), device="cuda", dtype=torch.float32)
    sequence = torch.arange(batch * beam * (step + 2), device="cuda", dtype=torch.int32).reshape(
        batch, beam, step + 2
    )

    expected = _beam_search_group_torch(log_probs, top_tokens, top_probs, sequence, step, top_k)
    actual = beam_search_group_triton(log_probs, top_tokens, top_probs, sequence, step, top_k)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_group_triton_stable_parent_grouping_and_cuda_graph() -> None:
    batch, beam, top_k, step = 1, 4, 4, 1
    log_probs = torch.zeros((beam, 1), device="cuda", dtype=torch.float32)
    top_probs = torch.full((beam, beam), -10.0, device="cuda", dtype=torch.float32)
    top_probs[3, 0] = 4.0
    top_probs[1, 0:3] = torch.tensor([3.0, 2.0, 1.0], device="cuda")
    top_tokens = torch.arange(beam * beam, device="cuda", dtype=torch.int32).reshape(beam, beam)
    sequence = torch.arange(beam * 3, device="cuda", dtype=torch.int32).reshape(batch, beam, 3)
    replay = partial(
        beam_search_group_triton,
        log_probs,
        top_tokens,
        top_probs,
        sequence,
        step,
        top_k,
    )

    expected = _beam_search_group_torch(log_probs, top_tokens, top_probs, sequence, step, top_k)
    replay()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = replay()
    graph.replay()
    torch.accelerator.synchronize()

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_fused_score_topk_stable_ties_and_cuda_graph() -> None:
    batch, beam, step = 1, 128, 2
    log_probs = torch.zeros((beam, 1), device="cuda", dtype=torch.float32)
    top_probs = torch.zeros((beam, beam), device="cuda", dtype=torch.float32)
    top_tokens = torch.arange(beam * beam, device="cuda", dtype=torch.int32).reshape(beam, beam)
    sequence = torch.arange(beam * (step + 2), device="cuda", dtype=torch.int32).reshape(
        batch, beam, step + 2
    )
    cases = (
        (
            partial(
                beam_search_rec_final_select_triton,
                log_probs,
                top_tokens,
                top_probs,
                sequence,
                step,
                beam,
            ),
            _beam_search_rec_final_select_torch(
                log_probs, top_tokens, top_probs, sequence, step, beam
            ),
        ),
        (
            partial(
                beam_search_group_triton,
                log_probs,
                top_tokens,
                top_probs,
                sequence,
                step,
                beam,
            ),
            _beam_search_group_torch(log_probs, top_tokens, top_probs, sequence, step, beam),
        ),
    )

    for replay, expected in cases:
        replay()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = replay()
        graph.replay()
        torch.accelerator.synchronize()
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
