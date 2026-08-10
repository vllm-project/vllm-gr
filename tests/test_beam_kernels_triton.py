# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA correctness tests for incremental BeamSearch KV Triton operators."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

from vllm_gr.v1.attention.backends.beam_kernels import (
    cache_unshared_kv,
    select_unshared_kv,
)
from vllm_gr.v1.attention.backends.beam_kernels_triton import (
    cache_unshared_kv_triton,
    select_unshared_kv_triton,
)

DTYPES = [torch.float16, torch.bfloat16]


def _pool(dtype: torch.dtype, layer: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (1, 6, 2, 4, 3)
    base = torch.arange(
        torch.tensor(shape).prod().item(), device="cuda", dtype=torch.float32
    ).reshape(shape)
    key = (base + layer * 1000).to(dtype)
    value = (base + layer * 2000 + 17).to(dtype)
    return key, value


@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore[misc]
@pytest.mark.parametrize("step", [0, 2, 3])  # type: ignore[misc]
def test_cache_unshared_kv_triton_matches_reference(
    dtype: torch.dtype,
    step: int,
) -> None:
    actual_key, actual_value = _pool(dtype)
    expected_key = actual_key.clone()
    expected_value = actual_value.clone()
    key_ptr = actual_key.data_ptr()
    value_ptr = actual_value.data_ptr()
    curr_key = torch.randn((4, 2, 3), device="cuda", dtype=dtype)
    curr_value = torch.randn_like(curr_key)

    cache_unshared_kv(expected_key, expected_value, curr_key, curr_value, step)
    result = cache_unshared_kv_triton(actual_key, actual_value, curr_key, curr_value, step)

    assert result is None
    assert actual_key.data_ptr() == key_ptr
    assert actual_value.data_ptr() == value_ptr
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore[misc]
@pytest.mark.parametrize("parent_dtype", [torch.int32, torch.int64])  # type: ignore[misc]
@pytest.mark.parametrize("step", [0, 3])  # type: ignore[misc]
@pytest.mark.parametrize(
    "parent_values",
    [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [3, 1, 1, 1],
    ],
    ids=["identity", "permutation", "duplicate-parent"],
)  # type: ignore[misc]
def test_select_unshared_kv_triton_matches_reference(
    dtype: torch.dtype,
    parent_dtype: torch.dtype,
    step: int,
    parent_values: list[int],
) -> None:
    actual_pairs = [_pool(dtype, layer=layer) for layer in range(2)]
    actual_keys = [pair[0] for pair in actual_pairs]
    actual_values = [pair[1] for pair in actual_pairs]
    expected_keys = [key.clone() for key in actual_keys]
    expected_values = [value.clone() for value in actual_values]
    key_ptrs = [key.data_ptr() for key in actual_keys]
    value_ptrs = [value.data_ptr() for value in actual_values]
    parent = torch.tensor(parent_values, dtype=parent_dtype, device="cuda")

    select_unshared_kv(parent, expected_keys, expected_values, step)
    result = select_unshared_kv_triton(parent, actual_keys, actual_values, step)

    assert result is None
    assert [key.data_ptr() for key in actual_keys] == key_ptrs
    assert [value.data_ptr() for value in actual_values] == value_ptrs
    for actual_key, expected_key in zip(actual_keys, expected_keys):
        torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)
    for actual_value, expected_value in zip(actual_values, expected_values):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def test_triton_operators_reject_cpu_tensors() -> None:
    shape = (1, 4, 2, 3, 4)
    key = torch.zeros(shape, dtype=torch.float16)
    value = torch.zeros_like(key)
    curr_key = torch.zeros((4, 2, 4), dtype=key.dtype)
    curr_value = torch.zeros_like(curr_key)
    parent = torch.arange(4, dtype=torch.int32)

    with pytest.raises(ValueError, match="require CUDA"):
        cache_unshared_kv_triton(key, value, curr_key, curr_value, 0)
    with pytest.raises(ValueError, match="require CUDA"):
        select_unshared_kv_triton(parent, [key], [value], 0)
