# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for incremental BeamSearch KV reference operators."""

from __future__ import annotations

import pytest
import torch

from vllm_gr.v1.attention.backends.beam_kernels import (
    cache_unshared_kv,
    select_unshared_kv,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.float16, torch.bfloat16]


def _pool(
    *,
    device: str,
    dtype: torch.dtype,
    layer: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (1, 6, 2, 4, 3)
    base = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    key = (base + layer * 1000).to(device=device, dtype=dtype)
    value = (base + layer * 2000 + 17).to(device=device, dtype=dtype)
    return key, value


@pytest.mark.parametrize("device", DEVICES)  # type: ignore[misc]
@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore[misc]
@pytest.mark.parametrize("step", [0, 2, 3])  # type: ignore[misc]
def test_cache_unshared_kv_writes_only_active_beams_at_step(
    device: str,
    dtype: torch.dtype,
    step: int,
) -> None:
    key, value = _pool(device=device, dtype=dtype)
    expected_key = key.clone()
    expected_value = value.clone()
    key_ptr = key.data_ptr()
    value_ptr = value.data_ptr()
    beam_width = 4
    curr_shape = (beam_width, key.shape[2], key.shape[4])
    curr_key = (
        torch.arange(torch.tensor(curr_shape).prod().item(), device=device, dtype=torch.float32)
        .reshape(curr_shape)
        .to(dtype)
    )
    curr_value = curr_key + 31
    expected_key[0, :beam_width, :, step, :] = curr_key
    expected_value[0, :beam_width, :, step, :] = curr_value

    result = cache_unshared_kv(key, value, curr_key, curr_value, step)

    assert result is None
    assert key.data_ptr() == key_ptr
    assert value.data_ptr() == value_ptr
    torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value, expected_value, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)  # type: ignore[misc]
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
def test_select_unshared_kv_rearranges_executed_history_only(
    device: str,
    dtype: torch.dtype,
    parent_dtype: torch.dtype,
    step: int,
    parent_values: list[int],
) -> None:
    pairs = [_pool(device=device, dtype=dtype, layer=layer) for layer in range(2)]
    keys = [pair[0] for pair in pairs]
    values = [pair[1] for pair in pairs]
    originals_key = [key.clone() for key in keys]
    originals_value = [value.clone() for value in values]
    expected_key = [key.clone() for key in keys]
    expected_value = [value.clone() for value in values]
    key_ptrs = [key.data_ptr() for key in keys]
    value_ptrs = [value.data_ptr() for value in values]
    parent = torch.tensor(parent_values, dtype=parent_dtype, device=device)
    parent_index = parent.to(torch.long)
    beam_width = parent.numel()

    for layer in range(len(keys)):
        expected_key[layer][0, :beam_width, :, : step + 1, :] = originals_key[layer][
            0, parent_index, :, : step + 1, :
        ]
        expected_value[layer][0, :beam_width, :, : step + 1, :] = originals_value[layer][
            0, parent_index, :, : step + 1, :
        ]

    result = select_unshared_kv(parent, keys, values, step)

    assert result is None
    assert [key.data_ptr() for key in keys] == key_ptrs
    assert [value.data_ptr() for value in values] == value_ptrs
    for layer in range(len(keys)):
        torch.testing.assert_close(keys[layer], expected_key[layer], rtol=0, atol=0)
        torch.testing.assert_close(values[layer], expected_value[layer], rtol=0, atol=0)


@pytest.mark.parametrize("operator", ["cache", "select"])  # type: ignore[misc]
def test_beam_kernels_reject_out_of_range_step(operator: str) -> None:
    key, value = _pool(device="cpu", dtype=torch.float16)
    if operator == "cache":
        curr_key = torch.zeros((4, 2, 3), dtype=key.dtype)
        curr_value = torch.zeros_like(curr_key)
        with pytest.raises(ValueError, match="step must be in"):
            cache_unshared_kv(key, value, curr_key, curr_value, key.shape[3])
    else:
        parent = torch.arange(4, dtype=torch.int32)
        with pytest.raises(ValueError, match="step must be in"):
            select_unshared_kv(parent, [key], [value], key.shape[3])


def test_cache_unshared_kv_rejects_dtype_mismatch_without_writing() -> None:
    key, value = _pool(device="cpu", dtype=torch.float16)
    original_key = key.clone()
    original_value = value.clone()
    curr_key = torch.zeros((4, 2, 3), dtype=torch.bfloat16)
    curr_value = torch.zeros_like(curr_key)

    with pytest.raises(TypeError, match="curr_key must have the same dtype"):
        cache_unshared_kv(key, value, curr_key, curr_value, 0)

    assert torch.equal(key, original_key)
    assert torch.equal(value, original_value)


def test_select_unshared_kv_rejects_invalid_parent_without_writing() -> None:
    key, value = _pool(device="cpu", dtype=torch.float16)
    original_key = key.clone()
    original_value = value.clone()
    parent = torch.tensor([0, 4, 1, 2], dtype=torch.int32)

    with pytest.raises(ValueError, match="parent values must be in"):
        select_unshared_kv(parent, [key], [value], 1)

    assert torch.equal(key, original_key)
    assert torch.equal(value, original_value)


def test_select_unshared_kv_validates_every_layer_before_writing() -> None:
    key0, value0 = _pool(device="cpu", dtype=torch.float16)
    key1, value1 = _pool(device="cpu", dtype=torch.float16, layer=1)
    bad_value1 = value1.to(torch.bfloat16)
    original_key0 = key0.clone()
    original_value0 = value0.clone()
    parent = torch.tensor([1, 0, 3, 2], dtype=torch.int32)

    with pytest.raises(TypeError, match="value cache 1 must have the same dtype"):
        select_unshared_kv(parent, [key0, key1], [value0, bad_value1], 1)

    assert torch.equal(key0, original_key0)
    assert torch.equal(value0, original_value0)
