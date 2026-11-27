# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Contract tests for fixed-address Beam Attention storage."""

from __future__ import annotations

import pytest
import torch

from vllm_gr.v1.beam.attention_pool import BeamAttentionPool


def _make_pool(
    *,
    max_batch: int = 1,
    max_beams: int = 4,
    num_kv_heads: int = 2,
    max_decode_steps: int = 3,
    head_dim: int = 8,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float16,
) -> BeamAttentionPool:
    return BeamAttentionPool(
        max_batch=max_batch,
        max_beams=max_beams,
        num_kv_heads=num_kv_heads,
        max_decode_steps=max_decode_steps,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_allocation_contract(dtype: torch.dtype) -> None:
    pool = _make_pool(dtype=dtype)
    expected_shape = (1, 4, 2, 3, 8)

    assert pool.dtype == dtype

    for tensor in (pool.unshared_key, pool.unshared_value):
        assert tensor.shape == expected_shape
        assert tensor.dtype == dtype
        assert tensor.device == torch.device("cpu")
        assert tensor.is_contiguous()
        assert not tensor.requires_grad

    assert pool.unshared_key.data_ptr() != pool.unshared_value.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float64, torch.int8])  # type: ignore[misc]
def test_pool_rejects_non_activation_dtype(dtype: torch.dtype) -> None:
    with pytest.raises(
        ValueError,
        match="dtype must be float16, bfloat16 or float32",
    ):
        _make_pool(dtype=dtype)


def test_pool_rejects_non_torch_dtype() -> None:
    with pytest.raises(TypeError, match="dtype must be a torch.dtype"):
        _make_pool(dtype="float16")


@pytest.mark.parametrize("max_batch", [1, 2, 4])  # type: ignore[misc]
def test_pool_shape_reflects_max_batch(max_batch: int) -> None:
    pool = _make_pool(
        max_batch=max_batch, max_beams=4, num_kv_heads=2, max_decode_steps=3, head_dim=8
    )
    expected_shape = (max_batch, 4, 2, 3, 8)
    assert pool.unshared_key.shape == expected_shape
    assert pool.unshared_value.shape == expected_shape


def test_reset_slot_zeros_only_target_slot() -> None:
    pool = _make_pool(max_batch=2, max_beams=4, num_kv_heads=2, max_decode_steps=3, head_dim=8)
    pool.unshared_key.fill_(1)
    pool.unshared_value.fill_(2)

    pool.reset_slot(0)

    assert torch.count_nonzero(pool.unshared_key[0]).item() == 0
    assert torch.count_nonzero(pool.unshared_value[0]).item() == 0
    # Slot 1 should be untouched
    assert torch.all(pool.unshared_key[1] == 1)
    assert torch.all(pool.unshared_value[1] == 2)

    # Reset slot 1 as well
    pool.reset_slot(1)
    assert torch.count_nonzero(pool.unshared_key[1]).item() == 0
    assert torch.count_nonzero(pool.unshared_value[1]).item() == 0


def test_reset_slot_preserves_tensor_identity_and_storage() -> None:
    pool = _make_pool(dtype=torch.bfloat16)
    pool.unshared_key.fill_(1)
    pool.unshared_value.fill_(2)
    key_id = id(pool.unshared_key)
    value_id = id(pool.unshared_value)
    key_ptr = pool.unshared_key.data_ptr()
    value_ptr = pool.unshared_value.data_ptr()
    key_properties = (pool.unshared_key.shape, pool.unshared_key.dtype, pool.unshared_key.device)
    value_properties = (
        pool.unshared_value.shape,
        pool.unshared_value.dtype,
        pool.unshared_value.device,
    )

    assert pool.reset_slot(0) is None
    assert torch.count_nonzero(pool.unshared_key[0]).item() == 0
    assert torch.count_nonzero(pool.unshared_value[0]).item() == 0
    assert pool.reset_slot(0) is None

    assert id(pool.unshared_key) == key_id
    assert id(pool.unshared_value) == value_id
    assert pool.unshared_key.data_ptr() == key_ptr
    assert pool.unshared_value.data_ptr() == value_ptr
    assert (
        pool.unshared_key.shape,
        pool.unshared_key.dtype,
        pool.unshared_key.device,
    ) == key_properties
    assert (
        pool.unshared_value.shape,
        pool.unshared_value.dtype,
        pool.unshared_value.device,
    ) == value_properties


@pytest.mark.parametrize(  # type: ignore[misc]
    ("max_batch", "max_beams", "num_kv_heads", "max_decode_steps", "head_dim", "dtype"),
    [
        (1, 2, 1, 2, 4, torch.float16),
        (1, 3, 2, 5, 7, torch.bfloat16),
        (2, 4, 2, 3, 8, torch.float16),
        (1, 128, 8, 3, 128, torch.float16),
    ],
)
def test_allocation_shape_matches_parameters(
    max_batch: int,
    max_beams: int,
    num_kv_heads: int,
    max_decode_steps: int,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    pool = _make_pool(
        max_batch=max_batch,
        max_beams=max_beams,
        num_kv_heads=num_kv_heads,
        max_decode_steps=max_decode_steps,
        head_dim=head_dim,
        dtype=dtype,
    )
    expected_shape = (max_batch, max_beams, num_kv_heads, max_decode_steps, head_dim)
    assert pool.unshared_key.shape == expected_shape
    assert pool.unshared_value.shape == expected_shape
    element_size = torch.empty((), dtype=dtype).element_size()
    expected_nbytes = (
        2 * max_batch * max_beams * num_kv_heads * max_decode_steps * head_dim * element_size
    )
    assert (
        pool.unshared_key.numel() * element_size + pool.unshared_value.numel() * element_size
        == expected_nbytes
    )
    if (max_batch, max_beams, num_kv_heads, max_decode_steps, head_dim) == (1, 128, 8, 3, 128):
        assert (
            pool.unshared_key.numel() * element_size + pool.unshared_value.numel() * element_size
            == 1_572_864
        )


def test_pool_attributes_stored_correctly() -> None:
    pool = _make_pool(max_batch=2, max_beams=6, num_kv_heads=3, max_decode_steps=5, head_dim=16)
    assert pool.max_batch == 2
    assert pool.max_beams == 6
    assert pool.num_kv_heads == 3
    assert pool.max_decode_steps == 5
    assert pool.head_dim == 16
    assert pool.device == torch.device("cpu")


def _assert_accelerator_reset_contract(device: torch.device) -> None:
    pool = _make_pool(device=device)
    pool.unshared_key.fill_(1)
    pool.unshared_value.fill_(1)
    key_ptr = pool.unshared_key.data_ptr()
    value_ptr = pool.unshared_value.data_ptr()

    pool.reset_slot(0)

    assert torch.count_nonzero(pool.unshared_key[0]).item() == 0
    assert torch.count_nonzero(pool.unshared_value[0]).item() == 0
    assert pool.unshared_key.data_ptr() == key_ptr
    assert pool.unshared_value.data_ptr() == value_ptr


@pytest.mark.gpu_test  # type: ignore[misc]
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")  # type: ignore[misc]
def test_cuda_pool_reset_preserves_addresses() -> None:
    _assert_accelerator_reset_contract(torch.device("cuda"))


def _npu_is_available() -> bool:
    npu = getattr(torch, "npu", None)
    return npu is not None and bool(npu.is_available())


@pytest.mark.npu_test  # type: ignore[misc]
@pytest.mark.skipif(not _npu_is_available(), reason="NPU is not available")  # type: ignore[misc]
def test_npu_pool_reset_preserves_addresses() -> None:
    _assert_accelerator_reset_contract(torch.device("npu"))


pytestmark = pytest.mark.cpu_test
