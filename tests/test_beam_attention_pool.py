# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contract tests for fixed-address Beam Attention storage."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vllm_gr.v1.beam import BeamAttentionPool
from vllm_gr.v1.beam.beam_attention_pool import BeamAttentionPool as PoolImplementation


def _make_pool(
    *,
    max_beams: int = 4,
    local_num_kv_heads: int = 2,
    max_decode_steps: int = 3,
    head_dim: int = 8,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float16,
) -> BeamAttentionPool:
    return BeamAttentionPool(
        max_beams=max_beams,
        local_num_kv_heads=local_num_kv_heads,
        max_decode_steps=max_decode_steps,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_allocation_contract(dtype: torch.dtype) -> None:
    pool = _make_pool(dtype=dtype)
    expected_shape = (1, 4, 2, 3, 8)

    for tensor in (pool.unshared_key, pool.unshared_value):
        assert tensor.shape == expected_shape
        assert tensor.dtype == dtype
        assert tensor.device == torch.device("cpu")
        assert tensor.is_contiguous()
        assert not tensor.requires_grad

    assert pool.unshared_key.data_ptr() != pool.unshared_value.data_ptr()


@pytest.mark.parametrize("beam_width", [1, 2, 4])  # type: ignore[misc]
def test_active_views_share_storage(beam_width: int) -> None:
    pool = _make_pool()
    pool.unshared_key.fill_(-1)
    pool.unshared_value.fill_(-2)
    key_id = id(pool.unshared_key)
    value_id = id(pool.unshared_value)
    key_ptr = pool.unshared_key.data_ptr()
    value_ptr = pool.unshared_value.data_ptr()

    key_view = pool.active_key(beam_width)
    value_view = pool.active_value(beam_width)

    assert key_view.shape == (1, beam_width, 2, 3, 8)
    assert value_view.shape == (1, beam_width, 2, 3, 8)
    assert key_view.data_ptr() == key_ptr
    assert value_view.data_ptr() == value_ptr

    key_view.fill_(3)
    value_view.fill_(4)
    assert torch.all(pool.unshared_key[:, :beam_width] == 3)
    assert torch.all(pool.unshared_value[:, :beam_width] == 4)
    if beam_width < 4:
        assert torch.all(pool.unshared_key[:, beam_width:] == -1)
        assert torch.all(pool.unshared_value[:, beam_width:] == -2)

    assert id(pool.unshared_key) == key_id
    assert id(pool.unshared_value) == value_id
    assert pool.unshared_key.data_ptr() == key_ptr
    assert pool.unshared_value.data_ptr() == value_ptr


@pytest.mark.parametrize("beam_width", [1.5, "1", None, True, False])  # type: ignore[misc]
def test_active_width_rejects_non_integer_and_bool(beam_width: object) -> None:
    pool = _make_pool()
    with pytest.raises(TypeError):
        pool.active_key(beam_width)
    with pytest.raises(TypeError):
        pool.active_value(beam_width)


@pytest.mark.parametrize("beam_width", [0, -1, 5])  # type: ignore[misc]
def test_active_width_rejects_out_of_range_values(beam_width: int) -> None:
    pool = _make_pool()
    with pytest.raises(ValueError):
        pool.active_key(beam_width)
    with pytest.raises(ValueError):
        pool.active_value(beam_width)


def test_reset_preserves_tensor_identity_and_storage() -> None:
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

    assert pool.reset() is None
    assert torch.count_nonzero(pool.unshared_key).item() == 0
    assert torch.count_nonzero(pool.unshared_value).item() == 0
    assert pool.reset() is None

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
    ("max_beams", "local_num_kv_heads", "max_decode_steps", "head_dim", "dtype"),
    [
        (2, 1, 2, 4, torch.float16),
        (3, 2, 5, 7, torch.bfloat16),
        (128, 8, 3, 128, torch.float16),
    ],
)
def test_nbytes_matches_allocated_tensors(
    max_beams: int,
    local_num_kv_heads: int,
    max_decode_steps: int,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    pool = _make_pool(
        max_beams=max_beams,
        local_num_kv_heads=local_num_kv_heads,
        max_decode_steps=max_decode_steps,
        head_dim=head_dim,
        dtype=dtype,
    )
    expected = (
        2
        * max_beams
        * local_num_kv_heads
        * max_decode_steps
        * head_dim
        * torch.empty((), dtype=dtype).element_size()
    )
    assert pool.nbytes() == expected
    if (max_beams, local_num_kv_heads, max_decode_steps, head_dim) == (128, 8, 3, 128):
        assert pool.nbytes() == 1_572_864


@pytest.mark.parametrize(
    "field", ["max_beams", "local_num_kv_heads", "max_decode_steps", "head_dim"]
)  # type: ignore[misc]
@pytest.mark.parametrize("value", [0, -1])  # type: ignore[misc]
def test_constructor_rejects_non_positive_dimensions(field: str, value: int) -> None:
    kwargs: dict[str, Any] = {
        "max_beams": 4,
        "local_num_kv_heads": 2,
        "max_decode_steps": 3,
        "head_dim": 8,
        "device": torch.device("cpu"),
        "dtype": torch.float16,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        BeamAttentionPool(**kwargs)


@pytest.mark.parametrize(
    "field", ["max_beams", "local_num_kv_heads", "max_decode_steps", "head_dim"]
)  # type: ignore[misc]
@pytest.mark.parametrize("value", [1.5, "1", None, True, False])  # type: ignore[misc]
def test_constructor_rejects_non_integer_dimensions(field: str, value: object) -> None:
    kwargs: dict[str, Any] = {
        "max_beams": 4,
        "local_num_kv_heads": 2,
        "max_decode_steps": 3,
        "head_dim": 8,
        "device": torch.device("cpu"),
        "dtype": torch.float16,
    }
    kwargs[field] = value
    with pytest.raises(TypeError):
        BeamAttentionPool(**kwargs)


_UNSUPPORTED_DTYPES = [torch.float32, torch.int32]
_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
if isinstance(_FP8_DTYPE, torch.dtype):
    _UNSUPPORTED_DTYPES.append(_FP8_DTYPE)


@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)  # type: ignore[misc]
def test_constructor_rejects_unsupported_torch_dtypes(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError):
        _make_pool(dtype=dtype)


def test_constructor_rejects_non_torch_dtype() -> None:
    with pytest.raises(TypeError):
        _make_pool(dtype="float16")


def test_constructor_rejects_non_torch_device() -> None:
    with pytest.raises(TypeError):
        _make_pool(device="cpu")


def test_constructor_rejects_meta_device() -> None:
    with pytest.raises(ValueError):
        _make_pool(device=torch.device("meta"))


def test_constructor_validates_before_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_empty(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("torch.empty must not be called for invalid parameters")

    monkeypatch.setattr(torch, "empty", unexpected_empty)
    with pytest.raises(TypeError):
        _make_pool(device="cpu")


def test_public_export_is_stable() -> None:
    assert BeamAttentionPool is PoolImplementation


def _assert_accelerator_reset_contract(device: torch.device) -> None:
    pool = _make_pool(device=device)
    pool.unshared_key.fill_(1)
    pool.unshared_value.fill_(1)
    key_ptr = pool.unshared_key.data_ptr()
    value_ptr = pool.unshared_value.data_ptr()

    pool.reset()

    assert torch.count_nonzero(pool.unshared_key).item() == 0
    assert torch.count_nonzero(pool.unshared_value).item() == 0
    assert pool.unshared_key.data_ptr() == key_ptr
    assert pool.unshared_value.data_ptr() == value_ptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")  # type: ignore[misc]
def test_cuda_pool_reset_preserves_addresses() -> None:
    _assert_accelerator_reset_contract(torch.device("cuda"))


def _npu_is_available() -> bool:
    npu = getattr(torch, "npu", None)
    return npu is not None and bool(npu.is_available())


@pytest.mark.skipif(not _npu_is_available(), reason="NPU is not available")  # type: ignore[misc]
def test_npu_pool_reset_preserves_addresses() -> None:
    _assert_accelerator_reset_contract(torch.device("npu"))
