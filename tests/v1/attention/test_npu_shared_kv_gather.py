# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CPU guards for NPU shared-KV gather dispatch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest
import torch


@pytest.fixture  # type: ignore[misc]
def module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv("VLLM_GR_NPU_FUSED_SHARED_KV_GATHER", raising=False)
    monkeypatch.delenv("VLLM_GR_NPU_SHARED_KV_LENGTH_MASK", raising=False)
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm_gr/v1/attention/backends/npu_shared_kv_gather.py"
    )
    spec = importlib.util.spec_from_file_location("_test_npu_shared_kv_gather", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pool = torch.arange(8 * 2 * 4 * 2 * 8, dtype=torch.float32).reshape(8, 2, 4, 2, 8)
    return pool[:, 0], pool[:, 1], torch.tensor([[3, 1, 3], [0, 7, 2]], dtype=torch.int32)


def test_disabled_path_does_not_import_triton_or_allocate(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_GR_NPU_FUSED_SHARED_KV_GATHER", "0")
    key, value, table = inputs()

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Disabled path must not compile or allocate")

    monkeypatch.setattr(module, "_kernel", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    assert module.try_fused_shared_kv_gather(key, value, table, 0) is None
    assert module.fused_shared_kv_gather_stats()["host_launch_counts"] == {}


def test_cpu_fallback_is_observable_and_never_runs_npu_kernel(
    module: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("VLLM_GR_NPU_FUSED_SHARED_KV_GATHER", "1")

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("CPU fallback must not compile a device kernel")

    monkeypatch.setattr(module, "_kernel", forbidden)
    for _ in range(2):
        assert module.try_fused_shared_kv_gather(*inputs(), 0) is None
    assert module.fused_shared_kv_gather_stats()["host_launch_counts"] == {"fallback:device": 2}
    assert sum("unsupported (device)" in item.message for item in caplog.records) == 1


@pytest.mark.parametrize("value", ["", "yes", "true", "2", "-1"])  # type: ignore[misc]
def test_invalid_experiment_flag_fails(
    module: Any, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("VLLM_GR_NPU_FUSED_SHARED_KV_GATHER", value)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        module.try_fused_shared_kv_gather(*inputs(), 0)


def test_cpu_rejection_does_not_read_device_values(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_GR_NPU_FUSED_SHARED_KV_GATHER", "1")

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Dispatch must use metadata, not tensor contents")

    tensors = inputs()
    for name in ("item", "tolist", "cpu"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    assert module.try_fused_shared_kv_gather(*tensors, 0) is None


@pytest.mark.parametrize("format_id", [0, 2, 3, 4, 29, 30])  # type: ignore[misc]
@pytest.mark.parametrize("last_stride", [1, 2])  # type: ignore[misc]
def test_only_unblocked_nchw_and_nd_formats_are_admitted(
    module: Any, monkeypatch: pytest.MonkeyPatch, format_id: int, last_stride: int
) -> None:
    device = SimpleNamespace(type="npu")

    class MetadataOnlyTensor:
        def __init__(
            self,
            shape: tuple[int, ...],
            dtype: torch.dtype,
            strides: tuple[int, ...],
            npu_format: int,
        ) -> None:
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = dtype
            self.device = device
            self._strides = strides
            self.npu_format = npu_format

        def stride(self, dimension: int | None = None) -> tuple[int, ...] | int:
            return self._strides if dimension is None else self._strides[dimension]

    # Strided halves of an interleaved pool; no data or fake kernel runs.
    key = MetadataOnlyTensor((8, 4, 2, 8), torch.bfloat16, (128, 16, 8, last_stride), format_id)
    value = MetadataOnlyTensor((8, 4, 2, 8), torch.bfloat16, (128, 16, 8, last_stride), format_id)
    table = MetadataOnlyTensor((2, 3), torch.int32, (3, 1), 2)
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(get_npu_format=lambda tensor: tensor.npu_format)
    )
    expected = (
        "last_stride" if last_stride != 1 else None if format_id in (0, 2) else "blocked_format"
    )
    assert module._unsupported_reason(key, value, table) == expected
    assert module.fused_shared_kv_gather_stats()["observed_input_formats_key_value_table"] == (
        {f"{format_id},{format_id},2": 1} if last_stride == 1 else {}
    )


def test_integer_dtype_view_preserves_interleaved_cpu_geometry() -> None:
    for dtype, integer in (
        (torch.bfloat16, torch.int16),
        (torch.float16, torch.int16),
        (torch.float32, torch.int32),
    ):
        pool = torch.arange(9 * 2 * 4 * 2 * 8).to(dtype).reshape(9, 2, 4, 2, 8)
        for tensor in (pool[1:, 0], pool[1:, 1]):
            bits = tensor.view(integer)
            assert tensor.data_ptr() == bits.data_ptr()
            assert tensor.stride() == bits.stride()
            assert tensor.shape == bits.shape
            assert tensor.storage_offset() == bits.storage_offset()


def test_non_dense_inner_layout_is_explicitly_rejected(module: Any) -> None:
    device = SimpleNamespace(type="npu")

    def tensor(shape: tuple[int, ...], dtype: torch.dtype, strides: tuple[int, ...]) -> Any:
        return SimpleNamespace(
            device=device,
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            stride=lambda dimension=None: strides if dimension is None else strides[dimension],
        )

    key = tensor((8, 4, 2, 8), torch.bfloat16, (256, 32, 16, 1))
    value = tensor((8, 4, 2, 8), torch.bfloat16, (256, 32, 16, 1))
    table = tensor((2, 3), torch.int32, (3, 1))
    assert module._unsupported_reason(key, value, table) == "inner_layout"


pytestmark = [pytest.mark.cpu_test, pytest.mark.l2]
