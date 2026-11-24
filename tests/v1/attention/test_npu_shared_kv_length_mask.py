# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Metadata and activation guards for the shared-KV length mask."""

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
    spec = importlib.util.spec_from_file_location("_test_shared_kv_length_mask", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize(  # type: ignore[misc]
    "case,expected",
    [
        ("nd", None),
        ("nchw", None),
        ("missing", "prefix_lengths_missing"),
        ("cpu", "prefix_lengths_device"),
        ("int64", "prefix_lengths_dtype"),
        ("float32", "prefix_lengths_dtype"),
        ("scalar", "prefix_lengths_shape"),
        ("matrix", "prefix_lengths_shape"),
        ("short", "prefix_lengths_shape"),
        ("long", "prefix_lengths_shape"),
        ("zero_stride", "prefix_lengths_stride"),
        ("stride2", "prefix_lengths_stride"),
        ("blocked", "prefix_lengths_format"),
    ],
)
def test_bad_length_metadata_is_rejected_without_value_reads(
    module: Any, monkeypatch: pytest.MonkeyPatch, case: str, expected: str | None
) -> None:
    device = SimpleNamespace(type="npu", index=0)
    key = SimpleNamespace(device=device)
    table = SimpleNamespace(shape=(4, 8))
    lengths: Any = SimpleNamespace(
        device=device,
        dtype=torch.int32,
        ndim=1,
        shape=(4,),
        stride=lambda dimension: 1,
        npu_format=2,
    )
    if case == "missing":
        lengths = None
    elif case == "nchw":
        lengths.npu_format = 0
    elif case == "cpu":
        lengths.device = SimpleNamespace(type="cpu")
    elif case in ("int64", "float32"):
        lengths.dtype = getattr(torch, case)
    elif case == "scalar":
        lengths.ndim, lengths.shape = 0, ()
    elif case == "matrix":
        lengths.ndim, lengths.shape = 2, (4, 1)
    elif case in ("short", "long"):
        lengths.shape = (3 if case == "short" else 5,)
    elif case in ("zero_stride", "stride2"):
        lengths.stride = lambda dimension: 0 if case == "zero_stride" else 2
    elif case == "blocked":
        lengths.npu_format = 29

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Length dispatch must not inspect device values")

    if lengths is not None:
        lengths.item = lengths.cpu = lengths.tolist = forbidden
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(get_npu_format=lambda tensor: tensor.npu_format)
    )
    assert module._unsupported_length_reason(key, table, lengths) == expected


@pytest.mark.parametrize("value", ["", "yes", "true", "2", "-1"])  # type: ignore[misc]
def test_invalid_length_mask_flag_fails(
    module: Any, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("VLLM_GR_NPU_SHARED_KV_LENGTH_MASK", value)
    monkeypatch.setenv("VLLM_GR_NPU_FUSED_SHARED_KV_GATHER", "1")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        module.try_fused_shared_kv_gather(None, None, None, 0)


def test_length_mask_requires_explicit_fused_copy(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_GR_NPU_SHARED_KV_LENGTH_MASK", "1")
    monkeypatch.setenv("VLLM_GR_NPU_FUSED_SHARED_KV_GATHER", "0")
    with pytest.raises(ValueError, match="requires"):
        module.try_fused_shared_kv_gather(None, None, None, 0)


def test_fused_copy_and_length_mask_default_on(module: Any) -> None:
    stats = module.fused_shared_kv_gather_stats()
    assert stats["requested"] == stats["length_mask_requested"] == "1"


@pytest.mark.parametrize("fused", ["0", "1"])  # type: ignore[misc]
def test_unset_length_mask_follows_fused_request(
    module: Any, monkeypatch: pytest.MonkeyPatch, fused: str
) -> None:
    monkeypatch.setenv("VLLM_GR_NPU_FUSED_SHARED_KV_GATHER", fused)
    assert module.fused_shared_kv_gather_stats()["length_mask_requested"] == fused


def test_single_disabled_flag_uses_complete_torch_fallback(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_GR_NPU_FUSED_SHARED_KV_GATHER", "0")
    assert module.try_fused_shared_kv_gather(None, None, None, 0) is None
    assert module.fused_shared_kv_gather_stats()["host_launch_counts"] == {}


pytestmark = [pytest.mark.cpu_test, pytest.mark.l2]
