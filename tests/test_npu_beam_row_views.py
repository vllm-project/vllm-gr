# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Layout and device guards for NPU attention views and direct shared KV."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from tools.hardware_marks import hardware_test


class _MockNPUTensor(torch.Tensor):  # type: ignore[misc]
    @property
    def device(self) -> torch.device:
        # Tensor storage and all arithmetic stay on CPU; only the host guard
        # sees an NPU. This does not exercise the real native operator.
        return cast(torch.device, SimpleNamespace(type="npu", index=0))


class _MockCUDATensor(torch.Tensor):  # type: ignore[misc]
    @property
    def device(self) -> torch.device:
        return cast(torch.device, SimpleNamespace(type="cuda", index=0))


@pytest.fixture  # type: ignore[misc]
def row_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the real helper with only its native availability/format mocked."""
    from vllm_gr.ops.npu import beam_ops

    try:
        importlib.import_module("torch_npu")
    except (ImportError, RuntimeError):
        stub = ModuleType("torch_npu")
        stub.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
        stub.__file__ = "/mock/torch_npu/__init__.py"
        monkeypatch.setitem(sys.modules, "torch_npu", stub)
    monkeypatch.setattr(beam_ops, "is_available", lambda: True)
    source = (
        Path(__file__).resolve().parents[1] / "vllm_gr/v1/attention/backends/beam_attn_npu_xllm.py"
    )
    spec = importlib.util.spec_from_file_location("_test_npu_beam_row_views_xllm", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "torch_npu", SimpleNamespace(get_npu_format=lambda _tensor: 2))
    return module


def _metadata(rows: int) -> SimpleNamespace:
    return SimpleNamespace(
        npu_prefix_rows_are_identity=True,
        is_beam_decode=True,
        std_indices=None,
        num_actual_tokens=rows,
        num_prefix_indices=rows,
        prefix_indices=torch.arange(rows, dtype=torch.int64),
    )


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("mixed", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "invalid_state", ["not_decode", "missing_session", "inactive_session", "multiple_queries"]
)
def test_xllm_rejects_invalid_beam_state_before_dispatch(
    row_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    mixed: bool,
    invalid_state: str,
) -> None:
    metadata = _metadata(4)
    metadata.beam_session = SimpleNamespace(is_active=True)
    metadata.suffix_max_q_len = 1
    if mixed:
        metadata.std_indices = torch.tensor([4])
    if invalid_state == "not_decode":
        metadata.is_beam_decode = False
    elif invalid_state == "missing_session":
        metadata.beam_session = None
    elif invalid_state == "inactive_session":
        metadata.beam_session.is_active = False
    else:
        metadata.suffix_max_q_len = 2

    fused = Mock()
    cascade = Mock()
    monkeypatch.setattr(row_module, "_xllm_incremental_decode", fused)
    monkeypatch.setattr(row_module.beam_attn_npu, "cascade_attention", cascade)
    tensor = torch.empty(0)
    error = ValueError if invalid_state == "multiple_queries" else RuntimeError
    match = "one query token" if error is ValueError else "active beam session"
    with pytest.raises(error, match=match):
        row_module.cascade_attention(
            tensor, tensor, tensor, tensor, metadata, None, 1.0, 0.0, 2, beam_pool=object()
        )
    fused.assert_not_called()
    cascade.assert_not_called()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "case", ["eligible", "no_window", "standard", "mixed", "no_prefix", "session_pool"]
)
def test_xllm_valid_batches_keep_fused_or_cascade_routing(
    row_module: ModuleType, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    metadata = _metadata(4)
    metadata.beam_session = SimpleNamespace(is_active=True)
    metadata.suffix_max_q_len = 1
    pool = None if case == "session_pool" else object()
    window = (-1, -1) if case == "no_window" else None
    if case == "standard":
        metadata.is_beam_decode = False
        metadata.beam_session = None
        metadata.prefix_indices = None
        metadata.std_indices = torch.arange(4)
    elif case == "mixed":
        metadata.std_indices = torch.tensor([4])
    elif case == "no_prefix":
        metadata.prefix_indices = None
        metadata.std_indices = torch.arange(4)

    fused = Mock()
    cascade = Mock()
    monkeypatch.setattr(row_module, "_xllm_incremental_decode", fused)
    monkeypatch.setattr(row_module.beam_attn_npu, "cascade_attention", cascade)
    tensor = torch.empty(0)
    row_module.cascade_attention(
        tensor, tensor, tensor, tensor, metadata, window, 1.0, 0.0, 2, beam_pool=pool
    )
    if case in {"eligible", "no_window"}:
        fused.assert_called_once_with(
            tensor, tensor, tensor, tensor, metadata, metadata.beam_session, pool, 1.0
        )
        cascade.assert_not_called()
    else:
        cascade.assert_called_once_with(
            tensor,
            tensor,
            tensor,
            tensor,
            metadata,
            window,
            1.0,
            0.0,
            2,
            s_aux=None,
            alibi_slopes=None,
            beam_pool=pool,
        )
        fused.assert_not_called()


@pytest.mark.parametrize("interleaved", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("max_valid_cols", [0, 1, 2, 8, 10])  # type: ignore[misc]
def test_shared_prefix_gather_preserves_key_value_and_block_routing(
    row_module: ModuleType, interleaved: bool, max_valid_cols: int
) -> None:
    """Exercise the real gather on CPU; native graph replay needs an NPU."""
    pool = torch.arange(16 * 2 * 4 * 2 * 8, dtype=torch.float32).reshape(16, 2, 4, 2, 8)
    key, value = pool[:, 0], pool[:, 1]
    if not interleaved:
        key, value = key.contiguous(), value.contiguous()
    table = torch.tensor([[5, 2, 7, 7, 7, 7, 7, 7], [3, 5, 7, 7, 7, 7, 7, 7]], dtype=torch.int32)
    width = min(max_valid_cols, table.shape[1]) if max_valid_cols > 0 else table.shape[1]
    for iteration in range(2):
        if iteration:
            # Fixed-shape inputs can change their referenced blocks and data.
            table[:, 0] = torch.tensor([9, 1], dtype=table.dtype)
            key.add_(1000)
            value.sub_(1000)
        gathered_key, gathered_value, remapped = row_module._gather_shared_blocks(
            key, value, table, max_valid_cols
        )
        assert gathered_key.shape[0] == table.shape[0] * width
        assert gathered_value.shape == gathered_key.shape
        assert gathered_key.is_contiguous() and gathered_value.is_contiguous()
        assert remapped.shape == (table.shape[0], width)
        torch.testing.assert_close(gathered_key[remapped.long()], key[table[:, :width].long()])
        torch.testing.assert_close(gathered_value[remapped.long()], value[table[:, :width].long()])


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])  # type: ignore[misc]
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])  # type: ignore[misc]
@pytest.mark.parametrize("max_valid_cols", [0, 2])  # type: ignore[misc]
def test_interleaved_shared_gather_owns_disjoint_contiguous_key_value_views(
    row_module: ModuleType,
    batch_size: int,
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    max_valid_cols: int,
) -> None:
    """Value's offset view must preserve routing and never alias paged KV.

    This validates CPU storage math, including nonzero source offsets. The
    native x_attention operator still needs its own NPU parity/replay check.
    """
    generator = torch.Generator().manual_seed(17)
    pool = torch.randn(18, 2, 4, 2, 8, generator=generator).to(dtype)
    layer_pool = pool[3:15]
    key, value = layer_pool[:, 0], layer_pool[:, 1]
    assert key.storage_offset() > 0
    table = torch.tensor(
        [[5, 2, 5, 11], [0, 9, 4, 4], [8, 3, 1, 10], [11, 2, 0, 7]][:batch_size],
        dtype=index_dtype,
    )
    width = max_valid_cols or table.shape[1]
    reference_key = key.index_select(0, table[:, :width].reshape(-1))
    reference_value = value.index_select(0, table[:, :width].reshape(-1))
    source_before = layer_pool.clone()

    gathered_key, gathered_value, remapped = row_module._gather_shared_blocks(
        key, value, table, max_valid_cols
    )

    torch.testing.assert_close(gathered_key, reference_key, rtol=0, atol=0)
    torch.testing.assert_close(gathered_value, reference_value, rtol=0, atol=0)
    assert gathered_key.dtype == dtype and gathered_value.dtype == dtype
    assert gathered_key.is_contiguous() and gathered_value.is_contiguous()
    assert gathered_key.untyped_storage().data_ptr() == gathered_value.untyped_storage().data_ptr()
    assert gathered_key.untyped_storage().data_ptr() != pool.untyped_storage().data_ptr()
    assert gathered_key.storage_offset() == 0
    assert gathered_value.storage_offset() == gathered_key.numel()
    assert remapped.dtype == index_dtype
    torch.testing.assert_close(
        gathered_key[remapped.long()], key[table[:, :width].long()], rtol=0, atol=0
    )
    torch.testing.assert_close(
        gathered_value[remapped.long()], value[table[:, :width].long()], rtol=0, atol=0
    )

    # The two offset views must not overlap, and neither may mutate paged KV.
    gathered_key.zero_()
    torch.testing.assert_close(gathered_value, reference_value, rtol=0, atol=0)
    torch.testing.assert_close(layer_pool, source_before, rtol=0, atol=0)


@pytest.mark.parametrize("interleaved", [False, True])  # type: ignore[misc]
def test_shared_gather_preserves_empty_batch(row_module: ModuleType, interleaved: bool) -> None:
    pool = torch.arange(8 * 2 * 4 * 2 * 8, dtype=torch.float32).reshape(8, 2, 4, 2, 8)
    key, value = pool[:, 0], pool[:, 1]
    if not interleaved:
        key, value = key.contiguous(), value.contiguous()
    table = torch.empty((0, 4), dtype=torch.int32)

    gathered_key, gathered_value, remapped = row_module._gather_shared_blocks(key, value, table, 0)

    assert gathered_key.shape == gathered_value.shape == (0, 4, 2, 8)
    assert gathered_key.is_contiguous() and gathered_value.is_contiguous()
    assert remapped.shape == table.shape
    assert remapped.dtype == table.dtype


def test_npu_dense_identity_rows_share_original_storage(
    row_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensor = torch.arange(32).reshape(4, 2, 4).as_subclass(_MockNPUTensor)

    def forbid_device_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("identity selection must not read device values on host")

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "cpu", forbid_device_read)
        guard.setattr(torch.Tensor, "item", forbid_device_read)
        guard.setattr(torch.Tensor, "tolist", forbid_device_read)
        selected = row_module.select_beam_rows(tensor, _metadata(4))
    assert selected.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
    assert selected.stride() == tensor.stride()
    assert selected.storage_offset() == 0
    torch.testing.assert_close(selected.as_subclass(torch.Tensor), tensor.as_subclass(torch.Tensor))


@pytest.mark.parametrize(  # type: ignore[misc]
    "case",
    [
        "unproved",
        "mixed",
        "not-decode",
        "padding",
        "row-count",
        "actual-count",
        "noncontiguous",
        "offset",
        "format",
    ],
)
def test_npu_unproved_or_unsupported_layout_keeps_gather(row_module: ModuleType, case: str) -> None:
    base = torch.arange(40).reshape(5, 2, 4).as_subclass(_MockNPUTensor)
    tensor = base[:4]
    metadata = _metadata(4)
    if case == "unproved":
        metadata.npu_prefix_rows_are_identity = False
        metadata.prefix_indices = torch.tensor([3, 1, 2, 0])
    elif case == "mixed":
        metadata.std_indices = torch.tensor([0])
    elif case == "not-decode":
        metadata.is_beam_decode = False
    elif case == "padding":
        tensor = base
    elif case == "row-count":
        metadata.num_prefix_indices = 3
        metadata.prefix_indices = torch.tensor([0, 1, 3])
    elif case == "actual-count":
        metadata.num_actual_tokens = 5
    elif case == "noncontiguous":
        tensor = tensor.transpose(1, 2)
    elif case == "offset":
        tensor = base[1:]
        assert tensor.is_contiguous() and tensor.storage_offset() != 0
    elif case == "format":
        row_module.torch_npu.get_npu_format = lambda _tensor: 29

    selected = row_module.select_beam_rows(tensor, metadata)
    assert selected.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
    torch.testing.assert_close(
        selected.as_subclass(torch.Tensor),
        tensor[metadata.prefix_indices].as_subclass(torch.Tensor),
    )


@pytest.mark.parametrize("mock_cuda", [False, True])  # type: ignore[misc]
def test_cpu_and_cuda_keep_gather_without_querying_npu_format(
    row_module: ModuleType, mock_cuda: bool
) -> None:
    tensor = torch.arange(32).reshape(4, 2, 4)
    if mock_cuda:
        tensor = tensor.as_subclass(_MockCUDATensor)

    def forbidden_format(_tensor: torch.Tensor) -> int:
        raise AssertionError("non-NPU tensors must not query the NPU format")

    row_module.torch_npu.get_npu_format = forbidden_format
    selected = row_module.select_beam_rows(tensor, _metadata(4))
    assert selected.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()


def _shared_decode_case(offset: int = 0) -> SimpleNamespace:
    rows, heads, kv_heads, dim = 4, 4, 2, 128
    key = torch.randn(7 + offset, 4, kv_heads, dim, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key = key[offset:].as_subclass(_MockNPUTensor)
    value = value[offset:].as_subclass(_MockNPUTensor)
    query = torch.randn(rows, heads, dim, dtype=key.dtype).as_subclass(_MockNPUTensor)
    pool = SimpleNamespace(
        unshared_key=torch.zeros(1, rows, kv_heads, 3, dim, dtype=key.dtype).as_subclass(
            _MockNPUTensor
        ),
        unshared_value=torch.zeros(1, rows, kv_heads, 3, dim, dtype=key.dtype).as_subclass(
            _MockNPUTensor
        ),
    )
    metadata = _metadata(rows)
    metadata.prefix_block_table = torch.tensor([[5, 2, 5]], dtype=torch.int32)
    metadata.prefix_seqlens_kv_device = torch.tensor([9], dtype=torch.int32)
    metadata.max_prefix_blocks_count = 3
    step = torch.tensor([1], dtype=torch.int32)
    session = SimpleNamespace(
        slot=0,
        beam_width=rows,
        decode_step=0,
        context=SimpleNamespace(decode_step_1based_buf=step),
    )
    return SimpleNamespace(
        key=key,
        value=value,
        query=query,
        pool=pool,
        metadata=metadata,
        session=session,
        output=torch.empty(rows, heads, dim, dtype=key.dtype),
    )


def _run_shared_decode(module: ModuleType, case: SimpleNamespace) -> None:
    module._xllm_incremental_decode(
        case.output,
        case.query,
        case.key,
        case.value,
        case.metadata,
        case.session,
        case.pool,
        case.key.shape[-1] ** -0.5,
    )


@pytest.mark.parametrize("offset", [0, 2])  # type: ignore[misc]
@pytest.mark.parametrize("capturing", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("layout", [None, "", "NHD", "LBNHC"])  # type: ignore[misc]
def test_npu_direct_shared_kv_preserves_original_table_and_storage(
    row_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    offset: int,
    capturing: bool,
    layout: str | None,
) -> None:
    """The actual decode dispatcher must bypass all shared-prefix gathers."""
    if layout is None:
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", layout)
    case = _shared_decode_case(offset)
    expected = torch.arange(case.output.numel(), dtype=torch.float32).reshape(case.output.shape)
    expected = expected.to(case.output.dtype)
    calls: list[tuple[Any, ...]] = []

    def attention(*args: Any, out: torch.Tensor | None = None) -> torch.Tensor:
        assert out is None
        calls.append(args)
        assert args[0].data_ptr() == case.query.data_ptr()
        assert args[1] is case.key and args[2] is case.value
        assert args[5] is case.metadata.prefix_block_table
        assert args[6] is case.metadata.prefix_seqlens_kv_device
        assert args[7] is case.session.context.decode_step_1based_buf
        assert args[3].data_ptr() == case.pool.unshared_key.data_ptr()
        assert args[4].data_ptr() == case.pool.unshared_value.data_ptr()
        assert args[8] == 0.0
        return expected

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("direct shared KV must not gather or read device values on host")

    monkeypatch.setattr(row_module, "_npu_stream_capturing", lambda: capturing)
    monkeypatch.setattr(row_module, "_gather_shared_blocks", forbidden)
    monkeypatch.setattr(row_module.beam_ops, "x_attention", attention)
    storage = tuple((t.data_ptr(), t.storage_offset(), t.stride()) for t in (case.key, case.value))
    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "cpu", forbidden)
        guard.setattr(torch.Tensor, "item", forbidden)
        guard.setattr(torch.Tensor, "tolist", forbidden)
        _run_shared_decode(row_module, case)
    assert len(calls) == 1
    assert storage == tuple(
        (t.data_ptr(), t.storage_offset(), t.stride()) for t in (case.key, case.value)
    )
    assert case.key.is_contiguous() and case.value.is_contiguous()
    assert (case.key.storage_offset() != 0) == (offset != 0)
    torch.testing.assert_close(case.output, expected, rtol=0, atol=0)


@pytest.mark.parametrize(  # type: ignore[misc]
    "reason",
    [
        "interleaved",
        "format",
        "suffix-heads",
        "query-heads",
        "query-dim",
        "value-shape",
        "value-dtype",
        "value-device",
        "unsupported-dtype",
        "HND",
        "LBHNC",
        "unknown-layout",
        "cpu",
        "cuda",
    ],
)
def test_unsupported_shared_kv_keeps_original_gather_dispatch(
    row_module: ModuleType, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """Guard failures retain the prior routing and block-table remapping."""
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    case = _shared_decode_case()
    if reason == "interleaved":
        pool = torch.randn(7, 2, 4, 2, 128, dtype=case.key.dtype).as_subclass(_MockNPUTensor)
        case.key, case.value = pool[:, 0], pool[:, 1]
        assert not case.key.is_contiguous()
    elif reason == "format":
        row_module.torch_npu.get_npu_format = lambda _tensor: 29
    elif reason == "suffix-heads":
        case.pool.unshared_key = case.pool.unshared_key[:, :, :1].contiguous()
    elif reason == "query-heads":
        case.query = case.query[:, :3].contiguous()
    elif reason == "query-dim":
        case.query = case.query[..., :64].contiguous()
    elif reason == "value-shape":
        case.value = case.value[:, :3].contiguous()
    elif reason == "value-dtype":
        case.value = case.value.to(torch.float16)
    elif reason == "value-device":
        case.value = case.value.as_subclass(torch.Tensor).as_subclass(_MockCUDATensor)
    elif reason == "unsupported-dtype":
        case.key, case.value = case.key.float(), case.value.float()
    elif reason in ("HND", "LBHNC", "unknown-layout"):
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", reason)
    elif reason in ("cpu", "cuda"):
        for name in ("key", "value", "query"):
            tensor = getattr(case, name).as_subclass(torch.Tensor)
            if reason == "cuda":
                tensor = tensor.as_subclass(_MockCUDATensor)
            setattr(case, name, tensor)

        def forbidden_format(_tensor: torch.Tensor) -> int:
            raise AssertionError("non-NPU tensors must not query the NPU format")

        row_module.torch_npu.get_npu_format = forbidden_format
    # Real gathering still executes; only the native attention invocation is mocked.
    original_gather = row_module._gather_shared_blocks
    gathered: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def gather(*args: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = original_gather(*args)
        gathered.append(result)
        return cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], result)

    def attention(*args: Any, out: torch.Tensor | None = None) -> torch.Tensor:
        assert out is None
        assert len(gathered) == 1
        assert args[1] is gathered[0][0] and args[2] is gathered[0][1]
        assert args[5] is gathered[0][2]
        assert args[5] is not case.metadata.prefix_block_table
        return torch.zeros_like(case.output)

    monkeypatch.setattr(row_module, "_npu_stream_capturing", lambda: False)
    monkeypatch.setattr(row_module, "_gather_shared_blocks", gather)
    monkeypatch.setattr(row_module.beam_ops, "x_attention", attention)
    _run_shared_decode(row_module, case)
    assert len(gathered) == 1
    table = case.metadata.prefix_block_table
    expected_table = torch.arange(table.numel(), dtype=table.dtype).reshape(table.shape)
    torch.testing.assert_close(gathered[0][2], expected_table, rtol=0, atol=0)
    for actual, original in zip(gathered[0][:2], (case.key, case.value)):
        torch.testing.assert_close(
            actual.as_subclass(torch.Tensor),
            original[table.reshape(-1)].as_subclass(torch.Tensor),
            rtol=0,
            atol=0,
        )


def test_cuda_forward_keeps_original_kv_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.attention.backends import beam_attn

    calls: dict[str, torch.Tensor] = {}

    def cache(*args: Any, **_kwargs: Any) -> None:
        calls["key"], calls["value"] = args[2:4]

    def forbid_select(*_args: Any) -> torch.Tensor:
        raise AssertionError("the CUDA forward must not call an NPU row selector")

    execution = SimpleNamespace(
        cache_unshared_kv=cache,
        cascade_attention=lambda **_kwargs: None,
        select_beam_rows=forbid_select,
    )
    monkeypatch.setattr(beam_attn, "_get_platform_execution_module", lambda: execution)
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", False)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", True)
    pool = SimpleNamespace(unshared_key=torch.empty(1), unshared_value=torch.empty(1))
    session = SimpleNamespace(
        is_active=True,
        pools=[pool],
        slot=0,
        decode_step=0,
        context=SimpleNamespace(decode_step_buf=None),
    )
    metadata = _metadata(4)
    metadata.beam_session = session
    metadata.suffix_max_q_len = 1
    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_dtype = "auto"
    impl.sliding_window = (-1, -1)
    impl.scale = 0.5
    impl.logits_soft_cap = 0.0
    impl.vllm_flash_attn_version = 3
    impl.prefill_flash_attn_version = impl.vllm_flash_attn_version
    impl.sinks = None
    impl.alibi_slopes = None
    query = torch.zeros(4, 2, 4)
    key = torch.arange(16).reshape(4, 1, 4).float()
    value = key + 100
    output = torch.empty_like(query)
    returned = impl.forward(
        layer=SimpleNamespace(_beam_pool_index=0),
        query=query,
        key=key,
        value=value,
        kv_cache=(torch.empty(1), torch.empty(1)),
        attn_metadata=metadata,
        output=output,
    )
    assert returned is output
    for name, original in (("key", key), ("value", value)):
        assert calls[name].data_ptr() != original.data_ptr()
        torch.testing.assert_close(calls[name], original)


def _native_modules() -> tuple[Any, Any]:
    torch_npu = pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    from vllm_gr.ops.npu import beam_ops

    if not beam_ops.is_available():
        pytest.skip("xllm-ops is not available")
    module = importlib.import_module("vllm_gr.v1.attention.backends.beam_attn_npu_xllm")
    return torch_npu, module


@hardware_test(res={"npu": "A3"}, num_cards=1)  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
def test_npu_row_view_cache_attention_and_graph_replay_are_exact(
    dtype: torch.dtype, batch_size: int
) -> None:
    """Real graph replay changes data while keeping the producer storage fixed."""
    _torch_npu, module = _native_modules()
    from vllm_gr.ops.npu import beam_ops

    rows, heads, kv_heads, dim = batch_size * 128, 16, 8, 128
    device = torch.device("npu")
    torch.manual_seed(1789 + batch_size)
    metadata = _metadata(rows)
    metadata.prefix_indices = metadata.prefix_indices.to(device)
    query = torch.randn(rows, heads, dim, dtype=dtype, device=device) * 0.1
    key = torch.randn(rows, kv_heads, dim, dtype=dtype, device=device) * 0.1
    value = torch.randn_like(key) * 0.1
    shared_k = torch.randn(batch_size, 128, kv_heads, dim, dtype=dtype, device=device) * 0.1
    shared_v = torch.randn_like(shared_k) * 0.1
    blocks = torch.arange(batch_size, dtype=torch.int32, device=device)
    shared_blocks = blocks.unsqueeze(1)
    lengths = torch.full((batch_size,), 128, dtype=torch.int32, device=device)
    step = torch.ones(1, dtype=torch.int32, device=device)
    pool_shape = (batch_size, 128, kv_heads, 3, dim)
    pools = [
        (
            torch.zeros(pool_shape, dtype=dtype, device=device),
            torch.zeros(pool_shape, dtype=dtype, device=device),
        )
        for _ in range(2)
    ]
    for tensor in (query, key, value):
        assert module.select_beam_rows(tensor, metadata).data_ptr() == tensor.data_ptr()

    def run(mode: int) -> torch.Tensor:
        if mode == 0:
            q = query[metadata.prefix_indices]
            k = key[metadata.prefix_indices]
            v = value[metadata.prefix_indices]
        else:
            q = module.select_beam_rows(query, metadata)
            k = module.select_beam_rows(key, metadata)
            v = module.select_beam_rows(value, metadata)
        pool_k, pool_v = pools[mode]
        beam_ops.cache_unshared_kv(pool_k, pool_v, k, v, blocks, step)
        return beam_ops.x_attention(
            q,
            shared_k,
            shared_v,
            pool_k.view(rows, kv_heads, 3, dim),
            pool_v.view(rows, kv_heads, 3, dim),
            shared_blocks,
            lengths,
            step,
        )

    with torch.inference_mode():
        for mode in (0, 1):
            for _ in range(2):
                run(mode)
        torch.npu.synchronize()
        graphs, outputs = [], []
        for mode in (0, 1):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = run(mode)
            graphs.append(graph)
            outputs.append(output)
        addresses = (query.data_ptr(), key.data_ptr(), value.data_ptr())
        for iteration in range(3):
            query.copy_(torch.randn_like(query) * (0.1 + iteration * 0.03))
            key.copy_(torch.randn_like(key) * 0.1)
            value.copy_(torch.randn_like(value) * 0.1)
            step.fill_(1 + iteration % 2)
            for graph in graphs:
                graph.replay()
            expected = run(0)
            torch.npu.synchronize()
            assert addresses == (query.data_ptr(), key.data_ptr(), value.data_ptr())
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            torch.testing.assert_close(outputs[1], expected, rtol=0, atol=0)
            for reference, actual in zip(pools[0], pools[1]):
                torch.testing.assert_close(reference, actual, rtol=0, atol=0)


@hardware_test(res={"npu": "A3"}, num_cards=1)  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
def test_npu_packed_qkv_views_still_materialize_gather(dtype: torch.dtype, batch_size: int) -> None:
    """Construct views on NPU so host-to-device transfer cannot hide the stride."""
    _torch_npu, module = _native_modules()
    rows, heads, kv_heads, dim = batch_size * 128, 16, 8, 128
    packed = torch.randn(rows, heads + 2 * kv_heads, dim, dtype=dtype, device="npu")
    query, key, value = packed.split((heads, kv_heads, kv_heads), dim=1)
    metadata = _metadata(rows)
    metadata.prefix_indices = metadata.prefix_indices.to("npu")
    for tensor in (query, key, value):
        assert not tensor.is_contiguous()
        selected = module.select_beam_rows(tensor, metadata)
        assert selected.is_contiguous()
        assert selected.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
        torch.testing.assert_close(selected, tensor[metadata.prefix_indices], rtol=0, atol=0)


@hardware_test(res={"npu": "A3"}, num_cards=1)  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "batch_size,beam_width,heads,kv_heads",
    [(1, 128, 16, 8), (4, 128, 16, 8), (1, 32, 8, 8), (4, 32, 8, 8)],
)
def test_npu_shared_gather_attention_and_graph_replay_are_exact(
    batch_size: int, beam_width: int, heads: int, kv_heads: int
) -> None:
    """Check offset K/V views at the real custom-op and ACL graph boundary.

    The control keeps the old gather-then-two-copies layout. Both arms gather
    a fixed 32-column table, including duplicate and out-of-order block IDs,
    and must consume changing shared KV and table values on every replay.
    """
    _torch_npu, module = _native_modules()
    from vllm_gr.ops.npu import beam_ops

    dtype, device = torch.bfloat16, torch.device("npu")
    rows, dim = batch_size * beam_width, 128
    num_blocks, block_size, table_width = 7, 128, 32
    torch.manual_seed(1907 + batch_size)
    storage = torch.randn(10, 2, block_size, kv_heads, dim, dtype=dtype, device=device) * 0.1
    shared_pool = storage[2:9]
    shared_key, shared_value = shared_pool[:, 0], shared_pool[:, 1]
    assert shared_key.storage_offset() > 0
    table_cpu = (
        torch.arange(batch_size * table_width, dtype=torch.int32).reshape(batch_size, table_width)
        * 3
        + 1
    ) % num_blocks
    table_cpu[:, :3] = torch.tensor(
        [[5, 2, 5], [3, 1, 3], [6, 0, 2], [1, 6, 1]][:batch_size], dtype=torch.int32
    )
    table = table_cpu.to(device)
    lengths_cpu = torch.tensor([127, 128, 129, 257][:batch_size], dtype=torch.int32)
    lengths = lengths_cpu.to(device)
    blocks = torch.arange(batch_size, dtype=torch.int32, device=device)
    step = torch.ones(1, dtype=torch.int32, device=device)
    query = torch.randn(rows, heads, dim, dtype=dtype, device=device) * 0.1
    key = torch.randn(rows, kv_heads, dim, dtype=dtype, device=device) * 0.1
    value = torch.randn_like(key) * 0.1
    pools = [
        (
            torch.zeros(batch_size, beam_width, kv_heads, 3, dim, dtype=dtype, device=device),
            torch.zeros(batch_size, beam_width, kv_heads, 3, dim, dtype=dtype, device=device),
        )
        for _ in range(2)
    ]

    def gather(mode: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode == 1:
            return cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                module._gather_shared_blocks(shared_key, shared_value, table, 0),
            )
        flat = table.reshape(-1)
        gathered = shared_pool.index_select(0, flat)
        remapped = torch.arange(flat.numel(), dtype=table.dtype, device=device).reshape(table.shape)
        return gathered[:, 0].contiguous(), gathered[:, 1].contiguous(), remapped

    def run(mode: int) -> torch.Tensor:
        gathered_key, gathered_value, remapped = gather(mode)
        pool_key, pool_value = pools[mode]
        beam_ops.cache_unshared_kv(pool_key, pool_value, key, value, blocks, step)
        return beam_ops.x_attention(
            query,
            gathered_key,
            gathered_value,
            pool_key.view(rows, kv_heads, 3, dim),
            pool_value.view(rows, kv_heads, 3, dim),
            remapped,
            lengths,
            step,
        )

    with torch.inference_mode():
        control = gather(0)
        candidate = gather(1)
        assert candidate[0].is_contiguous() and candidate[1].is_contiguous()
        assert candidate[1].storage_offset() == candidate[0].numel() > 0
        for expected, actual in zip(control, candidate):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for mode in (0, 1):
            for _ in range(2):
                run(mode)
        torch.npu.synchronize()

        graphs, outputs = [], []
        for mode in (0, 1):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = run(mode)
            graphs.append(graph)
            outputs.append(output)
        inputs = (shared_pool, table, lengths, query, key, value, step)
        addresses = tuple(tensor.data_ptr() for tensor in inputs)
        for iteration in range(3):
            shared_pool.copy_(torch.randn_like(shared_pool) * (0.1 + iteration * 0.03))
            table.copy_((table_cpu + 2 * iteration + 1) % num_blocks)
            if batch_size == 1:
                lengths.fill_((127, 129, 257)[iteration])
            else:
                lengths.copy_(lengths_cpu.roll(iteration))
            query.copy_(torch.randn_like(query) * 0.1)
            key.copy_(torch.randn_like(key) * 0.1)
            value.copy_(torch.randn_like(value) * 0.1)
            step.fill_(1 + iteration % 2)

            for expected, actual in zip(gather(0), gather(1)):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            eager_control, eager_candidate = run(0), run(1)
            for graph in graphs:
                graph.replay()
            torch.npu.synchronize()

            assert addresses == tuple(tensor.data_ptr() for tensor in inputs)
            torch.testing.assert_close(eager_control, eager_candidate, rtol=0, atol=0)
            torch.testing.assert_close(outputs[0], eager_control, rtol=0, atol=0)
            torch.testing.assert_close(outputs[1], eager_control, rtol=0, atol=0)
            for expected, actual in zip(pools[0], pools[1]):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)


@hardware_test(res={"npu": "A3"}, num_cards=1)  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "batch_size,beam_width,heads,kv_heads",
    [(1, 128, 16, 8), (4, 128, 16, 8), (1, 32, 8, 8), (4, 32, 8, 8)],
)
def test_npu_direct_shared_kv_attention_and_graph_replay_are_exact(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    batch_size: int,
    beam_width: int,
    heads: int,
    kv_heads: int,
) -> None:
    """Compare direct paged reads with the prior gather/remap at real replay.

    Nonzero-offset dense caches and duplicate, unsorted page IDs must remain
    valid as lengths cross page boundaries. This covers operator/graph
    behavior; allocator selection and full-model parity are separate checks.
    """
    _torch_npu, module = _native_modules()
    from vllm_gr.ops.npu import beam_ops

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")
    device = torch.device("npu")
    rows, dim = batch_size * beam_width, 128
    num_blocks, block_size, table_width = 7, 128, 32
    torch.manual_seed(2029 + batch_size)
    # Rank-4/5 factories can choose NCHW/NCDHW on Ascend even when logical
    # strides are dense. Match the native allocator's unblocked ABI before
    # taking offset views; a later format conversion could erase the offset.
    storage_key = _torch_npu.npu_format_cast(
        torch.randn(10, block_size, kv_heads, dim, dtype=dtype, device=device) * 0.1, 2
    )
    storage_value = _torch_npu.npu_format_cast(torch.randn_like(storage_key) * 0.1, 2)
    shared_key, shared_value = storage_key[2:9], storage_value[2:9]
    assert shared_key.is_contiguous() and shared_value.is_contiguous()
    assert shared_key.storage_offset() > 0 and shared_value.storage_offset() > 0
    assert _torch_npu.get_npu_format(shared_key) == 2
    assert _torch_npu.get_npu_format(shared_value) == 2
    table_cpu = (
        torch.arange(batch_size * table_width, dtype=torch.int32).reshape(batch_size, table_width)
        * 3
        + 1
    ) % num_blocks
    table_cpu[:, :3] = torch.tensor(
        [[5, 2, 5], [3, 1, 3], [6, 0, 2], [1, 6, 1]][:batch_size], dtype=torch.int32
    )
    table = table_cpu.to(device)
    lengths_cpu = torch.tensor([127, 128, 129, 257][:batch_size], dtype=torch.int32)
    lengths = lengths_cpu.to(device)
    blocks = torch.arange(batch_size, dtype=torch.int32, device=device)
    step = torch.ones(1, dtype=torch.int32, device=device)
    query = torch.randn(rows, heads, dim, dtype=dtype, device=device) * 0.1
    key = torch.randn(rows, kv_heads, dim, dtype=dtype, device=device) * 0.1
    value = torch.randn_like(key) * 0.1
    pools = [
        (
            _torch_npu.npu_format_cast(
                torch.zeros(batch_size, beam_width, kv_heads, 3, dim, dtype=dtype, device=device), 2
            ),
            _torch_npu.npu_format_cast(
                torch.zeros(batch_size, beam_width, kv_heads, 3, dim, dtype=dtype, device=device), 2
            ),
        )
        for _ in range(2)
    ]
    for pool_key, pool_value in pools:
        assert _torch_npu.get_npu_format(pool_key) == 2
        assert _torch_npu.get_npu_format(pool_value) == 2
    metadata = _metadata(rows)
    metadata.prefix_indices = metadata.prefix_indices.to(device)
    metadata.prefix_block_table = table
    metadata.prefix_seqlens_kv_device = lengths
    metadata.max_prefix_blocks_count = 3
    session = SimpleNamespace(
        slot=0,
        beam_width=beam_width,
        attention_width=rows,
        decode_step=0,
        context=SimpleNamespace(decode_step_1based_buf=step),
    )
    candidate_pool = SimpleNamespace(
        unshared_key=pools[1][0].view(1, rows, kv_heads, 3, dim),
        unshared_value=pools[1][1].view(1, rows, kv_heads, 3, dim),
    )
    assert _torch_npu.get_npu_format(candidate_pool.unshared_key[0]) == 2
    assert _torch_npu.get_npu_format(candidate_pool.unshared_value[0]) == 2
    candidate_output = torch.empty_like(query)
    eager_output = torch.empty_like(query)
    original_gather = module._gather_shared_blocks

    def forbidden_gather(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("eligible dense NPU caches must reach x_attention without gathering")

    monkeypatch.setattr(module, "_gather_shared_blocks", forbidden_gather)

    def run(mode: int, destination: torch.Tensor | None = None) -> torch.Tensor:
        pool_key, pool_value = pools[mode]
        beam_ops.cache_unshared_kv(pool_key, pool_value, key, value, blocks, step)
        if mode == 0:
            gathered_key, gathered_value, remapped = original_gather(
                shared_key, shared_value, table, 0
            )
            return beam_ops.x_attention(
                query,
                gathered_key,
                gathered_value,
                pool_key.view(rows, kv_heads, 3, dim),
                pool_value.view(rows, kv_heads, 3, dim),
                remapped,
                lengths,
                step,
            )
        output = candidate_output if destination is None else destination
        module._xllm_incremental_decode(
            output, query, shared_key, shared_value, metadata, session, candidate_pool, dim**-0.5
        )
        return output

    with torch.inference_mode():
        assert module._can_use_direct_shared_kv(
            shared_key, shared_value, query, candidate_pool.unshared_key[0]
        )
        for mode in (0, 1):
            for _ in range(2):
                run(mode)
        torch.npu.synchronize()
        graphs, outputs = [], []
        for mode in (0, 1):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = run(mode)
            graphs.append(graph)
            outputs.append(output)
        inputs = (shared_key, shared_value, table, lengths, query, key, value, step)
        addresses = tuple(tensor.data_ptr() for tensor in inputs)
        for iteration in range(3):
            shared_key.copy_(torch.randn_like(shared_key) * (0.1 + iteration * 0.03))
            shared_value.copy_(torch.randn_like(shared_value) * 0.1)
            table.copy_((table_cpu + 2 * iteration + 1) % num_blocks)
            if batch_size == 1:
                lengths.fill_((127, 129, 257)[iteration])
            else:
                lengths.copy_(lengths_cpu.roll(iteration))
            query.copy_(torch.randn_like(query) * 0.1)
            key.copy_(torch.randn_like(key) * 0.1)
            value.copy_(torch.randn_like(value) * 0.1)
            step.fill_(1 + iteration % 2)
            eager_control = run(0)
            # Use a separate output so eager execution cannot pre-populate
            # the captured candidate's destination and hide a broken replay.
            eager_candidate = run(1, eager_output)
            for output in outputs:
                output.fill_(float("nan"))
            for graph in graphs:
                graph.replay()
            torch.npu.synchronize()

            assert addresses == tuple(tensor.data_ptr() for tensor in inputs)
            torch.testing.assert_close(eager_control, eager_candidate, rtol=0, atol=0)
            torch.testing.assert_close(outputs[0], eager_control, rtol=0, atol=0)
            torch.testing.assert_close(outputs[1], eager_control, rtol=0, atol=0)
            for expected, actual in zip(pools[0], pools[1]):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)


pytestmark = pytest.mark.cpu_test
