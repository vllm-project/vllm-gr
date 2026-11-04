# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout and device guards for NPU identity-row attention views."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch


class _MockNPUTensor(torch.Tensor):  # type: ignore[misc]
    @property
    def device(self) -> torch.device:
        # Tensor storage and all arithmetic stay on CPU; only the host guard
        # sees an NPU. This does not exercise the real native operator.
        return cast(torch.device, SimpleNamespace(type="npu"))


class _MockCUDATensor(torch.Tensor):  # type: ignore[misc]
    @property
    def device(self) -> torch.device:
        return cast(torch.device, SimpleNamespace(type="cuda"))


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


@pytest.mark.npu_integration  # type: ignore[misc]
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


@pytest.mark.npu_integration  # type: ignore[misc]
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


@pytest.mark.npu_integration  # type: ignore[misc]
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
