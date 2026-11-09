# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""CPU contracts for the optional, isolated XAttention output binding."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[1]


class Device(NamedTuple):
    type: str


class Tensor:
    def __init__(
        self: Any,
        shape: Any = (2, 4, 8),
        *,
        dtype: Any = "bf16",
        device: Any = "npu",
        contiguous: Any = True,
        storage: Any = None,
        value: Any = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device if isinstance(device, Device) else Device(device)
        self.contiguous = contiguous
        self.storage_id = id(self) if storage is None else storage
        self.value = value
        self.copies: list[Any] = []
        self.writes: list[Any] = []

    def is_contiguous(self: Any) -> Any:
        return self.contiguous

    def untyped_storage(self: Any) -> Any:
        return SimpleNamespace(data_ptr=lambda: self.storage_id)

    def copy_(self: Any, other: Any) -> Any:
        self.copies.append(other)
        self.value = other.value
        return self

    def __getitem__(self: Any, index: Any) -> Any:
        assert isinstance(index, slice)
        stop = self.shape[0] if index.stop is None else index.stop
        start = 0 if index.start is None else index.start
        return Tensor(
            (stop - start, *self.shape[1:]),
            dtype=self.dtype,
            device=self.device.type,
            contiguous=self.contiguous,
            storage=self.storage_id,
        )

    def __setitem__(self: Any, index: Any, value: Any) -> Any:
        self.writes.append((index, value))


def load_file(path: Any, name: Any, monkeypatch: Any) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def environment(monkeypatch: Any) -> Any:
    torch = ModuleType("torch")  # type: Any
    torch.Tensor = Tensor
    torch.int32 = "int32"
    torch.tensor = lambda values, **kw: Tensor((len(values),), value=values, **kw)
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("vllm_gr", "vllm_gr.ops", "vllm_gr.ops.npu"):
        package = ModuleType(name)  # type: Any
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    native = SimpleNamespace(x_attention=Mock(return_value=Tensor(value=9)))
    loader = ModuleType("vllm_gr.ops.npu._xllm_loader")  # type: Any
    loader.custom_ops = lambda: native
    loader.has_op = lambda name: True
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    optional = ModuleType("vllm_gr.ops.npu._xattention_out")  # type: Any
    optional.get_x_attention_out = Mock(return_value=None)
    optional.x_attention_out_available = Mock(return_value=True)
    monkeypatch.setitem(sys.modules, optional.__name__, optional)
    module = load_file(ROOT / "vllm_gr/ops/npu/beam_ops.py", "_out_contract_ops", monkeypatch)
    inputs = tuple((Tensor() for _ in range(7))) + (Tensor((1,), dtype="int32"),)
    return SimpleNamespace(ops=module, native=native, optional=optional, inputs=inputs)


def test_direct_out_preserves_storage_and_abi_without_copy(environment: Any) -> None:
    env = environment
    output = Tensor(storage=123)
    operation = Mock(side_effect=lambda *args: setattr(args[-1], "value", 7))
    env.optional.get_x_attention_out.return_value = operation
    result = env.ops.x_attention(*env.inputs, out=output)
    assert result is output and output.value == 7 and (output.storage_id == 123)
    assert output.copies == []
    args = operation.call_args.args
    assert args[:6] == env.inputs[:6]
    assert args[6] is None and args[7] is env.inputs[6] and (args[8] is env.inputs[7])
    assert args[9] == 0.0 and args[10] is output
    env.native.x_attention.assert_not_called()


def test_missing_addon_uses_original_kernel_and_one_copy(environment: Any) -> None:
    env = environment
    output = Tensor()
    assert env.ops.x_attention(*env.inputs, out=output) is output
    env.native.x_attention.assert_called_once()
    assert output.copies == [env.native.x_attention.return_value] and output.value == 9


def test_no_out_preserves_original_allocating_api(environment: Any) -> None:
    env = environment
    assert env.ops.x_attention(*env.inputs) is env.native.x_attention.return_value
    env.native.x_attention.assert_called_once()
    env.optional.get_x_attention_out.assert_not_called()


def test_native_partial_write_failure_is_not_retried(environment: Any) -> None:
    env = environment
    output = Tensor()
    error = RuntimeError("partial native output")

    def failing(*args: Any) -> Any:
        args[-1].value = 123
        raise error

    operation = Mock(side_effect=failing)
    env.optional.get_x_attention_out.return_value = operation
    with pytest.raises(RuntimeError) as caught:
        env.ops.x_attention(*env.inputs, out=output)
    assert caught.value is error and output.value == 123 and (output.copies == [])
    operation.assert_called_once()
    env.native.x_attention.assert_not_called()


@pytest.mark.parametrize("change", ["shape", "dtype", "device", "contiguous"])
def test_invalid_output_fails_before_native_dispatch(environment: Any, change: Any) -> None:
    env = environment
    output = Tensor()
    setattr(
        output,
        change,
        {"shape": (1, 4, 8), "dtype": "fp32", "device": Device("cpu"), "contiguous": False}[change],
    )
    with pytest.raises(ValueError):
        env.ops.x_attention(*env.inputs, out=output)
    env.native.x_attention.assert_not_called()
    env.optional.get_x_attention_out.assert_not_called()


@pytest.mark.parametrize("input_index", range(8))
def test_output_cannot_alias_any_input_allocation(environment: Any, input_index: Any) -> None:
    env = environment
    output = Tensor(storage=env.inputs[input_index].storage_id)
    with pytest.raises(ValueError, match="share storage"):
        env.ops.x_attention(*env.inputs, out=output)
    env.native.x_attention.assert_not_called()
    env.optional.get_x_attention_out.assert_not_called()


def helper_namespace(monkeypatch: Any) -> Any:
    path = ROOT / "vllm_gr/v1/attention/backends/beam_attn_npu_xllm.py"
    names = {"_direct_attention_output", "_xllm_incremental_decode"}
    nodes = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]  # type: list[ast.stmt]
    assert {n.name if isinstance(n, ast.FunctionDef) else "unexpected" for n in nodes} == names
    namespace = {
        "math": math,
        "torch": SimpleNamespace(Tensor=Tensor, float16="fp16", bfloat16="bf16"),
        # Keep output contracts on the shared-gather path; direct shared-KV
        # routing is covered by test_npu_beam_row_views.py.
        "_can_use_direct_shared_kv": lambda *args: False,
        "_gather_shared_blocks": lambda key, value, table, cols: (key, value, table),
        "_npu_stream_capturing": lambda: False,
        "select_beam_rows": lambda query, metadata: query,
        "BeamAttentionMetadata": SimpleNamespace,
    }
    binding = ModuleType("vllm_gr.ops.npu._xattention_out")  # type: Any
    binding.x_attention_out_available = Mock(return_value=True)
    monkeypatch.setitem(sys.modules, binding.__name__, binding)
    namespace["binding"] = binding
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("mixed", [False, True])
def test_backend_direct_leading_view_or_original_mixed_scatter(
    mixed: Any, monkeypatch: Any
) -> None:
    ns = helper_namespace(monkeypatch)
    query, output = (Tensor(), Tensor())
    operation = Mock(side_effect=lambda *args, **kw: kw["out"] or Tensor(value=5))
    ns["beam_ops"] = SimpleNamespace(x_attention=operation)
    metadata = SimpleNamespace(
        std_indices=[2] if mixed else None,
        num_prefix_indices=2,
        prefix_indices="indices",
        prefix_seqlens_kv_device=Tensor((1,)),
        prefix_block_table=Tensor((1, 1)),
    )

    class PoolRows:
        def __getitem__(self: Any, index: Any) -> Any:
            return Tensor((2, 2, 3, 8))

    pool = SimpleNamespace(unshared_key=PoolRows(), unshared_value=PoolRows())
    session = SimpleNamespace(
        decode_step=0,
        slot=0,
        beam_width=2,
        context=SimpleNamespace(decode_step_1based_buf=Tensor((1,))),
    )
    ns["_xllm_incremental_decode"](
        output,
        query,
        Tensor((1, 4, 2, 8)),
        Tensor((1, 4, 2, 8)),
        metadata,
        session,
        pool,
        8 ** (-0.5),
    )
    target = operation.call_args.kwargs["out"]
    if mixed:
        assert target is None and len(output.writes) == 1
        assert output.writes[0][0] == "indices"
    else:
        assert target.storage_id == output.storage_id and target.shape == query.shape
        assert output.writes == []


@pytest.mark.parametrize("change", ["contiguous", "shape", "dtype", "device", "unsupported_dtype"])
def test_backend_unsupported_output_declines_direct_view(change: Any, monkeypatch: Any) -> None:
    ns = helper_namespace(monkeypatch)
    output, query = (Tensor(), Tensor())
    if change == "shape":
        query.shape = (3, 4, 8)
    elif change == "unsupported_dtype":
        query.dtype = output.dtype = "fp32"
    else:
        setattr(
            output, change, {"contiguous": False, "dtype": "fp32", "device": Device("cpu")}[change]
        )
    metadata = SimpleNamespace(std_indices=None, num_prefix_indices=2)
    assert ns["_direct_attention_output"](output, query, metadata) is None


def test_missing_addon_declines_before_inspecting_or_slicing_output(monkeypatch: Any) -> None:
    ns = helper_namespace(monkeypatch)
    ns["binding"].x_attention_out_available.return_value = False
    metadata = SimpleNamespace(std_indices=None)
    # Neither object has tensor attributes: touching an output view, dtype,
    # format or storage would fail instead of preserving the original route.
    assert ns["_direct_attention_output"](object(), object(), metadata) is None
    ns["binding"].x_attention_out_available.assert_called_once_with()


def test_missing_addon_backend_keeps_original_leading_output_copy(monkeypatch: Any) -> None:
    ns = helper_namespace(monkeypatch)
    ns["binding"].x_attention_out_available.return_value = False
    query, output = Tensor(), Tensor()
    allocated = Tensor(value=19)
    operation = Mock(return_value=allocated)
    ns["beam_ops"] = SimpleNamespace(x_attention=operation)
    metadata = SimpleNamespace(
        std_indices=None,
        num_prefix_indices=2,
        prefix_seqlens_kv_device=Tensor((1,)),
        prefix_block_table=None,
    )

    class PoolRows:
        def __getitem__(self: Any, index: Any) -> Any:
            return Tensor((2, 2, 3, 8))

    pool = SimpleNamespace(unshared_key=PoolRows(), unshared_value=PoolRows())
    session = SimpleNamespace(
        decode_step=0,
        slot=0,
        beam_width=2,
        context=SimpleNamespace(decode_step_1based_buf=Tensor((1,))),
    )
    key, value = Tensor((1, 4, 2, 8)), Tensor((1, 4, 2, 8))
    key.contiguous = value.contiguous = Mock()
    ns["_xllm_incremental_decode"](output, query, key, value, metadata, session, pool, 8 ** (-0.5))
    assert operation.call_args.kwargs["out"] is None
    assert output.writes == [(slice(None, 2), allocated)]


@pytest.mark.parametrize("required", [False, True])
def test_native_test_missing_prerequisite_is_optional_or_required(
    required: bool, monkeypatch: Any
) -> None:
    path = ROOT / "tests/kernels/test_x_attention_out_npu.py"
    nodes: list[ast.stmt] = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "_unavailable"
    ]
    assert len(nodes) == 1
    namespace = {"os": __import__("os"), "pytest": pytest}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setenv("VLLM_GR_TEST_XATTENTION_OUT_REQUIRED", "1" if required else "0")
    expected = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(expected, match="missing optional binding"):
        namespace["_unavailable"]("missing optional binding")
