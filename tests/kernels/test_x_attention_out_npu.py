# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Compare the optional output binding with the allocating XAttention exactly.

Set VLLM_GR_TEST_XATTENTION_OUT_REQUIRED=1 for feature acceptance: missing
runtime or binding prerequisites then fail rather than skip this module.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from tools.hardware_marks import hardware_marks

pytestmark = [*hardware_marks(res={"npu": "A3"}, num_cards=1), pytest.mark.l2]


def _unavailable(reason: str) -> None:
    if os.environ.get("VLLM_GR_TEST_XATTENTION_OUT_REQUIRED") == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason, allow_module_level=True)


try:
    import torch_npu
except (ImportError, OSError, RuntimeError) as exc:
    _unavailable(f"torch-npu integration dependency unavailable: {exc}")

try:
    import custom_ops_lib
except (ImportError, OSError, RuntimeError) as exc:
    _unavailable(f"xLLM integration dependency unavailable: {exc}")
if not hasattr(torch, "npu") or not torch.npu.is_available():
    _unavailable("Ascend NPU unavailable")
from vllm_gr.ops.npu import _xattention_out as optional_binding  # noqa: E402
from vllm_gr.ops.npu import beam_ops  # noqa: E402


if not optional_binding.x_attention_out_available():
    _unavailable("optional XAttention output binding unavailable or incompatible")

DEVICE = torch.device("npu")
WIDTH, HEADS, KV_HEADS, DIM, BLOCK, STEPS = (128, 16, 8, 128, 128, 3)


def _inputs(batch: int, prefix: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1731 + batch + prefix)
    rows = batch * WIDTH
    blocks = prefix // BLOCK
    floating_shapes = (
        (rows, HEADS, DIM),
        (batch * blocks, BLOCK, KV_HEADS, DIM),
        (batch * blocks, BLOCK, KV_HEADS, DIM),
        (rows, KV_HEADS, STEPS, DIM),
        (rows, KV_HEADS, STEPS, DIM),
    )
    floating = tuple((torch.randn(shape, dtype=dtype, device=DEVICE) for shape in floating_shapes))
    table = (
        torch.arange(batch * blocks, dtype=torch.int32, device=DEVICE)
        .flip(0)
        .reshape(batch, blocks)
    )
    lengths = torch.tensor(
        [prefix - 17 * (i % 2) for i in range(batch)], dtype=torch.int32, device=DEVICE
    )
    step = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    return (*floating, table, lengths, step)


def _old(args: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return custom_ops_lib.x_attention(*args[:6], None, args[6], args[7], 0.0)


def _output(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    storage = torch.full((query.shape[0] + 4, HEADS, DIM), -113, dtype=query.dtype, device=DEVICE)
    output = storage[2:-2]
    assert output.storage_offset() != 0 and output.is_contiguous()
    assert torch_npu.get_npu_format(output) == 2
    return (storage, output)


def _assert_sentinels(storage: torch.Tensor) -> None:
    assert bool(torch.all(storage[:2].cpu() == -113))
    assert bool(torch.all(storage[-2:].cpu() == -113))


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("prefix", [128, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("step", [1, 2])
def test_x_attention_out_exact_eager(
    batch: int, prefix: int, dtype: torch.dtype, step: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(batch, prefix, dtype)
    args[-1].fill_(step)
    storage, output = _output(args[0])
    operation = optional_binding.get_x_attention_out(output)
    assert callable(operation), "private addon is mandatory for this focused test"
    calls = []

    def observed(*native_args: Any) -> Any:
        calls.append(native_args[-1].data_ptr())
        return operation(*native_args)

    monkeypatch.setattr(optional_binding, "get_x_attention_out", lambda out: observed)
    expected = _old(args)
    returned = beam_ops.x_attention(*args, out=output)
    torch.npu.synchronize()
    assert returned is output and calls == [output.data_ptr()]
    assert torch.equal(expected.cpu(), output.cpu())
    _assert_sentinels(storage)


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("prefix", [128, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_x_attention_out_exact_graph_replay(
    batch: int, prefix: int, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(batch, prefix, dtype)
    storage, output = _output(args[0])
    operation = optional_binding.get_x_attention_out(output)
    assert callable(operation), "private addon is mandatory for graph acceptance"
    calls = []

    def observed(*native_args: Any) -> Any:
        calls.append(native_args[-1].data_ptr())
        return operation(*native_args)

    monkeypatch.setattr(optional_binding, "get_x_attention_out", lambda out: observed)
    for _ in range(3):
        beam_ops.x_attention(*args, out=output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        returned = beam_ops.x_attention(*args, out=output)
    assert returned is output
    capture_count = len(calls)
    pointers = [tensor.data_ptr() for tensor in (*args, output)]
    for replay, step in enumerate((1, 2, 1, 2)):
        args[0].add_(0.03125)
        args[1].add_(0.001953125)
        args[2].sub_(0.001953125)
        args[3].add_(0.00390625)
        args[4].sub_(0.00390625)
        args[5].copy_(args[5].roll(1, 0))
        args[6].fill_(prefix - replay % 2)
        args[7].fill_(step)
        expected = _old(args)
        output.fill_(-113)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(expected.cpu(), output.cpu())
        assert [tensor.data_ptr() for tensor in (*args, output)] == pointers
        assert len(calls) == capture_count, "replay must use the captured native call"
        _assert_sentinels(storage)


@pytest.mark.parametrize("invalid", ["alias", "noncontiguous", "dtype", "device"])
def test_x_attention_out_native_rejects_invalid_output(invalid: str) -> None:
    args = _inputs(1, 128, torch.bfloat16)
    storage, output = _output(args[0])
    operation = optional_binding.get_x_attention_out(output)
    assert callable(operation), "private addon is mandatory for native guards"
    if invalid == "alias":
        output = args[0]
    elif invalid == "noncontiguous":
        output = torch.empty((WIDTH, DIM, HEADS), dtype=args[0].dtype, device=DEVICE).transpose(
            1, 2
        )
    elif invalid == "dtype":
        output = output.float()
    else:
        output = output.cpu()
    output.fill_(-113)
    before = output.cpu().clone()
    with pytest.raises(RuntimeError):
        operation(*args[:6], None, args[6], args[7], 0.0, output)
    torch.npu.synchronize()
    assert torch.equal(output.cpu(), before)
    _assert_sentinels(storage)
