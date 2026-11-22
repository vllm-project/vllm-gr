# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from enum import Enum, auto
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_gr.ops.sliced_logits import (
    SlicedLMHeadProxy,
    compute_sliced_logits,
    get_rank_vocab_bounds,
)
from vllm_gr.utils.npu_adapter import is_cuda_available


@pytest.fixture(autouse=True)  # type: ignore[misc]
def cpu_embedding_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    original = UnquantizedEmbeddingMethod.apply

    def apply(self: Any, layer: Any, x: torch.Tensor, bias: Any = None) -> torch.Tensor:
        # Mock heads bypass native model loading and its cpu_linear setup.
        if x.device.type == "cpu":
            return F.linear(x, layer.weight, bias)
        return original(self, layer, x, bias)

    monkeypatch.setattr(UnquantizedEmbeddingMethod, "apply", apply)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("mode_name", ["REPLICATED", "STANDARD", "FINE_GRAINED", "PCP_X_TP"])  # type: ignore[misc]
def test_sliced_proxy_uses_source_head_replicated_mode(mode_name: str) -> None:
    class ParallelMode(Enum):
        REPLICATED = auto()
        STANDARD = auto()
        FINE_GRAINED = auto()
        PCP_X_TP = auto()

    original = SimpleNamespace(parallel_mode=ParallelMode[mode_name])
    proxy = SlicedLMHeadProxy(original, torch.ones(4, 2), 4)
    assert proxy.parallel_mode is ParallelMode.REPLICATED
    assert proxy.tp_size == 1
    assert original.parallel_mode is ParallelMode[mode_name]


@pytest.mark.l1  # type: ignore[misc]
def test_sliced_proxy_without_parallel_mode_needs_no_vendor_enum() -> None:
    proxy = SlicedLMHeadProxy(SimpleNamespace(), torch.ones(4, 2), 4)
    assert not hasattr(proxy, "parallel_mode")
    assert proxy.tp_size == 1


class MockLMHead(nn.Module):  # type: ignore[misc]
    def __init__(self, vocab_size: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(vocab_size))
        else:
            self.bias = None


def test_sliced_logits_equivalence() -> None:
    batch_size = 4
    hidden_dim = 128
    full_vocab_size = 170000
    offset = 20000
    slice_size = 24000

    torch.manual_seed(42)
    lm_head = MockLMHead(full_vocab_size, hidden_dim, bias=True)
    hidden_states = torch.randn(batch_size, hidden_dim)

    # 1. Compute full logits
    full_logits = torch.matmul(hidden_states, lm_head.weight.T) + lm_head.bias
    expected_slice_logits = full_logits[:, offset : offset + slice_size]

    # 2. Compute sliced logits
    sliced_logits = compute_sliced_logits(
        hidden_states=hidden_states,
        lm_head=lm_head,
        embedding_bias=lm_head.bias,
        offset=offset,
        slice_size=slice_size,
        quant_group_size=128,
    )

    assert sliced_logits is not None
    assert sliced_logits.shape == (batch_size, slice_size)
    assert torch.allclose(sliced_logits, expected_slice_logits, atol=1e-5, rtol=1e-5)


def test_quantization_group_alignment() -> None:
    batch_size = 2
    hidden_dim = 64
    full_vocab_size = 50000
    unaligned_offset = 20013
    slice_size = 24000
    group_size = 128

    lm_head = MockLMHead(full_vocab_size, hidden_dim, bias=False)
    hidden_states = torch.randn(batch_size, hidden_dim)

    full_logits = torch.matmul(hidden_states, lm_head.weight.T)
    expected_slice_logits = full_logits[:, unaligned_offset : unaligned_offset + slice_size]

    sliced_logits = compute_sliced_logits(
        hidden_states=hidden_states,
        lm_head=lm_head,
        embedding_bias=None,
        offset=unaligned_offset,
        slice_size=slice_size,
        quant_group_size=group_size,
    )

    assert sliced_logits is not None
    assert sliced_logits.shape == (batch_size, slice_size)
    assert torch.allclose(sliced_logits, expected_slice_logits, atol=1e-5, rtol=1e-5)


class MockTPHead(nn.Module):  # type: ignore[misc]
    """Simulates a shard of lm_head.weight for a specific TP rank."""

    def __init__(self, full_weight: torch.Tensor, rank: int, tp_size: int):
        super().__init__()
        vocab_size = full_weight.shape[0]
        shard_size = vocab_size // tp_size
        self.rank = rank
        self.tp_size = tp_size
        self.vocab_start_index = rank * shard_size
        self.vocab_end_index = (rank + 1) * shard_size
        self.weight = nn.Parameter(
            full_weight[self.vocab_start_index : self.vocab_end_index].clone(),
            requires_grad=False,
        )


def test_get_rank_vocab_bounds() -> None:
    full_w = torch.randn(1000, 64)
    head_rank0 = MockTPHead(full_w, rank=0, tp_size=2)
    head_rank1 = MockTPHead(full_w, rank=1, tp_size=2)

    assert get_rank_vocab_bounds(head_rank0) == (0, 500)
    assert get_rank_vocab_bounds(head_rank1) == (500, 1000)


def test_tp_sliced_logits_equivalence_tp2() -> None:
    """Verify that sliced logits under TP=2 match un-sliced logits across ranks."""
    torch.manual_seed(42)
    vocab_size = 10000
    embed_dim = 32
    batch_size = 4
    hidden_states = torch.randn(batch_size, embed_dim)
    full_weight = torch.randn(vocab_size, embed_dim)

    offset = 2000
    slice_size = 3000

    # Un-sliced reference logits for the slice [2000:5000]
    full_logits = F.linear(hidden_states, full_weight)
    ref_sliced_logits = full_logits[:, offset : offset + slice_size]

    # Rank 0 covers [0:5000], requested slice [2000:5000] -> rank 0 local [2000:5000]
    rank0_head = MockTPHead(full_weight, rank=0, tp_size=2)
    rank0_logits = compute_sliced_logits(
        hidden_states, rank0_head, offset=offset, slice_size=slice_size
    )

    # Rank 1 covers [5000:10000], requested slice [2000:5000] -> rank 1 has no overlap
    rank1_head = MockTPHead(full_weight, rank=1, tp_size=2)
    rank1_logits = compute_sliced_logits(
        hidden_states, rank1_head, offset=offset, slice_size=slice_size
    )

    assert rank0_logits is not None
    assert rank1_logits is not None
    # Rank 0 should compute exact logits for the slice [2000:5000]
    assert torch.allclose(rank0_logits, ref_sliced_logits, atol=1e-5, rtol=1e-5)
    # Rank 1 overlap is empty
    assert rank1_logits.shape[-1] == 0


def test_tp_sliced_logits_spanning_multiple_ranks() -> None:
    """Verify slice spanning across Rank 0 and Rank 1 under TP=2."""
    torch.manual_seed(123)
    vocab_size = 10000
    embed_dim = 32
    batch_size = 2
    hidden_states = torch.randn(batch_size, embed_dim)
    full_weight = torch.randn(vocab_size, embed_dim)

    # Slice [4000:6000] spans Rank 0 ([0:5000]) and Rank 1 ([5000:10000])
    offset = 4000
    slice_size = 2000
    full_logits = F.linear(hidden_states, full_weight)
    ref_sliced_logits = full_logits[:, offset : offset + slice_size]

    rank0_head = MockTPHead(full_weight, rank=0, tp_size=2)
    rank0_logits = compute_sliced_logits(
        hidden_states, rank0_head, offset=offset, slice_size=slice_size
    )

    rank1_head = MockTPHead(full_weight, rank=1, tp_size=2)
    rank1_logits = compute_sliced_logits(
        hidden_states, rank1_head, offset=offset, slice_size=slice_size
    )

    assert rank0_logits is not None
    assert rank1_logits is not None
    # Rank 0 computes [4000:5000] (1000 logits), Rank 1 computes [5000:6000] (1000 logits)
    assert rank0_logits.shape[-1] == 1000
    assert rank1_logits.shape[-1] == 1000

    gathered_logits = torch.cat([rank0_logits, rank1_logits], dim=-1)
    assert torch.allclose(gathered_logits, ref_sliced_logits, atol=1e-5, rtol=1e-5)


def test_quantized_lm_head_fallback() -> None:
    """Verify that quantized lm_head uses safe logits post-slicing fallback."""
    torch.manual_seed(99)
    vocab_size = 5000
    embed_dim = 16
    batch_size = 2
    hidden_states = torch.randn(batch_size, embed_dim)
    full_weight = torch.randn(vocab_size, embed_dim)

    projected_weight_shapes = []

    class DummyQuantMethod:
        def apply(self, layer: Any, x: torch.Tensor, bias: Any = None) -> torch.Tensor:
            projected_weight_shapes.append(tuple(layer.weight.shape))
            return F.linear(x, layer.weight, bias)

    class MockQuantizedHead(nn.Module):  # type: ignore[misc]
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.quant_method = DummyQuantMethod()

    quant_head = MockQuantizedHead(full_weight)
    offset = 1000
    slice_size = 500

    full_logits = F.linear(hidden_states, full_weight)
    ref_sliced = full_logits[:, offset : offset + slice_size]

    sliced_logits = compute_sliced_logits(
        hidden_states, quant_head, offset=offset, slice_size=slice_size
    )

    assert sliced_logits is not None
    assert sliced_logits.shape == (batch_size, slice_size)
    assert sliced_logits.is_contiguous()
    assert projected_weight_shapes == [(vocab_size, embed_dim)]
    assert torch.allclose(sliced_logits, ref_sliced, atol=1e-5, rtol=1e-5)
    # Ensure pre-sliced weight proxy was NOT created for quantized head
    assert not hasattr(quant_head, "_sliced_weight")


def _native_processor(vocab_size: int, head_dtype: torch.dtype | None = None) -> Any:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    from vllm_gr.v1.worker.sliced_logits_patch import apply_sliced_logits_patch

    apply_sliced_logits_patch()
    with set_current_vllm_config(VllmConfig()):
        processor = LogitsProcessor(vocab_size)
    processor.head_dtype = head_dtype
    return processor


@pytest.mark.parametrize(  # type: ignore[misc]
    "device,dtype",
    [
        ("cpu", torch.float32),
        pytest.param("cuda", torch.float16, marks=pytest.mark.gpu_test),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.gpu_test),
    ],
)
@pytest.mark.parametrize("head_dtype", [None, torch.float32])  # type: ignore[misc]
@pytest.mark.parametrize("skip_gather", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("slice_mode", ["disabled", "window"])  # type: ignore[misc]
def test_native_logits_forward(
    device: str,
    dtype: torch.dtype,
    head_dtype: torch.dtype | None,
    skip_gather: bool,
    slice_mode: str,
) -> None:
    """Exercise the real 0.30 forward, projection and dtype handling."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    if device == "cuda" and not is_cuda_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(43)
    head = MockLMHead(384, 32).to(device=device, dtype=dtype)
    head.weight.data.mul_(0.1)
    head.bias.data.mul_(0.1)
    head.quant_method = UnquantizedEmbeddingMethod()
    head.tp_size = 1
    hidden = torch.randn(3, 32, device=device, dtype=dtype) * 0.1
    processor = _native_processor(350, head_dtype)
    processor.scale = 0.7
    processor.soft_cap = 1.5
    window = (269, 57) if slice_mode == "window" else None
    if window is not None:
        processor._gr_vocab_slice = window

    output = processor(head, hidden, head.bias, skip_gather)
    expected_dtype = head_dtype or dtype
    expected = F.linear(
        hidden.to(expected_dtype), head.weight.to(expected_dtype), head.bias.to(expected_dtype)
    )
    if window is not None:
        expected = expected[..., window[0] : window[0] + window[1]]
        assert head._sliced_weight.shape[0] < head.weight.shape[0]
        assert head._sliced_proxy.tp_size == 1
    elif not skip_gather:
        expected = expected[..., :350]
    expected = torch.tanh(expected / processor.soft_cap) * processor.soft_cap
    expected = expected * processor.scale
    assert output is not None
    assert output.dtype == expected_dtype
    torch.testing.assert_close(
        output, expected, atol=1e-5 if expected_dtype == torch.float32 else 2e-3, rtol=2e-3
    )
    if window is None:
        assert not hasattr(head, "_sliced_weight")
    else:
        cached = head._sliced_weight
        processor(head, hidden, head.bias, skip_gather)
        assert head._sliced_weight is cached


@pytest.mark.parametrize("quantized", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("skip_gather", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("rank", [0, 1])  # type: ignore[misc]
@pytest.mark.parametrize("window", [(192, 160), (5, 11)])  # type: ignore[misc]
def test_native_logits_tp_contract(
    quantized: bool,
    skip_gather: bool,
    rank: int,
    window: tuple[int, int],
) -> None:
    """Simulate the collective, but use the real upstream projection/forward."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    torch.manual_seed(44)
    weight = torch.randn(512, 32)
    hidden = torch.randn(2, 32)
    head = MockTPHead(weight, rank, 2)
    projected = []

    class DummyQuantMethod:
        def apply(self, layer: Any, x: torch.Tensor, bias: Any = None) -> torch.Tensor:
            projected.append(layer.weight.shape[0])
            return F.linear(x, layer.weight, bias)

    head.quant_method = DummyQuantMethod() if quantized else UnquantizedEmbeddingMethod()
    processor = _native_processor(500)
    processor._gr_vocab_slice = window
    full = F.linear(hidden, weight)
    gathered = []

    def gather(local: torch.Tensor) -> torch.Tensor:
        gathered.append(local.shape[-1])
        torch.testing.assert_close(local, full[..., rank * 256 : (rank + 1) * 256])
        return full

    processor._gather_logits = gather
    output = processor(head, hidden, None, skip_gather)
    start, end = window[0], window[0] + window[1]
    if skip_gather:
        start = max(start, rank * 256)
        end = max(start, min(end, (rank + 1) * 256))
    assert output is not None
    torch.testing.assert_close(output, full[..., start:end])
    assert gathered == ([] if skip_gather else [256])
    if quantized:
        assert projected == [256]
    if quantized or not skip_gather:
        assert not hasattr(head, "_sliced_weight")


@pytest.mark.parametrize("quantized", [False, True])  # type: ignore[misc]
def test_native_logits_non_receiver_preserves_none(quantized: bool) -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    head = MockTPHead(torch.randn(512, 32), rank=1, tp_size=2)

    class DummyQuantMethod:
        def apply(self, layer: Any, x: torch.Tensor, bias: Any = None) -> torch.Tensor:
            return F.linear(x, layer.weight, bias)

    head.quant_method = DummyQuantMethod() if quantized else UnquantizedEmbeddingMethod()
    processor = _native_processor(500)
    processor._gr_vocab_slice = (5, 11)  # this rank has no overlap, but must gather
    calls = []

    def gather(local: torch.Tensor) -> None:
        calls.append(local.shape[-1])

    processor._gather_logits = gather
    assert processor(head, torch.randn(2, 32)) is None
    assert calls == [256]


def test_native_quantized_head_dtype_rejection() -> None:
    class DummyQuantMethod:
        def apply(self, layer: Any, x: torch.Tensor, bias: Any = None) -> torch.Tensor:
            return F.linear(x, layer.weight, bias)

    head = MockLMHead(384, 32)
    head.quant_method = DummyQuantMethod()
    head.tp_size = 1
    processor = _native_processor(350, torch.float64)
    processor._gr_vocab_slice = (129, 157)
    with pytest.raises(ValueError, match="only supported for an unquantized lm_head"):
        processor(head, torch.randn(2, 32))


def test_v2_topk_clamp_reuses_an_already_wrapped_import() -> None:
    from vllm_gr.v1.worker.sliced_logits_patch import _make_clamped_v2_topk

    calls = []

    def native(logits: torch.Tensor, num_logprobs: int, *, logits_mode: bool) -> Any:
        calls.append((num_logprobs, logits_mode))
        return logits.topk(num_logprobs, dim=-1)

    early_import = _make_clamped_v2_topk(native)
    late_import = _make_clamped_v2_topk(early_import)
    assert late_import is early_import
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    actual = late_import(logits, 8, logits_mode=True)
    torch.testing.assert_close(actual.values, torch.tensor([[3.0, 2.0, 1.0]]))
    assert calls == [(3, True)]


@pytest.mark.gpu_test  # type: ignore[misc]
@pytest.mark.parametrize("logits_mode", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("width", [1, 7])  # type: ignore[misc]
@pytest.mark.parametrize("num_logprobs", [0, 1, 5])  # type: ignore[misc]
def test_v2_compute_topk_scores_clamped_to_sliced_width(
    logits_mode: bool,
    width: int,
    num_logprobs: int,
) -> None:
    """Check all 0.30 importer bindings and score modes against dense results."""
    import importlib

    from vllm_gr.v1.worker import sliced_logits_patch

    if not is_cuda_available():
        pytest.skip("V2 score helper uses a Triton kernel (CUDA only)")
    # Load importers before installation too: existing bindings must be patched.
    modules = [
        importlib.import_module(name)
        for name in (
            "vllm.v1.worker.gpu.sample.logprob",
            "vllm.v1.worker.gpu.sample.sampler",
            "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
        )
    ]
    sliced_logits_patch.apply_sliced_logits_patch()
    logits = torch.arange(3 * width, device="cuda", dtype=torch.float32).reshape(3, width) / 3
    sampled = torch.zeros(3, dtype=torch.int64, device="cuda")
    k = min(num_logprobs, width)
    ids = torch.cat((sampled[:, None], logits.topk(k, dim=-1).indices), dim=-1)
    expected = (logits if logits_mode else logits.log_softmax(-1)).gather(-1, ids)
    for module in modules:
        output = module.compute_topk_scores(logits, num_logprobs, sampled, logits_mode=logits_mode)
        torch.testing.assert_close(output.logprob_token_ids, ids)
        torch.testing.assert_close(output.logprobs, expected)


pytestmark = pytest.mark.cpu_test
