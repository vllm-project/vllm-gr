# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-window numerical parity of actual streamed CUDA candidate kernels."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.kernels.test_rec_constrained_topk_cuda import _wide_window_inputs
from vllm_gr.ops.cuda.rec_constrained_topk import _rec_constrained_topk_cuda_out_unchecked
from vllm_gr.v1.worker.streaming_constrained_head import StreamingConstrainedHead

# mypy: disable-error-code="misc,no-untyped-call"
pytestmark = [
    pytest.mark.gpu_test,
    pytest.mark.l2,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]


def outputs(rows: int, width: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((rows, width), dtype=torch.int32, device="cuda"),
        torch.empty((rows, width), dtype=torch.float32, device="cuda"),
        torch.empty((rows, width), dtype=torch.bool, device="cuda"),
        torch.empty((rows,), dtype=torch.int32, device="cuda"),
    )


def prepare(monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, vocab: int) -> tuple[Any, ...]:
    inputs = _wide_window_inputs(dtype, 3, vocab, 64)
    metadata = SimpleNamespace(
        vocab_size=vocab, trie_count=1, max_prefix1_degree=64, max_prefix2_degree=64
    )
    backend = SimpleNamespace(
        info=SimpleNamespace(table_metadata=metadata),
        vocab_slice=None,
        _tables=lambda: inputs[2:8],
        _first_token_offsets=None,
        _max_first_degree=64,
    )
    weight = torch.empty((vocab, 1), dtype=dtype, device="cuda")
    head = StreamingConstrainedHead(weight, backend, 3, 16)

    def project(hidden: torch.Tensor, block: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
        # Isolate collection/normalization from GEMM rounding: the projected
        # values are identical to the full-window oracle, still on the GPU.
        start = block.storage_offset()
        out.copy_(inputs[0][:, start : start + out.shape[1]])
        return out

    monkeypatch.setattr(torch, "mm", project)
    hidden = torch.empty((3, 1), dtype=dtype, device="cuda")
    return inputs, head, hidden


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab", [2049, 8193, 24576, 65537])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_streamed_candidates_match_identical_full_logits(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    vocab: int,
    step: int,
) -> None:
    inputs, head, hidden = prepare(monkeypatch, dtype, vocab)
    reference, actual = outputs(3, 16), outputs(3, 16)
    _rec_constrained_topk_cuda_out_unchecked(
        *inputs,
        step,
        16,
        64,
        64,
        *reference,
        vocab_offset=0,
        table_vocab_size=inputs[0].shape[1],
        out_wire_logprobs=None,
    )
    head.select(hidden, inputs[1], inputs[-1], actual[:3], actual[3], None, current_step=step)
    for expected, value in zip(reference, actual):
        torch.testing.assert_close(value, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("poison", ["leading_negative_infinity", "nan", "positive_infinity"])
def test_streamed_lse_preserves_nonfinite_window_semantics(
    monkeypatch: pytest.MonkeyPatch,
    poison: str,
) -> None:
    inputs, head, hidden = prepare(monkeypatch, torch.float32, 8193)
    if poison == "leading_negative_infinity":
        inputs[0][:, :2048].fill_(-float("inf"))
    else:
        inputs[0][:, 1024] = float("nan") if poison == "nan" else float("inf")
    reference, actual = outputs(3, 16), outputs(3, 16)
    _rec_constrained_topk_cuda_out_unchecked(
        *inputs,
        0,
        16,
        64,
        64,
        *reference,
        vocab_offset=0,
        table_vocab_size=inputs[0].shape[1],
        out_wire_logprobs=None,
    )
    head.select(hidden, inputs[1], inputs[-1], actual[:3], actual[3], None)
    for expected, value in zip(reference, actual):
        torch.testing.assert_close(value, expected, atol=0, rtol=0, equal_nan=True)
    if poison == "leading_negative_infinity":
        denominator = head.stats[:, 0] + torch.log(head.stats[:, 1])
        torch.testing.assert_close(
            denominator,
            torch.logsumexp(
                inputs[0] / torch.where(inputs[-1] == 0, 1.0, inputs[-1])[:, None], dim=1
            ),
            atol=2e-6,
            rtol=2e-6,
        )


def test_graph_replay_reads_updated_logits_and_step(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, head, hidden = prepare(monkeypatch, torch.bfloat16, 8193)
    actual, reference = outputs(3, 16), outputs(3, 16)
    step = torch.zeros(1, dtype=torch.int32, device="cuda")
    head.select(hidden, inputs[1], inputs[-1], actual[:3], actual[3], step)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        head.select(hidden, inputs[1], inputs[-1], actual[:3], actual[3], step)
    for current in (0, 1, 2):
        step.fill_(current)
        inputs[0].add_(0.25)
        graph.replay()
        _rec_constrained_topk_cuda_out_unchecked(
            *inputs,
            current,
            16,
            64,
            64,
            *reference,
            vocab_offset=0,
            table_vocab_size=inputs[0].shape[1],
            out_wire_logprobs=None,
        )
        for expected, value in zip(reference, actual):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)


def test_changed_output_geometry_is_rejected_before_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, head, hidden = prepare(monkeypatch, torch.bfloat16, 8193)
    wrong = outputs(3, 8)
    with pytest.raises(ValueError, match="output geometry changed"):
        head.select(hidden, inputs[1], inputs[-1], wrong[:3], wrong[3], None)
