# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Full-vocabulary selection on an actual Ascend device, without custom ops."""

from __future__ import annotations

import math

import pytest
import torch

from vllm_gr.ops.npu.unconstrained_topk import select_unconstrained_candidates

pytestmark = pytest.mark.npu_test


@pytest.fixture
def npu_device() -> torch.device:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    return torch.device("npu")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("top_k", [1, 128, 544])
def test_npu_topk_matches_independent_host_enumeration(
    npu_device: torch.device, dtype: torch.dtype, batch_size: int, top_k: int
) -> None:
    vocab = 521
    host = torch.randn((batch_size, vocab), generator=torch.Generator().manual_seed(131)).to(dtype)
    host[0].zero_()
    if batch_size == 4:
        host[1].fill_(-float("inf"))
        host[1, 17], host[1, 30] = 1.0, 0.0
        host[1, 1], host[1, 2] = float("nan"), float("inf")
        host[2].fill_(-float("inf"))
        host[2, 0] = float("nan")
    temperatures = torch.tensor(([0.0, 0.5, 1.0, 2.0])[:batch_size])
    # A padded row stride must not make adjacent requests bleed into one another.
    storage = torch.empty((batch_size, vocab + 7), device=npu_device, dtype=dtype)
    logits = storage[:, :vocab]
    logits.copy_(host)
    actual = select_unconstrained_candidates(logits, temperatures.to(npu_device), top_k)
    ids, scores, masks = (
        actual.token_ids.cpu(),
        actual.logprobs.cpu(),
        actual.valid_mask.cpu(),
    )
    assert ids.dtype == torch.int32 and scores.dtype == torch.float32 and masks.dtype == torch.bool
    for row, values in enumerate(host.float().tolist()):
        temperature = temperatures[row].item() or 1.0
        finite = [
            (index, value / temperature)
            for index, value in enumerate(values)
            if math.isfinite(value)
        ]
        ordered = sorted(finite, key=lambda pair: (-pair[1], pair[0]))[:top_k]
        count = len(ordered)
        assert ids[row, :count].tolist() == [index for index, _ in ordered]
        assert masks[row].tolist() == [True] * count + [False] * (top_k - count)
        if count:
            maximum = max(value for _, value in finite)
            denominator = math.log(math.fsum(math.exp(value - maximum) for _, value in finite))
            expected = torch.tensor([(value - maximum) - denominator for _, value in ordered])
            torch.testing.assert_close(scores[row, :count], expected, rtol=1e-5, atol=2e-5)
        assert torch.isneginf(scores[row, count:]).all()
        assert (ids[row, count:] == 0).all()
        assert not torch.isnan(scores[row]).any()


def test_npu_selection_consumes_new_logits_at_the_same_storage_address(
    npu_device: torch.device,
) -> None:
    logits = torch.arange(17, dtype=torch.float32, device=npu_device).view(1, 17)
    temperatures = torch.ones(1, device=npu_device)
    pointer = logits.data_ptr()
    first = select_unconstrained_candidates(logits, temperatures, 3)
    first_ids = first.token_ids.cpu().tolist()
    logits.neg_()
    second = select_unconstrained_candidates(logits, temperatures, 3)
    assert logits.data_ptr() == pointer
    assert first_ids == [[16, 15, 14]]
    assert second.token_ids.cpu().tolist() == [[0, 1, 2]]


def test_npu_extreme_temperatures_preserve_finite_probability_mass(
    npu_device: torch.device,
) -> None:
    host = torch.tensor(
        [
            [2.0, 1.0, float("nan"), float("inf"), -float("inf")],
            [-2.0, -1.0, float("nan"), float("inf"), -float("inf")],
            [2.0, 2.0, 1.0, float("nan"), float("inf")],
            [3e38, 3e38, 2e38, float("nan"), float("inf")],
            [-3e38, -3e38, -3.1e38, float("nan"), float("inf")],
            [-3e38, 3e38, float("nan"), float("inf"), -float("inf")],
            [-3e38, 3e38, float("nan"), float("inf"), -float("inf")],
            [float("nan"), float("inf"), -float("inf"), float("nan"), float("inf")],
            [1.0, 2.0, float("nan"), float("inf"), -float("inf")],
            [0.0, 1e-40, float("nan"), float("inf"), -float("inf")],
            [2e-40, 1e-40, float("nan"), float("inf"), -float("inf")],
        ]
    )
    temperatures = torch.tensor(
        [1e-40, 1e-40, 1e-40, 0.0, 0.5, 3e38, 1.0, 1e-40, 3e38, 1e-40, 1e-40]
    )
    result = select_unconstrained_candidates(host.to(npu_device), temperatures.to(npu_device), 7)
    ids, scores, valid = result.token_ids.cpu(), result.logprobs.cpu(), result.valid_mask.cpu()
    for row, values in enumerate(host.tolist()):
        finite = sorted(
            ((index, value) for index, value in enumerate(values) if math.isfinite(value)),
            key=lambda pair: (-pair[1], pair[0]),
        )
        expected: list[tuple[int, float]] = []
        if finite:
            maximum, divisor = finite[0][1], temperatures[row].item() or 1.0
            centered = [(index, (value - maximum) / divisor) for index, value in finite]
            normalizer = math.log(math.fsum(math.exp(value) for _, value in centered))
            expected = [
                (index, value - normalizer)
                for index, value in centered
                if value - normalizer >= -torch.finfo(torch.float32).max
            ]
        count = len(expected)
        assert ids[row].tolist() == [index for index, _ in expected] + [0] * (7 - count)
        assert valid[row].tolist() == [True] * count + [False] * (7 - count)
        torch.testing.assert_close(
            scores[row, :count],
            torch.tensor([value for _, value in expected]),
            rtol=1e-5,
            atol=2e-5,
        )
        assert torch.isneginf(scores[row, count:]).all()
    assert not torch.isnan(scores).any()
