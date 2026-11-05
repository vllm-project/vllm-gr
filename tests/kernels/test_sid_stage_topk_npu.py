# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"
"""Actual-NPU SID stage normalization, global-ID mapping, and padding."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from vllm_gr.ops.npu.sid_stage_vocab import (
    build_sid_stage_vocab,
    select_sid_stage_candidates,
)

pytestmark = pytest.mark.npu_integration


@pytest.fixture
def npu_device() -> torch.device:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    return torch.device("npu")


def test_npu_full_and_compact_sid_stages_match_independent_normalization(
    npu_device: torch.device,
) -> None:
    entries = {
        "<s_a_0>": 17,
        "<s_a_1>": 2,
        "<s_a_2>": 7,
        "<s_b_0>": 11,
        "<s_c_0>": 19,
        "<s_c_1>": 4,
    }
    maps = build_sid_stage_vocab(SimpleNamespace(get_vocab=lambda: entries), 31, npu_device)
    host = torch.full((4, 31), 1000.0, dtype=torch.bfloat16)
    host[:, [2, 7, 17]] = torch.tensor(
        [[0, 0, 0], [float("nan"), float("inf"), -float("inf")], [1, 2, 3], [2, 2, -float("inf")]],
        dtype=torch.bfloat16,
    )
    host[:, 11] = torch.tensor([0, float("nan"), 5, float("inf")], dtype=torch.bfloat16)
    host[:, [4, 19]] = torch.tensor(
        [[0, 0], [float("inf"), float("nan")], [3, 1], [2, -float("inf")]], dtype=torch.bfloat16
    )
    temperature = torch.tensor([0.0, 0.5, 1.0, 2.0])
    for stage, ids in enumerate(([2, 7, 17], [11], [4, 19])):
        compact = torch.full((4, maps.vocab_size), 10000.0, dtype=host.dtype)
        compact[:, : len(ids)] = host[:, ids]
        for values, is_compact in ((host, False), (compact, True)):
            result = select_sid_stage_candidates(
                values.to(npu_device),
                temperature.to(npu_device),
                8,
                stage,
                maps,
                compact_logits=is_compact,
            )
            tokens, scores, valid = (
                result.token_ids.cpu(),
                result.logprobs.cpu(),
                result.valid_mask.cpu(),
            )
            for row in range(4):
                divisor = float(temperature[row]) or 1.0
                finite = [
                    (token, float(host[row, token]) / divisor)
                    for token in ids
                    if math.isfinite(float(host[row, token]))
                ]
                ordered = sorted(finite, key=lambda pair: (-pair[1], pair[0]))
                count = len(ordered)
                assert tokens[row].tolist() == [token for token, _ in ordered] + [0] * (8 - count)
                assert valid[row].tolist() == [True] * count + [False] * (8 - count)
                if count:
                    maximum = max(value for _, value in finite)
                    normalizer = math.log(
                        math.fsum(math.exp(value - maximum) for _, value in finite)
                    )
                    expected = torch.tensor(
                        [(value - maximum) - normalizer for _, value in ordered]
                    )
                    torch.testing.assert_close(scores[row, :count], expected, rtol=1e-5, atol=2e-5)
                assert torch.isneginf(scores[row, count:]).all()
                assert not torch.isnan(scores[row]).any()


def test_npu_sid_selection_consumes_updated_logits_without_reusing_padding(
    npu_device: torch.device,
) -> None:
    maps = build_sid_stage_vocab(
        SimpleNamespace(
            get_vocab=lambda: {
                "<s_a_0>": 2,
                "<s_a_1>": 7,
                "<s_a_2>": 17,
                "<s_b_0>": 11,
                "<s_c_0>": 4,
                "<s_c_1>": 19,
            }
        ),
        31,
        npu_device,
    )
    logits = torch.tensor([[0.0, 0.0, 1e9]], device=npu_device)
    temperature = torch.ones(1, device=npu_device)
    pointer = logits.data_ptr()
    first = select_sid_stage_candidates(logits, temperature, 4, 2, maps, compact_logits=True)
    assert first.token_ids.cpu().tolist() == [[4, 19, 0, 0]]
    logits[:, 0] = -float("inf")
    logits[:, 1] = 2.0
    second = select_sid_stage_candidates(logits, temperature, 4, 2, maps, compact_logits=True)
    assert logits.data_ptr() == pointer
    assert second.token_ids.cpu().tolist() == [[19, 0, 0, 0]]
    assert second.valid_mask.cpu().tolist() == [[True, False, False, False]]
    assert second.logprobs[0, 0].cpu() == 0


@pytest.mark.parametrize("compact", [False, True])
def test_npu_tiny_temperature_preserves_sid_ties_and_excludes_projection_padding(
    npu_device: torch.device, compact: bool
) -> None:
    entries = {
        "<s_a_0>": 2,
        "<s_a_1>": 7,
        "<s_a_2>": 17,
        "<s_b_0>": 11,
        "<s_c_0>": 4,
        "<s_c_1>": 19,
    }
    maps = build_sid_stage_vocab(SimpleNamespace(get_vocab=lambda: entries), 31, npu_device)
    host = torch.full((3, 31), 1e9)
    host[:, [4, 19]] = torch.tensor([[2.0, 1.0], [2.0, 2.0], [float("nan"), float("inf")]])
    compact_logits = torch.full((3, maps.vocab_size), 1e9)
    compact_logits[:, :2] = host[:, [4, 19]]
    result = select_sid_stage_candidates(
        (compact_logits if compact else host).to(npu_device),
        torch.tensor([1e-40], device=npu_device),
        4,
        2,
        maps,
        compact_logits=compact,
    )
    ids, scores, valid = result.token_ids.cpu(), result.logprobs.cpu(), result.valid_mask.cpu()
    assert ids.tolist() == [[4, 0, 0, 0], [4, 19, 0, 0], [0, 0, 0, 0]]
    assert valid.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, False, False, False],
    ]
    assert scores[0, 0] == 0
    torch.testing.assert_close(scores[1, :2], torch.full((2,), -math.log(2)))
    assert torch.isneginf(scores[~valid]).all()
    assert not torch.isnan(scores).any()
