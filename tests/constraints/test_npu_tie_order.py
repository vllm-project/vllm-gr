# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"

"""Tie-order and scoring-domain parity between the NPU wrapper and the oracle.

The fused NPU operator does not promise the CUDA oracle's exact-tie order
(score descending, then lower constraint-table offset) and normalizes over
each row's legal candidate set.  The NPU wrapper must re-score the operator's
selection on the shared full-window domain (like the CPU/CUDA paths) and
restore that order, so a fake fused operator that shuffles tie groups and
emits candidate-set scores must still produce byte-identical output to the
Torch reference.
"""

from __future__ import annotations

import importlib
from typing import cast

import numpy as np
import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.ops.rec_constrained_topk import rec_constrained_topk_torch

npu_op_module = importlib.import_module("vllm_gr.ops.npu.rec_constrained_topk")

_TABLE = make_table()


def _table_tensors() -> dict[str, torch.Tensor]:
    return {
        "first_token_ids": torch.from_numpy(np.array(_TABLE.first_token_ids, copy=True)),
        "prefix1_offsets": torch.from_numpy(np.array(_TABLE.prefix1_offsets, copy=True)),
        "prefix1_values": torch.from_numpy(np.array(_TABLE.prefix1_values, copy=True)),
        "prefix1_pair_keys": torch.from_numpy(np.array(_TABLE.prefix1_pair_keys, copy=True)),
        "prefix2_value_offsets": torch.from_numpy(
            np.array(_TABLE.prefix2_value_offsets, copy=True)
        ),
        "prefix2_values": torch.from_numpy(np.array(_TABLE.prefix2_values, copy=True)),
    }


class _ShufflingFusedOp:
    """Stand-in for the fused kernel: candidate-set scores, shuffled ties.

    Mirrors the real fused operator's output contract: the valid prefix of
    every row is reversed — scrambling exact-tie order (and non-tie order)
    while keeping valid-before-invalid packing — and every valid score is
    shifted by a per-row constant, i.e. the candidate-set domain
    (``logit/T - logsumexp`` over the row's legal slice) differs from the
    shared full-window domain by exactly such a constant.  A wrapper that
    passed the operator's scores through would therefore fail the byte-equal
    comparison against the full-window reference; only a wrapper that
    re-scores the selection can match it.
    """

    def rec_constrained_topk(self, *args: object) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, logprobs = rec_constrained_topk_torch(*args)
        valid = torch.isfinite(logprobs) & logprobs.gt(-1.0e19)
        counts = valid.sum(dim=1)
        tokens = tokens.clone()
        logprobs = logprobs.clone()
        for row in range(tokens.shape[0]):
            count = int(counts[row])
            tokens[row, :count] = torch.flip(tokens[row, :count], dims=(0,))
            logprobs[row, :count] = torch.flip(logprobs[row, :count], dims=(0,))
            # Candidate-set domain stand-in: a per-row constant offset from
            # the full-window scores the reference computes.
            logprobs[row, :count] -= float(row + 1) * 0.5
        return tokens.to(torch.int32), logprobs


@pytest.fixture()
def shuffled_fused_op(monkeypatch: pytest.MonkeyPatch) -> _ShufflingFusedOp:
    class _FakeModule:
        rec_constrained_topk = _ShufflingFusedOp().rec_constrained_topk  # type: ignore[method-assign]

    monkeypatch.setattr(npu_op_module, "_custom_ops", lambda: cast("object", _FakeModule()))
    # Force chunked slice scans and single-column top-k blocks so the chunk-
    # and block-boundary logic of the tie-order scan is exercised.
    monkeypatch.setattr(npu_op_module, "_TIE_SCAN_CHUNK", 2)
    monkeypatch.setattr(npu_op_module, "_TIE_TOPK_CHUNK", 1)
    return _ShufflingFusedOp()


def _assert_matches_oracle(
    logits: torch.Tensor,
    sequence_group: torch.Tensor | None,
    current_step: int,
    top_k: int,
    shuffled_fused_op: _ShufflingFusedOp,
) -> tuple[torch.Tensor, torch.Tensor]:
    tables = _table_tensors()
    if sequence_group is None:
        sequence_group = torch.zeros((logits.shape[0], 1), dtype=torch.int64)
    temperatures = torch.ones(logits.shape[0], dtype=torch.float32)
    got = npu_op_module.rec_constrained_topk_npu(
        logits,
        sequence_group,
        tables["first_token_ids"],
        tables["prefix1_offsets"],
        tables["prefix1_values"],
        tables["prefix1_pair_keys"],
        tables["prefix2_value_offsets"],
        tables["prefix2_values"],
        temperatures,
        current_step,
        top_k,
        int(_TABLE.metadata.max_prefix1_degree),
        int(_TABLE.metadata.max_prefix2_degree),
    )
    expected = rec_constrained_topk_torch(
        logits,
        sequence_group,
        tables["first_token_ids"],
        tables["prefix1_offsets"],
        tables["prefix1_values"],
        tables["prefix1_pair_keys"],
        tables["prefix2_value_offsets"],
        tables["prefix2_values"],
        temperatures,
        current_step,
        top_k,
    )
    assert torch.equal(got[0].to(torch.int64), expected[0])
    assert torch.equal(got[1], expected[1])
    return got


def test_step0_tie_restores_table_offset_order(shuffled_fused_op: _ShufflingFusedOp) -> None:
    # first_token_ids = [1, 3]; equal logits for both -> the oracle keeps the
    # lower table offset (token 1) first; the shuffled op returns [3, 1].
    logits = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    tokens, _ = _assert_matches_oracle(logits, None, 0, 2, shuffled_fused_op)
    assert tokens[0, 0].item() == 1


def test_step1_tie_restores_table_offset_order(shuffled_fused_op: _ShufflingFusedOp) -> None:
    # sequence starts with t0=1 -> prefix1 slice values[0:2] = [2, 4]; tie
    # between them must resolve to token 2 (lower offset).
    logits = torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.5, 0.0]], dtype=torch.float32)
    sequence_group = torch.tensor([[1, 2]], dtype=torch.int64)
    tokens, _ = _assert_matches_oracle(logits, sequence_group, 1, 2, shuffled_fused_op)
    assert tokens[0, 0].item() == 2


def test_step2_tie_restores_table_offset_order(shuffled_fused_op: _ShufflingFusedOp) -> None:
    # pair (1, 2) -> key 1*6+2 = 8 -> prefix2_values[0:2] = [0, 1]; tie must
    # resolve to token 0 (lower offset).
    logits = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    sequence_group = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    tokens, _ = _assert_matches_oracle(logits, sequence_group, 2, 2, shuffled_fused_op)
    assert tokens[0, 0].item() == 0


def test_distinct_scores_are_restored_exactly(shuffled_fused_op: _ShufflingFusedOp) -> None:
    # Candidates are constrained to first_token_ids {1, 3}; distinct scores
    # must come back in score order even after the fused-op shuffle.
    logits = torch.tensor([[0.0, 0.9, 0.0, 0.3, 0.0, 0.0]], dtype=torch.float32)
    tokens, _ = _assert_matches_oracle(logits, None, 0, 2, shuffled_fused_op)
    assert tokens[0].tolist() == [1, 3]


def test_invalid_tail_stays_behind_valid_entries(shuffled_fused_op: _ShufflingFusedOp) -> None:
    # Degree-1 row: one valid candidate, one invalid padding slot. The
    # padding must stay in the tail column after the reorder.
    logits = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    tables = _table_tensors()
    tokens, logprobs = npu_op_module.rec_constrained_topk_npu(
        logits[:1],
        torch.zeros((1, 1), dtype=torch.int64),
        torch.tensor([3], dtype=torch.int32),
        tables["prefix1_offsets"],
        tables["prefix1_values"],
        tables["prefix1_pair_keys"],
        tables["prefix2_value_offsets"],
        tables["prefix2_values"],
        torch.ones(1, dtype=torch.float32),
        0,
        2,
        2,
        2,
    )
    assert tokens[0, 0].item() == 3
    # fp32 cannot represent -1e20 exactly; assert the invalid sentinel magnitude.
    assert logprobs[0, 1].item() < -1.0e19
