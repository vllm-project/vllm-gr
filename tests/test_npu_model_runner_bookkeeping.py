# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="no-untyped-def"
"""Tests for NPU ModelRunner Beam sampler bookkeeping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.worker.npu import npu_model_runner_bookkeeping as bookkeeping


def test_logprob_remux_preserves_flattening_and_placeholder_rows() -> None:
    st_ids = torch.tensor([[10], [20], [30]])
    logprobs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    token_ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
    ranks = torch.tensor([7, 8, 9])

    final_st, final_lps, final_ids, final_ranks = bookkeeping._remux_logprobs_tensors(
        [
            ("ordinary", 0, 1, []),
            ("beam", 1, 2, [0, 1]),
            ("prefill", 3, 0, []),
        ],
        max_chained_dim=4,
        target_K=2,
        orig_lps=logprobs,
        orig_lp_ids=token_ids,
        orig_ranks=ranks,
        st_ids=st_ids,
    )

    assert final_st.tolist() == [[10], [20], [-1]]
    assert final_lps[:2].tolist() == [
        [1.0, 2.0, float("-inf"), float("-inf")],
        [3.0, 4.0, 5.0, 6.0],
    ]
    assert torch.isneginf(final_lps[2]).all()
    assert final_ids.tolist() == [[1, 2, 0, 0], [3, 4, 5, 6], [0, 0, 0, 0]]
    assert final_ranks.tolist() == [[7, 0, 0, 0], [8, 9, 0, 0], [0, 0, 0, 0]]


def test_beam_collapse_accumulates_partial_rows_and_clears_cache() -> None:
    runner = SimpleNamespace()
    beam_data = {"beam": {"is_beam_decode": True, "beam_width": 2}}

    def sampler(values: list[float], token_ids: list[int], rank: int) -> SimpleNamespace:
        return SimpleNamespace(
            sampled_token_ids=torch.tensor([[token_ids[0]]]),
            logprobs_tensors=SimpleNamespace(
                logprobs=torch.tensor([values]),
                logprob_token_ids=torch.tensor([token_ids]),
                selected_token_ranks=torch.tensor([rank]),
            ),
        )

    first = sampler([0.1, 0.2], [1, 2], 3)
    bookkeeping._beam_collapse_bookkeeping(
        runner,
        [("beam", 0, 1, [0])],
        beam_data,
        first,
    )
    assert runner._beam_logprobs_cache["beam"]["completed_beams"] == {0}
    assert first.logprobs_tensors.logprobs.shape == (1, 4)

    second = sampler([0.3, 0.4], [3, 4], 5)
    collapsed = bookkeeping._beam_collapse_bookkeeping(
        runner,
        [("beam", 0, 1, [1])],
        beam_data,
        second,
    )

    assert collapsed == (["beam"], {"beam": 0}, 1)
    assert torch.allclose(
        second.logprobs_tensors.logprobs,
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
    )
    assert second.logprobs_tensors.logprob_token_ids.tolist() == [[1, 2, 3, 4]]
    assert "beam" not in runner._beam_logprobs_cache


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
