# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import numpy as np
import pytest
from vllm.exceptions import VLLMValidationError

from vllm_gr.beam_search_decision_utils import (
    _EXCEPT_MASKIN,
    GlobalRunningStats,
    create_stat,
    mask_beam_star_candidates,
)
from vllm_gr.entrypoints.openai.protocol import to_beam_search_params
from vllm_gr.sampling_params import BeamSearchParams


def test_mask_beam_star_candidates_prunes_below_threshold() -> None:
    stats = GlobalRunningStats()
    scores = np.array([-1.0, -2.0, -3.0, -10.0])
    indices = np.array([0, 1, 2, 3])
    # finite_scores = [-1.0, -2.0, -3.0, -10.0] -> s_max = -1.0, std = sqrt(12.5) ~= 3.5355
    # token_idx = 1 (since 0 is in _EXCEPT_MASKIN), coef = 0.5, token_idx_offset = 2
    # cutoff = -1.0 - 0.5 * (1 + 2) * 3.5355 = -6.3033
    masked_indices = mask_beam_star_candidates(
        scores, indices, stats, token_idx=1, token_idx_offset=2, coef=0.5
    )
    # finite scores >= -6.3033 are [-1.0, -2.0, -3.0], corresponding to indices [0, 1, 2]
    np.testing.assert_array_equal(masked_indices, np.array([0, 1, 2]))


def test_mask_beam_star_candidates_token_idx_offset() -> None:
    stats_offset_2 = GlobalRunningStats()
    stats_offset_1 = GlobalRunningStats()
    scores = np.array([0.0, -1.0, -2.0, -3.0, -4.0])
    indices = np.arange(5)
    # Population std of [0, -1, -2, -3, -4] is sqrt(2) ~= 1.4142; s_max = 0.
    # With token_idx_offset = 2: cutoff = 0 - 1.0 * (1 + 2) * 1.4142 = -4.2426 -> keep all
    masked_offset_2 = mask_beam_star_candidates(
        scores, indices, stats_offset_2, token_idx=1, token_idx_offset=2, coef=1.0
    )
    np.testing.assert_array_equal(masked_offset_2, indices)
    # With token_idx_offset = 1: cutoff = 0 - 1.0 * (1 + 1) * 1.4142 = -2.8284 -> keep [0, 1, 2]
    masked_offset_1 = mask_beam_star_candidates(
        scores, indices, stats_offset_1, token_idx=1, token_idx_offset=1, coef=1.0
    )
    np.testing.assert_array_equal(masked_offset_1, np.array([0, 1, 2]))


def test_mask_beam_star_candidates_except_maskin_no_pruning() -> None:
    stats = GlobalRunningStats()
    scores = np.array([-1.0, -2.0, -3.0, -10.0])
    indices = np.array([0, 1, 2, 3])
    # token_idx in _EXCEPT_MASKIN is exempt from pruning
    for t_idx in _EXCEPT_MASKIN:
        masked_indices = mask_beam_star_candidates(
            scores, indices, stats, token_idx=t_idx, token_idx_offset=2, coef=1.0
        )
        np.testing.assert_array_equal(masked_indices, indices)


def test_mask_beam_star_candidates_equal_scores_no_over_pruning() -> None:
    stats = GlobalRunningStats()
    scores = np.array([-1.5, -1.5, -1.5])
    indices = np.array([0, 1, 2])
    # s_max = -1.5, std = 0.0, token_idx = 1, coef = 3.3
    # cutoff = -1.5 - 3.3 * (1 + 2) * 0.0 = -1.5
    masked_indices = mask_beam_star_candidates(
        scores, indices, stats, token_idx=1, token_idx_offset=2, coef=3.3
    )
    np.testing.assert_array_equal(masked_indices, indices)


def test_mask_beam_star_candidates_handles_pre_existing_neg_inf() -> None:
    stats = GlobalRunningStats()
    scores = np.array([-1.0, -2.0, -float("inf"), -10.0])
    indices = np.array([0, 1, 2, 3])
    # finite_scores = [-1.0, -2.0, -10.0] -> s_max = -1.0, std ~= 4.0276
    # token_idx = 1, coef = 0.5, token_idx_offset = 2
    # cutoff = -1.0 - 0.5 * (1 + 2) * 4.0276 = -7.0414
    masked_indices = mask_beam_star_candidates(
        scores, indices, stats, token_idx=1, token_idx_offset=2, coef=0.5
    )
    # finite scores >= -7.0414 are [-1.0, -2.0] at indices [0, 1]
    np.testing.assert_array_equal(masked_indices, np.array([0, 1]))


def test_mask_beam_star_candidates_all_neg_inf() -> None:
    stats = GlobalRunningStats()
    scores = np.array([-float("inf"), -float("inf")])
    indices = np.array([0, 1])
    # token_idx 1 (active pruning step)
    masked_indices = mask_beam_star_candidates(
        scores, indices, stats, token_idx=1, token_idx_offset=1, coef=3.3
    )
    np.testing.assert_array_equal(masked_indices, indices)


@pytest.mark.parametrize("s_max", [-3.0, -6.0, -9.0, -12.0])  # type: ignore[misc]
@pytest.mark.parametrize("sigma", [2.0, 3.0, 4.0])  # type: ignore[misc]
@pytest.mark.parametrize("token_idx", [0, 1, 2, 3])  # type: ignore[misc]
@pytest.mark.parametrize("token_idx_offset", [1, 2])  # type: ignore[misc]
def test_mask_beam_star_candidates_parameterized(
    s_max: float, sigma: float, token_idx: int, token_idx_offset: int
) -> None:
    stats = GlobalRunningStats()
    base = np.array([s_max, s_max - 2.0 * sigma])
    stats.push(np.tile(base, 100000))
    coef = 2.0
    cutoff = s_max - coef * (token_idx + token_idx_offset) * sigma
    scores = np.array(
        [cutoff + 2.0, cutoff + 0.5, cutoff, cutoff - 0.5, cutoff - 10.0, -float("inf")]
    )
    indices = np.array([0, 1, 2, 3, 4, 5])
    masked_indices = mask_beam_star_candidates(
        scores, indices, stats, token_idx=token_idx, token_idx_offset=token_idx_offset, coef=coef
    )
    if token_idx in _EXCEPT_MASKIN:
        # Exempt from pruning
        expected_indices = indices
    else:
        expected_indices = np.array([0, 1, 2])
    np.testing.assert_array_equal(masked_indices, expected_indices)


def test_mask_beam_star_candidates_subset_indices() -> None:
    stats = GlobalRunningStats()
    scores = np.array([-1.0, -8.0, -2.0, -15.0, -3.0, -0.5])
    indices = np.array([1, 2, 3, 4])
    masked_indices = mask_beam_star_candidates(
        scores, indices, stats, token_idx=1, token_idx_offset=2, coef=1.0
    )
    sigma = stats.std
    s_max = stats.max
    cutoff = s_max - 1.0 * (1 + 2) * sigma
    expected = indices[scores[indices] >= cutoff]
    np.testing.assert_array_equal(masked_indices, expected)


def test_create_stat_singleton() -> None:
    if hasattr(create_stat, "_stats"):
        delattr(create_stat, "_stats")
    stats1 = create_stat(10)
    assert len(stats1) == 10
    assert hasattr(create_stat, "_stats")
    assert len(create_stat._stats) == 10

    stats2 = create_stat(5)
    assert len(stats2) == 5
    assert stats2[0] is stats1[0]

    stats3 = create_stat(15)
    assert len(stats3) == 15
    assert stats3[0] is stats1[0]
    assert len(create_stat._stats) == 15


def test_create_stat_thread_safety() -> None:
    import concurrent.futures

    if hasattr(create_stat, "_stats"):
        delattr(create_stat, "_stats")

    def worker(token_count: int) -> int:
        s = create_stat(token_count)
        for stat in s:
            stat.push(np.array([1.0, 2.0, 3.0]))
        return len(s)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, 50) for _ in range(20)]
        results = [f.result() for f in futures]

    assert all(r == 50 for r in results)
    assert len(create_stat._stats) == 50
    # Each of 20 workers pushed 3 items to each of 50 stats -> total n = 60 per stat
    assert create_stat._stats[0].n == 60


def test_beam_search_params_validation() -> None:
    params = BeamSearchParams(
        beam_width=2,
        max_tokens=10,
        beam_star=True,
        beam_star_coef=2.5,
    )
    assert params.beam_star is True
    assert params.beam_star_coef == 2.5

    with pytest.raises(VLLMValidationError) as exc_info:
        BeamSearchParams(beam_width=2, max_tokens=10, beam_star_coef=-0.5)
    assert "beam_star_coef must be positive" in str(exc_info.value)

    with pytest.raises(VLLMValidationError) as exc_info:
        BeamSearchParams(beam_width=2, max_tokens=10, beam_star_coef=0.0)
    assert "beam_star_coef must be positive" in str(exc_info.value)


def test_protocol_to_beam_search_params_parsing() -> None:
    dummy_req = types.SimpleNamespace(
        vllm_xargs={"beam_star": True, "beam_star_coef": 2.5},
        n=4,
        temperature=None,
        ignore_eos=False,
        length_penalty=1.0,
        include_stop_str_in_output=False,
        _DEFAULT_SAMPLING_PARAMS={"temperature": 0.0},
    )
    params = to_beam_search_params(dummy_req, max_tokens=20, default_sampling_params={})
    assert params.beam_width == 4
    assert params.max_tokens == 20
    assert params.beam_star is True
    assert params.beam_star_coef == 2.5


def test_pruned_beams_stride_extraction() -> None:
    from vllm_gr.beam_search_decision_utils import extract_and_dedup_segment

    # Simulate beam_width=128, pruned to 28 beams at token_idx=1
    beam_width = 128
    logprobs_num = beam_width
    active_beams_count = 28
    fork_info = [(i, 100 + i) for i in range(active_beams_count)]

    # Engine generates logprobs_num candidates for each active beam
    raw_token_ids = [1000 + i for i in range(active_beams_count * logprobs_num)]
    raw_logprobs = [-float(i) * 0.01 for i in range(active_beams_count * logprobs_num)]
    total_logprobs = len(raw_token_ids)

    stride_k = total_logprobs // len(fork_info)
    assert stride_k == 128

    all_beams_token_id = []
    all_beams_logprob = []
    for b_idx in range(len(fork_info)):
        start_offset = b_idx * stride_k
        end_offset = start_offset + stride_k
        token_ids_pos, logprobs_pos, _, _ = extract_and_dedup_segment(
            raw_token_ids,
            raw_logprobs,
            None,
            None,
            start_offset,
            end_offset,
            logprobs_num,
        )
        all_beams_token_id.extend(token_ids_pos)
        all_beams_logprob.extend(logprobs_pos)

    assert len(all_beams_token_id) == active_beams_count * logprobs_num
    assert len(all_beams_logprob) == 3584
    assert len(all_beams_logprob) != active_beams_count**2
