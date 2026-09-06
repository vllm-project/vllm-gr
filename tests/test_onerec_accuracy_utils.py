# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from benchmarks.open_one_rec.one_rec_acc_test import frame_sid_generation
from benchmarks.open_one_rec.tasks.recommendation.utils_by_pid import (
    apply_sid_to_pid_strategy,
)


def test_frame_sid_generation_restores_prompt_injected_begin_marker() -> None:
    tail = "<s_a_1><s_b_2><s_c_3><|sid_end|>"

    assert frame_sid_generation(tail, "<|sid_begin|>") == (
        "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"
    )


def test_frame_sid_generation_does_not_duplicate_marker() -> None:
    framed = "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"

    assert frame_sid_generation(framed, "<|sid_begin|>") == framed


def test_downsampled_popularity_tie_preserves_mapping_order() -> None:
    candidates = [
        {"pid": 17, "count": 9, "count_after_downsample": 4},
        {"pid": 23, "count": 8, "count_after_downsample": 4},
        {"pid": 31, "count": 7, "count_after_downsample": 2},
    ]

    assert apply_sid_to_pid_strategy(candidates, "most_popular_after_downsampling") == 17


def test_downsampled_popularity_selects_later_strict_maximum() -> None:
    candidates = [
        {"iid": 17, "count": 9, "count_after_downsample": 3},
        {"iid": 23, "count": 8, "count_after_downsample": 5},
    ]

    assert apply_sid_to_pid_strategy(candidates, "most_popular_after_downsampling") == 23
