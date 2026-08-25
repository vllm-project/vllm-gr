# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import torch

from tests.constraints.test_table import make_table
from vllm_gr.constraints.validity import (
    constraint_candidate_counts,
    valid_mask_from_counts,
)


def _tables() -> tuple[torch.Tensor, ...]:
    table = make_table()
    return (
        torch.from_numpy(np.array(table.first_token_ids, copy=True)),
        torch.from_numpy(np.array(table.prefix1_offsets, copy=True)),
        torch.from_numpy(np.array(table.prefix1_pair_keys, copy=True)),
        torch.from_numpy(np.array(table.prefix2_value_offsets, copy=True)),
    )


def test_candidate_counts_cover_all_steps_and_invalid_prefixes() -> None:
    first_ids, offsets1, pair_keys, offsets2 = _tables()
    step0 = torch.zeros(2, 1, 1, dtype=torch.int32)
    assert constraint_candidate_counts(
        step0,
        first_ids,
        offsets1,
        pair_keys,
        offsets2,
        current_step=0,
        vocab_size=6,
    ).tolist() == [2, 2]

    step1 = torch.tensor([[[1, 0, 0], [3, 0, 0], [-1, 0, 0], [6, 0, 0]]])
    assert constraint_candidate_counts(
        step1,
        first_ids,
        offsets1,
        pair_keys,
        offsets2,
        current_step=1,
        vocab_size=6,
    ).tolist() == [2, 1, 0, 0]

    step2 = torch.tensor([[[1, 2, 0], [1, 4, 0], [3, 5, 0], [1, 3, 0], [-1, 2, 0]]])
    assert constraint_candidate_counts(
        step2,
        first_ids,
        offsets1,
        pair_keys,
        offsets2,
        current_step=2,
        vocab_size=6,
    ).tolist() == [2, 1, 2, 0, 0]


def test_valid_mask_is_a_contiguous_prefix_independent_of_scores() -> None:
    mask = valid_mask_from_counts(torch.tensor([3, 1, 0]), top_k=2)

    assert mask.is_contiguous()
    assert mask.tolist() == [[True, True], [True, False], [False, False]]
