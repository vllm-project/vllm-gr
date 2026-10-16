# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Heterogeneous early-EOS batches must preserve active members' packed KV."""

import pytest
import torch

from tests.test_gr_input_binding import exercise_mixed_batch_kv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("batch_size, finished_index", [(2, 0), (2, 1), (4, 0), (4, 2)])
@pytest.mark.parametrize("step", [0, 1])
@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
def test_early_eos_does_not_change_live_members_packed_kv(
    batch_size: int, finished_index: int, step: int, graph_mode: bool
) -> None:
    exercise_mixed_batch_kv(batch_size, finished_index, step, device="cuda", graph_mode=graph_mode)
