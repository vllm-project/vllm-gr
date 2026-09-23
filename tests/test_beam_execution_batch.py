# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the synchronized B x W execution view."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.beam.execution_batch import (
    BeamExecutionBatch,
    restore_cache_pool_layout,
    select_execution_pool,
    split_batch_decision,
    validate_batch_order,
)


def _sessions(batch_size: int = 2, width: int = 3, capacity: int | None = None):
    capacity = capacity or batch_size
    pool = SimpleNamespace(
        max_batch=capacity,
        max_beams=width,
        unshared_key=torch.arange(capacity * width).view(capacity, width, 1, 1, 1),
        unshared_value=torch.arange(capacity * width).view(capacity, width, 1, 1, 1),
    )
    context = SimpleNamespace(
        max_beams=width,
        max_decode_steps=4,
        pools=[pool],
        suffix_k_buf=torch.empty(batch_size * width * 4),
        suffix_v_buf=torch.empty(batch_size * width * 4),
    )
    sessions = []
    for slot in range(batch_size):
        state = SimpleNamespace(is_last_step=False)
        sessions.append(
            SimpleNamespace(
                session_id=f"req-{slot}",
                slot=slot,
                context=context,
                beam_width=width,
                decode_step=1,
                is_active=True,
                _state=state,
            )
        )
    return sessions


def test_execution_batch_flattens_contiguous_pool_without_copy() -> None:
    sessions = _sessions()
    batch = BeamExecutionBatch(sessions)

    assert batch.batch_size == 2
    assert batch.request_width == 3
    assert batch.beam_width == 3
    assert batch.attention_width == 6
    assert batch.pools[0].unshared_key.shape == (1, 6, 1, 1, 1)
    assert (
        batch.pools[0].unshared_key.data_ptr()
        == sessions[0].context.pools[0].unshared_key.data_ptr()
    )


def test_execution_batch_keeps_fixed_storage_view_for_smaller_active_batch() -> None:
    sessions = _sessions(batch_size=2, capacity=4)
    batch = BeamExecutionBatch(sessions)

    assert batch.attention_width == 6
    assert batch.pools[0].max_beams == 12
    assert batch.pools[0].unshared_key.shape == (1, 12, 1, 1, 1)


def test_execution_batch_rejects_slot_holes() -> None:
    sessions = _sessions()
    sessions[1].slot = 2

    with pytest.raises(RuntimeError, match="contiguous slots"):
        BeamExecutionBatch(sessions)


def test_validate_batch_order_rejects_scheduler_reordering() -> None:
    batch = BeamExecutionBatch(_sessions())
    beam_data = {"req-0": {}, "req-1": {}}

    validate_batch_order(batch, ["req-0", "req-1"], beam_data)
    with pytest.raises(RuntimeError, match="split, mixed, or reordered"):
        validate_batch_order(batch, ["req-1", "req-0"], beam_data)


def test_split_batch_decision_localizes_parent_indices() -> None:
    host = {
        "out_token_ids": [[10, 11, 12], [20, 21, 22]],
        "out_token_index": [[2, 0, 1], [4, 3, 5]],
        "out_log_probs": [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]],
    }

    result = split_batch_decision(["req-0", "req-1"], 3, host, False)

    assert result["req-0"]["out_token_index"] == [[2, 0, 1]]
    assert result["req-1"]["out_token_index"] == [[1, 0, 2]]
    assert result["req-1"]["session_id"] == "req-1"


def test_split_batch_decision_rejects_cross_request_parent() -> None:
    host = {
        "out_token_ids": [[10, 11], [20, 21]],
        "out_token_index": [[0, 1], [1, 2]],
    }

    with pytest.raises(RuntimeError, match="cross-request"):
        split_batch_decision(["req-0", "req-1"], 2, host, False)


def test_restore_cache_pool_layout_recovers_block_and_beam_dimensions() -> None:
    flat = torch.arange(24).view(1, 6, 2, 2)

    restored = restore_cache_pool_layout(flat, request_width=3)

    assert restored.shape == (2, 3, 2, 2)


def test_select_execution_pool_replaces_a_mismatched_cached_layout() -> None:
    cached = SimpleNamespace(unshared_key=torch.empty(2, 3, 1))
    current = SimpleNamespace(unshared_key=torch.empty(1, 6, 1))

    same_shape_other_storage = SimpleNamespace(unshared_key=torch.empty(1, 6, 1))
    assert select_execution_pool(same_shape_other_storage, current) is current
    assert select_execution_pool(cached, current) is current
    assert select_execution_pool(current, current) is current
    assert select_execution_pool(None, current) is current
