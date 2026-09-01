# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_gr.v1.beam.beam_decision import (
    BeamPostprocessWorkspace,
    BeamPostprocessWorkspaceKey,
    BeamPostprocessWorkspacePool,
)

# pytest parametrization decorators are untyped under strict mypy.
# mypy: disable-error-code="misc"


def _workspace(max_batch: int = 2) -> BeamPostprocessWorkspace:
    return BeamPostprocessWorkspace.allocate(
        max_batch=max_batch,
        beam_width=4,
        top_k=4,
        max_decode_steps=3,
        device="cpu",
        logits_dtype=torch.float16,
        pin_host=False,
    )


def test_workspace_reserves_batch_dimension_and_fixed_protocol_buffers() -> None:
    workspace = _workspace()

    assert workspace.topk_token_ids.shape == (2, 4, 4)
    assert workspace.topk_scores.shape == (2, 4, 4)
    assert workspace.topk_valid_mask.shape == (2, 4, 4)
    assert workspace.topk_valid_counts.shape == (2, 4)
    assert workspace.selected_token_ids.shape == (2, 4)
    assert workspace.selected_parent_ids.shape == (2, 4)
    assert workspace.selected_scores.shape == (2, 4)
    assert workspace.next_sequence.shape == (2, 4, 3)
    assert workspace.final_token_ids.shape == (2, 4)
    assert workspace.final_parent_ids.shape == (2, 4)
    assert workspace.final_scores.shape == (2, 4)
    assert workspace.packed_result.shape == (24,)
    assert workspace.host_result.shape == (24,)
    assert workspace.host_result_numpy.shape == (24,)
    workspace.host_result[0] = 17
    assert int(workspace.host_result_numpy[0]) == 17
    assert workspace.copy_event is None
    assert workspace.prefill_staging.shape == (26,)
    assert workspace.host_prefill_staging.shape == (26,)
    assert workspace.host_prefill_staging_numpy.shape == (26,)
    workspace.host_prefill_staging[1] = 23
    assert int(workspace.host_prefill_staging_numpy[1]) == 23
    assert workspace.prefill_copy_event is None


def test_workspace_batch_slots_do_not_overwrite_each_other() -> None:
    workspace = _workspace()
    workspace.topk_token_ids.fill_(-1)
    workspace.selected_parent_ids.fill_(-1)

    workspace.topk_token_ids[0].fill_(11)
    workspace.selected_parent_ids[0].copy_(torch.tensor([3, 2, 1, 0], dtype=torch.int32))

    assert torch.equal(workspace.topk_token_ids[1], torch.full((4, 4), -1, dtype=torch.int32))
    assert torch.equal(workspace.selected_parent_ids[1], torch.full((4,), -1, dtype=torch.int32))

    workspace.topk_token_ids[1].fill_(22)
    workspace.selected_parent_ids[1].copy_(torch.tensor([0, 0, 2, 2], dtype=torch.int32))

    assert torch.equal(workspace.topk_token_ids[0], torch.full((4, 4), 11, dtype=torch.int32))
    assert torch.equal(
        workspace.selected_parent_ids[0],
        torch.tensor([3, 2, 1, 0], dtype=torch.int32),
    )


def test_workspace_views_are_contiguous_and_addresses_stay_stable() -> None:
    workspace = _workspace()
    addresses = tuple(tensor.data_ptr() for tensor in workspace.device_tensors())

    for batch_size in (1, 2, 1, 2):
        decode = workspace.candidate_rows(workspace.topk_token_ids, batch_size)
        compact = workspace.candidate_rows(
            workspace.topk_token_ids,
            batch_size,
            compact=True,
        )
        selected = workspace.active_beams(workspace.selected_token_ids, batch_size)
        assert decode.shape == (batch_size * 4, 4)
        assert compact.shape == (batch_size, 4)
        assert selected.shape == (batch_size, 4)
        assert decode.is_contiguous()
        assert compact.is_contiguous()
        assert selected.is_contiguous()

    assert tuple(tensor.data_ptr() for tensor in workspace.device_tensors()) == addresses


def test_workspace_pool_never_creates_a_bucket_on_require() -> None:
    workspace = _workspace()
    pool = BeamPostprocessWorkspacePool()
    pool.add(workspace)

    assert pool.require(workspace.key) is workspace
    assert pool.keys == (workspace.key,)
    assert pool.device_memory_bytes == workspace.device_memory_bytes
    with pytest.raises(ValueError, match="duplicate"):
        pool.add(workspace)
    with pytest.raises(KeyError, match="not prewarmed"):
        pool.require(BeamPostprocessWorkspaceKey(8, 8, torch.float16))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_batch": 0}, "max_batch"),
        ({"beam_width": 0}, "beam_width"),
        ({"top_k": 0}, "top_k"),
        ({"max_decode_steps": 0}, "max_decode_steps"),
    ],
)
def test_workspace_rejects_invalid_geometry(kwargs: dict[str, int], match: str) -> None:
    params = {
        "max_batch": 1,
        "beam_width": 4,
        "top_k": 4,
        "max_decode_steps": 3,
        "device": "cpu",
        "logits_dtype": torch.float16,
        "pin_host": False,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        BeamPostprocessWorkspace.allocate(**params)


def test_workspace_result_count_bounds() -> None:
    workspace = _workspace()
    assert workspace.result_count(1) == 4
    assert workspace.result_count(2, 3) == 6
    with pytest.raises(ValueError, match="batch_size"):
        workspace.result_count(3)
    with pytest.raises(ValueError, match="result_width"):
        workspace.result_count(1, 5)
