# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent selection and KV ownership checks shared by both platforms."""

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.constraints.backend import ConstraintOutput
from vllm_gr.v1.beam.beam_attention_pool import BeamAttentionPool
from vllm_gr.v1.beam.dynamic_beam import (
    DynamicBeamState,
    get_stage_pools,
    initialize_packed_stage_pools,
    prune_stage_pools,
    reorder_stage_kv,
)


def context(batch, device):
    return SimpleNamespace(
        max_batch=batch,
        max_beams=4,
        max_decode_steps=3,
        pools=[
            BeamAttentionPool(batch, 4, 1, 3, 8, torch.device(device), torch.float16)
            for _ in range(2)
        ],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("source,target", [(4, 2), (2, 4), (3, 3)])
def test_request_local_kv_fanout(device, batch, source, target):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    ctx = context(batch, device)
    banks = get_stage_pools(ctx, source)
    before = []
    for layer, pool in enumerate(banks):
        pool.unshared_key.copy_(
            torch.arange(pool.unshared_key.numel(), device=device).reshape_as(pool.unshared_key)
            + layer * 10000
        )
        pool.unshared_value.copy_(-pool.unshared_key)
        before.append((pool.unshared_key.clone(), pool.unshared_value.clone()))
    parents = torch.tensor([[source - 1, 0, source - 1, 0][:target]] * batch, device=device)
    reorder_stage_kv(ctx, source, target, parents, 2)
    for layer, pool in enumerate(get_stage_pools(ctx, target)):
        for b in range(batch):
            for child, parent in enumerate(parents[b].tolist()):
                torch.testing.assert_close(
                    pool.unshared_key[b, child, :, :2],
                    before[layer][0][b, parent, :, :2],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    pool.unshared_value[b, child, :, :2],
                    before[layer][1][b, parent, :, :2],
                    rtol=0,
                    atol=0,
                )


def packed_context(batch):
    """An idle context whose pools are laid out as one packed width bank."""
    return SimpleNamespace(
        max_batch=batch,
        max_beams=4,
        max_decode_steps=3,
        registry=SimpleNamespace(num_active_sessions=0),
        pools=[
            BeamAttentionPool(batch, 4, 1, 3, 8, torch.device("cpu"), torch.float16)
            for _ in range(2)
        ],
    )


def fill_packed_bank(ctx, width):
    # Layer offsets stay inside FP16 range so copies can be compared exactly.
    for layer, pool in enumerate(get_stage_pools(ctx, width)):
        pool.unshared_key.copy_(
            torch.arange(pool.unshared_key.numel(), dtype=torch.float16).reshape_as(
                pool.unshared_key
            )
            + layer * 1000
        )
        pool.unshared_value.copy_(-pool.unshared_key)


@pytest.mark.parametrize(
    "max_batch,batch,source,target",
    [(4, 2, 4, 2), (4, 4, 2, 4), (4, 4, 3, 3), (3, 2, 1, 3)],
)
def test_packed_reorder_matches_parent_gather(max_batch, batch, source, target):
    """NPU packed banks take separate code paths from the per-pool gather."""
    ctx = packed_context(max_batch)
    initialize_packed_stage_pools(ctx)
    fill_packed_bank(ctx, source)
    before = [
        (pool.unshared_key.clone(), pool.unshared_value.clone())
        for pool in get_stage_pools(ctx, source)
    ]
    parents = torch.tensor([[(source - 1), 0, (source - 1), 0][:target]] * batch)
    reorder_stage_kv(ctx, source, target, parents, 2)
    for layer, pool in enumerate(get_stage_pools(ctx, target)):
        for b in range(batch):
            for child, parent in enumerate(parents[b].tolist()):
                torch.testing.assert_close(
                    pool.unshared_key[b, child, :, :2],
                    before[layer][0][b, parent, :, :2],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    pool.unshared_value[b, child, :, :2],
                    before[layer][1][b, parent, :, :2],
                    rtol=0,
                    atol=0,
                )


def test_packed_bank_pruning_releases_only_uncaptured_widths():
    ctx = packed_context(2)
    initialize_packed_stage_pools(ctx)
    get_stage_pools(ctx, 2)
    get_stage_pools(ctx, 3)
    assert prune_stage_pools(ctx, (2,)) == (3,)
    assert set(ctx._npu_beam_width_pools) == {2}
    # The capacity bank is never released by pruning.
    assert set(ctx._npu_packed_width_pools) == {2, 4}


def test_packed_initialization_rejects_active_sessions():
    ctx = packed_context(2)
    ctx.registry.num_active_sessions = 1
    with pytest.raises(RuntimeError, match="active"):
        initialize_packed_stage_pools(ctx)


def test_widening_keeps_multiple_winners_from_one_parent():
    ctx = context(2, "cpu")
    state = DynamicBeamState(ctx, 2, 2)
    state.initialize(
        ConstraintOutput(
            torch.tensor([[1], [9]], dtype=torch.int32),
            torch.tensor([[-0.1], [-0.2]]),
            torch.ones((2, 1), dtype=torch.bool),
        ),
        1,
    )
    tokens = torch.tensor([[2, 3, 4, 5], [12, 13, 14, 15]], dtype=torch.int32)
    local = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.4, -0.3, -0.2, -0.1]])
    result = state.select(
        ConstraintOutput(tokens, local, torch.ones_like(tokens, dtype=torch.bool)),
        1,
        4,
        1,
        is_last=True,
        result_width=2,
    )
    assert result["out_sequence"].tolist() == [[[1, 2], [1, 3]], [[9, 15], [9, 14]]]
    assert result["out_token_index"].tolist() == [[0, 0], [1, 1]]
    torch.testing.assert_close(result["out_log_probs"], torch.tensor([[-0.2, -0.3], [-0.3, -0.4]]))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_branch_candidates_are_independent_from_retained_widths(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    ctx = context(1, device)
    state = DynamicBeamState(ctx, 1, 2)
    state.initialize(
        ConstraintOutput(
            torch.tensor([[12, 14]], dtype=torch.int32, device=device),
            torch.tensor([[-0.1, -0.2]], device=device),
            torch.ones((1, 2), dtype=torch.bool, device=device),
        ),
        2,
    )
    assert state.sequence[:, :, 0].tolist() == [[12, 14]]

    result = state.select(
        ConstraintOutput(
            torch.tensor([[21, 22], [31, 32]], dtype=torch.int32, device=device),
            torch.tensor([[-0.1, -0.2], [-0.3, -0.4]], device=device),
            torch.ones((2, 2), dtype=torch.bool, device=device),
        ),
        input_width=2,
        output_width=4,
        stage_index=1,
        is_last=True,
    )
    assert result["out_sequence"].shape == (1, 4, 2)


def test_initialize_treats_non_finite_scores_as_invalid():
    ctx = context(2, "cpu")
    state = DynamicBeamState(ctx, 2, 2)
    state.initialize(
        ConstraintOutput(
            torch.tensor([[1], [9]], dtype=torch.int32),
            torch.tensor([[float("nan")], [float("inf")]]),
            torch.ones((2, 1), dtype=torch.bool),
        ),
        1,
    )
    assert state.scores.tolist() == [[float("-inf")], [float("-inf")]]
    assert state.sequence[:, :, 0].tolist() == [[0], [0]]


def test_select_never_promotes_non_finite_candidates():
    """NaN sorts above every finite value in torch.topk and must not win."""
    ctx = context(1, "cpu")
    state = DynamicBeamState(ctx, 1, 2)
    state.initialize(
        ConstraintOutput(
            torch.tensor([[1]], dtype=torch.int32),
            torch.tensor([[-1.0]]),
            torch.ones((1, 1), dtype=torch.bool),
        ),
        1,
    )
    candidates = ConstraintOutput(
        torch.tensor([[2, 3, 4, 5]], dtype=torch.int32),
        torch.tensor([[float("nan"), float("inf"), -0.4, -0.5]]),
        torch.ones((1, 4), dtype=torch.bool),
    )
    result = state.select(candidates, 1, 2, 1, is_last=True, result_width=2)
    assert result["out_token_ids"].tolist() == [[4, 5]]
    torch.testing.assert_close(
        result["out_log_probs"], torch.tensor([[-1.4, -1.5]]), rtol=0, atol=1e-6
    )


def test_select_ignores_non_finite_parent_scores_per_batch():
    """NaN/+Inf parents must not eliminate another request's finite winner."""
    ctx = context(3, "cpu")
    state = DynamicBeamState(ctx, 3, 2)
    state.initialize(
        ConstraintOutput(
            torch.tensor([[1], [9], [5]], dtype=torch.int32),
            torch.tensor([[-1.0], [-1.0], [-1.0]]),
            torch.ones((3, 1), dtype=torch.bool),
        ),
        1,
    )
    state.scores[0, 0] = float("nan")
    state.scores[1, 0] = float("inf")
    candidates = ConstraintOutput(
        torch.tensor([[2], [3], [4]], dtype=torch.int32),
        torch.tensor([[0.0], [0.0], [-0.5]]),
        torch.ones((3, 1), dtype=torch.bool),
    )
    result = state.select(candidates, 1, 1, 1, is_last=True)
    assert result["out_token_ids"].tolist() == [[0], [0], [4]]
    torch.testing.assert_close(
        result["out_log_probs"],
        torch.tensor([[float("-inf")], [float("-inf")], [-1.5]]),
        rtol=0,
        atol=1e-6,
    )
    assert not torch.isnan(result["out_log_probs"]).any()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("returned", [1, 2])
def test_gpu_single_step_packing_filters_before_return_limit(bad, returned):
    from vllm_gr.v1.worker.gpu_dynamic_beam import GPUDynamicBeam

    adapter = GPUDynamicBeam.__new__(GPUDynamicBeam)
    adapter.state = DynamicBeamState(context(1, "cpu"), 1, 1)
    adapter.state.initialize(
        ConstraintOutput(
            torch.tensor([[11, 12]], dtype=torch.int32),
            torch.tensor([[bad, -0.2]], dtype=torch.float32),
            torch.ones((1, 2), dtype=torch.bool),
        ),
        2,
    )
    output = SimpleNamespace(
        options=SimpleNamespace(num_return_sequences=returned),
        layout=SimpleNamespace(
            steps=1, width=returned, row_stride=6, member_stride=1 + 6 * returned
        ),
        device=torch.zeros(4 + 6 * returned, dtype=torch.int32),
    )
    adapter.pack_final(output)
    assert output.device[3].item() == 1
    assert output.device[4].item() == 12
    assert output.device[6:7].view(torch.float32).item() == pytest.approx(-0.2)
