# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the NPU persistent lineage workspace.

These tests cover tensor/ABI semantics with CPU storage only. They are not
evidence of NPU data residency, stream behavior, D2H count, or performance.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.beam.beam_decision import BeamPostprocessWorkspace
from vllm_gr.v1.worker.gpu_model_runner_patch import CudaGRDeviceBackend
from vllm_gr.v1.worker.gr_model_runner_state import unpack_gr_final_output
from vllm_gr.v1.worker.npu_gr_workspace import (
    NPUWorkspace,
    NPUWorkspaceLease,
)


def _canonical(width: int, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((1, width, steps), dtype=torch.int32),
        torch.zeros((1, width), dtype=torch.float32),
    )


def test_workspace_binds_resolved_allocation_device_not_requested_alias() -> None:
    # cpu:0 resolves to cpu, just as an index-free accelerator request binds
    # to an indexed device. This is a real allocation, not a fake NPU tensor.
    workspace = NPUWorkspace.allocate(beam_width=4, max_steps=3, device="cpu:0")
    assert workspace.device == workspace.final_staging_i32.device == torch.device("cpu")
    sequence, scores = _canonical(4, 3)
    lease = workspace.acquire(
        "device-alias",
        generation_steps=3,
        result_width=2,
        canonical_sequence=sequence,
        canonical_scores=scores,
    )
    workspace.reset(lease)
    assert workspace.release(lease)


def _acquire(
    *, width: int = 4, steps: int = 3, result_width: int = 2
) -> tuple[NPUWorkspace, Any, torch.Tensor, torch.Tensor]:
    workspace = NPUWorkspace.allocate(
        beam_width=width,
        max_steps=steps,
        device="cpu",
    )
    sequence, scores = _canonical(width, steps)
    lease = workspace.acquire(
        "request",
        generation_steps=steps,
        result_width=result_width,
        canonical_sequence=sequence,
        canonical_scores=scores,
    )
    workspace.reset(lease)
    return workspace, lease, sequence, scores


def _record_all_valid_bootstrap(workspace, lease, scores, active):
    """Existing finite test fixtures have W structurally valid candidates."""
    return workspace.record_bootstrap(
        lease,
        scores,
        active,
        candidate_valid_mask=torch.ones_like(scores, dtype=torch.bool),
    )


@pytest.mark.parametrize(
    "values,candidates,active,expected_status",
    [
        (
            [-1.0, -float("inf"), -float("inf"), -float("inf")],
            [True, False, False, False],
            [True, False, False, False],
            0,
        ),
        ([-float("inf")] * 4, [True, True, False, False], [False] * 4, 0),
        ([-float("inf")] * 4, [False] * 4, [False] * 4, 1),
        ([-1.0, -float("inf"), -float("inf"), -float("inf")], [False] * 4, [False] * 4, 1),
        (
            [-1.0, -float("inf"), -float("inf"), -float("inf")],
            [True, False, False, False],
            [False] * 4,
            1,
        ),
        (
            [float("nan"), -float("inf"), -float("inf"), -float("inf")],
            [True, False, False, False],
            [False] * 4,
            1,
        ),
        (
            [float("inf"), -float("inf"), -float("inf"), -float("inf")],
            [True, False, False, False],
            [True, False, False, False],
            1,
        ),
        (
            [-float("inf"), -1.0, -float("inf"), -float("inf")],
            [False, True, False, False],
            [False, True, False, False],
            1,
        ),
    ],
)
def test_prefill_structural_validity_is_separate_from_beam_activity(
    values, candidates, active, expected_status
):
    workspace, lease, sequence, scores = _acquire(width=4, steps=1, result_width=4)
    scores.copy_(torch.tensor([values], dtype=torch.float32))
    sequence[0, :, 0] = torch.tensor([10, 0, 0, 0], dtype=torch.int32)
    assert workspace.record_bootstrap(
        lease,
        scores,
        torch.tensor([active], dtype=torch.bool),
        candidate_valid_mask=torch.tensor([candidates], dtype=torch.bool),
    )
    packed = workspace.pack_final(lease, sequence, scores)
    assert packed.status.tolist() == [expected_status]
    if expected_status:
        return
    assert workspace.active_mask.tolist() == [active]
    assert workspace.cumulative_delta_history[0, :, 0].tolist() == values
    # Exercise the real GPU packer on CPU, with its Prefill count-based
    # status and unmodified raw score history, including legal -inf padding.
    cuda_workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=4,
        top_k=4,
        max_decode_steps=1,
        device="cpu",
        logits_dtype=torch.float16,
        pin_host=False,
    )
    cuda_workspace.gr_invalid_status.zero_()
    cuda_workspace.gr_parent_history.zero_()
    cuda_workspace.gr_logprob_history[0, :, 0].copy_(scores[0])
    cuda_state = SimpleNamespace(
        next_token_ids=torch.empty((1, 4), dtype=torch.int32),
        step=1,
        beam_scores=scores,
        session=SimpleNamespace(
            context=SimpleNamespace(decision=SimpleNamespace(sequence=sequence))
        ),
    )
    cuda_staging, words, _ = CudaGRDeviceBackend._pack_final_result(cuda_state, cuda_workspace)
    assert torch.equal(cuda_staging[: 1 + words], packed.staging)


@pytest.mark.parametrize(
    "wrong,error",
    [(torch.ones((4,), dtype=torch.bool), ValueError), (torch.ones((1, 4)), TypeError)],
)
def test_prefill_candidate_mask_structure_validated_before_writes(wrong, error):
    workspace, lease, _, scores = _acquire(width=4, steps=1, result_width=4)
    with pytest.raises(error, match="candidate_valid_mask"):
        workspace.record_bootstrap(
            lease,
            scores,
            torch.ones((1, 4), dtype=torch.bool),
            candidate_valid_mask=wrong,
        )
    assert workspace.status.tolist() == [0]
    assert workspace._recorded_steps == 0


def _record_three_stage_lineage() -> tuple[NPUWorkspace, Any, torch.Tensor, torch.Tensor]:
    workspace, lease, sequence, scores = _acquire()

    # Bootstrap: four distinct initial beams.
    sequence[0, :, 0] = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    scores.copy_(torch.tensor([[-1.0, -2.0, -3.0, -4.0]]))
    assert not _record_all_valid_bootstrap(
        workspace,
        lease,
        scores,
        torch.ones((1, 4), dtype=torch.bool),
    )

    # Step 1: parent 1 produces two surviving children (rows 0 and 2).
    snapshot = workspace.before_decision(
        lease,
        scores,
        expected_device_step=1,
    )
    assert snapshot.data_ptr() == workspace.previous_scores.data_ptr()
    parents = torch.tensor([[1, 0, 1, 3]], dtype=torch.int32)
    scores.copy_(torch.tensor([[-2.5, -1.25, -2.75, -5.0]]))
    sequence[0, :, :2] = torch.tensor(
        [[[11, 20], [10, 21], [11, 22], [13, 23]]],
        dtype=torch.int32,
    )
    assert not workspace.record_decision(
        lease,
        parents,
        scores,
        torch.ones((1, 4), dtype=torch.bool),
        expected_device_step=1,
    )

    # Final R=2 < W=4. Final row 0 follows prior row 0; row 1 follows row 2.
    workspace.before_decision(lease, scores, expected_device_step=2)
    final_parents = torch.tensor([[0, 2]], dtype=torch.int32)
    scores[0, :2] = torch.tensor([-2.625, -3.125], dtype=torch.float32)
    # Deliberately pollute unused canonical tails: result_width must determine
    # the final ABI rather than workspace capacity or inactive beam contents.
    scores[0, 2:] = torch.tensor([999.0, 998.0])
    sequence[0, :2] = torch.tensor(
        [[[11, 20, 30], [11, 22, 31]]],
        dtype=torch.int32,
    )
    sequence[0, 2:] = 777
    assert workspace.record_decision(
        lease,
        final_parents,
        scores,
        torch.ones((1, 2), dtype=torch.bool),
        expected_device_step=2,
    )
    return workspace, lease, sequence, scores


def _unpack(pack: Any) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    decoded = unpack_gr_final_output(
        {
            "buffer_format": pack.buffer_format,
            "shape": list(pack.shape),
            "buffer": pack.payload.numpy().tobytes(),
        }
    )
    assert decoded is not None
    return decoded


def test_three_stage_multi_parent_lineage_r_lt_w_and_real_unpack() -> None:
    workspace, lease, sequence, scores = _record_three_stage_lineage()
    pack = workspace.pack_final(lease, sequence, scores)

    assert pack.shape == (2, 3)
    assert pack.payload_words == 20
    assert pack.staging.numel() == 21
    assert pack.payload.data_ptr() == pack.staging[1:].data_ptr()
    assert pack.status.data_ptr() == pack.staging.data_ptr()
    assert pack.status.tolist() == [0]
    sequences, parents, local_deltas, final_scores = _unpack(pack)
    assert sequences == [[11, 20, 30], [11, 22, 31]]
    assert parents == [[0, 1, 0], [0, 1, 2]]
    assert local_deltas == [[-2.0, -0.5, -0.125], [-2.0, -0.75, -0.375]]
    assert final_scores == [-2.625, -3.125]


def test_npu_and_cuda_cpu_packers_emit_bit_identical_payload() -> None:
    npu_workspace, lease, sequence, scores = _record_three_stage_lineage()
    npu_pack = npu_workspace.pack_final(lease, sequence, scores)

    cuda_workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=4,
        top_k=4,
        max_decode_steps=3,
        device="cpu",
        logits_dtype=torch.float16,
        pin_host=False,
    )
    cuda_workspace.gr_invalid_status.zero_()
    cuda_workspace.gr_parent_history.copy_(npu_workspace.parent_history)
    cuda_workspace.gr_logprob_history.copy_(npu_workspace.cumulative_delta_history)
    cuda_state = SimpleNamespace(
        next_token_ids=torch.empty((1, 2), dtype=torch.int32),
        step=3,
        beam_scores=scores,
        session=SimpleNamespace(
            context=SimpleNamespace(decision=SimpleNamespace(sequence=sequence))
        ),
    )
    cuda_staging, cuda_payload_words, cuda_width = CudaGRDeviceBackend._pack_final_result(
        cuda_state, cuda_workspace
    )
    cuda_payload = cuda_staging[
        cuda_workspace.max_batch : cuda_workspace.max_batch + cuda_payload_words
    ]

    assert cuda_width == npu_pack.shape[0] == 2
    assert cuda_payload_words == npu_pack.payload_words
    assert torch.equal(cuda_payload, npu_pack.payload)
    assert cuda_payload.numpy().tobytes() == npu_pack.payload.numpy().tobytes()
    assert _unpack(npu_pack) == unpack_gr_final_output(
        {
            "buffer_format": npu_pack.buffer_format,
            "shape": list(npu_pack.shape),
            "buffer": cuda_payload.numpy().tobytes(),
        }
    )


def test_cumulative_fp32_delta_is_intentionally_not_raw_candidate_logprob() -> None:
    workspace, lease, sequence, scores = _acquire(
        width=8,
        steps=2,
        result_width=8,
    )
    raw_local = torch.tensor(0.1, dtype=torch.float32)
    scores.fill_(-1000.0)
    assert not _record_all_valid_bootstrap(
        workspace, lease, scores, torch.ones((1, 8), dtype=torch.bool)
    )
    workspace.before_decision(lease, scores, expected_device_step=1)
    selected = scores[0, 0] + raw_local
    scores.fill_(selected)
    assert workspace.record_decision(
        lease,
        torch.zeros((1, 8), dtype=torch.int32),
        scores,
        torch.ones((1, 8), dtype=torch.bool),
        expected_device_step=1,
    )

    recovered = workspace.cumulative_delta_history[0, 0, 1]
    assert recovered == selected - torch.tensor(-1000.0, dtype=torch.float32)
    assert recovered.view(torch.int32) != raw_local.view(torch.int32)
    assert recovered == torch.tensor(0.0999755859375, dtype=torch.float32)


def test_fixed_addresses_survive_release_reset_and_new_lease() -> None:
    workspace, first, _sequence, first_scores = _acquire(
        width=8,
        steps=3,
        result_width=4,
    )
    tensors = (
        workspace.parent_history,
        workspace.cumulative_delta_history,
        workspace.previous_scores,
        workspace.gather_indices,
        workspace.backtrace_indices,
        workspace.status,
        workspace.final_staging_i32,
        workspace.control_storage_0,
        workspace.control_storage_1,
    )
    addresses = tuple(tensor.data_ptr() for tensor in tensors)
    _record_all_valid_bootstrap(
        workspace,
        first,
        first_scores,
        torch.ones((1, 8), dtype=torch.bool),
    )
    assert workspace.control_slot(first, device_step=1) is workspace.control_storage_1
    assert workspace.release(first)
    assert not workspace.release(first)

    sequence, scores = _canonical(8, 3)
    second = workspace.acquire(
        "request-2",
        generation_steps=3,
        result_width=4,
        canonical_sequence=sequence,
        canonical_scores=scores,
    )
    workspace.reset(second)
    _record_all_valid_bootstrap(
        workspace,
        second,
        scores,
        torch.ones((1, 8), dtype=torch.bool),
    )
    assert second.epoch == first.epoch + 1
    assert tuple(tensor.data_ptr() for tensor in tensors) == addresses
    assert torch.count_nonzero(workspace.parent_history) == 0
    assert torch.count_nonzero(workspace.cumulative_delta_history) == 0
    assert workspace.control_slot(second, device_step=1) is workspace.control_storage_1

    for operation in (
        lambda: workspace.reset(first),
        lambda: workspace.control_slot(first, device_step=1),
        lambda: workspace.pack_final(first, sequence, scores),
    ):
        with pytest.raises(RuntimeError, match="stale or foreign"):
            operation()
    foreign = NPUWorkspaceLease("foreign", second.epoch, 3, 4)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        workspace.release(foreign)


def test_reset_is_required_and_early_pack_is_rejected() -> None:
    workspace = NPUWorkspace.allocate(beam_width=8, max_steps=3, device="cpu")
    sequence, scores = _canonical(8, 3)
    lease = workspace.acquire(
        "request",
        generation_steps=3,
        result_width=4,
        canonical_sequence=sequence,
        canonical_scores=scores,
    )
    with pytest.raises(RuntimeError, match="reset before use"):
        _record_all_valid_bootstrap(workspace, lease, scores, torch.ones((1, 8), dtype=torch.bool))
    workspace.reset(lease)
    with pytest.raises(RuntimeError, match="already reset"):
        workspace.reset(lease)
    _record_all_valid_bootstrap(workspace, lease, scores, torch.ones((1, 8), dtype=torch.bool))
    with pytest.raises(RuntimeError, match="before terminal"):
        workspace.pack_final(lease, sequence, scores)


@pytest.mark.parametrize("invalid_lease", [None, object()])
def test_reset_and_release_reject_non_lease_objects(invalid_lease: object) -> None:
    workspace, _lease, _sequence, _scores = _acquire()
    with pytest.raises(RuntimeError, match="stale or foreign"):
        workspace.reset(invalid_lease)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="stale or foreign"):
        workspace.release(invalid_lease)  # type: ignore[arg-type]


def test_snapshot_and_record_ordering_rejects_missing_duplicate_and_stale_steps() -> None:
    workspace, lease, _sequence, scores = _acquire(
        width=8,
        steps=3,
        result_width=4,
    )
    active = torch.ones((1, 8), dtype=torch.bool)
    parents = torch.zeros((1, 8), dtype=torch.int32)
    with pytest.raises(RuntimeError, match="snapshot/device step"):
        workspace.before_decision(lease, scores, expected_device_step=1)
    _record_all_valid_bootstrap(workspace, lease, scores, active)
    with pytest.raises(RuntimeError, match="bootstrap lineage"):
        _record_all_valid_bootstrap(workspace, lease, scores, active)
    with pytest.raises(RuntimeError, match="no matching score snapshot"):
        workspace.record_decision(
            lease,
            parents,
            scores,
            active,
            expected_device_step=1,
        )
    workspace.before_decision(lease, scores, expected_device_step=1)
    with pytest.raises(RuntimeError, match="not consumed"):
        workspace.before_decision(lease, scores, expected_device_step=1)
    workspace.record_decision(
        lease,
        parents,
        scores,
        active,
        expected_device_step=1,
    )
    with pytest.raises(RuntimeError, match="no matching score snapshot"):
        workspace.record_decision(
            lease,
            parents,
            scores,
            active,
            expected_device_step=1,
        )


@pytest.mark.parametrize("invalid_step", [True, False, 1.0, "1", None])
def test_step_arguments_reject_bool_and_non_integer_values(invalid_step: object) -> None:
    workspace, lease, _sequence, scores = _acquire(
        width=8,
        steps=3,
        result_width=4,
    )
    active = torch.ones((1, 8), dtype=torch.bool)
    parents = torch.zeros((1, 8), dtype=torch.int32)
    _record_all_valid_bootstrap(workspace, lease, scores, active)

    with pytest.raises((TypeError, RuntimeError), match="step"):
        workspace.before_decision(
            lease,
            scores,
            expected_device_step=invalid_step,  # type: ignore[arg-type]
        )
    with pytest.raises(RuntimeError, match="step"):
        workspace.control_slot(
            lease,
            device_step=invalid_step,  # type: ignore[arg-type]
        )

    workspace.before_decision(lease, scores, expected_device_step=1)
    with pytest.raises((TypeError, RuntimeError), match="integer|step"):
        workspace.record_decision(
            lease,
            parents,
            scores,
            active,
            expected_device_step=invalid_step,  # type: ignore[arg-type]
        )


def test_fp32_delta_overflow_sets_status_and_clears_affected_beams() -> None:
    workspace, lease, sequence, scores = _acquire(
        width=4,
        steps=2,
        result_width=2,
    )
    finfo = torch.finfo(torch.float32)
    scores.copy_(torch.tensor([[-finfo.max, 0.0, 0.0, 0.0]]))
    _record_all_valid_bootstrap(workspace, lease, scores, torch.ones((1, 4), dtype=torch.bool))
    workspace.before_decision(lease, scores, expected_device_step=1)
    scores[0, :2] = torch.tensor([finfo.max, 1.0])
    assert workspace.record_decision(
        lease,
        torch.tensor([[0, 1]], dtype=torch.int32),
        scores,
        torch.ones((1, 2), dtype=torch.bool),
        expected_device_step=1,
    )

    pack = workspace.pack_final(lease, sequence, scores)
    assert pack.status.tolist() == [1]
    assert workspace.active_mask.tolist() == [[False, True, False, False]]
    assert torch.isfinite(workspace.cumulative_delta_history).all()


def test_bound_canonical_score_resize_is_detected_before_snapshot() -> None:
    workspace, lease, _sequence, scores = _acquire(
        width=4,
        steps=2,
        result_width=2,
    )
    _record_all_valid_bootstrap(workspace, lease, scores, torch.ones((1, 4), dtype=torch.bool))
    scores.resize_(1, 3)
    with pytest.raises(ValueError, match="canonical scores must have shape"):
        workspace.before_decision(lease, scores, expected_device_step=1)


def test_bound_canonical_sequence_resize_is_detected_before_final_pack() -> None:
    workspace, lease, sequence, scores = _acquire(
        width=4,
        steps=2,
        result_width=2,
    )
    _record_all_valid_bootstrap(workspace, lease, scores, torch.ones((1, 4), dtype=torch.bool))
    workspace.before_decision(lease, scores, expected_device_step=1)
    workspace.record_decision(
        lease,
        torch.tensor([[0, 1]], dtype=torch.int32),
        scores,
        torch.ones((1, 2), dtype=torch.bool),
        expected_device_step=1,
    )
    sequence.resize_(1, 4, 1)
    with pytest.raises(ValueError, match="canonical sequence must have shape"):
        workspace.pack_final(lease, sequence, scores)


@pytest.mark.parametrize(
    ("parents", "active", "score_value"),
    [
        (torch.tensor([[99, -1]], dtype=torch.int32), torch.ones((1, 2), dtype=torch.bool), 0.0),
        (torch.tensor([[0, 1]], dtype=torch.int32), torch.tensor([[True, False]]), 0.0),
        (
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.ones((1, 2), dtype=torch.bool),
            float("nan"),
        ),
        (
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.ones((1, 2), dtype=torch.bool),
            float("inf"),
        ),
    ],
)
def test_invalid_parent_inactive_or_nonfinite_sets_device_status_without_oob(
    parents: torch.Tensor,
    active: torch.Tensor,
    score_value: float,
) -> None:
    workspace, lease, sequence, scores = _acquire(
        width=4,
        steps=2,
        result_width=2,
    )
    scores.fill_(-1.0)
    _record_all_valid_bootstrap(workspace, lease, scores, torch.ones((1, 4), dtype=torch.bool))
    workspace.before_decision(lease, scores, expected_device_step=1)
    scores[0, :2].fill_(score_value)
    assert workspace.record_decision(
        lease,
        parents,
        scores,
        active,
        expected_device_step=1,
    )
    pack = workspace.pack_final(lease, sequence, scores)
    assert pack.status.tolist() == [1]
    expected_valid = torch.logical_and(
        active,
        torch.logical_and(
            torch.logical_and(parents >= 0, parents < 4),
            torch.isfinite(scores[:, :2]),
        ),
    )
    assert torch.equal(workspace.active_mask[:, :2], expected_valid)
    assert not torch.any(workspace.active_mask[:, 2:])
    assert torch.all(torch.isfinite(workspace.cumulative_delta_history))


@pytest.mark.parametrize(
    ("parents", "active", "match"),
    [
        (torch.zeros((4,), dtype=torch.int32), torch.ones((1, 4), dtype=torch.bool), "shape"),
        (torch.zeros((1, 4), dtype=torch.int64), torch.ones((1, 4), dtype=torch.bool), "int32"),
        (torch.zeros((1, 4), dtype=torch.int32), torch.ones((4,), dtype=torch.bool), "shape"),
    ],
)
def test_record_decision_rejects_invalid_parent_or_mask_layout_before_commit(
    parents: torch.Tensor,
    active: torch.Tensor,
    match: str,
) -> None:
    workspace, lease, _sequence, scores = _acquire(
        width=4,
        steps=3,
        result_width=2,
    )
    _record_all_valid_bootstrap(workspace, lease, scores, torch.ones((1, 4), dtype=torch.bool))
    workspace.before_decision(lease, scores, expected_device_step=1)
    with pytest.raises((TypeError, ValueError), match=match):
        workspace.record_decision(
            lease,
            parents,
            scores,
            active,
            expected_device_step=1,
        )
    assert workspace._recorded_steps == 1


def test_workspace_methods_do_not_materialize_device_tensors_on_host() -> None:
    for method in (
        NPUWorkspace.reset,
        NPUWorkspace.record_bootstrap,
        NPUWorkspace.before_decision,
        NPUWorkspace.record_decision,
        NPUWorkspace.pack_final,
        NPUWorkspace.control_slot,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert attributes.isdisjoint({"cpu", "tolist", "item", "numpy"})
