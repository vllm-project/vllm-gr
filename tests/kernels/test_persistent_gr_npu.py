# SPDX-License-Identifier: Apache-2.0
"""Real-NPU decision ABI probes; not ModelRunner, KV, or ACL acceptance.

In particular, test a fixed sequence capacity larger than the current valid
length. Existing kernel smoke cases used capacity == current_step + 1.
Missing NPU hardware is an unvalidated environment, never an acceptance PASS.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.ops.onerec_decode import (
    _beam_search_group_torch,
    _beam_search_rec_final_select_torch,
)
from vllm_gr.v1.beam.beam_attention_context import BeamDecisionBuffers
from vllm_gr.v1.worker.gr_model_runner_state import unpack_gr_final_output
from vllm_gr.v1.worker.npu_gr_decision import (
    advance_fused_decision,
    bootstrap_from_constrained_prefill,
)
from vllm_gr.v1.worker.npu_gr_workspace import NPUWorkspace

pytestmark = pytest.mark.npu_integration

try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"NPU environment not validated: {exc}", allow_module_level=True)

if not torch.npu.is_available():
    pytest.skip("NPU environment not validated: no device", allow_module_level=True)


@pytest.mark.parametrize("width", [64, 128, 256])
@pytest.mark.parametrize("capacity,generation_steps", [(3, 2), (5, 3)])
def test_persistent_decision_fixed_capacity_matches_cpu_golden(
    width: int, capacity: int, generation_steps: int
) -> None:
    device = torch.device("npu")
    decision = BeamDecisionBuffers(1, width, capacity, device, torch.float32)
    session = SimpleNamespace(
        is_active=True,
        slot=0,
        beam_width=width,
        decode_step=0,
        context=SimpleNamespace(decision=decision),
    )
    sequence_ptr = decision.sequence.data_ptr()
    scores_ptr = decision.beam_scores.data_ptr()
    generator = torch.Generator().manual_seed(54)
    tokens = torch.arange(width, dtype=torch.int32).reshape(1, width)
    scores = -torch.rand((1, width), generator=generator)
    reference_sequence = torch.zeros((1, width, capacity), dtype=torch.int32)
    reference_sequence[:, :, 0].copy_(tokens)
    reference_scores = scores
    reference_parents = [torch.zeros((1, width), dtype=torch.int32)]
    reference_deltas = [scores]
    workspace = NPUWorkspace.allocate(beam_width=width, max_steps=capacity, device=device)
    lease = workspace.acquire(
        "npu-decision-probe",
        generation_steps=generation_steps,
        result_width=width // 2,
        canonical_sequence=decision.sequence,
        canonical_scores=decision.beam_scores,
    )
    workspace.reset(lease)
    staging_ptr = workspace.final_staging_i32.data_ptr()

    # No injected ops: these calls must use the actual NPU decision exports.
    bootstrap = bootstrap_from_constrained_prefill(
        session, tokens.to(device), scores.to(device), expected_width=width
    )
    assert bootstrap.device_step == 1
    assert bootstrap.reorder_parents is None
    assert bootstrap.reorder_prefix is None
    torch.testing.assert_close(bootstrap.token_ids.cpu(), tokens, rtol=0, atol=0)
    torch.testing.assert_close(bootstrap.scores.cpu(), scores, rtol=0, atol=0)
    assert not workspace.record_bootstrap(
        lease,
        decision.beam_scores,
        torch.isfinite(bootstrap.scores),
        candidate_valid_mask=torch.ones_like(bootstrap.scores, dtype=torch.bool),
    )

    for step in range(1, generation_steps):
        candidate_tokens = torch.randint(
            0, 10000, (width, width), generator=generator, dtype=torch.int32
        )
        candidate_scores = -torch.rand((width, width), generator=generator)
        final = step == generation_steps - 1
        # Exercise a narrower final result as well as fixed-width Group.
        result_width = width // 2 if final else width
        golden = _beam_search_rec_final_select_torch if final else _beam_search_group_torch
        expected = golden(
            reference_scores.reshape(width, 1),
            candidate_tokens,
            candidate_scores,
            reference_sequence,
            step,
            result_width,
        )
        parents = expected[1].reshape(1, result_width)
        reference_parents.append(parents)
        reference_deltas.append(
            expected[2].reshape(1, result_width) - reference_scores[:, parents[0].to(torch.int64)]
        )
        workspace.before_decision(lease, decision.beam_scores, expected_device_step=step)
        result = advance_fused_decision(
            session,
            candidate_tokens.to(device),
            candidate_scores.to(device),
            expected_device_step=step,
            is_last_step=final,
            result_width=result_width,
        )
        assert (
            workspace.record_decision(
                lease,
                result.parent_ids,
                decision.beam_scores,
                torch.isfinite(result.scores),
                expected_device_step=step,
            )
            is final
        )
        assert result.device_step == step + 1
        assert decision.worker_decode_steps == step + 1
        assert session.decode_step == step - 1  # No hidden session/KV advance.
        assert decision.sequence.data_ptr() == sequence_ptr
        assert decision.beam_scores.data_ptr() == scores_ptr
        assert result.sequence.data_ptr() == sequence_ptr
        assert result.scores.data_ptr() == scores_ptr
        torch.testing.assert_close(
            result.token_ids.cpu(), expected[0].reshape(1, result_width), rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.parent_ids.cpu(), expected[1].reshape(1, result_width), rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.scores.cpu(), expected[2].reshape(1, result_width), rtol=0, atol=1e-5
        )
        # The unused capacity tail is not a generated sequence field.
        torch.testing.assert_close(
            result.sequence[:, :, : step + 1].cpu(),
            expected[-1][:, :, : step + 1],
            rtol=0,
            atol=0,
        )
        if final:
            assert result.reorder_parents is None
            assert result.reorder_prefix is None
        else:
            assert result.reorder_parents is result.parent_ids
            torch.testing.assert_close(result.reorder_prefix.cpu(), expected[3], rtol=0, atol=0)
            torch.testing.assert_close(decision.group_token_num.cpu(), expected[3], rtol=0, atol=0)
            # This is only a decision probe. No KV is allocated or reordered;
            # simulate the caller's cursor update for the next decision.
            session.decode_step += 1
            reference_scores = expected[2]
            reference_sequence = expected[-1]

    pack = workspace.pack_final(lease, decision.sequence, decision.beam_scores)
    assert pack.staging.data_ptr() == staging_ptr
    assert pack.shape == (width // 2, generation_steps)
    # Test-only D2H: neither this probe nor pack_final is a transfer benchmark.
    host = pack.staging.cpu()
    assert host[0].item() == 0
    unpacked = unpack_gr_final_output(
        {
            "buffer_format": pack.buffer_format,
            "shape": pack.shape,
            "buffer": host[1:].numpy().tobytes(),
        }
    )
    assert unpacked is not None
    sequences, parents, deltas, final_scores = unpacked
    expected_parent_paths = []
    expected_delta_paths = []
    for row in range(width // 2):
        parent_path = [0] * generation_steps
        delta_path = [0.0] * generation_steps
        ancestor = row
        for step in range(generation_steps - 1, -1, -1):
            parent_path[step] = int(reference_parents[step][0, ancestor])
            delta_path[step] = float(reference_deltas[step][0, ancestor])
            ancestor = parent_path[step]
        expected_parent_paths.append(parent_path)
        expected_delta_paths.append(delta_path)
    assert sequences == expected[-1][0, :, :generation_steps].tolist()
    assert parents == expected_parent_paths
    torch.testing.assert_close(
        torch.tensor(deltas), torch.tensor(expected_delta_paths), rtol=0, atol=2e-5
    )
    torch.testing.assert_close(
        torch.tensor(final_scores), expected[2].reshape(-1), rtol=0, atol=1e-5
    )
    assert workspace.release(lease)
    assert not workspace.release(lease)
