# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the real wide-candidate Triton indexing/EOS kernels on a CPU."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_wide_candidate_kernels_in_cpu_interpreter() -> None:
    # Triton language helpers choose their interpreter implementation at import
    # time. Isolate that setting from the ordinary CPU/CUDA pytest process.
    result = subprocess.run(
        [sys.executable, __file__],
        env={
            **os.environ,
            "TRITON_INTERPRET": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")]
            ),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _exercise_kernels() -> None:
    import torch

    from vllm_gr.ops.cuda.beam_selection import _group_beam_outputs_kernel
    from vllm_gr.ops.cuda.beam_state import (
        FINISH_STOP,
        BeamStateScratch,
        _append_completed_kernel,
        _prepare_candidates_kernel,
        _rank_final_beams_kernel,
        _validate_step_kernel,
    )
    from vllm_gr.v1.beam.beam_attention_context import BeamDecisionBuffers
    from vllm_gr.v1.beam.beam_state import BeamSearchState

    # Request-major rows remain B*W=4. Candidate indices use K=4, whereas
    # returned parents use the global B*W row convention expected by KV reorder.
    tokens = torch.arange(16, dtype=torch.int32).reshape(4, 4)
    selected = torch.tensor([[4, 1], [6, 0]], dtype=torch.int32)
    scores = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    outputs = [
        torch.empty((2, 2), dtype=dtype)
        for dtype in (torch.int32, torch.int32, torch.float32, torch.int32)
    ]
    _group_beam_outputs_kernel[(2,)](
        scores, selected, tokens, *outputs, 2, 4, 2, WRITE_FLAT_INDICES=True, BLOCK=2
    )
    assert outputs[0].tolist() == [[1, 4], [8, 14]]
    assert outputs[1].tolist() == [[0, 1], [2, 3]]
    torch.testing.assert_close(outputs[2], torch.tensor([[0.2, 0.1], [0.4, 0.3]]))
    assert outputs[3].tolist() == [[1, 4], [0, 6]]

    state = BeamSearchState(
        BeamDecisionBuffers(1, 2, 3, torch.device("cpu"), torch.float16),
        beam_width=2,
        candidate_width=4,
        max_new_tokens=3,
        stop_token_ids=(99,),
    )
    state.beam_scores[0].copy_(torch.tensor([-0.01, -0.02]))
    state.active_mask.fill_(True)
    state.is_initialized.fill_(True)
    state.num_generated_tokens.fill_(1)
    state.num_completed.fill_(1)
    state.completed_token_ids[0, 0] = 99
    state.completed_scores[0, 0] = -0.03
    state.completed_lengths[0, 0] = 1
    state.completed_finish_reasons[0, 0] = FINISH_STOP
    scratch = BeamStateScratch.allocate(1, 2, torch.device("cpu"), candidate_width=4)
    candidates = torch.tensor([[20, 99, 21, 22], [30, 31, 99, 32]], dtype=torch.int32)
    logprobs = torch.tensor([[-2, -0.8, -3, -4], [-0.1, -0.2, -0.9, -0.3]])
    _prepare_candidates_kernel[(2,)](
        candidates,
        logprobs,
        torch.ones_like(candidates, dtype=torch.bool),
        state.beam_scores,
        state.active_mask,
        state.num_generated_tokens,
        state.max_new_tokens_tensor,
        state.stop_token_ids,
        state.num_stop_tokens,
        state.ignore_stop_tokens,
        scratch.parent_scores,
        scratch.selection_mask,
        scratch.terminal_ranks,
        scratch.finish_reasons,
        scratch.cumulative_scores,
        scratch.row_counts,
        scratch.row_errors,
        scratch.row_valid_counts,
        WIDTH=4,
        BLOCK=4,
        PREFILL=False,
    )
    assert scratch.selection_mask.tolist() == [
        [[True, False, True, True], [True, True, False, True]]
    ]
    assert scratch.row_counts.tolist() == [[1, 1]]
    _validate_step_kernel[(1,)](
        state.is_initialized,
        state.is_finished,
        state.error_code,
        state.num_generated_tokens,
        state.num_decode_steps,
        state.max_new_tokens_tensor,
        state.num_completed,
        scratch.row_counts,
        scratch.row_errors,
        scratch.row_valid_counts,
        scratch.row_offsets,
        scratch.execute,
        scratch.reorder,
        scratch.next_completed,
        ROWS=2,
        BLOCK=2,
        CAPACITY=state.completed_capacity,
        KV_CAPACITY=3,
        PREFILL=False,
    )
    _append_completed_kernel[(2,)](
        candidates,
        scratch.terminal_ranks,
        scratch.finish_reasons,
        scratch.cumulative_scores,
        scratch.row_offsets,
        state.num_generated_tokens,
        scratch.execute,
        state.completed_token_ids,
        state.completed_scores,
        state.completed_parent_ids,
        state.completed_lengths,
        state.completed_finish_reasons,
        state.completed_tie_break_indices,
        WIDTH=4,
        BLOCK=4,
        PREFILL=False,
    )
    assert state.error_code.item() == 0
    assert scratch.next_completed.item() == 3
    assert state.completed_token_ids[0, :3].tolist() == [99, 99, 99]
    torch.testing.assert_close(state.completed_scores[0, :3], torch.tensor([-0.03, -0.81, -0.92]))
    assert state.completed_parent_ids[0, 1:3].tolist() == [0, 1]
    assert state.completed_lengths[0, :3].tolist() == [1, 2, 2]
    assert state.completed_tie_break_indices[0, 1:3].tolist() == [1, 6]
    ranked = torch.empty(3)
    _rank_final_beams_kernel[(1,)](
        state.completed_scores[0],
        state.completed_lengths[0],
        state.completed_finish_reasons[0],
        ranked,
        CAPACITY=3,
        PROMPT_LENGTH=5,
        MAX_TOKENS=3,
        LENGTH_PENALTY=1.0,
        BLOCK=4,
    )
    # EOS is excluded from the total decoder length, including prompt tokens.
    torch.testing.assert_close(ranked, torch.tensor([-0.03 * 8 / 5, -0.81 * 8 / 6, -0.92 * 8 / 6]))

    from tests.test_gr_input_binding import exercise_mixed_batch_kv

    for batch_size, finished_index in ((2, 0), (2, 1), (4, 0), (4, 2)):
        for step in (0, 1):
            exercise_mixed_batch_kv(
                batch_size, finished_index, step, device="cpu", graph_mode=False
            )


if __name__ == "__main__":
    _exercise_kernels()
