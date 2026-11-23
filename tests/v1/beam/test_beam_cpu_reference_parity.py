# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch

from vllm_gr.ops.onerec_decode import (
    _beam_search_group_torch,
    _beam_search_rec_final_select_torch,
)
from vllm_gr.v1.beam.cpu_reference import (
    BeamItemCandidates,
    BeamReferenceBeam,
    BeamReferenceItemState,
    BeamReferenceState,
    normalize_constraint_candidates,
    select_final_sequences,
    select_next_beams,
)


def _case() -> tuple[
    BeamReferenceState,
    tuple[BeamItemCandidates, ...],
    npt.NDArray[np.int32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.float32],
]:
    batch_size, beam_width = 2, 3
    sequences = np.asarray([[10, 11, 12], [20, 21, 22]], dtype=np.int32)
    parent_scores = np.asarray(
        [[-0.10, -0.20, -0.30], [-0.05, -0.25, -0.40]],
        dtype=np.float32,
    )
    tokens = np.asarray(
        [
            [[100, 101, 102], [110, 111, 112], [120, 121, 122]],
            [[200, 201, 202], [210, 211, 212], [220, 221, 222]],
        ],
        dtype=np.int32,
    )
    logprobs = np.asarray(
        [
            [[-0.50, -0.10, -0.90], [-0.05, -0.40, -0.60], [-0.20, -0.30, -0.70]],
            [[-0.60, -0.15, -0.80], [-0.10, -0.45, -0.55], [-0.05, -0.35, -0.75]],
        ],
        dtype=np.float32,
    )
    state = BeamReferenceState(
        beam_width=beam_width,
        num_generated_tokens=1,
        items=tuple(
            BeamReferenceItemState(
                item_index=item,
                beams=tuple(
                    BeamReferenceBeam(
                        is_active=True,
                        token_ids=(int(sequences[item, beam]),),
                        cumulative_logprob=parent_scores[item, beam],
                        parent_beam_indices=(0,),
                    )
                    for beam in range(beam_width)
                ),
                num_active_beams=beam_width,
                completed_beams=(),
            )
            for item in range(batch_size)
        ),
    )
    candidates = tuple(
        normalize_constraint_candidates(
            item_index=item,
            parent_beam_indices=np.arange(beam_width, dtype=np.int64),
            token_ids=tokens[item],
            logprobs=logprobs[item],
            valid_mask=np.ones((beam_width, beam_width), dtype=np.bool_),
        )
        for item in range(batch_size)
    )
    return state, candidates, sequences, parent_scores, tokens, logprobs


def test_next_beams_matches_onerec_selection_and_parent_grouping() -> None:
    state, candidates, sequences, parent_scores, tokens, logprobs = _case()
    batch_size, beam_width = parent_scores.shape
    reference = select_next_beams(state, candidates)
    sequence_buffer = torch.zeros((batch_size, beam_width, 2), dtype=torch.int32)
    sequence_buffer[:, :, 0] = torch.from_numpy(sequences)
    out_tokens, out_parents, out_scores, _, out_sequences = _beam_search_group_torch(
        torch.from_numpy(parent_scores.reshape(-1, 1)),
        torch.from_numpy(tokens.reshape(-1, beam_width)),
        torch.from_numpy(logprobs.reshape(-1, beam_width)),
        sequence_buffer,
        1,
        beam_width,
    )

    for item_index, item in enumerate(reference.items):
        assert [beam.token_ids[-1] for beam in item.beams] == out_tokens[item_index].tolist()
        assert [beam.parent_beam_indices[-1] for beam in item.beams] == (
            out_parents[item_index] - item_index * beam_width
        ).tolist()
        np.testing.assert_array_equal(
            [beam.cumulative_logprob for beam in item.beams],
            out_scores[item_index].numpy(),
        )
        assert [beam.token_ids for beam in item.beams] == [
            tuple(row) for row in out_sequences[item_index].tolist()
        ]


def test_final_sequences_matches_onerec_final_select() -> None:
    state, candidates, sequences, parent_scores, tokens, logprobs = _case()
    batch_size, beam_width = parent_scores.shape
    num_return_sequences = 2
    reference = select_final_sequences(
        state,
        candidates,
        beam_width=beam_width,
        num_return_sequences=num_return_sequences,
    )
    sequence_buffer = torch.zeros((batch_size, beam_width, 2), dtype=torch.int32)
    sequence_buffer[:, :, 0] = torch.from_numpy(sequences)
    out_tokens, out_parents, out_scores, out_sequences = _beam_search_rec_final_select_torch(
        torch.from_numpy(parent_scores.reshape(-1, 1)),
        torch.from_numpy(tokens.reshape(-1, beam_width)),
        torch.from_numpy(logprobs.reshape(-1, beam_width)),
        sequence_buffer,
        1,
        num_return_sequences,
    )
    out_tokens = out_tokens.reshape(batch_size, num_return_sequences)
    out_parents = out_parents.reshape(batch_size, num_return_sequences)
    out_scores = out_scores.reshape(batch_size, num_return_sequences)

    for item_index, item in enumerate(reference.items):
        assert [sequence.token_ids[-1] for sequence in item.sequences] == out_tokens[
            item_index
        ].tolist()
        assert [sequence.parent_beam_indices[-1] for sequence in item.sequences] == (
            out_parents[item_index] - item_index * beam_width
        ).tolist()
        np.testing.assert_array_equal(
            [sequence.cumulative_logprob for sequence in item.sequences],
            out_scores[item_index].numpy(),
        )
        assert [sequence.token_ids for sequence in item.sequences] == [
            tuple(row) for row in out_sequences[item_index].tolist()
        ]


def test_exact_tie_parity_uses_lower_logical_flat_index() -> None:
    beam_width = 2
    state = BeamReferenceState(
        beam_width=beam_width,
        num_generated_tokens=1,
        items=(
            BeamReferenceItemState(
                item_index=0,
                beams=(
                    BeamReferenceBeam(True, (10,), np.float32(0), (0,)),
                    BeamReferenceBeam(True, (11,), np.float32(0), (0,)),
                ),
                num_active_beams=2,
                completed_beams=(),
            ),
        ),
    )
    tokens = np.asarray([[20, 21], [22, 23]], dtype=np.int32)
    logprobs: npt.NDArray[np.float32] = np.zeros((2, 2), dtype=np.float32)
    candidates = normalize_constraint_candidates(
        item_index=0,
        parent_beam_indices=np.asarray([0, 1]),
        token_ids=tokens,
        logprobs=logprobs,
        valid_mask=np.ones_like(tokens, dtype=np.bool_),
    )
    reference = select_next_beams(state, (candidates,))
    out_tokens, out_parents, _, _, _ = _beam_search_group_torch(
        torch.zeros((2, 1), dtype=torch.float32),
        torch.from_numpy(tokens),
        torch.from_numpy(logprobs),
        torch.tensor([[[10, 0], [11, 0]]], dtype=torch.int32),
        1,
        beam_width,
    )
    assert [beam.token_ids[-1] for beam in reference.items[0].beams] == [20, 21]
    assert [beam.token_ids[-1] for beam in reference.items[0].beams] == out_tokens[0].tolist()
    assert [beam.parent_beam_indices[-1] for beam in reference.items[0].beams] == (
        out_parents[0].tolist()
    )


pytestmark = pytest.mark.cpu_test
