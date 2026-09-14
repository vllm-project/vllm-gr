# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_gr.ops.cuda.beam_state import FINISH_LENGTH, FINISH_STOP
from vllm_gr.v1.beam.batch_contracts import BeamOutputOptions
from vllm_gr.v1.worker.beam_final_output import _build_result, _PayloadLayout


def _compact_payload() -> tuple[_PayloadLayout, torch.Tensor]:
    layout = _PayloadLayout.create(width=2, steps=3)
    payload = torch.zeros(layout.size, dtype=torch.int32)
    payload[0] = 2
    first = payload[1 : 1 + layout.row_stride]
    first[:3] = torch.tensor([99, 0, 0])
    first[3:6] = torch.tensor([0, 0, 0])
    first[6:7].view(torch.float32).fill_(-0.1)
    first[7:] = torch.tensor([1, FINISH_STOP, 99])
    second = payload[1 + layout.row_stride : 1 + 2 * layout.row_stride]
    second[:3] = torch.tensor([10, 20, 32])
    second[3:6] = torch.tensor([0, 0, 0])
    second[6:7].view(torch.float32).fill_(-0.2)
    second[7:] = torch.tensor([3, FINISH_LENGTH, -1])
    return layout, payload


def test_wraps_gpu_ranked_stop_and_length_results_without_reranking() -> None:
    layout, payload = _compact_payload()
    result = _build_result(
        payload,
        layout,
        session_id="batch",
        request_id="request",
        item_index=0,
        options=BeamOutputOptions(2, False, True, True),
    )

    sequences = result.output.items[0].sequences
    assert sequences[0].token_ids == ()
    assert sequences[0].parent_beam_indices == (0,)
    assert sequences[0].finish_reason == "stop"
    assert sequences[0].stop_token_id == 99
    assert sequences[1].token_ids == (10, 20, 32)
    assert sequences[1].parent_beam_indices == (0, 0, 0)
    assert sequences[1].finish_reason == "length"
    assert sequences[1].cumulative_logprob == pytest.approx(-0.2)


def test_applies_output_options_and_empty_result_to_compact_rows() -> None:
    layout, payload = _compact_payload()
    result = _build_result(
        payload,
        layout,
        session_id="batch",
        request_id="request",
        item_index=0,
        options=BeamOutputOptions(1, True, False, False),
    )
    sequence = result.output.items[0].sequences[0]
    assert sequence.token_ids == (99,)
    assert sequence.cumulative_logprob is None
    assert sequence.parent_beam_indices == ()

    payload[0] = 0
    empty = _build_result(
        payload,
        layout,
        session_id="batch",
        request_id="request",
        item_index=0,
        options=BeamOutputOptions(1, True, True, True),
    )
    assert empty.output is None
    assert empty.failure.code == "no_valid_beams"
