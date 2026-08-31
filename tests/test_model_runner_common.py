# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for device-neutral ModelRunner Beam helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.worker import model_runner_common as common


def test_position_helpers_support_buffer_forms_and_partial_beam() -> None:
    device = torch.full((8,), -1, dtype=torch.long)
    host = torch.full((8,), -1, dtype=torch.long)

    buffered_runner = SimpleNamespace(positions=SimpleNamespace(gpu=device, cpu=host))
    assert common._get_positions_buffers(buffered_runner) == (device, host)

    plain_runner = SimpleNamespace(positions=device, _positions_cpu_buf=host)
    assert common._get_positions_buffers(plain_runner) == (device, host)

    logits_indices: list[int] = []
    result = common._process_beam_decode_request(
        {
            "prefix_len": 10,
            "beam_width": 2,
            "beam_decode_steps": 4,
            "cache_len": 12,
        },
        seq_len=3,
        token_offset=1,
        gpu_pos=device,
        cpu_pos=host,
        new_logits_indices=logits_indices,
    )

    assert result == (2, 1, 4, [0])
    assert logits_indices == [2]
    assert device.tolist() == [-1, 12, 13, 10, -1, -1, -1, -1]
    assert torch.equal(device, host)

    common._write_position_segments(0, [], [], device, host)
    assert torch.equal(device, host)


def test_mixed_batch_remap_installs_functional_proxy() -> None:
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([10, 20, 30, 40]),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    input_batch = SimpleNamespace(
        req_ids=["ordinary", "beam", None, "prefill"],
        logprob_token_ids={"stale": [1]},
        num_computed_tokens_cpu=[5, 0, 0, 2],
        num_prompt_tokens=[5, 0, 0, 4],
        num_tokens_no_spec=[10, 20, 30, 40],
        num_tokens=[11, 21, 31, 41],
        token_ids_cpu=torch.arange(16).view(4, 4),
        sampling_metadata=sampling_metadata,
        passthrough="original",
    )
    positions = SimpleNamespace(
        gpu=torch.full((12,), -1, dtype=torch.long),
        cpu=torch.full((12,), -1, dtype=torch.long),
    )
    runner = SimpleNamespace(input_batch=input_batch, positions=positions)
    scheduler_output = SimpleNamespace(
        beam_data={
            "beam": {
                "is_beam_decode": True,
                "prefix_len": 4,
                "beam_width": 2,
                "beam_decode_steps": 2,
                "cache_len": 4,
            }
        }
    )

    logits_indices = common._beam_remap_inputs(runner, scheduler_output, [2, 4, 1, 3])

    assert logits_indices == [3, 5, 9]
    assert positions.gpu[2:6].tolist() == [4, 5, 4, 5]
    assert torch.equal(positions.gpu, positions.cpu)
    assert scheduler_output._gr_request_row_configs == [
        ("ordinary", 0, 0, []),
        ("beam", 1, 2, [0, 1]),
        ("prefill", 4, 1, []),
    ]

    proxy = runner.input_batch
    assert isinstance(proxy, common.InputBatchProxy)
    assert proxy.req_ids.copy() == ["ordinary", "beam", "beam", None, "prefill"]
    assert proxy.sampling_metadata.temperature.tolist() == [20, 20, 40]
    assert proxy.num_tokens_no_spec[2] == 20
    proxy.num_tokens[2] = 99
    assert input_batch.num_tokens[1] == 99
    proxy.is_token_ids[4, 0] = 77
    assert input_batch.token_ids_cpu[3, 0].item() == 77
    proxy.passthrough = "updated"
    assert input_batch.passthrough == "updated"
    assert input_batch.logprob_token_ids == {}


def test_logprob_remux_preserves_flattening_and_placeholder_rows() -> None:
    st_ids = torch.tensor([[10], [20], [30]])
    logprobs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    token_ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
    ranks = torch.tensor([7, 8, 9])

    final_st, final_lps, final_ids, final_ranks = common._remux_logprobs_tensors(
        [
            ("ordinary", 0, 1, []),
            ("beam", 1, 2, [0, 1]),
            ("prefill", 3, 0, []),
        ],
        max_chained_dim=4,
        target_K=2,
        orig_lps=logprobs,
        orig_lp_ids=token_ids,
        orig_ranks=ranks,
        st_ids=st_ids,
    )

    assert final_st.tolist() == [[10], [20], [-1]]
    assert final_lps[:2].tolist() == [
        [1.0, 2.0, float("-inf"), float("-inf")],
        [3.0, 4.0, 5.0, 6.0],
    ]
    assert torch.isneginf(final_lps[2]).all()
    assert final_ids.tolist() == [[1, 2, 0, 0], [3, 4, 5, 6], [0, 0, 0, 0]]
    assert final_ranks.tolist() == [[7, 0, 0, 0], [8, 9, 0, 0], [0, 0, 0, 0]]


def test_beam_collapse_accumulates_partial_rows_and_clears_cache() -> None:
    runner = SimpleNamespace()
    beam_data = {"beam": {"is_beam_decode": True, "beam_width": 2}}

    def sampler(values: list[float], token_ids: list[int], rank: int) -> SimpleNamespace:
        return SimpleNamespace(
            sampled_token_ids=torch.tensor([[token_ids[0]]]),
            logprobs_tensors=SimpleNamespace(
                logprobs=torch.tensor([values]),
                logprob_token_ids=torch.tensor([token_ids]),
                selected_token_ranks=torch.tensor([rank]),
            ),
        )

    first = sampler([0.1, 0.2], [1, 2], 3)
    common._beam_collapse_bookkeeping(
        runner,
        object(),
        [("beam", 0, 1, [0])],
        beam_data,
        first,
    )
    assert runner._beam_logprobs_cache["beam"]["completed_beams"] == {0}
    assert first.logprobs_tensors.logprobs.shape == (1, 4)

    second = sampler([0.3, 0.4], [3, 4], 5)
    collapsed = common._beam_collapse_bookkeeping(
        runner,
        object(),
        [("beam", 0, 1, [1])],
        beam_data,
        second,
    )

    assert collapsed == (["beam"], {"beam": 0}, 1)
    assert torch.allclose(
        second.logprobs_tensors.logprobs,
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
    )
    assert second.logprobs_tensors.logprob_token_ids.tolist() == [[1, 2, 3, 4]]
    assert "beam" not in runner._beam_logprobs_cache


def test_beam_collapse_cache_completion_copes_with_batch_padding() -> None:
    """A narrower beam request must copy its own footprint, not the padded row.

    When requests with different beam widths share a batch, _remux_logprobs_tensors
    pads every row to the batch-wide max_chained_dim. Completing the logprobs cache
    for the narrower request must write only log_probs_num * K elements, leaving the
    -inf tail untouched (previously this raised a size-mismatch RuntimeError).
    """
    runner = SimpleNamespace()
    beam_data = {
        "narrow": {"is_beam_decode": True, "beam_width": 2},
        "wide": {"is_beam_decode": True, "beam_width": 3},
    }
    row_configs = [("narrow", 0, 1, [0]), ("wide", 1, 3, [0, 1, 2])]

    def sampler(
        narrow_values: list[float], narrow_ids: list[int], narrow_rank: int
    ) -> SimpleNamespace:
        # orig GPU rows: 1 for narrow + 3 for wide; st_ids: one row per request.
        return SimpleNamespace(
            sampled_token_ids=torch.tensor([[narrow_ids[0]], [10], [20], [30]]),
            logprobs_tensors=SimpleNamespace(
                logprobs=torch.tensor([narrow_values, [9.0, 9.0], [9.0, 9.0], [9.0, 9.0]]),
                logprob_token_ids=torch.tensor([narrow_ids, [11, 12], [21, 22], [31, 32]]),
                selected_token_ranks=torch.tensor([narrow_rank, 7, 8, 9]),
            ),
        )

    first = sampler([0.1, 0.2], [1, 2], 3)
    common._beam_collapse_bookkeeping(
        runner,
        object(),
        row_configs,
        beam_data,
        first,
    )
    assert runner._beam_logprobs_cache["narrow"]["completed_beams"] == {0}
    assert first.logprobs_tensors.logprobs.shape == (2, 6)

    second = sampler([0.3, 0.4], [3, 4], 5)
    collapsed = common._beam_collapse_bookkeeping(
        runner,
        object(),
        [("narrow", 0, 1, [1]), ("wide", 1, 3, [0, 1, 2])],
        beam_data,
        second,
    )

    assert collapsed == (["narrow", "wide"], {"narrow": 0, "wide": 1}, 2)
    assert torch.allclose(
        second.logprobs_tensors.logprobs[0],
        torch.tensor([0.1, 0.2, 0.3, 0.4, float("-inf"), float("-inf")]),
    )
    assert second.logprobs_tensors.logprob_token_ids[0].tolist() == [1, 2, 3, 4, 0, 0]
    assert "narrow" not in runner._beam_logprobs_cache


def test_execute_wrapper_is_idempotent_and_restores_outer_context() -> None:
    class ExpectedError(RuntimeError):
        pass

    scheduler_output = SimpleNamespace(beam_data={"beam": {"is_beam_decode": True}})
    runner = SimpleNamespace()
    should_raise = False

    def execute_model(self: Any, output: Any) -> str:
        assert output is scheduler_output
        assert BEAM_DATA_VAR.get() is self._current_beam_info
        assert self._current_beam_info == {
            "beam_data": scheduler_output.beam_data,
            "req_ids": [],
        }
        if should_raise:
            raise ExpectedError
        return "result"

    wrapped = common.make_patched_execute_model(execute_model)
    assert common.make_patched_execute_model(wrapped) is wrapped

    outer = {"level": "outer"}
    token = BEAM_DATA_VAR.set(outer)
    try:
        assert wrapped(runner, scheduler_output) == "result"
        assert BEAM_DATA_VAR.get() is outer

        should_raise = True
        with pytest.raises(ExpectedError):
            wrapped(runner, scheduler_output)
        assert BEAM_DATA_VAR.get() is outer
    finally:
        BEAM_DATA_VAR.reset(token)

    assert runner._current_beam_data == {}
    assert runner._current_beam_info is None
