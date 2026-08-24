# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_gr.constraints.backend import INVALID_LOGIT, ConstraintOutput
from vllm_gr.v1.worker.constraint_hook import (
    build_constrained_sampler_output,
    build_sequence_group_for_step,
    collect_constrained_rows,
    run_constrained_partition,
    validate_constraint_step,
)


def _decode(step: int = 1, width: int = 2) -> dict[str, object]:
    return {
        "is_constrained": True,
        "constraint_step": step,
        "beam_width": width,
        "beam_suffix": [[1, 3], [2, 4]],
        "max_decode_steps": 3,
    }


def test_constraint_step_accepts_only_zero_one_two() -> None:
    assert [validate_constraint_step(value) for value in (0, 1, 2, np.int64(2))] == [
        0,
        1,
        2,
        2,
    ]
    for value in (-1, 3, True, "1", None):
        with pytest.raises(RuntimeError, match="constraint_step"):
            validate_constraint_step(value)


def test_collect_rows_uses_sampler_row_cursor() -> None:
    configs = [("plain", 0, 1, []), ("constrained", 0, 2, [])]
    rows = collect_constrained_rows(configs, {"constrained": _decode()})
    assert [(req_id, start, width) for req_id, start, width, _ in rows] == [("constrained", 1, 2)]


def test_build_prefix_tensor_is_flat_rows_by_steps() -> None:
    rows = [("req", 0, 2, _decode(step=2))]
    prefixes = build_sequence_group_for_step(rows, torch.device("cpu"), 2)
    assert prefixes.dtype == torch.int32
    assert prefixes.shape == (2, 3)
    assert prefixes[:, :2].tolist() == [[1, 3], [2, 4]]


def test_prefill_prefix_has_one_row_per_logits_row() -> None:
    rows = [("a", 0, 1, {**_decode(step=0), "beam_suffix": []})]
    prefixes = build_sequence_group_for_step(rows, torch.device("cpu"), 0)
    assert prefixes.shape == (1, 3)


def test_prefill_rejects_more_than_one_logits_row() -> None:
    rows = [("a", 0, 2, {**_decode(step=0), "beam_suffix": []})]
    with pytest.raises(RuntimeError, match="expected exactly 1"):
        build_sequence_group_for_step(rows, torch.device("cpu"), 0)


def test_partition_calls_preloaded_backend() -> None:
    observed = []

    class Backend:
        def select(self, logits, metadata):
            observed.append(metadata)
            return ConstraintOutput(
                token_ids=torch.tensor([[3, 4], [4, 3]], dtype=torch.int32),
                logprobs=torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
                valid_mask=torch.ones((2, 2), dtype=torch.bool),
            )

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            sampling_metadata=SimpleNamespace(temperature=torch.tensor([0.0, 0.5]))
        ),
    )
    rows = [("req", 0, 2, _decode(step=1))]
    tokens, probs, valid_mask, global_rows = run_constrained_partition(
        runner,
        torch.zeros((2, 8)),
        rows,
        2,
        Backend(),
    )
    assert tokens.tolist() == [[3, 4], [4, 3]]
    assert probs.shape == (2, 2)
    assert valid_mask.tolist() == [[True, True], [True, True]]
    assert global_rows == [0, 1]
    assert observed[0].current_step == 1
    assert observed[0].prefix_token_ids.shape == (2, 3)
    assert observed[0].prefix_token_ids.dtype == torch.int32
    assert observed[0].prefix_token_ids.is_contiguous()
    assert observed[0].temperatures.tolist() == [1.0, 0.5]
    assert observed[0].temperatures.is_contiguous()


def test_partition_merges_requests_with_the_same_step() -> None:
    calls: list[torch.Tensor] = []

    class Backend:
        def select(self, logits, metadata):
            calls.append(logits)
            row_count = logits.shape[0]
            return ConstraintOutput(
                token_ids=torch.ones((row_count, 2), dtype=torch.int32),
                logprobs=torch.zeros((row_count, 2), dtype=torch.float32),
                valid_mask=torch.ones((row_count, 2), dtype=torch.bool),
            )

    logits = torch.zeros((4, 8))
    rows = [
        ("a", 0, 2, _decode(step=1)),
        ("b", 2, 2, _decode(step=1)),
    ]
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(sampling_metadata=SimpleNamespace(temperature=torch.ones(4)))
    )

    run_constrained_partition(runner, logits, rows, 2, Backend())

    assert len(calls) == 1
    assert calls[0].shape == logits.shape
    assert calls[0].data_ptr() == logits.data_ptr()


def test_partition_restores_global_order_across_step_groups() -> None:
    calls: list[int] = []

    class Backend:
        def select(self, logits, metadata):
            calls.append(metadata.current_step)
            row_count = logits.shape[0]
            token = metadata.current_step + 1
            return ConstraintOutput(
                token_ids=torch.full((row_count, 2), token, dtype=torch.int32),
                logprobs=torch.zeros((row_count, 2), dtype=torch.float32),
                valid_mask=torch.ones((row_count, 2), dtype=torch.bool),
            )

    rows = [
        ("prefill", 0, 1, {**_decode(step=0), "beam_suffix": []}),
        ("decode", 1, 2, _decode(step=1)),
    ]
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(sampling_metadata=SimpleNamespace(temperature=torch.ones(3)))
    )

    tokens, _, _, global_rows = run_constrained_partition(
        runner,
        torch.zeros((3, 8)),
        rows,
        2,
        Backend(),
    )

    assert calls == [0, 1]
    assert global_rows == [0, 1, 2]
    assert tokens[:, 0].tolist() == [1, 2, 2]


def test_direct_sampler_output_uses_fixed_width_backend_output() -> None:
    sampler = build_constrained_sampler_output(
        torch.tensor([[3, 4]], dtype=torch.int32),
        torch.tensor([[-0.1, -0.2]], dtype=torch.float32),
        torch.tensor([[True, False]], dtype=torch.bool),
        2,
    )
    assert sampler.sampled_token_ids.tolist() == [[3]]
    assert sampler.logprobs_tensors.logprob_token_ids.tolist() == [[3, 4]]
    assert sampler.logprobs_tensors.logprobs.shape == (1, 2)
    assert (
        sampler.logprobs_tensors.logprobs[0, 1].item()
        == torch.tensor(INVALID_LOGIT, dtype=torch.float32).item()
    )


def test_direct_prefill_output_expands_through_beam_remux() -> None:
    from vllm_gr.v1.worker.model_runner_common import _beam_collapse_bookkeeping

    sampler = build_constrained_sampler_output(
        torch.tensor([[3, 4]], dtype=torch.int32),
        torch.tensor([[-0.1, -0.2]], dtype=torch.float32),
        torch.ones((1, 2), dtype=torch.bool),
        2,
    )
    runner = SimpleNamespace()

    _beam_collapse_bookkeeping(
        runner,
        SimpleNamespace(),
        [("req", 0, 1, [])],
        {"req": {"beam_width": 2, "is_beam_decode": False}},
        sampler,
    )

    assert sampler.sampled_token_ids.tolist() == [[3]]
    assert sampler.logprobs_tensors.logprobs.shape == (1, 4)
    assert sampler.logprobs_tensors.logprob_token_ids.shape == (1, 4)
    assert sampler.logprobs_tensors.logprobs[0, :2].tolist() == pytest.approx([-0.1, -0.2])
    assert torch.isneginf(sampler.logprobs_tensors.logprobs[0, 2:]).all()
