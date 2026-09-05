# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU composition of the persistent sampler and real six-table NPU Backend.

Only the fused NPU operator is explicitly replaced with the PyTorch reference.
These tests prove table dispatch and sampler ABI composition, not NPU kernel,
graph, transfer, or stream correctness. CPU reference inspection is intentional.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintTableBackendEnum
from vllm_gr.constraints.backend import INVALID_LOGIT, ConstraintBackendInfo, ConstraintMetadata
from vllm_gr.constraints.npu_backend import NPUConstraintBackend
from vllm_gr.constraints.table import CONSTRAINT_TABLE_TENSOR_NAMES, validate_constraint_table
from vllm_gr.ops.rec_constrained_topk import rec_constrained_topk_torch
from vllm_gr.v1.worker.npu_gr_sampler import (
    NPUConstrainedCandidates,
    NPUPersistentConstraintSampler,
    NPU_SAMPLER_STATUS_EMPTY,
    NPU_SAMPLER_STATUS_OK,
)


class _CPUReferenceOp:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.fail_step: int | None = None
        self.error = RuntimeError("injected fused operator failure")

    def __call__(self, *args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(
            tuple(value.clone() if torch.is_tensor(value) else value for value in args)
        )
        if args[9] == self.fail_step:
            raise self.error
        tokens, logprobs = rec_constrained_topk_torch(*args)
        return tokens.to(torch.int32), logprobs


def _composition(
    temperature: float = 1.0, *, width: int = 4
) -> tuple[NPUPersistentConstraintSampler, NPUConstraintBackend, _CPUReferenceOp]:
    table = make_table()
    validate_constraint_table(table)
    tensors = {
        name: torch.from_numpy(np.array(getattr(table, name), copy=True))
        for name in CONSTRAINT_TABLE_TENSOR_NAMES
    }
    op = _CPUReferenceOp()
    backend = NPUConstraintBackend(
        **tensors,
        info=ConstraintBackendInfo(
            table_metadata=table.metadata,
            backend=ConstraintTableBackendEnum.NPU,
            max_rows=width,
            max_top_k=width,
            host_memory_bytes=0,
            device_memory_bytes=table.total_bytes,
        ),
        op=op,
    )
    backend._select_for_readiness(  # noqa: SLF001
        torch.zeros(1, 6),
        ConstraintMetadata(
            prefix_token_ids=torch.zeros((1, 3), dtype=torch.int32),
            temperatures=torch.ones(1),
            current_step=0,
            top_k=width,
        ),
    )
    backend._mark_ready()  # noqa: SLF001
    op.calls.clear()
    sampler = NPUPersistentConstraintSampler(
        "composition",
        width,
        3,
        temperature,
        torch.device("cpu"),
        backend,
        cpu_test_backend=True,
    )
    return sampler, backend, op


def _check_candidates(
    result: NPUConstrainedCandidates,
    logits: torch.Tensor,
    allowed_rows: list[list[int]],
    temperature: float,
) -> None:
    raw = result.constraint_output
    wire = result.sampler_output.logprobs_tensors
    assert wire is not None
    assert wire.logprob_token_ids is raw.token_ids
    assert result.sampler_output._gr_constraint_logprobs is raw.logprobs
    assert result.sampler_output._gr_constraint_valid_mask is raw.valid_mask
    assert result.valid_counts.tolist() == [len(row) for row in allowed_rows]
    assert result.status.tolist() == [
        NPU_SAMPLER_STATUS_OK if row else NPU_SAMPLER_STATUS_EMPTY for row in allowed_rows
    ]
    for row, allowed in enumerate(allowed_rows):
        count = len(allowed)
        assert raw.valid_mask[row].tolist() == [True] * count + [False] * (
            raw.token_ids.shape[1] - count
        )
        if count:
            scores = logits[row, allowed].float() / (temperature or 1.0)
            order = torch.argsort(scores, descending=True, stable=True)
            expected_tokens = torch.tensor(allowed, dtype=torch.int32)[order]
            expected_logprobs = torch.log_softmax(scores, dim=0)[order]
            torch.testing.assert_close(raw.token_ids[row, :count], expected_tokens)
            torch.testing.assert_close(raw.logprobs[row, :count], expected_logprobs)
            torch.testing.assert_close(wire.logprobs[row, :count], expected_logprobs)
            assert result.sampler_output.sampled_token_ids[row, 0] == expected_tokens[0]
        else:
            assert result.sampler_output.sampled_token_ids[row, 0] == 0
        assert torch.all(raw.logprobs[row, count:] == INVALID_LOGIT)
        assert torch.all(torch.isneginf(wire.logprobs[row, count:]))
    assert result.sampler_output.logprobs_tensors.selected_token_ranks.dtype == torch.int64


@pytest.mark.parametrize("temperature", [0.0, 0.25, 2.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_tables_dispatch_all_stages_and_preserve_temperature_and_padding(
    temperature: float, dtype: torch.dtype
) -> None:
    sampler, _, op = _composition(temperature)
    sequence = torch.full((1, 4, 3), 5, dtype=torch.int32)
    prefix_ptr = sampler.prefix_token_ids.data_ptr()
    base_logits = torch.tensor([0.125, -0.75, 1.125, 0.625, -0.375, 2.25], dtype=dtype)
    allowed_stages = [
        [[1, 3]],
        [[2, 4], [5], [2, 4], [5]],
        [[0, 1], [3], [1, 4], [0, 1]],
    ]
    expected_prefixes = [
        [[[0, 0, 0]]],
        [[[1, 0, 0]], [[3, 0, 0]], [[1, 0, 0]], [[3, 0, 0]]],
        [[[1, 2, 0]], [[1, 4, 0]], [[3, 5, 0]], [[1, 2, 0]]],
    ]
    for step, allowed in enumerate(allowed_stages):
        if step == 1:
            sequence[0, :, 0] = torch.tensor([1, 3, 1, 3], dtype=torch.int32)
        elif step == 2:
            # Simulate a Beam decision reordering the canonical parent paths.
            sequence[0, :, :2] = torch.tensor([[1, 2], [1, 4], [3, 5], [1, 2]])
        logits = base_logits.repeat(len(allowed), 1)
        result = sampler.select(logits, sequence, expected_device_step=step)
        _check_candidates(result, logits, allowed, temperature)
        args = op.calls[-1]
        assert len(args) == 13
        assert args[9:] == (step, 4, 2, 2)
        assert args[1].tolist() == expected_prefixes[step]
        assert args[8].tolist() == [temperature]
        # BF16 is losslessly widened by the real Backend before the op call;
        # neither FP16 nor FP32 is forced through the BF16 path.
        assert args[0].dtype == (torch.float32 if dtype == torch.bfloat16 else dtype)
        torch.testing.assert_close(args[0].float(), logits.float(), rtol=0, atol=0)
        assert sampler.prefix_token_ids.data_ptr() == prefix_ptr
        sampler.consume(result)
    assert len(op.calls) == 3


@pytest.mark.parametrize("temperature", [0.0, 0.25, 2.0])
def test_top_one_normalizes_over_all_legal_candidates_at_every_stage(
    temperature: float,
) -> None:
    sampler, _, op = _composition(temperature, width=1)
    sequence = torch.zeros((1, 1, 3), dtype=torch.int32)
    logits = torch.tensor([[0.5, 0.25, 0.5, -0.25, 0.0, 0.75]])
    for step, allowed in enumerate(([1, 3], [2, 4], [0, 1])):
        result = sampler.select(logits, sequence, expected_device_step=step)
        scores = logits[0, allowed] / (temperature or 1.0)
        order = torch.argsort(scores, descending=True, stable=True)
        expected_token = torch.tensor(allowed, dtype=torch.int32)[order[:1]]
        # The denominator includes both legal entries, although W=1 retains
        # only the best. Normalizing after truncation would incorrectly yield 0.
        expected_logprob = torch.log_softmax(scores, dim=0)[order[:1]]
        raw = result.constraint_output
        wire = result.sampler_output.logprobs_tensors
        assert wire is not None
        assert result.valid_counts.tolist() == [1]
        assert raw.valid_mask.tolist() == [[True]]
        assert result.status.tolist() == [NPU_SAMPLER_STATUS_OK]
        torch.testing.assert_close(raw.token_ids[0], expected_token)
        torch.testing.assert_close(raw.logprobs[0], expected_logprob)
        torch.testing.assert_close(wire.logprobs[0], expected_logprob)
        assert raw.logprobs[0, 0] < 0
        assert op.calls[-1][9:] == (step, 1, 2, 2)
        sampler.consume(result)
        sequence[0, 0, step] = expected_token[0]
    assert sequence.tolist() == [[[1, 2, 0]]]
    assert len(op.calls) == 3


@pytest.mark.parametrize("missing_step", [1, 2])
def test_missing_prefix_uses_real_table_validity_and_deferred_empty_status(
    missing_step: int,
) -> None:
    sampler, _, op = _composition(width=2)
    sequence = torch.zeros((1, 2, 3), dtype=torch.int32)
    prefill = sampler.select(torch.zeros(1, 6), sequence, expected_device_step=0)
    sampler.consume(prefill)
    sequence[0, :, 0] = torch.tensor([0 if missing_step == 1 else 1, 3])
    first = sampler.select(torch.zeros(2, 6), sequence, expected_device_step=1)
    _check_candidates(
        first, torch.zeros(2, 6), [[], [5]] if missing_step == 1 else [[2, 4], [5]], 1.0
    )
    if missing_step == 2:
        sampler.consume(first)
        sequence[0, :, :2] = torch.tensor([[1, 5], [3, 5]], dtype=torch.int32)
        final = sampler.select(torch.zeros(2, 6), sequence, expected_device_step=2)
        _check_candidates(final, torch.zeros(2, 6), [[], [1, 4]], 1.0)
    assert len(op.calls) == missing_step + 1
    assert not sampler.poisoned


def test_real_backend_operator_failure_is_not_retried_or_replaced() -> None:
    sampler, _, op = _composition(width=2)
    sequence = torch.zeros((1, 2, 3), dtype=torch.int32)
    prefill = sampler.select(torch.zeros(1, 6), sequence, expected_device_step=0)
    sampler.consume(prefill)
    sequence[0, :, 0] = torch.tensor([1, 3], dtype=torch.int32)
    op.fail_step = 1
    with pytest.raises(RuntimeError) as captured:
        sampler.select(torch.zeros(2, 6), sequence, expected_device_step=1)
    assert captured.value is op.error
    assert sampler.poisoned
    assert len(op.calls) == 2
    with pytest.raises(RuntimeError, match="poisoned"):
        sampler.select(torch.zeros(2, 6), sequence, expected_device_step=1)
    assert len(op.calls) == 2


def test_real_backend_closed_after_prefill_cannot_fall_back() -> None:
    sampler, backend, op = _composition(width=2)
    sequence = torch.zeros((1, 2, 3), dtype=torch.int32)
    prefill = sampler.select(torch.zeros(1, 6), sequence, expected_device_step=0)
    sampler.consume(prefill)
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        sampler.select(torch.zeros(2, 6), sequence, expected_device_step=1)
    assert sampler.poisoned
    assert len(op.calls) == 1
