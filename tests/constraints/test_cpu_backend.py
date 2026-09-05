# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.config import CONSTRAINT_TABLE_FORMAT, ConstraintTableBackendEnum
from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
)
from vllm_gr.constraints.cpu_backend import CPUConstraintBackend
from vllm_gr.constraints.table import ConstraintTable, ConstraintTableMetadata


def _backend(
    table: ConstraintTable | None = None,
    *,
    max_top_k: int = 8,
    max_rows: int = 8,
) -> CPUConstraintBackend:
    table = table or make_table()
    return CPUConstraintBackend(
        table,
        ConstraintBackendInfo(
            table_metadata=table.metadata,
            backend=ConstraintTableBackendEnum.CPU,
            max_top_k=max_top_k,
            max_rows=max_rows,
            host_memory_bytes=table.total_bytes,
            device_memory_bytes=0,
        ),
    )


def _select(
    backend: CPUConstraintBackend,
    logits: torch.Tensor,
    prefixes: list[list[int]],
    *,
    step: int,
    top_k: int,
    temperatures: torch.Tensor | None = None,
) -> ConstraintOutput:
    return backend.select(
        logits,
        ConstraintMetadata(
            prefix_token_ids=torch.tensor(prefixes, dtype=torch.int32, device=logits.device),
            temperatures=(
                temperatures
                if temperatures is not None
                else torch.ones(1, dtype=torch.float32, device=logits.device)
            ),
            current_step=step,
            top_k=top_k,
        ),
    )


@pytest.mark.parametrize(
    ("step", "prefix", "allowed"),
    [
        (0, [0, 0, 0], [1, 3]),
        (1, [1, 0, 0], [2, 4]),
        (2, [1, 2, 0], [0, 1]),
    ],
)
def test_cpu_backend_step_zero_one_two(step: int, prefix: list[int], allowed: list[int]) -> None:
    logits = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    output = _select(_backend(), logits, [prefix], step=step, top_k=3)

    assert set(output.token_ids[0][output.valid_mask[0]].tolist()) == set(allowed)
    assert output.valid_mask.sum().item() == len(allowed)
    assert torch.equal(
        output.logprobs[0][~output.valid_mask[0]],
        torch.full((3 - len(allowed),), INVALID_LOGIT),
    )


def test_cpu_backend_normalizes_over_shared_logits_window_before_topk() -> None:
    logits = torch.tensor([[0.0, -100.0, 1.0, -100.0, 2.0, 100.0]])
    output = _select(_backend(), logits, [[1, 0, 0]], step=1, top_k=1)

    # Token 5 is illegal and cannot be returned, but remains part of the
    # model-probability denominator shared by every parent beam.
    assert output.token_ids.item() == 4
    expected = torch.tensor(2.0) - torch.logsumexp(logits[0], dim=0)
    torch.testing.assert_close(output.logprobs.squeeze(), expected)


def test_cpu_backend_validity_does_not_collide_with_low_legal_logprob() -> None:
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0e20, 0.0]])
    output = _select(_backend(), logits, [[1, 0, 0]], step=1, top_k=3)

    assert output.valid_mask.tolist() == [[True, True, False]]
    assert set(output.token_ids[0][output.valid_mask[0]].tolist()) == {2, 4}
    legal_logprobs = output.logprobs[0][output.valid_mask[0]]
    assert torch.isfinite(legal_logprobs).all()
    assert bool((legal_logprobs < (INVALID_LOGIT / 2)).any())
    assert torch.equal(
        output.logprobs[0][~output.valid_mask[0]],
        torch.tensor([INVALID_LOGIT]),
    )


def test_cpu_backend_empty_prefix_and_pair_alias_are_padding_only() -> None:
    backend = _backend()
    logits = torch.zeros(2, 6)
    output = _select(
        backend,
        logits,
        [[-1, 14, 0], [0, 8, 0]],
        step=2,
        top_k=2,
    )
    assert not output.valid_mask.any()
    assert torch.equal(output.logprobs, torch.full((2, 2), INVALID_LOGIT))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cpu_backend_scalar_per_row_and_zero_temperature(dtype: torch.dtype) -> None:
    logits = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=dtype)
    output = _select(
        _backend(),
        logits,
        [[1, 0, 0], [1, 0, 0]],
        step=1,
        top_k=2,
        temperatures=torch.tensor([0.0, 2.0]),
    )
    assert output.logprobs.dtype == torch.float32
    assert output.token_ids.dtype == torch.int32
    assert output.token_ids[:, 0].tolist() == [4, 4]


def _wide_table(vocab_size: int = 300) -> ConstraintTable:
    first = np.arange(vocab_size, dtype=np.int32)
    offsets = np.arange(vocab_size + 1, dtype=np.int32)
    values1 = np.zeros(vocab_size, dtype=np.int32)
    pair_keys = np.arange(vocab_size, dtype=np.int64) * vocab_size
    values2 = np.zeros(vocab_size, dtype=np.int32)
    total_bytes = sum(
        value.nbytes for value in (first, offsets, values1, pair_keys, offsets, values2)
    )
    return ConstraintTable(
        first_token_ids=first,
        prefix1_offsets=offsets,
        prefix1_values=values1,
        prefix1_pair_keys=pair_keys,
        prefix2_value_offsets=offsets.copy(),
        prefix2_values=values2,
        metadata=ConstraintTableMetadata(
            format=CONSTRAINT_TABLE_FORMAT,
            artifact_digest="sha256:" + "a" * 64,
            tokenizer_digest="sha256:" + "b" * 64,
            vocab_size=vocab_size,
            max_prefix1_degree=1,
            max_prefix2_degree=1,
            total_bytes=total_bytes,
        ),
    )


def test_cpu_backend_supports_top_k_above_npu_limit() -> None:
    backend = _backend(_wide_table(), max_top_k=300, max_rows=1)
    output = _select(
        backend,
        torch.arange(300, dtype=torch.float32).reshape(1, 300),
        [[0, 0, 0]],
        step=0,
        top_k=257,
    )
    assert output.token_ids.shape == (1, 257)
    assert output.valid_mask.all()


def test_cpu_close_is_idempotent_and_preserves_info() -> None:
    backend = _backend()
    info = backend.info
    backend.close()
    backend.close()
    assert backend.info == info
    with pytest.raises(RuntimeError, match="closed"):
        _select(backend, torch.zeros(1, 6), [[0, 0, 0]], step=0, top_k=1)
