# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintBackendKind
from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintBackendInfo,
    ConstraintMetadata,
)
from vllm_gr.constraints.npu_backend import NPUConstraintBackend
from vllm_gr.constraints.table import CONSTRAINT_TABLE_TENSOR_NAMES
from vllm_gr.ops.rec_constrained_topk import rec_constrained_topk_torch


def _backend(op: Callable[..., tuple[torch.Tensor, torch.Tensor]]) -> NPUConstraintBackend:
    table = make_table()
    tensors = {
        name: torch.from_numpy(np.array(getattr(table, name), copy=True))
        for name in CONSTRAINT_TABLE_TENSOR_NAMES
    }
    return NPUConstraintBackend(
        **tensors,
        info=ConstraintBackendInfo(
            table_metadata=table.metadata,
            backend=ConstraintBackendKind.NPU,
            max_top_k=4,
            max_rows=2,
            host_memory_bytes=0,
            device_memory_bytes=table.total_bytes,
        ),
        op=op,
    )


def _reference_op(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, logprobs = rec_constrained_topk_torch(*args)
    return tokens.to(torch.int32), logprobs


def _metadata(step: int = 0, top_k: int = 2) -> ConstraintMetadata:
    return ConstraintMetadata(
        prefix_token_ids=torch.tensor([[1, 2, 0]], dtype=torch.int32),
        temperatures=torch.ones(1, dtype=torch.float32),
        current_step=step,
        top_k=top_k,
    )


def test_npu_backend_is_not_publicly_callable_until_ready() -> None:
    backend = _backend(_reference_op)
    with pytest.raises(RuntimeError, match="not ready"):
        backend.select(torch.zeros(1, 6), _metadata())

    backend._select_for_readiness(torch.zeros(1, 6), _metadata())  # noqa: SLF001
    backend._mark_ready()  # noqa: SLF001
    output = backend.select(torch.zeros(1, 6), _metadata())
    assert output.token_ids.dtype == torch.int32


def test_npu_backend_derives_validity_without_reading_logprob_values() -> None:
    def op(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([[4, 2, 0]], dtype=torch.int32),
            torch.tensor([[0.0, INVALID_LOGIT, INVALID_LOGIT]]),
        )

    output = _backend(op)._select_for_readiness(  # noqa: SLF001
        torch.zeros(1, 6),
        _metadata(step=1, top_k=3),
    )

    assert output.valid_mask.is_contiguous()
    assert output.valid_mask.tolist() == [[True, True, False]]


@pytest.mark.parametrize(
    ("bad_op", "message"),
    [
        (lambda *args: (torch.zeros(1, 1, dtype=torch.int32), torch.zeros(1, 1)), "shape"),
        (lambda *args: (torch.zeros(1, 2, dtype=torch.int64), torch.zeros(1, 2)), "int32"),
        (
            lambda *args: (
                torch.zeros(1, 2, dtype=torch.int32),
                torch.zeros(1, 2, dtype=torch.float16),
            ),
            "float32",
        ),
        (
            lambda *args: (
                torch.zeros(1, 4, dtype=torch.int32)[:, ::2],
                torch.zeros(1, 2),
            ),
            "contiguous",
        ),
        (
            lambda *args: (
                torch.zeros(1, 2, dtype=torch.int32, device="meta"),
                torch.zeros(1, 2, device="meta"),
            ),
            "device",
        ),
    ],
)
def test_npu_backend_rejects_raw_operator_contract_errors(
    bad_op: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    message: str,
) -> None:
    backend = _backend(bad_op)
    with pytest.raises(RuntimeError, match=message):
        backend._select_for_readiness(torch.zeros(1, 6), _metadata())  # noqa: SLF001


def test_npu_backend_passes_the_six_tensors_and_exact_attributes() -> None:
    captured: dict[str, tuple[object, ...]] = {}

    def op(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
        captured["args"] = args
        return torch.zeros(1, 2, dtype=torch.int32), torch.zeros(1, 2)

    backend = _backend(op)
    backend._select_for_readiness(torch.zeros(1, 6), _metadata(step=2))  # noqa: SLF001
    args = captured["args"]
    assert len(args) == 13
    prefixes = args[1]
    assert isinstance(prefixes, torch.Tensor)
    assert tuple(prefixes.shape) == (1, 1, 3)
    assert args[9:] == (2, 2, 2, 2)


def test_npu_close_is_idempotent_preserves_info_and_releases_owned_state() -> None:
    backend = _backend(_reference_op)
    info = backend.info
    backend.close()
    backend.close()
    assert backend.info == info
    assert backend._op is None  # noqa: SLF001
    assert backend._first_token_ids is None  # noqa: SLF001
    with pytest.raises(RuntimeError, match="closed"):
        backend.select(torch.zeros(1, 6), _metadata())


pytestmark = pytest.mark.cpu_test
