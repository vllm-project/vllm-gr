# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintBackendKind
from vllm_gr.constraints.backend import ConstraintBackendInfo, ConstraintMetadata
from vllm_gr.constraints.cpu_backend import CPUConstraintBackend
from vllm_gr.constraints.npu_backend import NPUConstraintBackend
from vllm_gr.constraints.table import CONSTRAINT_TABLE_TENSOR_NAMES
from vllm_gr.ops.rec_constrained_topk import rec_constrained_topk_torch


def _backends() -> tuple[CPUConstraintBackend, NPUConstraintBackend]:
    table = make_table()
    common = {
        "table_metadata": table.metadata,
        "max_top_k": 4,
        "max_rows": 2,
    }
    cpu = CPUConstraintBackend(
        table,
        ConstraintBackendInfo(
            **common,
            backend=ConstraintBackendKind.CPU,
            host_memory_bytes=table.total_bytes,
            device_memory_bytes=0,
        ),
    )
    tensors = {
        name: torch.from_numpy(np.array(getattr(table, name), copy=True))
        for name in CONSTRAINT_TABLE_TENSOR_NAMES
    }

    def op(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, logprobs = rec_constrained_topk_torch(*args)
        return tokens.to(torch.int32), logprobs

    npu = NPUConstraintBackend(
        **tensors,
        info=ConstraintBackendInfo(
            **common,
            backend=ConstraintBackendKind.NPU,
            host_memory_bytes=0,
            device_memory_bytes=table.total_bytes,
        ),
        op=op,
    )
    npu._mark_ready()  # noqa: SLF001
    return cpu, npu


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cpu_and_npu_backends_match_for_non_tie_inputs(step: int, dtype: torch.dtype) -> None:
    cpu, npu = _backends()
    prefixes = torch.tensor([[1, 2, 0], [3, 5, 0]], dtype=torch.int32)
    # Every token has a distinct, exactly representable score in all dtypes.
    logits = torch.tensor(
        [[-8, -5, 1, 7, 3, 9], [-7, -4, 2, 8, 4, 10]],
        dtype=dtype,
    )
    metadata = ConstraintMetadata(
        prefix_token_ids=prefixes,
        temperatures=torch.tensor([0.0, 2.0]),
        current_step=step,
        top_k=2,
    )
    expected = cpu.select(logits, metadata)
    actual = npu.select(logits, metadata)

    assert torch.equal(actual.valid_mask, expected.valid_mask)
    for row in range(logits.shape[0]):
        mask = expected.valid_mask[row]
        torch.testing.assert_close(
            actual.token_ids[row][mask],
            expected.token_ids[row][mask],
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(actual.logprobs, expected.logprobs, rtol=0, atol=0)


def test_exact_tie_contract_does_not_assert_slot_order() -> None:
    cpu, npu = _backends()
    metadata = ConstraintMetadata(
        prefix_token_ids=torch.tensor([[1, 0, 0]], dtype=torch.int32),
        temperatures=torch.ones(1),
        current_step=1,
        top_k=2,
    )
    logits = torch.zeros(1, 6)
    cpu_output = cpu.select(logits, metadata)
    npu_output = npu.select(logits, metadata)

    # Without a kernel tie-break, only legality/count/logprob semantics are
    # contractual. Tests must not compare exact slot ordering for equal scores.
    assert cpu_output.valid_mask.sum() == npu_output.valid_mask.sum() == 2
    assert set(cpu_output.token_ids[0].tolist()) <= {2, 4}
    assert set(npu_output.token_ids[0].tolist()) <= {2, 4}
    torch.testing.assert_close(
        cpu_output.logprobs[0].sort().values,
        npu_output.logprobs[0].sort().values,
    )


pytestmark = pytest.mark.cpu_test
