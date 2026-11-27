# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Standalone MLU operator acceptance; does not start a vLLM runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from vllm_gr_mlu.provider import MLU_BACKEND

from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintBackendKind, ConstraintTableSpec
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.backend import ConstraintMetadata
from vllm_gr.constraints.cpu_backend import CPUConstraintBackend
from vllm_gr.constraints.loader import load_constraint_backend
from vllm_gr.platforms import activate_backend, register_backend
from vllm_gr.platforms.registry import _reset_backend_registry_for_tests


@pytest.fixture  # type: ignore[misc]
def mlu_device():
    mlu = pytest.importorskip("torch_mlu").mlu
    if not mlu.is_available():
        pytest.skip("no MLU device available")
    _reset_backend_registry_for_tests()
    register_backend(MLU_BACKEND)
    activate_backend("mlu")
    try:
        yield torch.device("mlu", mlu.current_device())
    finally:
        _reset_backend_registry_for_tests()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_mlu_constraints_full_vocabulary_parity(
    tmp_path: Path, mlu_device: torch.device, dtype: torch.dtype
) -> None:
    table = make_table()
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(table, path)
    spec = ConstraintTableSpec(
        enabled=True,
        backend=ConstraintBackendKind.MLU,
        max_top_k=128,
        path=str(path),
        artifact_digest=digest,
        tokenizer_digest=table.metadata.tokenizer_digest,
    )
    backend = load_constraint_backend(
        spec, device=mlu_device, vocab_size=6, dtype=dtype, max_rows=128
    )
    assert backend is not None
    reference = CPUConstraintBackend(table, backend.info)
    try:
        logits = torch.tensor([0, -1, 1, 2, 3, -2], dtype=dtype).repeat(128, 1)
        temperatures = torch.tensor([0.0, 2.0] * 64, dtype=torch.float32)
        for step in range(3):
            prefixes = torch.tensor([[1, 2, 0], [5, 2, 0]] * 64, dtype=torch.int32)
            result = backend.select(
                logits.to(mlu_device),
                ConstraintMetadata(prefixes.to(mlu_device), temperatures.to(mlu_device), step, 128),
            )
            expected = reference.select(
                logits, ConstraintMetadata(prefixes, temperatures, step, 128)
            )
            assert (
                result.token_ids.device
                == result.logprobs.device
                == result.valid_mask.device
                == mlu_device
            )
            assert torch.equal(result.valid_mask.cpu(), expected.valid_mask)
            mask = expected.valid_mask
            assert torch.equal(result.token_ids.cpu()[mask], expected.token_ids[mask])
            torch.testing.assert_close(
                result.logprobs.cpu(), expected.logprobs, rtol=1e-4, atol=1e-3
            )
    finally:
        backend.close()
        reference.close()
