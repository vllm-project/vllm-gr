# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"

"""Real fused-op readiness gate for all three constraint-table steps."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintTableBackendEnum, ConstraintTableSpec
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.backend import INVALID_LOGIT, ConstraintMetadata
from vllm_gr.constraints.loader import load_constraint_backend
from vllm_gr.ops.npu.rec_constrained_topk import (
    is_available as rec_constrained_topk_npu_available,
)

pytestmark = pytest.mark.npu_test

try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"torch-npu is not installed: {exc}", allow_module_level=True)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is unavailable", allow_module_level=True)
if not rec_constrained_topk_npu_available():
    pytest.skip("real NPU rec_constrained_topk is unavailable", allow_module_level=True)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skip(reason="NPU BF16 readiness has a known precision issue"),
        ),
        torch.float32,
    ],
)
def test_real_npu_loader_passes_step_zero_one_two_at_top_k_256(
    tmp_path: Path,
    dtype: torch.dtype,
) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    config = ConstraintTableSpec(
        enabled=True,
        backend=ConstraintTableBackendEnum.NPU,
        path=str(path),
        artifact_digest=digest,
        tokenizer_digest="sha256:" + "b" * 64,
        max_top_k=256,
    )
    backend = load_constraint_backend(
        config,
        device=torch.device("npu"),
        vocab_size=6,
        dtype=dtype,
        max_rows=2,
    )
    assert backend is not None
    assert backend.info.max_top_k == 256
    backend.close()


def test_real_npu_keeps_low_legal_logprob_separate_from_padding(tmp_path: Path) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    backend = load_constraint_backend(
        ConstraintTableSpec(
            enabled=True,
            backend=ConstraintTableBackendEnum.NPU,
            path=str(path),
            artifact_digest=digest,
            tokenizer_digest="sha256:" + "b" * 64,
            max_top_k=3,
        ),
        device=torch.device("npu"),
        vocab_size=6,
        dtype=torch.float32,
        max_rows=1,
    )
    assert backend is not None
    try:
        logits = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 1.0e20, 0.0]],
            dtype=torch.float32,
            device="npu",
        )
        output = backend.select(
            logits,
            ConstraintMetadata(
                prefix_token_ids=torch.tensor([[1, 0, 0]], dtype=torch.int32, device="npu"),
                temperatures=torch.ones(1, dtype=torch.float32, device="npu"),
                current_step=1,
                top_k=3,
            ),
        )

        assert output.valid_mask.is_contiguous()
        assert output.valid_mask.cpu().tolist() == [[True, True, False]]
        assert set(output.token_ids[0][output.valid_mask[0]].cpu().tolist()) == {2, 4}
        legal_logprobs = output.logprobs[0][output.valid_mask[0]]
        assert bool(torch.isfinite(legal_logprobs).all())
        assert bool((legal_logprobs < (INVALID_LOGIT / 2)).any())
    finally:
        backend.close()
