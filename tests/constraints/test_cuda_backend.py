# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintTableBackendEnum, ConstraintTableConfig
from vllm_gr.constraints import loader
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.backend import ConstraintMetadata
from vllm_gr.constraints.loader import load_constraint_backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def test_cuda_loader_readiness_select_and_close(tmp_path: Path) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    backend = load_constraint_backend(
        ConstraintTableConfig(
            enabled=True,
            backend=ConstraintTableBackendEnum.CUDA,
            path=str(path),
            artifact_digest=digest,
            tokenizer_digest="sha256:" + "b" * 64,
            max_top_k=4,
        ),
        device=torch.device("cuda"),
        vocab_size=6,
        dtype=torch.bfloat16,
        max_rows=2,
    )
    assert backend is not None
    assert backend.info.backend is ConstraintTableBackendEnum.CUDA
    assert backend.info.host_memory_bytes == 0
    assert backend.info.device_memory_bytes == make_table().total_bytes
    output = backend.select(
        torch.tensor([[-8, -5, 1, 7, 3, 9]], dtype=torch.bfloat16, device="cuda"),
        ConstraintMetadata(
            prefix_token_ids=torch.tensor([[1, 0, 0]], dtype=torch.int32, device="cuda"),
            temperatures=torch.ones(1, dtype=torch.float32, device="cuda"),
            current_step=1,
            top_k=4,
        ),
    )
    assert output.token_ids.dtype == torch.int32
    assert output.logprobs.dtype == torch.float32
    assert output.valid_mask.cpu().tolist() == [[True, True, False, False]]
    assert set(output.token_ids[0][output.valid_mask[0]].cpu().tolist()) == {2, 4}
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="not ready"):
        backend.select(
            torch.zeros((1, 6), device="cuda"),
            ConstraintMetadata(
                prefix_token_ids=torch.zeros((1, 3), dtype=torch.int32, device="cuda"),
                temperatures=torch.ones(1, device="cuda"),
                current_step=0,
                top_k=1,
            ),
        )


def test_cuda_readiness_failure_closes_unpublished_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    uploaded: list[object] = []
    original_upload = loader._upload_cuda_backend  # noqa: SLF001

    def record_upload(*args: object, **kwargs: object) -> object:
        result = original_upload(*args, **kwargs)
        uploaded.append(result[0])
        return result

    monkeypatch.setattr(loader, "_upload_cuda_backend", record_upload)
    monkeypatch.setattr(
        loader,
        "_run_cuda_readiness",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("readiness failed")),
    )
    with pytest.raises(RuntimeError, match="readiness failed"):
        load_constraint_backend(
            ConstraintTableConfig(
                enabled=True,
                backend=ConstraintTableBackendEnum.CUDA,
                path=str(path),
                artifact_digest=digest,
                tokenizer_digest="sha256:" + "b" * 64,
                max_top_k=4,
            ),
            device=torch.device("cuda"),
            vocab_size=6,
            dtype=torch.float16,
            max_rows=2,
        )
    assert len(uploaded) == 1
    with pytest.raises(RuntimeError, match="closed"):
        uploaded[0]._tables()  # type: ignore[attr-defined]  # noqa: SLF001


def test_cuda_loader_rejects_fp32_logits_before_upload(tmp_path: Path) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    with pytest.raises(ValueError, match="fp16/bf16"):
        load_constraint_backend(
            ConstraintTableConfig(
                enabled=True,
                backend=ConstraintTableBackendEnum.CUDA,
                path=str(path),
                artifact_digest=digest,
                tokenizer_digest="sha256:" + "b" * 64,
                max_top_k=4,
            ),
            device=torch.device("cuda"),
            vocab_size=6,
            dtype=torch.float32,
            max_rows=2,
        )
