# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
import torch

from vllm_gr.config import ConstraintTableBackend, ConstraintTableConfig
from vllm_gr.constraints.artifact import write_constraint_table_artifact
from vllm_gr.constraints.cpu_backend import CPUConstraintBackend
from vllm_gr.constraints.loader import load_constraint_backend

from .helpers import TOKENIZER_DIGEST, make_table


def test_cpu_loader_has_no_framework_max_top_k_cap(tmp_path: Path) -> None:
    path = tmp_path / "constraint_table.safetensors"
    digest = write_constraint_table_artifact(make_table(), path)
    config = ConstraintTableConfig(
        enabled=True,
        backend=ConstraintTableBackend.CPU,
        path=str(path),
        artifact_digest=digest,
        tokenizer_digest=TOKENIZER_DIGEST,
        max_top_k=257,
    )

    backend = load_constraint_backend(
        config=config,
        device=torch.device("cpu"),
        vocab_size=8,
        dtype=torch.float32,
    )

    assert backend is not None
    assert backend.info.max_top_k == 257
    backend.close()


def test_cpu_loader_reuses_artifact_table_when_applying_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.constraints.loader as loader

    table = make_table()
    config = ConstraintTableConfig(
        enabled=True,
        backend=ConstraintTableBackend.CPU,
        path="/unused/constraint_table.safetensors",
        artifact_digest="sha256:" + "1" * 64,
        tokenizer_digest=TOKENIZER_DIGEST,
        max_top_k=257,
    )
    monkeypatch.setattr(loader, "load_constraint_table_artifact", lambda *args, **kwargs: table)

    backend = load_constraint_backend(
        config=config,
        device=torch.device("cpu"),
        vocab_size=8,
        dtype=torch.float32,
    )

    assert isinstance(backend, CPUConstraintBackend)
    assert backend._table is table
    assert backend.info.table is table.info
    assert backend.info.max_top_k == 257
    backend.close()


def test_npu_capacity_is_rejected_before_artifact_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.constraints.loader as loader

    config = ConstraintTableConfig(
        enabled=True,
        backend=ConstraintTableBackend.NPU,
        path="/missing/constraint_table.safetensors",
        artifact_digest="sha256:" + "1" * 64,
        tokenizer_digest=TOKENIZER_DIGEST,
        max_top_k=257,
    )

    def fake_npu_device(value: str | torch.device) -> SimpleNamespace:
        del value
        return SimpleNamespace(type="npu")

    def forbidden_artifact_read(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("NPU capacity must fail before Artifact access")

    monkeypatch.setattr(loader, "_device", fake_npu_device)
    monkeypatch.setattr(loader, "load_constraint_table_artifact", forbidden_artifact_read)
    with pytest.raises(ValueError, match=r"NPU.*max_top_k <= 256.*257"):
        loader.load_constraint_backend(
            config=config,
            device="npu",
            vocab_size=8,
            dtype=torch.float16,
        )
