# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One Worker reload_trie operation backed by VersionedConstraintTable."""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from vllm_gr.constraints.versioned_table import VersionedConstraintTable


def _sync(runner: Any) -> None:
    device = torch.device(runner.device)
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def _load_trie(runner: Any, path: str) -> Any:
    from vllm_gr.config import GRConfig
    from vllm_gr.constraints.loader import load_constraint_backend

    config = GRConfig.from_vllm_config(runner.vllm_config).constraint_table
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    config = replace(config, path=path, artifact_digest="sha256:" + digest.hexdigest())
    backend = load_constraint_backend(
        config,
        device=torch.device(runner.device),
        vocab_size=runner.model_config.get_vocab_size(),
        dtype=runner.model_config.dtype,
        max_rows=runner.constraint_backend.info.max_rows,
    )
    return backend


def versioned_constraint_table(runner: Any, backend: Any) -> VersionedConstraintTable:
    """Wrap the loaded backend once, keeping the runner's original attribute."""
    return VersionedConstraintTable(
        backend,
        loader=lambda path: _load_trie(runner, path),
        retention_seconds=float(os.environ.get("VLLM_GR_TRIE_RETENTION_SECONDS", "60")),
        before_release=lambda: _sync(runner),
    )


def reload_trie(self: Any, path: str, version: str) -> dict[str, Any]:
    self.model_runner.constraint_backend.reload(path, version)
    return {"version": version, "ready": True}


def install_worker_hot_update_hooks(worker_cls: type) -> None:
    setattr(worker_cls, "reload_trie", reload_trie)
