# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""One Worker reload_trie operation backed by VersionedConstraintTable."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from vllm_gr.constraints.versioned_table import VersionedConstraintTable
from vllm_gr.platforms.binding import runner_constraint_backend


def _sync(runner: Any) -> None:
    device = torch.device(runner.device)
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def _load_trie(runner: Any, path: str) -> Any:
    from vllm_gr.config import GRConfig
    from vllm_gr.constraints.loader import load_runner_constraint_backend
    from vllm_gr.platforms.plan import read_backend_plan

    gr_config = GRConfig.from_vllm_config(runner.vllm_config)
    if gr_config.constraint_table is None or not gr_config.constraint_table.enabled:
        raise ValueError("Cannot reload a disabled constraint table")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    table = replace(
        gr_config.constraint_table, path=path, artifact_digest="sha256:" + digest.hexdigest()
    )
    binding = getattr(runner, "_gr_backend_binding", None)
    return load_runner_constraint_backend(
        runner,
        replace(gr_config, constraint_table=table),
        plan=binding.plan if binding is not None else read_backend_plan(runner.vllm_config),
        vocab_size=runner.model_config.get_vocab_size(),
        max_rows=runner_constraint_backend(runner).info.max_rows,
    )


def versioned_constraint_table(
    runner: Any,
    backend: Any,
    *,
    prepare_backend: Callable[[Any, Any], None] | None = None,
) -> VersionedConstraintTable:
    """Finish runner-specific preparation before initial or replacement publication."""
    if prepare_backend is not None:
        prepare_backend(runner, backend)

    def load(path: str) -> Any:
        candidate = _load_trie(runner, path)
        try:
            if prepare_backend is not None:
                prepare_backend(runner, candidate)
        except BaseException:
            candidate.close()
            raise
        return candidate

    return VersionedConstraintTable(
        backend,
        loader=load,
        retention_seconds=float(os.environ.get("VLLM_GR_TRIE_RETENTION_SECONDS", "60")),
        before_release=lambda: _sync(runner),
    )


def reload_trie(self: Any, path: str, version: str) -> dict[str, Any]:
    runner_constraint_backend(self.model_runner).reload(path, version)
    return {"version": version, "ready": True}


def install_worker_hot_update_hooks(worker_cls: type) -> None:
    setattr(worker_cls, "reload_trie", reload_trie)
