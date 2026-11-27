# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""One Worker reload_trie operation backed by VersionedConstraintTable."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from vllm_gr.config.models import deployment_for, runtime_constraint_table
from vllm_gr.constraints.versioned_table import VersionedConstraintTable


def _sync(runner: Any) -> None:
    device = torch.device(runner.device)
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def _load_trie(runner: Any, path: str) -> Any:
    from vllm_gr.config import GRConfig
    from vllm_gr.constraints.loader import load_constraint_backend

    config = runtime_constraint_table(GRConfig.from_vllm_config(runner.vllm_config))
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert config is not None
    config = replace(config, path=path, artifact_digest="sha256:" + digest.hexdigest())
    backend = load_constraint_backend(
        config,
        device=torch.device(runner.device),
        vocab_size=runner.model_config.get_vocab_size(),
        dtype=runner.model_config.dtype,
        max_rows=runner.constraint_backend.info.max_rows,
    )
    return backend


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

    settings = deployment_for(getattr(runner, "vllm_config", None))
    before_publish: Callable[[Any], None] | None = None
    if settings.enable_streaming_constrained_head and torch.device(runner.device).type == "cuda":
        from vllm_gr.config import GRConfig, resolve_candidate_width
        from vllm_gr.v1.worker.cuda.streaming_constrained_head import (
            prewarm_streaming_constrained_head,
        )

        def prepare_streaming(candidate: Any) -> None:
            config = GRConfig.from_vllm_config(runner.vllm_config)
            if config.beam_max_width is not None:
                prewarm_streaming_constrained_head(
                    runner, candidate, resolve_candidate_width(config)
                )

        before_publish = prepare_streaming

    return VersionedConstraintTable(
        backend,
        loader=load,
        retention_seconds=settings.trie_retention_seconds,
        before_release=lambda: _sync(runner),
        before_publish=before_publish,
    )


def reload_trie(self: Any, path: str, version: str) -> dict[str, Any]:
    self.model_runner.constraint_backend.reload(path, version)
    return {"version": version, "ready": True}


def install_worker_hot_update_hooks(worker_cls: type) -> None:
    setattr(worker_cls, "reload_trie", reload_trie)
