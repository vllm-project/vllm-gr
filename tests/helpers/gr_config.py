# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Fixtures for GR deployment settings and constraint artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vllm_gr.config import (
    ConstraintBackendKind,
    ConstraintConfig,
    ConstraintMode,
    ConstraintTableSpec,
    GRConfig,
)
from vllm_gr.config.models import DeploymentOptions


def table_constraint(*, max_top_k: int = 128) -> ConstraintConfig:
    return ConstraintConfig(
        mode=ConstraintMode.TABLE,
        table=ConstraintTableSpec(
            backend=ConstraintBackendKind.CUDA,
            path="/tmp/constraint_table.safetensors",
            max_top_k=max_top_k,
            artifact_digest="sha256:" + "a" * 64,
            tokenizer_digest="sha256:" + "b" * 64,
        ),
    )


def engine_args_from_file(tmp_path: Path, mapping: Mapping[str, object]) -> SimpleNamespace:
    """Pass deployment input through the same path accepted by EngineArgs."""
    import json
    from types import SimpleNamespace

    path = tmp_path / "scenario.yaml"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return SimpleNamespace(config=path)


def engine_config(**options: object) -> SimpleNamespace:
    """A test engine carrying a validated GR snapshot."""
    host = SimpleNamespace(additional_config={})
    set_deployment(host, **options)
    return host


def set_deployment(host: SimpleNamespace, **options: object) -> None:
    config = GRConfig.from_vllm_config(host)
    if config.beam_max_width is None:
        config = replace(config, beam_max_width=128)
    deployment = DeploymentOptions.from_mapping({**config.deployment.to_mapping(), **options})
    host._vllm_gr_parsed_config = replace(config, deployment=deployment)


def npu_dynamic_engine_args(
    model: str,
    artifact: Path,
    artifact_digest: str,
    tokenizer_digest: str,
    *,
    beam_width: int,
    graph: bool,
    explicit_widths: bool,
) -> dict[str, Any]:
    """Build the real NPU dynamic-beam ST startup arguments without device imports."""
    config: dict[str, Any] = {
        "prefill_graph_enabled": False,
        "worker_decision": True,
        "beam_width": [beam_width] * 3,
        "max_decode_steps": 3,
        "beam_graph_enabled": False,
        "decode_graph_batch_size": 2,
        "decode_graph_widths": [8, 16, 32],
        "npu_decode_execution": "xllm",
        "constraint_mode": "table",
        "constraint_table": {
            "backend": "npu",
            "max_top_k": beam_width,
            "path": str(artifact),
            "format": "constraint_table_v1",
            "digests": {"artifact": artifact_digest, "tokenizer": tokenizer_digest},
        },
        "candidate_width": [beam_width] * 3,
    }
    kwargs: dict[str, Any] = dict(
        model=model,
        trust_remote_code=True,
        dtype="float16",
        seed=0,
        async_scheduling=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=64,
        max_logprobs=beam_width,
        gpu_memory_utilization=0.5,
        worker_extension_cls="tests.e2e.test_npu_dynamic_beam_st.DynamicBeamTestWorkerExtension",
    )
    config["beam_graph_enabled"] = graph
    if not explicit_widths:
        config.pop("decode_graph_widths")
    return dict(kwargs, gr=config, attention_config={"backend": "CUSTOM"}, enforce_eager=not graph)


def serving_beam_engine_args(
    model: str, config: dict[str, Any], gpu_memory_utilization: float
) -> dict[str, Any]:
    """Build the serving regression's real engine inputs from width schedules."""
    return dict(
        model=model,
        trust_remote_code=True,
        async_scheduling=True,
        enable_chunked_prefill=False,
        max_model_len=2048,
        max_num_batched_tokens=8192,
        # V1 owns one logical request; beam rows are managed by the worker.
        max_num_seqs=1,
        enforce_eager=not config["beam_graph_enabled"],
        enable_log_requests=False,
        shutdown_timeout=10,
        max_logprobs=2 * max(config["candidate_width"]),
        gpu_memory_utilization=gpu_memory_utilization,
        gr=config,
        attention_config={"backend": "CUSTOM"},
    )
