# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Test-only schema-v2 config factory.

Keeps legacy fixtures readable while production migrates to
``GRConfig(constraint=ConstraintConfig(...))``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from vllm_gr.config import (
    ConstraintBackendKind,
    ConstraintConfig,
    ConstraintMode,
    ConstraintTableSpec,
    GRConfig,
)
from vllm_gr.config.models import DeploymentOptions

_DEFAULT_PATH = "/tmp/constraint_table.safetensors"
_DEFAULT_ARTIFACT_DIGEST = "sha256:" + "a" * 64
_DEFAULT_TOKENIZER_DIGEST = "sha256:" + "b" * 64


def _table_constraint(table: object | None, catalog_path: str | None) -> ConstraintConfig:
    path = getattr(table, "path", None) if table is not None else None
    path = path or catalog_path or _DEFAULT_PATH
    if not Path(path).is_absolute():
        # Legacy catalog_path was a marker, not an artifact path.
        path = _DEFAULT_PATH
    backend = getattr(table, "backend", None) if table is not None else None
    backend_kind = (
        ConstraintBackendKind(backend.value) if backend is not None else ConstraintBackendKind.CUDA
    )
    return ConstraintConfig(
        mode=ConstraintMode.TABLE,
        table=ConstraintTableSpec(
            path=path,
            backend=backend_kind,
            max_top_k=(
                getattr(table, "max_top_k", None)
                if table is not None and getattr(table, "max_top_k", None) is not None
                else 128
            ),
            format=(
                getattr(table, "format", "constraint_table_v1")
                if table is not None
                else "constraint_table_v1"
            ),
            artifact_digest=(
                getattr(table, "artifact_digest", None) or _DEFAULT_ARTIFACT_DIGEST
                if table is not None
                else _DEFAULT_ARTIFACT_DIGEST
            ),
            tokenizer_digest=(
                getattr(table, "tokenizer_digest", None) or _DEFAULT_TOKENIZER_DIGEST
                if table is not None
                else _DEFAULT_TOKENIZER_DIGEST
            ),
        ),
    )


def make_gr_config(**kwargs: object) -> GRConfig:
    """Build a GRConfig from legacy keyword arguments used by old tests."""

    constraint = kwargs.pop("constraint", None)
    table = kwargs.pop("constraint_table", None)
    backend = kwargs.pop("constraint_backend", None)
    constrained = kwargs.pop("constrained_generation", None)
    catalog_path = cast(str | None, kwargs.pop("catalog_path", None))

    has_legacy_constraint = (
        table is not None
        or backend is not None
        or constrained is not None
        or catalog_path is not None
    )
    if constraint is None:
        if not has_legacy_constraint:
            return GRConfig(**kwargs)
        if table is None and backend is None and catalog_path is None:
            # No table/backend: constrained_generation alone is schema-v1 state.
            return GRConfig(constrained_generation=constrained, **kwargs)
        if constrained is False:
            constraint = ConstraintConfig(mode=ConstraintMode.NONE)
        elif table is not None and not getattr(table, "enabled", False) and backend is None:
            # A disabled table is still a valid sizing holder in schema-v1.
            return GRConfig(constraint_table=table, **kwargs)
        elif table is not None or backend == "constraint_table":
            constraint = _table_constraint(table, catalog_path)
        else:
            constraint = ConstraintConfig(mode=ConstraintMode.NONE)

    if "beam_candidate_width" not in kwargs and table is not None:
        candidate_width = getattr(table, "candidate_width", None)
        if candidate_width is not None:
            kwargs["beam_candidate_width"] = candidate_width
    return GRConfig(constraint=constraint, **kwargs)


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
        worker_extension_cls="tests.st.test_npu_dynamic_beam_st.DynamicBeamTestWorkerExtension",
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
