# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_gr.config import (
    CONSTRAINT_TABLE_FORMAT,
    ConstraintBackend,
    ConstraintTableBackendEnum,
    ConstraintTableConfig,
    GRConfig,
    NPUDecodeExecution,
    gr_config_from_mapping,
    gr_config_to_mapping,
    load_gr_config,
    write_canonical_gr_config,
)
from vllm_gr.config.parser import (
    CATALOG_PATH_KEY,
    GR_CONFIG_KEY,
    parse_gr_config,
)

_ARTIFACT_DIGEST = "sha256:" + "a" * 64
_TOKENIZER_DIGEST = "sha256:" + "b" * 64


def _canonical_mapping(path: str, *, enabled: bool = True) -> dict[str, object]:
    return {
        "schema_version": 1,
        "beam": {
            "graph_enabled": True,
            "max_width": 32,
            "max_decode_steps": 3,
        },
        "constraint_table": {
            "enabled": enabled,
            "backend": "cpu",
            "path": path,
            "format": CONSTRAINT_TABLE_FORMAT,
            "artifact_digest": _ARTIFACT_DIGEST,
            "tokenizer_digest": _TOKENIZER_DIGEST,
            "max_top_k": 128,
        },
        "attention_backend": None,
    }


def test_canonical_mapping_round_trip_is_exact(tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    config = gr_config_from_mapping(raw)

    assert config.beam_max_width == 32
    table = config.constraint_table
    assert table is not None
    assert table.backend is ConstraintTableBackendEnum.CPU
    assert gr_config_to_mapping(config) == raw


def test_canonical_write_invalidates_cached_defaults_for_uniproc(tmp_path: Path) -> None:
    host = SimpleNamespace(additional_config={})
    assert GRConfig.from_vllm_config(host).beam_max_width is None
    config = gr_config_from_mapping(_canonical_mapping(str(tmp_path / "table.safetensors")))
    write_canonical_gr_config(host, config)
    assert GRConfig.from_vllm_config(host) == config


@pytest.mark.parametrize("worker_decision", [True, False])  # type: ignore[misc]
def test_canonical_beam_runtime_options_round_trip(
    worker_decision: bool,
    tmp_path: Path,
) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["beam"]
    assert isinstance(beam, dict)
    beam["worker_decision"] = worker_decision
    beam["engine_driven"] = False

    config = gr_config_from_mapping(raw)

    assert config.beam_worker_decision is worker_decision
    assert config.beam_engine_driven is False
    assert gr_config_to_mapping(config) == raw


def test_canonical_self_advance_requires_worker_decision(tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["beam"]
    assert isinstance(beam, dict)
    beam["engine_driven"] = True

    with pytest.raises(ValueError, match="requires beam_worker_decision=True"):
        gr_config_from_mapping(raw)


def test_canonical_mapping_defaults_constraint_backend_to_cuda(tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"), enabled=False)
    table_mapping = raw["constraint_table"]
    assert isinstance(table_mapping, dict)
    del table_mapping["backend"]

    table = gr_config_from_mapping(raw).constraint_table

    assert table is not None
    assert table.backend is ConstraintTableBackendEnum.CUDA


def test_canonical_writer_uses_primitive_mapping(tmp_path: Path) -> None:
    config = gr_config_from_mapping(
        _canonical_mapping(str(tmp_path / "table.safetensors"), enabled=False)
    )
    vllm_config = SimpleNamespace(additional_config={"unrelated": 1})

    write_canonical_gr_config(vllm_config, config)

    payload = vllm_config.additional_config[GR_CONFIG_KEY]
    assert json.loads(json.dumps(payload)) == payload
    assert parse_gr_config(vllm_config) == config
    assert vllm_config.additional_config["unrelated"] == 1


@pytest.mark.parametrize("version", [True, 1.0, "1", None])  # type: ignore[misc]
def test_schema_version_is_strict(version: object, tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    raw["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        gr_config_from_mapping(raw)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("layer", "key"),
    [("top", "unknown"), ("beam", "width"), ("table", "digest")],
)
def test_unknown_keys_are_rejected(layer: str, key: str, tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    if layer == "top":
        raw[key] = 1
    elif layer == "beam":
        raw["beam"][key] = 1  # type: ignore[index]
    else:
        raw["constraint_table"][key] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown"):
        gr_config_from_mapping(raw)


def test_yaml_path_is_resolved_relative_to_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "gr.yaml"
    config_path.write_text(
        """schema_version: 1
beam:
  graph_enabled: true
  max_width: 32
  max_decode_steps: 3
constraint_table:
  enabled: true
  backend: cpu
  path: ../artifacts/table.safetensors
  format: constraint_table_v1
  artifact_digest: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  tokenizer_digest: sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  max_top_k: 128
attention_backend: null
""",
        encoding="utf-8",
    )

    config = load_gr_config(config_path)

    table = config.constraint_table
    assert table is not None
    assert table.path == str((tmp_path / "artifacts/table.safetensors").resolve())


def test_mapping_rejects_relative_artifact_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        gr_config_from_mapping(_canonical_mapping("table.safetensors"))


def test_mapping_base_dir_resolves_relative_artifact_path(tmp_path: Path) -> None:
    config = gr_config_from_mapping(_canonical_mapping("table.safetensors"), base_dir=tmp_path)
    table = config.constraint_table
    assert table is not None
    assert table.path == str((tmp_path / "table.safetensors").resolve())


def test_canonical_and_legacy_additional_config_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    vllm_config = SimpleNamespace(
        additional_config={
            GR_CONFIG_KEY: _canonical_mapping(str(tmp_path / "table.safetensors")),
            CATALOG_PATH_KEY: "/legacy/catalog.json",
        }
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_gr_config(vllm_config)


def test_canonical_conversion_rejects_legacy_only_fields() -> None:
    config = GRConfig(
        catalog_path="/legacy/catalog.json",
        constraint_backend=ConstraintBackend.CONSTRAINT_TABLE,
    )
    with pytest.raises(ValueError, match="legacy-only"):
        gr_config_to_mapping(config)

    with pytest.raises(ValueError, match="npu_decode_execution"):
        gr_config_to_mapping(GRConfig(npu_decode_execution=NPUDecodeExecution.XLLM))


def test_constraint_table_requires_complete_enabled_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact_digest"):
        ConstraintTableConfig(
            max_top_k=128,
            enabled=True,
            path=str(tmp_path / "table.safetensors"),
        )


def test_canonical_mapping_requires_explicit_max_top_k(tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    table = raw["constraint_table"]
    assert isinstance(table, dict)
    del table["max_top_k"]

    with pytest.raises(ValueError, match=r"constraint_table\.max_top_k"):
        gr_config_from_mapping(raw)


def test_constraint_max_top_k_has_no_npu_upper_bound(tmp_path: Path) -> None:
    config = ConstraintTableConfig(
        enabled=True,
        backend=ConstraintTableBackendEnum.NPU,
        path=str(tmp_path / "table.safetensors"),
        artifact_digest=_ARTIFACT_DIGEST,
        tokenizer_digest=_TOKENIZER_DIGEST,
        max_top_k=257,
    )
    assert config.max_top_k == 257


def test_engine_creation_writes_enabled_canonical_config(
    tmp_path: Path,
) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = SimpleNamespace(additional_config={})
    engine_args = SimpleNamespace(
        _vllm_gr_config_input=_canonical_mapping(str(tmp_path / "table.safetensors")),
        catalog_path=None,
        constraint_backend=None,
        attention_backend=None,
        attention_config=None,
    )
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        # This test covers canonical pass-through, not platform defaults.
        # Isolate both accelerator probes so it behaves identically on CUDA,
        # NPU, and CPU CI runners.
        mock.patch("vllm_gr.engine.arg_utils_gr._is_npu_platform", return_value=False),
        mock.patch("vllm_gr.engine.arg_utils_gr._is_cuda_platform", return_value=False),
    ):
        assert create_engine_config(engine_args) is vllm_config
    assert vllm_config.additional_config[GR_CONFIG_KEY] == _canonical_mapping(
        str(tmp_path / "table.safetensors")
    )

    conflicting_engine_args = SimpleNamespace(
        _vllm_gr_config_input=_canonical_mapping(
            str(tmp_path / "table.safetensors"), enabled=False
        ),
        catalog_path=None,
        constraint_backend=None,
        attention_backend="CUSTOM",
        attention_config=None,
    )
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=SimpleNamespace(additional_config={}),
        ),
        pytest.raises(ValueError, match="cannot be combined"),
    ):
        create_engine_config(conflicting_engine_args)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("async_scheduling", "expected_engine_driven"),
    [(False, True), (True, True)],
)
def test_engine_creation_serializes_cuda_decision_defaults(
    tmp_path: Path,
    async_scheduling: bool,
    expected_engine_driven: bool | None,
) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = SimpleNamespace(additional_config={})
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    engine_args = SimpleNamespace(
        _vllm_gr_config_input=raw,
        catalog_path=None,
        constraint_backend=None,
        attention_backend=None,
        attention_config=None,
        async_scheduling=async_scheduling,
    )

    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch("vllm_gr.engine.arg_utils_gr._is_cuda_platform", return_value=True),
    ):
        assert create_engine_config(engine_args) is vllm_config

    payload = vllm_config.additional_config[GR_CONFIG_KEY]
    assert payload["beam"]["worker_decision"] is True
    assert payload["beam"].get("engine_driven") is expected_engine_driven
    assert parse_gr_config(vllm_config).beam_worker_decision is True
    assert parse_gr_config(vllm_config).beam_engine_driven is expected_engine_driven
