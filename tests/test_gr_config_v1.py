# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
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


def test_engine_creation_fails_closed_for_enabled_canonical_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keep this unit test independent of optional vLLM runtime dependencies.
    # The production module only needs these three upstream symbols before the
    # patched EngineArgs method is invoked.
    class EngineArgs:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        @staticmethod
        def add_cli_args(parser: object) -> object:
            return parser

        def create_engine_config(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError

        @classmethod
        def from_cli_args(cls, args: object) -> object:
            raise NotImplementedError

    class AttentionBackendEnum(str, Enum):
        CUSTOM = "CUSTOM"

    def module(name: str, **values: Any) -> ModuleType:
        result = ModuleType(name)
        for key, value in values.items():
            setattr(result, key, value)
        return result

    fake_modules = {
        "vllm.engine": module("vllm.engine"),
        "vllm.engine.arg_utils": module("vllm.engine.arg_utils", EngineArgs=EngineArgs),
        "vllm.utils.argparse_utils": module(
            "vllm.utils.argparse_utils", FlexibleArgumentParser=object
        ),
        "vllm.v1.attention": module("vllm.v1.attention"),
        "vllm.v1.attention.backends": module("vllm.v1.attention.backends"),
        "vllm.v1.attention.backends.registry": module(
            "vllm.v1.attention.backends.registry",
            AttentionBackendEnum=AttentionBackendEnum,
        ),
    }
    for name, fake_module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, fake_module)
    sys.modules.pop("vllm_gr.engine.arg_utils_gr", None)
    engine_package = sys.modules.get("vllm_gr.engine")
    if engine_package is not None and hasattr(engine_package, "arg_utils_gr"):
        monkeypatch.delattr(engine_package, "arg_utils_gr")

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
        pytest.raises(RuntimeError, match="Worker integration"),
    ):
        create_engine_config(engine_args)

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
