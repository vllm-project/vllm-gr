# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_gr.config import (
    DEFAULT_BEAM_MAX_DECODE_STEPS,
    DEFAULT_CONSTRAINT_MAX_TOP_K,
    BeamConfig,
    ConstraintTableBackend,
    ConstraintTableConfig,
    GRAttentionBackend,
    GRConfig,
)
from vllm_gr.config.parser import (
    GR_CONFIG_KEY,
    gr_config_from_mapping,
    gr_config_to_mapping,
    load_gr_config,
    normalize_gr_config_input,
)

DIGEST = "sha256:" + "1" * 64
TOKENIZER_DIGEST = "sha256:" + "2" * 64


def _enabled_config() -> GRConfig:
    return GRConfig(
        beam=BeamConfig(max_width=128, max_decode_steps=3, graph_enabled=True),
        constraint_table=ConstraintTableConfig(
            enabled=True,
            backend=ConstraintTableBackend.NPU,
            path="/models/constraint_table.safetensors",
            artifact_digest=DIGEST,
            tokenizer_digest=TOKENIZER_DIGEST,
            max_top_k=128,
        ),
    )


def _vllm_config(additional_config: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(additional_config={} if additional_config is None else additional_config)


def test_nested_additional_config_round_trip() -> None:
    expected = _enabled_config()
    vllm_config = _vllm_config({"unrelated": True})
    expected.write_to_vllm_config(vllm_config)

    assert set(vllm_config.additional_config) == {"unrelated", GR_CONFIG_KEY}
    assert GRConfig.from_vllm_config(vllm_config) == expected
    assert (
        json.loads(json.dumps(vllm_config.additional_config))[GR_CONFIG_KEY]["constraint_table"][
            "backend"
        ]
        == "npu"
    )


def test_worker_reconstructs_without_reading_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _enabled_config()
    vllm_config = _vllm_config({GR_CONFIG_KEY: gr_config_to_mapping(config)})

    def forbidden(path: object) -> GRConfig:
        raise AssertionError("Worker attempted to read YAML")

    monkeypatch.setattr("vllm_gr.config.parser.load_gr_config", forbidden)
    assert GRConfig.from_vllm_config(vllm_config) == config


def test_load_yaml_schema(tmp_path: Path) -> None:
    path = tmp_path / "vllm_gr.yaml"
    path.write_text(
        """schema_version: 1
beam:
  max_width: 128
  max_decode_steps: 3
  graph_enabled: true
constraint_table:
  enabled: true
  backend: npu
  path: /models/constraint_table.safetensors
  format: constraint_table_v1
  artifact_digest: sha256:1111111111111111111111111111111111111111111111111111111111111111
  tokenizer_digest: sha256:2222222222222222222222222222222222222222222222222222222222222222
  max_top_k: 128
""",
        encoding="utf-8",
    )
    assert load_gr_config(path) == _enabled_config()


def test_normalize_python_config_inputs(tmp_path: Path) -> None:
    expected = _enabled_config()
    path = tmp_path / "vllm_gr.yaml"
    path.write_text(
        f"""schema_version: 1
beam:
  max_width: 128
constraint_table:
  enabled: true
  backend: npu
  path: /models/constraint_table.safetensors
  artifact_digest: {DIGEST}
  tokenizer_digest: {TOKENIZER_DIGEST}
  max_top_k: 128
""",
        encoding="utf-8",
    )

    assert normalize_gr_config_input(expected) is expected
    assert normalize_gr_config_input(gr_config_to_mapping(expected)) == expected
    assert normalize_gr_config_input(path) == expected
    assert normalize_gr_config_input(str(path)) == expected


def test_normalize_rejects_unsupported_python_input() -> None:
    with pytest.raises(TypeError, match="YAML path, mapping, or GRConfig"):
        normalize_gr_config_input(1)  # type: ignore[arg-type]


def test_mapping_uses_shared_defaults() -> None:
    config = gr_config_from_mapping({"schema_version": 1})

    assert config.beam.max_decode_steps == DEFAULT_BEAM_MAX_DECODE_STEPS
    assert config.constraint_table.max_top_k == DEFAULT_CONSTRAINT_MAX_TOP_K


def test_load_yaml_resolves_artifact_path_against_yaml_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "vllm_gr.yaml"
    path.write_text(
        f"""schema_version: 1
beam:
  max_width: {DEFAULT_CONSTRAINT_MAX_TOP_K}
constraint_table:
  enabled: true
  path: ./artifacts/constraint_table.safetensors
  artifact_digest: {DIGEST}
  tokenizer_digest: {TOKENIZER_DIGEST}
""",
        encoding="utf-8",
    )

    config = load_gr_config(path)
    expected = (config_dir / "artifacts/constraint_table.safetensors").resolve()

    assert config.constraint_table.path == str(expected)
    assert gr_config_to_mapping(config)["constraint_table"]["path"] == str(expected)


def test_mapping_input_rejects_relative_artifact_path() -> None:
    with pytest.raises(ValueError, match="path must be absolute"):
        normalize_gr_config_input(
            {
                "schema_version": 1,
                "beam": {"max_width": DEFAULT_CONSTRAINT_MAX_TOP_K},
                "constraint_table": {
                    "enabled": True,
                    "path": "./constraint_table.safetensors",
                    "artifact_digest": DIGEST,
                    "tokenizer_digest": TOKENIZER_DIGEST,
                },
            }
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"unknown": True}, "unknown"),
        ({"constraint_table": {"enabled": True}}, "requires"),
        (
            {
                "constraint_table": {
                    "enabled": True,
                    "backend": "cuda",
                    "path": "/table",
                    "artifact_digest": DIGEST,
                    "tokenizer_digest": TOKENIZER_DIGEST,
                }
            },
            "backend",
        ),
    ],
)  # type: ignore[misc]
def test_invalid_config_fails_before_runtime(
    update: dict[str, object],
    message: str,
) -> None:
    raw: dict[str, object] = {"schema_version": 1}
    raw.update(update)
    with pytest.raises(ValueError, match=message):
        gr_config_from_mapping(raw)


def test_digest_format_is_strict() -> None:
    with pytest.raises(ValueError, match="64 lowercase"):
        ConstraintTableConfig(
            enabled=True,
            path="/table",
            artifact_digest="abc",
            tokenizer_digest=TOKENIZER_DIGEST,
        )


def test_constraint_table_capacity_must_cover_beam_width() -> None:
    with pytest.raises(ValueError, match="beam.max_width must be <="):
        GRConfig(
            beam=BeamConfig(max_width=128),
            constraint_table=ConstraintTableConfig(
                enabled=True,
                path="/table",
                artifact_digest=DIGEST,
                tokenizer_digest=TOKENIZER_DIGEST,
                max_top_k=64,
            ),
        )


def test_constraint_table_capacity_may_exceed_beam_width() -> None:
    config = GRConfig(
        beam=BeamConfig(max_width=64),
        constraint_table=ConstraintTableConfig(
            enabled=True,
            backend=ConstraintTableBackend.NPU,
            path="/table",
            artifact_digest=DIGEST,
            tokenizer_digest=TOKENIZER_DIGEST,
            max_top_k=512,
        ),
    )

    assert config.constraint_table.max_top_k == 512


def test_enabled_constraint_table_requires_beam_capacity() -> None:
    with pytest.raises(ValueError, match="requires beam.max_width"):
        GRConfig(
            constraint_table=ConstraintTableConfig(
                enabled=True,
                path="/table",
                artifact_digest=DIGEST,
                tokenizer_digest=TOKENIZER_DIGEST,
            ),
        )


def test_legacy_flat_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="legacy"):
        GRConfig.from_vllm_config(_vllm_config({"vllm_gr_beam_max_width": 128}))


def test_only_one_gr_cli_entrypoint() -> None:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from vllm_gr.engine.arg_utils_gr import add_cli_args

    parser = FlexibleArgumentParser()
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_add_cli_args",
        side_effect=lambda value: value,
    ):
        add_cli_args(parser)
    gr_options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--vllm-gr")
    }
    assert gr_options == {"--vllm-gr-config"}
    for removed in (
        "--catalog-path",
        "--constraint-backend",
        "--beam-max-width",
        "--beam-max-decode-steps",
        "--no-beam-graph",
    ):
        assert removed not in parser._option_string_actions


def test_create_engine_config_reads_yaml_and_writes_nested_value(tmp_path: Path) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    path = tmp_path / "config.yaml"
    path.write_text(
        """schema_version: 1
beam:
  max_width: 4
  max_decode_steps: 3
  graph_enabled: false
constraint_table:
  enabled: false
""",
        encoding="utf-8",
    )
    engine_args = SimpleNamespace(
        _vllm_gr_config_input=str(path),
        attention_backend=None,
        attention_config=None,
    )
    target = _vllm_config({"unrelated": "kept"})
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=target,
    ):
        result = create_engine_config(engine_args)
    parsed = GRConfig.from_vllm_config(result)
    assert parsed.beam == BeamConfig(max_width=4, graph_enabled=False)
    assert result.additional_config["unrelated"] == "kept"


@pytest.mark.parametrize("as_mapping", [False, True])
def test_create_engine_config_accepts_python_config_objects(as_mapping: bool) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    expected = _enabled_config()
    config_input: object = gr_config_to_mapping(expected) if as_mapping else expected
    engine_args = SimpleNamespace(
        _vllm_gr_config_input=config_input,
        attention_backend=None,
        attention_config=None,
    )
    target = _vllm_config({"unrelated": "kept"})
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=target,
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result) == expected
    assert result.additional_config["unrelated"] == "kept"


def test_engine_args_stores_config_under_input_name() -> None:
    from vllm_gr.engine.arg_utils_gr import _init_engine_args

    expected = _enabled_config()
    engine_args = SimpleNamespace()
    with mock.patch("vllm_gr.engine.arg_utils_gr.origin_engine_args_init"):
        _init_engine_args(engine_args, vllm_gr_config=expected)  # type: ignore[arg-type]

    assert engine_args._vllm_gr_config_input is expected
    assert not hasattr(engine_args, "vllm_gr_config")


def test_cli_path_is_moved_to_engine_config_input() -> None:
    from vllm_gr.engine.arg_utils_gr import from_cli_args

    engine_args = SimpleNamespace()
    args = SimpleNamespace(
        attention_config=None,
        vllm_gr_config="/configs/vllm_gr.yaml",
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_from_cli_args",
        return_value=engine_args,
    ):
        result = from_cli_args(SimpleNamespace, args)

    assert result is engine_args
    assert engine_args._vllm_gr_config_input == "/configs/vllm_gr.yaml"
    assert not hasattr(engine_args, "vllm_gr_config")


def test_attention_marker_is_nested() -> None:
    config = GRConfig(attention_backend=GRAttentionBackend.CUSTOM)
    vllm_config = _vllm_config()
    config.write_to_vllm_config(vllm_config)
    assert vllm_config.additional_config[GR_CONFIG_KEY]["attention_backend"] == "CUSTOM"


def test_default_config_does_not_claim_additional_config() -> None:
    target = _vllm_config({"unrelated": 1})
    GRConfig().write_to_vllm_config(target)
    assert target.additional_config == {"unrelated": 1}
