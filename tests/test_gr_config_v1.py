# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_gr.config import (
    CONSTRAINT_TABLE_FORMAT,
    DEFAULT_ONLINE_BATCH_SIZES,
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


def test_canonical_graph_batch_shape_round_trip(tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["beam"]
    assert isinstance(beam, dict)
    beam["graph_batch_size"] = 2

    config = gr_config_from_mapping(raw)

    assert config.beam_graph_batch_size == 2
    assert gr_config_to_mapping(config) == raw


@pytest.mark.parametrize("sizes", [[1], [1, 3, 7], list(range(1, 9))])  # type: ignore[misc]
def test_online_batch_sizes_round_trip(sizes: list[int], tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["beam"]
    assert isinstance(beam, dict)
    beam["online_batch_sizes"] = sizes
    config = gr_config_from_mapping(raw)
    assert config.beam_online_batch_sizes == tuple(sizes)
    assert gr_config_to_mapping(config) == raw
    host = SimpleNamespace(additional_config={})
    write_canonical_gr_config(host, config)
    assert GRConfig.from_vllm_config(host).beam_online_batch_sizes == tuple(sizes)


def test_online_batch_sizes_default_and_normalization() -> None:
    assert GRConfig().beam_online_batch_sizes == DEFAULT_ONLINE_BATCH_SIZES == (1, 2, 4)
    assert GRConfig(beam_online_batch_sizes=(7, 1, 3)).beam_online_batch_sizes == (1, 3, 7)


@pytest.mark.parametrize("canonical", [False, True])  # type: ignore[misc]
def test_graph_widths_and_online_batch_sizes_survive_transport(
    tmp_path: Path, canonical: bool
) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["beam"]
    assert isinstance(beam, dict)
    beam.update(graph_widths=[8, 16, 32], online_batch_sizes=[1, 3, 7])
    config = gr_config_from_mapping(raw)
    assert gr_config_to_mapping(config) == raw

    host = SimpleNamespace(additional_config={})
    if canonical:
        write_canonical_gr_config(host, config)
    else:
        replace(config, constraint_table=None).write_to_vllm_config(host)
    restored = GRConfig.from_vllm_config(
        SimpleNamespace(additional_config=json.loads(json.dumps(host.additional_config)))
    )
    assert restored.beam_graph_widths == (8, 16, 32)
    assert restored.beam_online_batch_sizes == (1, 3, 7)


@pytest.mark.parametrize("max_num_seqs", [1, 2, 3])  # type: ignore[misc]
def test_default_online_policy_preserves_smaller_engine_capacity(
    monkeypatch: pytest.MonkeyPatch, max_num_seqs: int
) -> None:
    from vllm_gr.engine import arg_utils_gr

    host = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs)
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *args: host)
    monkeypatch.setattr(arg_utils_gr, "_is_npu_platform", lambda: False)
    monkeypatch.setattr(arg_utils_gr, "_is_cuda_platform", lambda: False)
    assert arg_utils_gr.create_engine_config(SimpleNamespace()) is host
    assert host.scheduler_config.max_num_seqs == max_num_seqs
    assert GRConfig.from_vllm_config(host).beam_online_batch_sizes == (1, 2, 4)


@pytest.mark.parametrize(  # type: ignore[misc]
    "sizes",
    [None, [], [3, 7], [1, 1], [True, 3], [1, False], [1, 3.0], [1, "3"], [1, 0], [1, 9], "1,3,7"],
)
def test_invalid_online_batch_sizes_fail_during_config_parsing(
    sizes: object, tmp_path: Path
) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["beam"]
    assert isinstance(beam, dict)
    beam["online_batch_sizes"] = sizes
    with pytest.raises(ValueError, match="online_batch_sizes"):
        gr_config_from_mapping(raw)
    with pytest.raises(ValueError, match="online_batch_sizes"):
        GRConfig.from_vllm_config(
            SimpleNamespace(additional_config={"vllm_gr_beam_online_batch_sizes": sizes})
        )


@pytest.mark.parametrize("max_num_seqs", [4, 7, 8])  # type: ignore[misc]
def test_engine_creation_preserves_online_batch_policy(
    monkeypatch: pytest.MonkeyPatch, max_num_seqs: int
) -> None:
    from vllm_gr.engine import arg_utils_gr

    host = SimpleNamespace(
        additional_config={"vllm_gr_beam_online_batch_sizes": [1, 3, 7]},
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *args: host)
    monkeypatch.setattr(arg_utils_gr, "_is_npu_platform", lambda: False)
    monkeypatch.setattr(arg_utils_gr, "_is_cuda_platform", lambda: False)
    if max_num_seqs < 7:
        with pytest.raises(ValueError, match="online_batch_sizes exceeds max_num_seqs"):
            arg_utils_gr.create_engine_config(SimpleNamespace())
        assert host.scheduler_config.max_num_seqs == max_num_seqs
        return
    assert arg_utils_gr.create_engine_config(SimpleNamespace()) is host
    assert GRConfig.from_vllm_config(host).beam_online_batch_sizes == (1, 3, 7)
    GRConfig().write_to_vllm_config(host)
    assert "vllm_gr_beam_online_batch_sizes" not in host.additional_config
    assert GRConfig.from_vllm_config(host).beam_online_batch_sizes == (1, 2, 4)


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

    config = gr_config_from_mapping(raw)

    assert config.beam_worker_decision is worker_decision
    assert config.beam_engine_driven is None
    assert gr_config_to_mapping(config) == raw


def test_retired_canonical_engine_control_does_not_require_worker_decision(tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["beam"]
    assert isinstance(beam, dict)
    beam["engine_driven"] = True

    with pytest.warns(FutureWarning, match="retired and ignored"):
        config = gr_config_from_mapping(raw)
    assert config.beam_worker_decision is None
    assert config.beam_engine_driven is None
    encoded_beam = gr_config_to_mapping(config)["beam"]
    assert isinstance(encoded_beam, dict)
    assert "engine_driven" not in encoded_beam


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
    stale = GRConfig.from_vllm_config(vllm_config)
    assert stale.beam_max_width is None

    write_canonical_gr_config(vllm_config, config)

    payload = vllm_config.additional_config[GR_CONFIG_KEY]
    assert json.loads(json.dumps(payload)) == payload
    assert GRConfig.from_vllm_config(vllm_config) == config
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


@pytest.mark.parametrize("batch_size", [1, 2, 4])  # type: ignore[misc]
def test_engine_creation_preserves_flat_graph_batch_size(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    from vllm_gr.engine import arg_utils_gr

    config = SimpleNamespace(additional_config={"vllm_gr_beam_graph_batch_size": batch_size})
    assert GRConfig.from_vllm_config(config).beam_graph_batch_size == batch_size
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *args: config)
    monkeypatch.setattr(arg_utils_gr, "_is_npu_platform", lambda: False)
    monkeypatch.setattr(arg_utils_gr, "_is_cuda_platform", lambda: False)

    assert arg_utils_gr.create_engine_config(SimpleNamespace()) is config
    assert GRConfig.from_vllm_config(config).beam_graph_batch_size == batch_size
    assert config.additional_config.get("vllm_gr_beam_graph_batch_size", 1) == batch_size


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


@pytest.mark.parametrize("device", ["cuda", "npu"])  # type: ignore[misc]
def test_benchmark_server_command_uses_canonical_attention_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    import huggingface_hub.constants
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_gr.engine import arg_utils_gr
    from vllm_gr.entrypoints.cli.serve import GRServeSubcommand

    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    raw["attention_backend"] = "CUSTOM"
    raw["beam"] = {
        "execution_mode": "v1",
        "max_width": 128,
        "max_decode_steps": 3,
        "graph_enabled": device == "cuda",
        "worker_decision": True,
    }
    table = raw["constraint_table"]
    assert isinstance(table, dict)
    table["backend"] = device
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))

    # Execute the actual server command with an argument recorder instead of
    # launching a model. This catches legacy flags left behind in the script.
    script = (Path(__file__).parent / "system/run_benchmark_and_verify.sh").read_text()
    command = script.split('echo "Starting OneRec online server"\n', 1)[1]
    command = command.split("\nSERVER_PID=$!", 1)[0]
    captured = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            r"""vllm-gr() { printf '%s\0' "$@"; }
"""
            + command
            + "\nwait",
        ],
        env={
            **os.environ,
            "config_path": str(config_path),
            "run_port": "7000",
            "MODEL_PATH": str(tmp_path),
        },
        capture_output=True,
        check=True,
    )
    parser = FlexibleArgumentParser()
    serve = GRServeSubcommand()
    serve.subparser_init(parser.add_subparsers(dest="subparser"))
    args = parser.parse_args(captured.stdout.decode().split("\0")[:-1])
    serve.validate(args)
    # ServeSubcommand.cmd normally normalizes the positional model before
    # constructing EngineArgs. Keep this config-only test local and offline.
    args.model = args.model_tag
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    engine_args = arg_utils_gr.from_cli_args(EngineArgs, args)
    assert engine_args.model == str(tmp_path)
    config = SimpleNamespace(
        additional_config={},
        attention_config=SimpleNamespace(backend=None, flash_attn_version=None),
        scheduler_config=SimpleNamespace(
            async_scheduling=args.async_scheduling,
            enable_chunked_prefill=args.enable_chunked_prefill,
        ),
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: config)
    monkeypatch.setattr(arg_utils_gr, "_is_cuda_platform", lambda: device == "cuda")
    monkeypatch.setattr(arg_utils_gr, "_is_npu_platform", lambda: device == "npu")

    result = arg_utils_gr.create_engine_config(engine_args)

    assert result.attention_config.backend is AttentionBackendEnum.CUSTOM
    # Suffix fusion must leave hardware-dependent prefix selection automatic.
    assert result.attention_config.flash_attn_version is None
    assert result.additional_config[GR_CONFIG_KEY]["attention_backend"] == "CUSTOM"
    assert result.additional_config[GR_CONFIG_KEY]["constraint_table"]["backend"] == device


@pytest.mark.parametrize("async_scheduling", [False, True])  # type: ignore[misc]
def test_engine_creation_serializes_cuda_decision_defaults(
    tmp_path: Path,
    async_scheduling: bool,
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
    assert "engine_driven" not in payload["beam"]
    assert parse_gr_config(vllm_config).beam_worker_decision is True
    assert parse_gr_config(vllm_config).beam_engine_driven is None


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("enabled", [False, True])  # type: ignore[misc]
def test_adapter_cache_switch_round_trip(enabled: bool, tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["beam"]
    assert isinstance(beam, dict)
    if enabled:
        beam["adapter_cache_enabled"] = True
    config = gr_config_from_mapping(raw)
    assert config.beam_adapter_cache_enabled is enabled
    assert gr_config_to_mapping(config) == raw
    host = SimpleNamespace(additional_config={})
    legacy = GRConfig(beam_adapter_cache_enabled=enabled)
    legacy.write_to_vllm_config(host)
    assert GRConfig.from_vllm_config(host).beam_adapter_cache_enabled is enabled


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])  # type: ignore[misc]
def test_adapter_cache_switch_requires_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="adapter_cache_enabled must be a boolean"):
        GRConfig(beam_adapter_cache_enabled=value)
