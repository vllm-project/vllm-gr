# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from tests.helpers.gr_config import engine_args_from_file
from tests.helpers.paths import REPO_ROOT, TESTS_DIR
from vllm_gr.config import (
    CONSTRAINT_TABLE_FORMAT,
    ConstraintBackendKind,
    ConstraintTableSpec,
    GRConfig,
    gr_config_from_mapping,
    gr_config_to_mapping,
    load_gr_config,
    write_canonical_gr_config,
)
from vllm_gr.config.parser import GR_CONFIG_KEY, parse_gr_config
from vllm_gr.platforms.legacy import legacy_config_service

CATALOG_PATH_KEY = "vllm_gr_catalog_path"

_ARTIFACT_DIGEST = "sha256:" + "a" * 64
_TOKENIZER_DIGEST = "sha256:" + "b" * 64


def _canonical_mapping(path: str, *, enabled: bool = True) -> dict[str, Any]:
    constraint: dict[str, object] = {"mode": "table" if enabled else "none"}
    if enabled:
        constraint["table"] = {
            "backend": "cpu",
            "path": path,
            "format": CONSTRAINT_TABLE_FORMAT,
            "digests": {
                "artifact": _ARTIFACT_DIGEST,
                "tokenizer": _TOKENIZER_DIGEST,
            },
            "max_top_k": 128,
        }
    return {
        "gr": {
            "beam_graph_enabled": True,
            "beam_width": [32] * 3,
            "max_decode_steps": 3,
            "constraint_mode": constraint.get("mode", "none"),
            "constraint_table": constraint.get("table"),
            "candidate_width": [32] * 3,
        }
    }


def test_canonical_mapping_round_trip_is_exact(tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    config = gr_config_from_mapping(raw)

    assert config.beam_max_width == 32
    table = config.constraint_config.table
    assert table is not None
    assert table.backend is ConstraintBackendKind.CPU
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config


def test_canonical_graph_batch_shape_round_trip(tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["gr"]
    assert isinstance(beam, dict)
    beam["decode_graph_batch_size"] = 2

    config = gr_config_from_mapping(raw)

    assert config.beam_graph_batch_size == 2
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config


@pytest.mark.parametrize("sizes", [[1], [1, 3, 7], list(range(1, 9))])  # type: ignore[misc]
def test_beam_batch_sizes_round_trip(sizes: list[int], tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    raw["gr"].update({"beam_batch_sizes": sizes})
    config = gr_config_from_mapping(raw)
    assert config.deployment.beam_batch_sizes == tuple(sizes)
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config
    host = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    write_canonical_gr_config(host, config)
    assert GRConfig.from_vllm_config(host).deployment.beam_batch_sizes == tuple(sizes)


def test_default_beam_batch_sizes() -> None:
    assert GRConfig().deployment.beam_batch_sizes == (1, 2, 4)


def test_graph_widths_and_beam_batch_sizes_survive_transport(tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["gr"]
    assert isinstance(beam, dict)
    beam["decode_graph_widths"] = [8, 16, 32]
    raw["gr"].update({"beam_batch_sizes": [1, 3, 7]})
    config = gr_config_from_mapping(raw)
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config

    host = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    config.write_to_vllm_config(host)
    restored = GRConfig.from_vllm_config(
        SimpleNamespace(additional_config=json.loads(json.dumps(host.additional_config)))
    )
    assert restored.beam_graph_widths == (8, 16, 32)
    assert restored.deployment.beam_batch_sizes == (1, 3, 7)


@pytest.mark.parametrize("max_num_seqs", [1, 2, 3])  # type: ignore[misc]
def test_default_capture_sizes_preserve_smaller_engine_capacity(
    monkeypatch: pytest.MonkeyPatch, max_num_seqs: int
) -> None:
    from vllm_gr.engine import arg_utils_gr

    host = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs, async_scheduling=True),
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *args: host)
    monkeypatch.setattr(arg_utils_gr, "engine_config_service", lambda: legacy_config_service(None))
    assert arg_utils_gr.create_engine_config(SimpleNamespace()) is host
    assert host.scheduler_config.max_num_seqs == max_num_seqs
    assert GRConfig.from_vllm_config(host).deployment.beam_batch_sizes == (1, 2, 4)


@pytest.mark.parametrize(  # type: ignore[misc]
    "sizes",
    [[], [7, 3], [1, 1], [True, 3], [1, False], [1, 3.0], [1, "3"], [1, 0], [1, 9], "1,3,7"],
)
def test_invalid_beam_batch_sizes_fail_during_config_parsing(sizes: object, tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    raw["gr"].update({"beam_batch_sizes": sizes})
    with pytest.raises(ValueError, match="beam.batch.sizes"):
        gr_config_from_mapping(raw)


@pytest.mark.parametrize("max_num_seqs", [4, 7, 8])  # type: ignore[misc]
def test_engine_creation_preserves_beam_batch_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_num_seqs: int
) -> None:
    from vllm_gr.engine import arg_utils_gr

    host = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs, async_scheduling=True),
    )
    args = engine_args_from_file(tmp_path, {"gr": {"beam_batch_sizes": [1, 3, 7]}})
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *args: host)
    monkeypatch.setattr(arg_utils_gr, "engine_config_service", lambda: legacy_config_service(None))
    assert arg_utils_gr.create_engine_config(args) is host
    assert GRConfig.from_vllm_config(host).deployment.beam_batch_sizes == (1, 3, 7)
    GRConfig().write_to_vllm_config(host)
    assert host.scheduler_config.max_num_seqs == max_num_seqs
    assert GRConfig.from_vllm_config(host).deployment.beam_batch_sizes == (1, 2, 4)


def test_canonical_write_invalidates_cached_defaults_for_uniproc(tmp_path: Path) -> None:
    host = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
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
    beam = raw["gr"]
    assert isinstance(beam, dict)
    beam["worker_decision"] = worker_decision

    config = gr_config_from_mapping(raw)

    assert config.beam_worker_decision is worker_decision
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config


def test_canonical_mapping_defaults_constraint_backend_to_auto(tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    table_mapping = raw["gr"]["constraint_table"]
    assert isinstance(table_mapping, dict)
    del table_mapping["backend"]

    table = gr_config_from_mapping(raw).constraint_config.table

    assert table is not None
    assert table.backend is ConstraintBackendKind.AUTO


def test_canonical_writer_uses_primitive_mapping(tmp_path: Path) -> None:
    config = gr_config_from_mapping(
        _canonical_mapping(str(tmp_path / "table.safetensors"), enabled=False)
    )
    vllm_config = SimpleNamespace(additional_config={"unrelated": 1})
    stale = GRConfig.from_vllm_config(vllm_config)
    assert stale.beam_max_width is None

    vllm_config.additional_config["vllm_gr_backend_options"] = {"provider": "example"}
    write_canonical_gr_config(vllm_config, config)

    payload = vllm_config.additional_config[GR_CONFIG_KEY]
    assert json.loads(json.dumps(payload)) == payload
    assert GRConfig.from_vllm_config(vllm_config) == config
    assert vllm_config.additional_config["unrelated"] == 1
    assert vllm_config.additional_config["vllm_gr_backend_options"] == {"provider": "example"}


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
        raw["gr"][key] = 1
    else:
        raw["gr"]["constraint_table"][key] = 1
    with pytest.raises(ValueError, match="(?i)unknown"):
        gr_config_from_mapping(raw)


def test_yaml_path_is_resolved_relative_to_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "gr.yaml"
    config_path.write_text(
        """gr:
  beam_graph_enabled: true
  beam_width:
  - 32
  - 32
  - 32
  max_decode_steps: 3
  constraint_mode: table
  constraint_table:
    backend: cpu
    path: ../artifacts/table.safetensors
    format: constraint_table_v1
    digests:
      artifact: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      tokenizer: sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    max_top_k: 128
  candidate_width:
  - 32
  - 32
  - 32
""",
        encoding="utf-8",
    )

    config = load_gr_config(config_path)

    table = config.constraint_config.table
    assert table is not None
    assert table.path == str((tmp_path / "artifacts/table.safetensors").resolve())


def test_mapping_rejects_relative_artifact_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        gr_config_from_mapping(_canonical_mapping("table.safetensors"))


def test_mapping_base_dir_resolves_relative_artifact_path(tmp_path: Path) -> None:
    config = gr_config_from_mapping(_canonical_mapping("table.safetensors"), base_dir=tmp_path)
    table = config.constraint_config.table
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
    with pytest.raises(ValueError, match="Removed vLLM-GR"):
        parse_gr_config(vllm_config)


def test_constraint_table_requires_complete_enabled_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact_digest"):
        ConstraintTableSpec(
            max_top_k=128,
            enabled=True,
            path=str(tmp_path / "table.safetensors"),
        )


def test_canonical_mapping_requires_explicit_max_top_k(tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    table = raw["gr"]["constraint_table"]
    assert isinstance(table, dict)
    del table["max_top_k"]

    with pytest.raises(ValueError, match=r"gr\.constraint_table\.max_top_k"):
        gr_config_from_mapping(raw)


def test_constraint_max_top_k_has_no_npu_upper_bound(tmp_path: Path) -> None:
    config = ConstraintTableSpec(
        enabled=True,
        backend=ConstraintBackendKind.NPU,
        path=str(tmp_path / "table.safetensors"),
        artifact_digest=_ARTIFACT_DIGEST,
        tokenizer_digest=_TOKENIZER_DIGEST,
        max_top_k=257,
    )
    assert config.max_top_k == 257


@pytest.mark.parametrize("batch_size", [1, 2, 4])  # type: ignore[misc]
def test_engine_creation_preserves_flat_graph_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    from vllm_gr.engine import arg_utils_gr

    config = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    args = engine_args_from_file(tmp_path, {"gr": {"decode_graph_batch_size": batch_size}})
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *args: config)
    monkeypatch.setattr(arg_utils_gr, "engine_config_service", lambda: legacy_config_service(None))

    if not hasattr(config, "scheduler_config"):
        config.scheduler_config = SimpleNamespace(async_scheduling=True)
    assert arg_utils_gr.create_engine_config(args) is config
    assert GRConfig.from_vllm_config(config).beam_graph_batch_size == batch_size


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("use_v2", [False, True])  # type: ignore[misc]
def test_canonical_inputs_produce_identical_worker_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_v2: bool
) -> None:
    from vllm_gr.engine import arg_utils_gr

    raw = {
        "gr": {
            "beam_width": [128, 128, 128],
            "candidate_width": [128, 128, 128],
            "max_decode_steps": 3,
            "worker_decision": True,
            "constraint_mode": "none",
            "constraint_table": None,
        }
    }
    original = json.dumps(raw, sort_keys=True)
    monkeypatch.setattr(
        arg_utils_gr, "engine_config_service", lambda: legacy_config_service("cuda")
    )
    outputs = []
    for from_file in (False, True):
        host = SimpleNamespace(
            additional_config={"unrelated": 1},
            scheduler_config=SimpleNamespace(async_scheduling=True),
            use_v2_model_runner=use_v2,
        )
        monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *args: host)
        args = engine_args_from_file(tmp_path, raw) if from_file else SimpleNamespace(gr=raw["gr"])
        arg_utils_gr.create_engine_config(args)
        assert host.additional_config["unrelated"] == 1
        assert json.dumps(raw, sort_keys=True) == original
        outputs.append(json.dumps(host.additional_config, sort_keys=True))
        # Workers reconstruct and reserialize the normalized transport snapshot.
        worker = SimpleNamespace(additional_config=json.loads(outputs[-1]))
        config = GRConfig.from_vllm_config(worker)
        assert config.beam_max_width == 128 and config.beam_worker_decision is True
        config.write_to_vllm_config(worker)
        assert json.dumps(worker.additional_config, sort_keys=True) == outputs[-1]
    assert outputs[0] == outputs[1]


def test_engine_creation_writes_enabled_canonical_config(
    tmp_path: Path,
) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    engine_args = engine_args_from_file(
        tmp_path, _canonical_mapping(str(tmp_path / "table.safetensors"))
    )
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.engine_config_service",
            return_value=legacy_config_service(None),
        ),
    ):
        assert create_engine_config(engine_args) is vllm_config
    assert gr_config_from_mapping(vllm_config.additional_config[GR_CONFIG_KEY]).beam_max_width == 32

    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config", return_value=vllm_config
        ),
        pytest.raises(ValueError, match="Supply GR settings"),
    ):
        create_engine_config(engine_args)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("explicit_defaults", [False, True], ids=["sparse", "explicit-defaults"])  # type: ignore[misc]
def test_canonical_input_paths_share_serialization_and_compile_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_defaults: bool
) -> None:
    from vllm.config import VllmConfig
    from vllm.config.kernel import IrOpPriorityConfig

    from vllm_gr.engine import arg_utils_gr

    raw = {
        "gr": {
            "beam_graph_enabled": True,
            "beam_width": [128] * 3,
            "max_decode_steps": 3,
            "worker_decision": True,
            "constraint_mode": "none",
            "constraint_table": None,
            "candidate_width": [128] * 3,
        },
        "attention-config": {"backend": "CUSTOM"},
    }
    expected = gr_config_from_mapping(raw)
    if explicit_defaults:
        raw = gr_config_to_mapping(expected)
    original_raw = deepcopy(raw)
    unrelated = {"enabled": True}
    offline = VllmConfig(additional_config={"other_plugin": unrelated})
    online = VllmConfig(additional_config={"other_plugin": unrelated})
    # Isolate configuration normalization from accelerator startup policy.
    monkeypatch.setattr(arg_utils_gr, "engine_config_service", lambda: legacy_config_service(None))
    for host, engine_args in (
        (offline, engine_args_from_file(tmp_path, raw)),
        (online, engine_args_from_file(tmp_path, gr_config_to_mapping(expected))),
    ):
        # Replace platform defaults after VllmConfig initialization so hashing
        # needs only native IR providers, without loading device kernels/models.
        host.kernel_config.ir_op_priority = IrOpPriorityConfig(
            rms_norm=["native"],
            fused_add_rms_norm=["native"],
            gelu_and_mul_sparse=["native"],
        )
        # Both entry paths may already have a cached parse before writeback.
        GRConfig.from_vllm_config(host)
        monkeypatch.setattr(
            arg_utils_gr, "origin_create_engine_config", mock.Mock(return_value=host)
        )
        host.scheduler_config.async_scheduling = True
        host.scheduler_config.enable_chunked_prefill = False
        assert arg_utils_gr.create_engine_config(engine_args) is host
        assert GRConfig.from_vllm_config(host) == expected
        assert isinstance(host.additional_config, dict)
        assert host.additional_config["other_plugin"] is unrelated

    assert raw == original_raw
    assert offline.additional_config == online.additional_config
    assert offline.compute_hash() == online.compute_hash()


@pytest.mark.parametrize("device", ["cuda", "npu"])  # type: ignore[misc]
def test_benchmark_server_command_uses_canonical_attention_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    import huggingface_hub.constants
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from benchmarks.open_one_rec.tasks.recommendation.config import VIDEO_CONFIG
    from vllm_gr.engine import arg_utils_gr
    from vllm_gr.entrypoints.cli.main import _add_mode_option, _mode_arguments
    from vllm_gr.entrypoints.cli.serve import GRServeSubcommand
    from vllm_gr.entrypoints.cli.serve_config import GRArgumentParser

    table = _canonical_mapping(str(tmp_path / "table.safetensors"))["gr"]["constraint_table"]
    table["backend"] = device
    overrides = {
        "constraint_table": table,
        "beam_width": [128] * 3,
        "max_decode_steps": 3,
        "beam_graph_enabled": device == "cuda",
        "worker_decision": True,
        "candidate_width": [128] * 3,
    }

    # Execute the actual server command with an argument recorder instead of
    # launching a model. This catches legacy flags left behind in the script.
    script = (TESTS_DIR / "system/run_benchmark_and_verify.sh").read_text()
    command = script.split('echo "Starting OneRec online server"\n', 1)[1]
    command = command.split("\nSERVER_PID=$!", 1)[0]
    captured = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            r"""python() { printf '%s\0' "$@"; }
"""
            + command
            + "\nwait",
        ],
        env={
            **os.environ,
            "gr_overrides": json.dumps(overrides),
            "run_port": "7000",
            "MODEL_PATH": str(tmp_path),
            "SCRIPT_DIR": str(TESTS_DIR / "system"),
        },
        capture_output=True,
        check=True,
    )
    monkeypatch.setattr(EngineArgs, "add_cli_args", staticmethod(arg_utils_gr.add_cli_args))
    parser = GRArgumentParser()
    serve = GRServeSubcommand()
    child = serve.subparser_init(parser.add_subparsers(dest="subparser"))
    _add_mode_option(child)
    argv = captured.stdout.decode().split("\0")[:-1]
    assert argv[:2] == ["-m", "vllm_gr.entrypoints.cli.main"]
    selected, serve_argv = _mode_arguments(argv[2:])
    assert selected
    args = parser.parse_args(serve_argv)
    template_name = VIDEO_CONFIG["prompt_config"]["custom_chat_template"]
    offline_template = REPO_ROOT / "benchmarks/open_one_rec" / template_name
    assert Path(args.chat_template).samefile(offline_template)
    serve.validate(args)
    # ServeSubcommand.cmd normally normalizes the positional model before
    # constructing EngineArgs. Keep this config-only test local and offline.
    args.model = args.model_tag
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    engine_args = arg_utils_gr.from_cli_args(EngineArgs, args)
    assert engine_args.model == str(tmp_path)
    config = SimpleNamespace(
        additional_config={},
        attention_config=engine_args.attention_config,
        scheduler_config=SimpleNamespace(
            async_scheduling=args.async_scheduling,
            enable_chunked_prefill=args.enable_chunked_prefill,
        ),
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: config)
    monkeypatch.setattr(
        arg_utils_gr, "engine_config_service", lambda: legacy_config_service(device)
    )

    result = arg_utils_gr.create_engine_config(engine_args)

    assert result.attention_config.backend is AttentionBackendEnum.CUSTOM
    # Suffix fusion must leave hardware-dependent prefix selection automatic.
    assert result.attention_config.flash_attn_version is None
    assert result.additional_config[GR_CONFIG_KEY].get("attention_backend") is None
    assert result.additional_config[GR_CONFIG_KEY]["gr"]["constraint_table"]["backend"] == device


@pytest.mark.parametrize("async_scheduling", [False, True])  # type: ignore[misc]
def test_engine_creation_serializes_cuda_decision_defaults(
    tmp_path: Path,
    async_scheduling: bool,
) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    engine_args = engine_args_from_file(tmp_path, raw)

    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.engine_config_service",
            return_value=legacy_config_service("cuda"),
        ),
    ):
        assert create_engine_config(engine_args) is vllm_config

    payload = vllm_config.additional_config[GR_CONFIG_KEY]
    assert payload["gr"]["worker_decision"] is True
    assert "engine_driven" not in payload["gr"]
    assert parse_gr_config(vllm_config).beam_worker_decision is True


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("enabled", [False, True])  # type: ignore[misc]
def test_adapter_cache_switch_round_trip(enabled: bool, tmp_path: Path) -> None:
    raw = _canonical_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["gr"]
    assert isinstance(beam, dict)
    if enabled:
        beam["adapter_cache_enabled"] = True
    config = gr_config_from_mapping(raw)
    assert config.beam_adapter_cache_enabled is enabled
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config
    host = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    config = GRConfig(beam_adapter_cache_enabled=enabled)
    config.write_to_vllm_config(host)
    assert GRConfig.from_vllm_config(host).beam_adapter_cache_enabled is enabled


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])  # type: ignore[misc]
@pytest.mark.parametrize("source", ["python", "canonical"])  # type: ignore[misc]
def test_adapter_cache_switch_requires_boolean(value: object, source: str) -> None:
    with pytest.raises(ValueError, match="adapter_cache_enabled must be a boolean"):
        if source == "python":
            GRConfig(beam_adapter_cache_enabled=value)
        else:
            gr_config_from_mapping({"gr": {"adapter_cache_enabled": value}})


@pytest.mark.l1  # type: ignore[misc]
def test_engine_cli_no_longer_exposes_legacy_schema_v1_flags() -> None:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from vllm_gr.engine import arg_utils_gr

    parser = FlexibleArgumentParser()
    with mock.patch.object(arg_utils_gr, "origin_add_cli_args", return_value=parser):
        arg_utils_gr.add_cli_args(parser)

    for flag in (
        "--catalog-path",
        "--constraint-backend",
        "--constrained-generation",
        "--no-constrained-generation",
        "--beam-engine-driven",
        "--no-beam-engine-driven",
    ):
        assert flag not in parser._option_string_actions


def _parser(monkeypatch: pytest.MonkeyPatch) -> Any:
    from vllm.engine.arg_utils import EngineArgs

    from vllm_gr.engine import arg_utils_gr
    from vllm_gr.entrypoints.cli.serve import GRServeSubcommand
    from vllm_gr.entrypoints.cli.serve_config import GRArgumentParser

    monkeypatch.setattr(EngineArgs, "add_cli_args", staticmethod(arg_utils_gr.add_cli_args))
    monkeypatch.setattr(EngineArgs, "__init__", arg_utils_gr._init_engine_args)
    parser = GRArgumentParser()
    from vllm_gr.entrypoints.cli.main import _add_mode_option

    child = GRServeSubcommand().subparser_init(parser.add_subparsers(dest="subparser"))
    _add_mode_option(child)
    return parser


def _serve_engine_args(monkeypatch: pytest.MonkeyPatch, args: Any) -> Any:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.cli import serve

    from vllm_gr.engine import arg_utils_gr
    from vllm_gr.entrypoints.cli.serve import GRServeSubcommand

    captured: list[Any] = []

    async def run_server(parsed: Any) -> None:
        captured.append(arg_utils_gr.from_cli_args(EngineArgs, parsed))

    # Native serve resolves model_tag before constructing EngineArgs.
    with monkeypatch.context() as patch:
        patch.setattr(serve, "run_server", run_server)
        GRServeSubcommand.cmd(args)
    assert len(captured) == 1
    return captured[0]


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("value, expected", [("4k", 4000), ("auto", -1), ("null", None)])  # type: ignore[misc]
def test_native_yaml_types_match_cli_and_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, expected: int | None
) -> None:
    from vllm.engine.arg_utils import EngineArgs

    path = tmp_path / "engine.yaml"
    path.write_text(
        f"max-model-len: {value}\nenable-chunked-prefill: false\nignore-patterns: []\ngr: {{}}\n"
    )
    parser = _parser(monkeypatch)
    cli = parser.parse_args(["serve", str(tmp_path), "--config", str(path)])
    args = EngineArgs(model=str(tmp_path), config=path)
    assert cli.max_model_len == args.max_model_len == expected
    assert cli.enable_chunked_prefill is args.enable_chunked_prefill is False
    assert cli.ignore_patterns == args.ignore_patterns == []
    explicit = EngineArgs(model=str(tmp_path), config=path, max_model_len=2048)
    cli_explicit = parser.parse_args(
        ["serve", str(tmp_path), "--config", str(path), "--max-model-len", "2048"]
    )
    assert explicit.max_model_len == cli_explicit.max_model_len == 2048
    # A parser may serve multiple invocations without retaining a prior YAML.
    fresh = parser.parse_args(["serve", str(tmp_path)])
    assert fresh.max_model_len == 8192
    from_cli = _serve_engine_args(monkeypatch, cli)
    assert from_cli.max_model_len == expected
    assert from_cli.model == str(tmp_path)


@pytest.mark.l1  # type: ignore[misc]
def test_import_null_clears_value_and_invalid_native_value_names_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.engine.arg_utils import EngineArgs

    base = tmp_path / "base.yaml"
    base.write_text("max-model-len: 4k\n")
    path = tmp_path / "engine.yaml"
    path.write_text("import: ./base.yaml\nmax-model-len: null\ngr: {}\n")
    parser = _parser(monkeypatch)
    assert EngineArgs(model=str(tmp_path), config=path).max_model_len is None
    assert parser.parse_args(["serve", str(tmp_path), "--config", str(path)]).max_model_len is None
    path.write_text("max-model-len: invalid\ngr: {}\n")
    with pytest.raises(ValueError, match="max-model-len"):
        EngineArgs(model=str(tmp_path), config=path)
    with pytest.raises(ValueError, match="max-model-len"):
        parser.parse_args(["serve", str(tmp_path), "--config", str(path)])


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("explicit", [False, True])  # type: ignore[misc]
def test_cli_budget_provenance_preserves_explicit_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    from vllm.engine.arg_utils import EngineArgs

    parser = _parser(monkeypatch)
    argv = ["serve", str(tmp_path), "--max-model-len", "16384", "--no-enable-chunked-prefill"]
    kwargs = {"max_num_batched_tokens": 8192} if explicit else {}
    if explicit:
        argv += ["--max-num-batched-tokens", "8192"]
    cli = _serve_engine_args(monkeypatch, parser.parse_args(argv))
    assert cli.model == str(tmp_path)
    python = EngineArgs(
        model=str(tmp_path), gr={}, max_model_len=16384, enable_chunked_prefill=False, **kwargs
    )
    assert (
        cli.max_num_batched_tokens == python.max_num_batched_tokens == (8192 if explicit else 16384)
    )


@pytest.mark.l1  # type: ignore[misc]
def test_native_engine_does_not_inherit_gr_defaults_or_parallelism_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.engine.arg_utils import EngineArgs

    from vllm_gr.engine import arg_utils_gr

    monkeypatch.setattr(EngineArgs, "__init__", arg_utils_gr._init_engine_args)
    monkeypatch.setattr(
        arg_utils_gr, "engine_config_service", lambda: legacy_config_service("cuda")
    )

    def native_config(args: Any, *unused: Any) -> Any:
        return SimpleNamespace(
            additional_config={},
            scheduler_config=SimpleNamespace(async_scheduling=True, max_num_seqs=8),
            parallel_config=SimpleNamespace(tensor_parallel_size=args.tensor_parallel_size),
        )

    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", native_config)
    native = EngineArgs(model=str(tmp_path), tensor_parallel_size=2)
    assert not native._gr_selected
    with mock.patch.object(
        arg_utils_gr, "resolve_native_settings", wraps=arg_utils_gr.resolve_native_settings
    ) as resolve:
        native_host = arg_utils_gr.create_engine_config(native)
        resolve.assert_not_called()
    assert native_host.additional_config == {}
    assert GRConfig.from_vllm_config(native_host).beam_max_width is None
    gr = EngineArgs(model=str(tmp_path), tensor_parallel_size=2, gr={})
    assert gr._gr_selected and gr.max_model_len == 8192
    with pytest.raises(ValueError, match="TP/PP/DP/DCP/PCP=1"):
        arg_utils_gr.create_engine_config(gr)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(
    "overrides",
    [
        ["--gr-config", '{"result_width":2}'],
        ["--gr.result_width", "2"],
    ],
)  # type: ignore[misc]
def test_gr_dispatcher_separates_mode_and_json_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: list[str]
) -> None:
    import sys

    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.cli import serve as native_serve

    from vllm_gr.engine import arg_utils_gr
    from vllm_gr.entrypoints.cli import main
    from vllm_gr.entrypoints.cli.serve import GRServeSubcommand

    captured: list[Any] = []

    async def capture_server(args: Any) -> None:
        captured.append(args)

    monkeypatch.setattr(EngineArgs, "add_cli_args", staticmethod(arg_utils_gr.add_cli_args))
    # Keep ServeSubcommand.cmd's positional model handling in the test.
    monkeypatch.setattr(native_serve, "run_server", capture_server)
    monkeypatch.setattr(GRServeSubcommand, "validate", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", str(tmp_path), "--gr", *overrides])
    main.dispatch_main()
    assert len(captured) == 1
    assert captured[0]._gr_mode is True
    assert captured[0].gr == [{"result_width": 2}]
    args = arg_utils_gr.from_cli_args(EngineArgs, captured[0])
    assert args.model == str(tmp_path)
    assert args._gr_selected and args.gr == {"result_width": 2}


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("worker_decision", [None, False, True])  # type: ignore[misc]
def test_npu_sid_stage_vocab_resolves_auto_before_validation(
    worker_decision: bool | None,
) -> None:
    from vllm_gr.engine import arg_utils_gr

    defaults = arg_utils_gr._beam_worker_decision_default(
        legacy_config_service("npu"), beam_width_configured=True, worker_decision=None
    )
    settings = {"sid_stage_vocab": True}
    if worker_decision is not None:
        settings["worker_decision"] = worker_decision
    if worker_decision is False:
        with pytest.raises(ValueError, match="worker_decision"):
            arg_utils_gr.load_engine_gr_config(None, {}, settings, worker_decision_default=defaults)
    else:
        config = arg_utils_gr.load_engine_gr_config(
            None, {}, settings, worker_decision_default=defaults
        )
        assert config.beam_sid_stage_vocab and config.beam_worker_decision is True


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("explicit", [False, True])  # type: ignore[misc]
def test_grllm_resolves_native_settings_once_with_explicit_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    from vllm.engine.arg_utils import EngineArgs

    from vllm_gr import plugin
    from vllm_gr.engine import arg_utils_gr
    from vllm_gr.entrypoints import gr as frontend

    path = tmp_path / "engine.yaml"
    path.write_text(f"model: {tmp_path}\nmax-model-len: 4k\ndtype: bfloat16\ngr: {{}}\n")
    captured: list[Any] = []

    def native_llm(self: Any, model: str, dtype: str = "auto", **kwargs: Any) -> None:
        captured.append(EngineArgs(model=model, dtype=dtype, **kwargs))
        self.llm_engine = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))

    monkeypatch.setattr(EngineArgs, "__init__", arg_utils_gr._init_engine_args)
    monkeypatch.setattr(frontend.LLM, "__init__", native_llm)
    monkeypatch.setattr(frontend, "_install_renderer_eos_token_id_cache", lambda: None)
    monkeypatch.setattr(plugin, "initialize_runtime", lambda: None)
    with mock.patch.object(
        arg_utils_gr, "resolve_native_settings", wraps=arg_utils_gr.resolve_native_settings
    ) as resolve:
        frontend.GRLLM(config=path, **({"max_model_len": 6144} if explicit else {}))
        assert resolve.call_count == 1
    assert captured[0].model == str(tmp_path)
    assert captured[0].dtype == "bfloat16"
    assert captured[0].max_model_len == (6144 if explicit else 4000)


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _isolate_backend_plan_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise configuration policy independently of a native backend install."""
    from vllm_gr.platforms import planning

    monkeypatch.setattr(planning, "admit_backend_plan", lambda config: None)
