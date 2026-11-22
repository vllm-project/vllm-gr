# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Public configuration, startup composition, and worker snapshot regressions."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.helpers.gr_config import table_constraint
from vllm_gr.config import (
    ConstraintConfig,
    GRConfig,
    gr_config_from_mapping,
    gr_config_to_mapping,
    load_gr_config,
)
from vllm_gr.config.models import DeploymentOptions
from vllm_gr.config.parser import (
    CAPACITY_FIELDS,
    COMMON_ENGINE_CONFIG,
    CONFIG_ATTRIBUTES,
    OPTION_KEYS,
    load_engine_gr_config,
    read_settings,
)
from vllm_gr.platforms import planning
from vllm_gr.platforms.config import BackendConfigService

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize("backend", ["cuda", "npu"])  # type: ignore[misc]
def test_decode_gate_configuration_reaches_public_loader(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = Path(__file__).parents[1] / "system" / "run_benchmark_and_verify.sh"
    producer = script.read_text().split("<<'PY_CONFIG'\n", 1)[1].split("\nPY_CONFIG", 1)[0]
    digests = ("sha256:" + "a" * 64, "sha256:" + "b" * 64)
    monkeypatch.setitem(
        sys.modules,
        "vllm_gr.constraints.artifact_builder",
        SimpleNamespace(build_single_trie_artifact=lambda *args: digests),
    )
    artifact = tmp_path / "table.safetensors"
    monkeypatch.setattr(
        sys,
        "argv",
        ["gate", str(tmp_path / "model"), str(tmp_path / "triples"), str(artifact), backend, "128"],
    )
    exec(compile(producer, str(script), "exec"), {"__name__": "__main__"})
    config = load_engine_gr_config(None, {}, json.loads(capsys.readouterr().out))
    assert config.constraint is not None and config.constraint.table is not None
    assert config.constraint.table.backend.value == backend
    assert config.constraint.table.artifact_digest == digests[0]
    assert config.constraint.table.tokenizer_digest == digests[1]
    assert config.beam_max_width == 128
    assert config.deployment.candidate_width == (128, 128, 128)
    assert config.beam_graph_enabled is (backend == "cuda")


def test_default_document_declares_every_public_option() -> None:
    settings = read_settings(COMMON_ENGINE_CONFIG)["gr"]
    # The packaged schedules determine both capacities.
    assert set(settings) == (set(CONFIG_ATTRIBUTES) - set(CAPACITY_FIELDS)) | OPTION_KEYS | {
        "constraint_mode",
        "constraint_table",
    }
    config = load_gr_config(COMMON_ENGINE_CONFIG)
    assert gr_config_to_mapping(config)["gr"] == settings


def test_performance_gate_emits_server_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.beam_search import measure_latency
    from tools.constraints import build_constraint_table

    digests = ("sha256:" + "a" * 64, "sha256:" + "b" * 64)
    monkeypatch.setattr(
        build_constraint_table, "build_constraint_table", lambda **kwargs: (*digests, None)
    )
    triples = tmp_path / "triples.json"
    triples.write_text("[]", encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Recommend the next video.", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "beam-performance",
            "prepare",
            "--model",
            str(tmp_path / "model"),
            "--triples",
            str(triples),
            "--prompt",
            str(prompt),
            "--output-dir",
            str(tmp_path),
            "--ab-prefill-grid",
        ],
    )
    measure_latency.prepare(measure_latency.parse_args())
    config_path = tmp_path / "gr-config.yaml"
    config = load_gr_config(config_path)
    assert config.attention_backend is not None and config.attention_backend.value == "CUSTOM"
    assert config.constraint_config.table is not None
    assert config.constraint_config.table.artifact_digest == digests[0]
    assert config.constraint_config.table.tokenizer_digest == digests[1]
    assert config.deployment.prefill_token_buckets == (1024, 2048, 4096)
    assert (
        read_settings(config_path)["gr"] == json.loads((tmp_path / "setup.json").read_text())["gr"]
    )


def test_public_snapshot_round_trip_without_file_io(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_gr_config(COMMON_ENGINE_CONFIG)
    mapping = gr_config_to_mapping(config)
    assert set(mapping) == {"gr"}
    assert not {"beam", "deployment", "schema_version"} & mapping.keys()
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("worker reopened configuration"))
    restored = gr_config_from_mapping(json.loads(json.dumps(mapping)))
    assert restored == config


@pytest.mark.parametrize(  # type: ignore[misc]
    "config,mode,enabled",
    [
        (GRConfig(), None, False),
        (GRConfig(beam_worker_decision=True), None, True),
        (GRConfig(constraint=ConstraintConfig()), "none", True),
        (GRConfig(constraint=table_constraint()), "table", True),
    ],
)
def test_driver_and_worker_preserve_constraint_state(
    config: GRConfig, mode: str | None, enabled: bool
) -> None:
    driver = SimpleNamespace(additional_config={})
    config.write_to_vllm_config(driver)
    wire = json.loads(json.dumps(driver.additional_config))
    cached = GRConfig.from_vllm_config(driver)
    worker = GRConfig.from_vllm_config(SimpleNamespace(additional_config=wire))

    assert cached is config
    assert worker.constraint == cached.constraint
    assert worker.enabled is cached.enabled is enabled
    gr = wire["vllm_gr"]["gr"]
    if mode is None:
        assert "constraint_mode" not in gr
        assert "constraint_table" not in gr
    else:
        assert gr["constraint_mode"] == mode


@pytest.mark.parametrize(  # type: ignore[misc]
    "config",
    [
        GRConfig(beam_max_width=8),
        GRConfig(beam_candidate_width=4),
        GRConfig(beam_max_width=8, beam_candidate_width=4, constraint=table_constraint()),
        GRConfig(deployment=DeploymentOptions(beam_width=(8, 4, 2), candidate_width=(8, 4, 2))),
        GRConfig(deployment=DeploymentOptions(beam_width=(8, 16, 16), candidate_width=(8, 4, 4))),
    ],
)
def test_resolved_widths_survive_worker_transport(config: GRConfig) -> None:
    driver = SimpleNamespace(additional_config={})
    config.write_to_vllm_config(driver)
    wire = json.loads(json.dumps(driver.additional_config))
    worker = GRConfig.from_vllm_config(SimpleNamespace(additional_config=wire))
    assert worker == GRConfig.from_vllm_config(driver) == config
    assert worker.beam_graph_active == config.beam_graph_active
    assert worker.enabled == config.enabled
    assert gr_config_to_mapping(worker) == wire["vllm_gr"]
    if config.deployment.beam_width:
        assert config.beam_max_width == max(config.deployment.beam_width)
    else:
        assert wire["vllm_gr"]["gr"]["beam_width"] is None
        assert wire["vllm_gr"]["gr"]["candidate_width"] is None


@pytest.mark.parametrize(  # type: ignore[misc]
    "kwargs, message",
    [
        ({"admin_port": 70000}, "admin_port"),
        ({"beam_width": (8, 4)}, "candidate_width"),
        ({"prefill_token_buckets": (256, 128)}, "strictly increasing"),
        ({"prefix_calibration_path": "relative.yaml"}, "absolute"),
    ],
)
def test_python_deployment_rejects_invalid_settings(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DeploymentOptions(**kwargs)


@pytest.mark.parametrize("capacity", [{"beam_max_width": 16}, {"beam_candidate_width": 4}])  # type: ignore[misc]
def test_python_capacity_cannot_override_step_widths(capacity: dict[str, int]) -> None:
    deployment = DeploymentOptions(beam_width=(8, 4), candidate_width=(8, 2))
    with pytest.raises(ValueError, match="conflicts"):
        GRConfig(deployment=deployment, **capacity)


def test_capacity_only_snapshot_can_be_loaded_as_startup_yaml(tmp_path: Path) -> None:
    config = GRConfig(beam_max_width=8, beam_candidate_width=4, constraint=ConstraintConfig())
    path = tmp_path / "startup.yaml"
    path.write_text(json.dumps(gr_config_to_mapping(config)))
    assert load_gr_config(path) == config


def test_public_mapping_pins_symlink_targets_before_worker_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "A", tmp_path / "B"
    for directory in (first, second):
        directory.mkdir()
        for name in ("table", "calibration"):
            (directory / name).write_text(directory.name)
    current = tmp_path / "current"
    current.symlink_to(first, target_is_directory=True)
    spec = table_constraint()
    assert spec.table is not None
    mapping = gr_config_to_mapping(
        GRConfig(
            constraint=replace(spec, table=replace(spec.table, path=str(current / "table"))),
            deployment=DeploymentOptions(prefix_calibration_path=str(current / "calibration")),
        )
    )
    driver = gr_config_from_mapping(mapping)
    host = SimpleNamespace(additional_config={})
    driver.write_to_vllm_config(host)
    snapshot = json.loads(json.dumps(host.additional_config))
    current.unlink()
    current.symlink_to(second, target_is_directory=True)
    monkeypatch.setattr(Path, "resolve", lambda *a, **k: pytest.fail("worker resolved a path"))
    worker = GRConfig.from_vllm_config(SimpleNamespace(additional_config=snapshot))
    assert driver == worker
    assert worker.constraint is not None and worker.constraint.table is not None
    assert worker.constraint.table.path == str(first / "table")
    assert worker.deployment.prefix_calibration_path == str(first / "calibration")
    assert Path(worker.constraint.table.path).read_text() == "A"
    assert Path(worker.deployment.prefix_calibration_path).read_text() == "A"


def test_worker_uses_driver_paths_and_frozen_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = table_constraint()
    assert table.table is not None
    path = str(tmp_path / "dir" / ".." / "artifact")
    batches = [1, 4]
    config = GRConfig(
        constraint=replace(table, table=replace(table.table, path=path)),
        deployment=DeploymentOptions.from_mapping(
            {"beam_batch_sizes": batches, "prefix_calibration_path": path}
        ),
    )
    batches.clear()
    assert config.deployment.beam_batch_sizes == (1, 4)
    assert config.deployment.prefix_calibration_path == str(tmp_path / "artifact")
    driver = SimpleNamespace(additional_config={})
    config.write_to_vllm_config(driver)
    worker = SimpleNamespace(additional_config=json.loads(json.dumps(driver.additional_config)))
    monkeypatch.setattr(Path, "resolve", lambda *a, **k: pytest.fail("worker resolved a path"))
    assert GRConfig.from_vllm_config(worker) == config


@pytest.mark.parametrize("key", ["execution_mode", "dynamic_beam_widths", "per_parent_beam_widths"])  # type: ignore[misc]
def test_retired_configuration_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="Unknown gr settings"):
        gr_config_from_mapping({"gr": {key: None}})


@pytest.mark.parametrize(  # type: ignore[misc]
    "document",
    [
        {"schema_version": 2, "beam": {"max_width": 8}},
        {"beam": {"max_width": 8}},
        {"deployment": {"beam_batch_sizes": [1]}},
    ],
)
def test_other_configuration_structures_are_rejected(document: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Unknown configuration"):
        gr_config_from_mapping(document)


def test_step_widths_and_capacity(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "gr:\n  beam_width: [8, 4, 2]\n  candidate_width: [8, 4, 2]\n  max_decode_steps: 5\n  worker_decision: true\n  result_width: 1\n"
    )
    config = load_gr_config(path)
    assert config.beam_max_width == 8
    assert config.beam_max_decode_steps == 5
    assert config.deployment.beam_width == (8, 4, 2)
    assert config.deployment.candidate_width == (8, 4, 2)
    assert config.deployment.result_width == 1
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config


@pytest.mark.parametrize(  # type: ignore[misc]
    ("settings", "message"),
    [
        ({"beam_width": [4, 4], "candidate_width": [4]}, "one entry"),
        ({"beam_width": [4], "candidate_width": [8]}, "first gr.candidate_width"),
        (
            {"beam_width": [4, 4, 4], "candidate_width": [4, 4, 4], "max_decode_steps": 2},
            "max_decode_steps",
        ),
        ({"beam_width": [4], "candidate_width": [4], "result_width": 5}, "result_width"),
        ({"beam_width": 4, "candidate_width": 4}, "nonempty list"),
        ({"dense_prefill": "flash"}, "gr.dense_prefill"),
        ({"prefix_min_chain_us": -1}, "gr.prefix_min_chain_us"),
        ({"prefix_min_chain_us": float("nan")}, "gr.prefix_min_chain_us"),
    ],
)
def test_invalid_gr_settings_fail_with_field_names(
    settings: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        gr_config_from_mapping({"gr": settings})


def test_cli_and_python_apply_one_document_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.engine.arg_utils import EngineArgs

    from vllm_gr.engine import arg_utils_gr
    from vllm_gr.entrypoints.cli.serve import GRServeSubcommand
    from vllm_gr.entrypoints.cli.serve_config import GRArgumentParser

    model = tmp_path / "model"
    model.mkdir()
    path = tmp_path / "scenario.yaml"
    path.write_text(
        "max-model-len: 4096\ngr:\n  beam_width: [8, 8, 8]\n  candidate_width: [8, 8, 8]\n"
        "  dense_prefill: fa2\n  prefix_min_chain_us: 12\n"
    )
    args = EngineArgs.__new__(EngineArgs)
    with patch.object(arg_utils_gr, "read_settings", wraps=read_settings) as read:
        arg_utils_gr._init_engine_args(args, model=str(model), config=path)
        assert args.max_model_len == 4096
        assert read.call_count == 1
        path.unlink()
        config = arg_utils_gr.load_engine_gr_config(
            args.config, None, startup_document=args._gr_startup_document
        )
        assert config.beam_max_width == 8
        assert config.deployment.dense_prefill == "fa2"
        assert config.deployment.prefix_min_chain_us == 12
        assert read.call_count == 1

    path.write_text(
        "max-model-len: 4096\ngr:\n  beam_width: [8, 8, 8]\n  candidate_width: [8, 8, 8]\n"
    )
    monkeypatch.setattr(EngineArgs, "add_cli_args", staticmethod(arg_utils_gr.add_cli_args))
    parser = GRArgumentParser()
    GRServeSubcommand().subparser_init(parser.add_subparsers(dest="subparser"))
    cli = parser.parse_args(["serve", str(model), "--config", str(path)])
    assert cli.max_model_len == args.max_model_len


def test_python_explicit_arguments_override_file(tmp_path: Path) -> None:
    from vllm.engine.arg_utils import EngineArgs

    from vllm_gr.engine import arg_utils_gr

    model = tmp_path / "model"
    model.mkdir()
    path = tmp_path / "scenario.yaml"
    path.write_text("max-model-len: 4096\ngr: {}\n")
    args = EngineArgs.__new__(EngineArgs)
    arg_utils_gr._init_engine_args(args, model=str(model), config=path, max_model_len=2048)
    assert args.max_model_len == 2048


def test_imports_and_overrides_preserve_declaring_file_paths(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "gr:\n  prefix_calibration_path: ./calibration.yaml\n  beam_batch_sizes: [1, 2, 4]\n"
    )
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("import: ./base.yaml\ngr:\n  beam_batch_sizes: [1, 4]\n")
    config = load_gr_config(scenario, {"beam_batch_sizes": [1]})
    assert config.deployment.beam_batch_sizes == (1,)
    assert config.deployment.prefix_calibration_path == str(tmp_path / "calibration.yaml")


def test_startup_logs_the_complete_public_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.engine import arg_utils_gr

    host = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(max_num_seqs=8, async_scheduling=True),
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: host)
    monkeypatch.setattr(arg_utils_gr, "engine_config_service", BackendConfigService)
    monkeypatch.setattr(planning, "admit_backend_plan", lambda config: None)
    with patch.object(arg_utils_gr.logger, "info") as info:
        assert arg_utils_gr.create_engine_config(SimpleNamespace(_gr_selected=True)) is host
    info.assert_called_once()
    assert json.loads(info.call_args.args[1]) == gr_config_to_mapping(
        GRConfig.from_vllm_config(host)
    )


def test_native_attention_backend_reaches_worker_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_gr.config import GRAttentionBackend
    from vllm_gr.engine import arg_utils_gr

    host = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(max_num_seqs=8, async_scheduling=True),
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.CUSTOM),
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: host)
    monkeypatch.setattr(arg_utils_gr, "engine_config_service", BackendConfigService)
    monkeypatch.setattr(planning, "admit_backend_plan", lambda config: None)
    arg_utils_gr.create_engine_config(SimpleNamespace(_gr_selected=True))
    assert GRConfig.from_vllm_config(host).attention_backend is GRAttentionBackend.CUSTOM
