# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""GR settings through CLI, Python and native engine creation."""

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

from tests.helpers.gr_config import (
    OpaqueAdditionalConfig,
    config_host,
    engine_args_from_file,
    fixed_beam_mapping,
    prepared_engine_args,
)
from tests.helpers.paths import REPO_ROOT, TESTS_DIR
from vllm_gr.config import (
    ConstraintMode,
    GRConfig,
    gr_config_from_mapping,
    gr_config_to_mapping,
)
from vllm_gr.config.parser import GR_CONFIG_KEY, parse_gr_config
from vllm_gr.platforms.legacy import legacy_config_service

ATTENTION_BACKEND_KEY = "vllm_gr_attention_backend"

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _isolate_backend_plan_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise configuration policy independently of a native backend install."""
    from vllm_gr.platforms import planning

    monkeypatch.setattr(planning, "admit_backend_plan", lambda config: None)


def test_engine_config_entrypoint_preserves_unrelated_config(tmp_path: Path) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = config_host({"unrelated": True})
    engine_args = engine_args_from_file(
        tmp_path, {"gr": {"beam_width": None, "candidate_width": None}}
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=vllm_config,
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result).beam_max_width is None
    assert result.additional_config["unrelated"] is True


def test_native_cli_preserves_upstream_defaults_and_explicit_options() -> None:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.max_model_len is None
    assert defaults.max_num_batched_tokens is None
    assert defaults.enable_chunked_prefill is None
    assert defaults.enable_prefix_caching is None

    overridden = parser.parse_args(
        [
            "--max-model-len",
            "4096",
            "--max-num-batched-tokens",
            "4096",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
        ]
    )
    assert overridden.max_model_len == 4096
    assert overridden.max_num_batched_tokens == 4096
    assert overridden.enable_chunked_prefill is True
    assert overridden.enable_prefix_caching is True


def test_engine_config_preserves_opaque_additional_config_without_gr_overrides() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    additional_config = OpaqueAdditionalConfig()
    vllm_config = config_host(additional_config)
    engine_args = SimpleNamespace(
        attention_backend=None,
        attention_config=None,
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=vllm_config,
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result).beam_max_width is None
    assert result.additional_config is additional_config


def test_engine_config_rejects_invalid_raw_gr_value_without_cli_override() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    engine_args = SimpleNamespace(
        attention_backend=None,
        attention_config=None,
    )
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=config_host({ATTENTION_BACKEND_KEY: "INVALID"}),
        ),
        pytest.raises(ValueError, match="Removed vLLM-GR configuration keys"),
    ):
        engine_args.additional_config = {ATTENTION_BACKEND_KEY: "INVALID"}
        create_engine_config(engine_args)


def test_attention_selector_config_reader_propagates_validation_errors() -> None:
    from vllm_gr.v1.attention.selector_patch import _get_current_gr_attention_backend

    with (
        mock.patch(
            "vllm.config.get_current_vllm_config_or_none",
            return_value=config_host(
                {"vllm_gr": {"attention-config": {"backend": "INVALID"}, "gr": {}}}
            ),
        ),
        pytest.raises(ValueError, match="unsupported.*attention_backend"),
    ):
        _get_current_gr_attention_backend()


def test_npu_beam_without_catalog_stays_unconstrained(tmp_path: Path) -> None:
    """A catalog-free NPU deployment uses full-vocabulary generation."""
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = config_host()
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.engine_config_service",
            return_value=legacy_config_service("npu"),
        ),
    ):
        result = create_engine_config(
            engine_args_from_file(
                tmp_path, {"gr": {"beam_width": [128] * 3, "candidate_width": [128] * 3}}
            )
        )

    config = GRConfig.from_vllm_config(result)
    assert config.constraint_config.mode is ConstraintMode.NONE
    assert config.constraint_config.table is None


def test_cuda_beam_enables_worker_decision_by_default(tmp_path: Path) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    engine_args = engine_args_from_file(
        tmp_path, {"gr": {"beam_width": [128] * 3, "candidate_width": [128] * 3}}
    )
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=config_host(),
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.engine_config_service",
            return_value=legacy_config_service("cuda"),
        ),
    ):
        result = create_engine_config(engine_args)

    config = GRConfig.from_vllm_config(result)
    assert config.beam_worker_decision is True


@pytest.mark.parametrize("platform", ["cpu", "cuda", "npu"])  # type: ignore[misc]
@pytest.mark.parametrize("worker_decision", [None, False, True])  # type: ignore[misc]
def test_engine_config_preserves_explicit_decision_controls(
    tmp_path: Path,
    platform: str,
    worker_decision: bool | None,
) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    initial = GRConfig(
        beam_max_width=32,
        beam_worker_decision=worker_decision,
    )
    vllm_config = config_host()
    engine_args = engine_args_from_file(tmp_path, gr_config_to_mapping(initial))
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config", return_value=vllm_config
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.engine_config_service",
            return_value=legacy_config_service(platform),
        ),
    ):
        result = GRConfig.from_vllm_config(create_engine_config(engine_args))
    expected_worker = worker_decision
    if platform != "cpu" and expected_worker is None:
        expected_worker = True
    assert result.beam_worker_decision is expected_worker


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
        monkeypatch.setattr(
            arg_utils_gr, "origin_create_engine_config", mock.Mock(return_value=host)
        )
        args = (
            engine_args_from_file(tmp_path, raw)
            if from_file
            else SimpleNamespace(additional_config={"vllm_gr": raw})
        )
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
        tmp_path, fixed_beam_mapping(str(tmp_path / "table.safetensors"))
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
    ):
        assert create_engine_config(engine_args) is vllm_config


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

    table = fixed_beam_mapping(str(tmp_path / "table.safetensors"))["gr"]["constraint_table"]
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
    engine_args = EngineArgs.from_cli_args(serve.prepare_args(args)[0])
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
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
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


def test_serve_parser_omits_removed_flags() -> None:
    parser = _parser()._subparsers._group_actions[0].choices["serve"]

    for flag in (
        "--catalog-path",
        "--constraint-backend",
        "--constrained-generation",
        "--no-constrained-generation",
        "--beam-engine-driven",
        "--no-beam-engine-driven",
    ):
        assert flag not in parser._option_string_actions


def _parser() -> Any:
    from vllm_gr.entrypoints.cli.serve import GRServeSubcommand
    from vllm_gr.entrypoints.cli.serve_config import GRArgumentParser

    parser = GRArgumentParser()
    from vllm_gr.entrypoints.cli.main import _add_mode_option

    child = GRServeSubcommand().subparser_init(parser.add_subparsers(dest="subparser"))
    _add_mode_option(child)
    return parser


def _serve_engine_args(monkeypatch: pytest.MonkeyPatch, args: Any) -> Any:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.cli import serve

    from vllm_gr.entrypoints.cli.serve import GRServeSubcommand

    captured: list[Any] = []

    async def run_server(parsed: Any) -> None:
        captured.append(EngineArgs.from_cli_args(parsed))

    # Native serve resolves model_tag before constructing EngineArgs.
    with monkeypatch.context() as patch:
        patch.setattr(serve, "run_server", run_server)
        GRServeSubcommand.cmd(args)
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("value, expected", [("4k", 4000), ("auto", -1), ("null", None)])  # type: ignore[misc]
def test_native_yaml_types_match_cli_and_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, expected: int | None
) -> None:
    path = tmp_path / "engine.yaml"
    path.write_text(
        f"max-model-len: {value}\nenable-chunked-prefill: false\nignore-patterns: []\ngr: {{}}\n"
    )
    parser = _parser()
    cli = parser.parse_args(["serve", str(tmp_path), "--config", str(path)])
    args = prepared_engine_args(model=str(tmp_path), config=path)
    assert cli.max_model_len == args.max_model_len == expected
    assert cli.enable_chunked_prefill is args.enable_chunked_prefill is False
    assert cli.ignore_patterns == args.ignore_patterns == []
    explicit = prepared_engine_args(model=str(tmp_path), config=path, max_model_len=2048)
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


def test_import_null_clears_value_and_invalid_native_value_names_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("max-model-len: 4k\n")
    path = tmp_path / "engine.yaml"
    path.write_text("import: ./base.yaml\nmax-model-len: null\ngr: {}\n")
    parser = _parser()
    assert prepared_engine_args(model=str(tmp_path), config=path).max_model_len is None
    assert parser.parse_args(["serve", str(tmp_path), "--config", str(path)]).max_model_len is None
    path.write_text("max-model-len: invalid\ngr: {}\n")
    with pytest.raises(ValueError, match="max-model-len"):
        prepared_engine_args(model=str(tmp_path), config=path)
    with pytest.raises(ValueError, match="max-model-len"):
        parser.parse_args(["serve", str(tmp_path), "--config", str(path)])


@pytest.mark.parametrize("explicit", [False, True])  # type: ignore[misc]
def test_cli_budget_provenance_preserves_explicit_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    parser = _parser()
    argv = ["serve", str(tmp_path), "--max-model-len", "16384", "--no-enable-chunked-prefill"]
    kwargs: dict[str, Any] = {"max_num_batched_tokens": 8192} if explicit else {}
    if explicit:
        argv += ["--max-num-batched-tokens", "8192"]
    cli = _serve_engine_args(monkeypatch, parser.parse_args(argv))
    assert cli.model == str(tmp_path)
    python = prepared_engine_args(
        model=str(tmp_path), gr={}, max_model_len=16384, enable_chunked_prefill=False, **kwargs
    )
    assert (
        cli.max_num_batched_tokens == python.max_num_batched_tokens == (8192 if explicit else 16384)
    )


def test_native_engine_does_not_inherit_gr_defaults_or_parallelism_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.engine.arg_utils import EngineArgs

    from vllm_gr.engine import arg_utils_gr

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
    assert not native.additional_config
    from vllm_gr.engine import startup

    with mock.patch.object(
        startup, "resolve_native_settings", wraps=startup.resolve_native_settings
    ) as resolve:
        native_host = arg_utils_gr.create_engine_config(native)
        resolve.assert_not_called()
    assert native_host.additional_config == {}
    assert GRConfig.from_vllm_config(native_host).beam_max_width is None
    gr = prepared_engine_args(model=str(tmp_path), tensor_parallel_size=2, gr={})
    assert "vllm_gr" in gr.additional_config and gr.max_model_len == 8192
    with pytest.raises(ValueError, match="TP/PP/DP/DCP/PCP=1"):
        arg_utils_gr.create_engine_config(gr)


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

    from vllm_gr.entrypoints.cli import main
    from vllm_gr.entrypoints.cli.serve import GRServeSubcommand

    captured: list[Any] = []

    async def capture_server(args: Any) -> None:
        captured.append(args)

    # Keep ServeSubcommand.cmd's positional model handling in the test.
    monkeypatch.setattr(native_serve, "run_server", capture_server)
    monkeypatch.setattr(GRServeSubcommand, "validate", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", str(tmp_path), "--gr", *overrides])
    main.dispatch_main()
    assert len(captured) == 1
    assert captured[0]._gr_mode is True
    assert captured[0].gr == [{"result_width": 2}]
    args = EngineArgs.from_cli_args(captured[0])
    assert args.model == str(tmp_path)
    assert GRConfig.from_vllm_config(args).deployment.result_width == 2


@pytest.mark.parametrize("worker_decision", [None, False, True])  # type: ignore[misc]
def test_npu_sid_stage_vocab_resolves_auto_before_validation(
    worker_decision: bool | None,
) -> None:
    from vllm_gr.config.parser import resolve_startup_config
    from vllm_gr.engine import arg_utils_gr

    defaults = arg_utils_gr._beam_worker_decision_default(
        legacy_config_service("npu"), beam_width_configured=True, worker_decision=None
    )
    settings = {"sid_stage_vocab": True}
    if worker_decision is not None:
        settings["worker_decision"] = worker_decision
    if worker_decision is False:
        with pytest.raises(ValueError, match="worker_decision"):
            resolve_startup_config({}, settings, worker_decision_default=defaults)[0]
    else:
        config = resolve_startup_config({}, settings, worker_decision_default=defaults)[0]
        assert config.beam_sid_stage_vocab and config.beam_worker_decision is True


@pytest.mark.parametrize("explicit", [False, True])  # type: ignore[misc]
def test_grllm_resolves_native_settings_once_with_explicit_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    from vllm.engine.arg_utils import EngineArgs

    from vllm_gr import plugin
    from vllm_gr.entrypoints import gr as frontend

    path = tmp_path / "engine.yaml"
    path.write_text(f"model: {tmp_path}\nmax-model-len: 4k\ndtype: bfloat16\ngr: {{}}\n")
    captured: list[Any] = []

    def native_llm(self: Any, model: str, dtype: str = "auto", **kwargs: Any) -> None:
        captured.append(EngineArgs(model=model, dtype=dtype, **kwargs))
        self.llm_engine = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))

    monkeypatch.setattr(frontend.LLM, "__init__", native_llm)
    monkeypatch.setattr(frontend, "_install_renderer_eos_token_id_cache", lambda: None)
    monkeypatch.setattr(plugin, "initialize_runtime", lambda: None)
    from vllm_gr.engine import startup

    with mock.patch.object(
        startup, "resolve_native_settings", wraps=startup.resolve_native_settings
    ) as resolve:
        frontend.GRLLM(config=path, **({"max_model_len": 6144} if explicit else {}))
        assert resolve.call_count == 1
    assert captured[0].model == str(tmp_path)
    assert captured[0].dtype == "bfloat16"
    assert captured[0].max_model_len == (6144 if explicit else 4000)


@pytest.mark.parametrize(  # type: ignore[misc]
    "argv,launcher",
    [(["--headless"], "run_headless"), (["--api-server-count", "2"], "run_multi_api_server")],
)
def test_native_launchers_receive_prepared_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str], launcher: str
) -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.cli import serve

    from vllm_gr.entrypoints.cli.serve import GRServeSubcommand

    parsed = _parser().parse_args(
        ["serve", str(tmp_path), *argv, "--gr-config", '{"result_width": 2, "admin_port": 9123}']
    )
    observed = []

    def launch(args: Any) -> None:
        engine_args = AsyncEngineArgs.from_cli_args(args)
        observed.append(GRConfig.from_vllm_config(engine_args))
        assert "admin_port" not in engine_args.additional_config["vllm_gr"]["gr"]

    monkeypatch.setattr(serve, launcher, launch)
    GRServeSubcommand.cmd(parsed)
    assert len(observed) == 1 and observed[0].deployment.result_width == 2


@pytest.mark.parametrize("extra", [{"vllm_gr": {}}, {"vllm_gr_beam_max_width": 8}])  # type: ignore[misc]
def test_driver_rejects_inline_settings(extra: Any) -> None:
    from vllm_gr.engine.startup import prepare_engine_settings

    with pytest.raises(ValueError, match="--config"):
        prepare_engine_settings(None, {"additional_config": extra}, {})
