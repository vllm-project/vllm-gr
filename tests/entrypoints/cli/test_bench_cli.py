# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm.utils.argparse_utils import FlexibleArgumentParser

from tests.helpers.paths import REPO_ROOT
from vllm_gr.benchmarks import scenarios
from vllm_gr.benchmarks.scenarios import SCENARIOS, scenario_names
from vllm_gr.entrypoints.cli import bench as bench_cli
from vllm_gr.entrypoints.cli import main as cli

pytestmark = pytest.mark.cpu_test


def _parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    bench_cli.GRBenchSubcommand().subparser_init(subparsers)
    return parser


def test_scenario_registry_covers_all_existing_benchmarks() -> None:
    assert set(scenario_names()) == {
        "open-one-rec",
        "beam-performance",
        "beam-performance-ab",
        "beam-recall",
        "qwen",
        "ab-offline",
    }
    assert all(SCENARIOS[name].module or SCENARIOS[name].path for name in SCENARIOS)


@pytest.mark.parametrize("separator", [[], ["--"]])  # type: ignore[misc]
def test_parser_forwards_scenario_arguments(
    monkeypatch: pytest.MonkeyPatch, separator: list[str]
) -> None:
    calls = []
    monkeypatch.setattr(bench_cli, "run_scenario", lambda name, argv: calls.append((name, argv)))
    forwarded = ["--model_path", "qwen", "--option.value=1", "--help"]
    args = bench_cli.parse_bench_args(_parser(), ["bench", "qwen", *separator, *forwarded])
    bench_cli.GRBenchSubcommand.cmd(args)
    assert calls == [("qwen", forwarded)]


def test_list_scenarios(capsys: pytest.CaptureFixture[str]) -> None:
    bench_cli.GRBenchSubcommand.cmd(_parser().parse_args(["bench", "--list"]))
    output = capsys.readouterr().out
    assert all(name in output for name in scenario_names())


def test_bare_bench_lists_scenarios(capsys: pytest.CaptureFixture[str]) -> None:
    bench_cli.GRBenchSubcommand.cmd(_parser().parse_args(["bench"]))
    assert "qwen:" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--gr", "serve", "model"], ["--gr"], ["--help", "--gr"]])  # type: ignore[misc]
def test_mode_switch_requires_command_first(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="follows the command"):
        cli._mode_arguments(argv)


def test_mode_switch_preserves_configuration_and_forwarding_boundary() -> None:
    argv = [
        "serve",
        "model",
        "--gr",
        "--gr.beam_width",
        "[128,128,128]",
        "--gr-config",
        '{"worker_decision":true}',
    ]
    selected, forwarded = cli._mode_arguments(argv)
    assert selected and forwarded == argv[:2] + argv[3:]
    assert cli._mode_arguments(["bench", "qwen", "--", "--gr"]) == (
        False,
        ["bench", "qwen", "--", "--gr"],
    )


def test_gr_dispatch_selects_mode_without_mutating_process_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []
    argv = ["vllm", "serve", "model", "--gr", "--port", "9000"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cli, "_gr_main", captured.append)
    cli.dispatch_main()
    assert captured == [["serve", "model", "--port", "9000"]]
    assert sys.argv is argv and "--gr" in argv


def test_mode_option_retains_typed_config_action() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gr-config",
        dest="gr",
        type=json.loads,
        action="append",
        default=[{"worker_decision": True}],
    )
    cli._add_mode_option(parser)
    args = parser.parse_args(["--gr", "--gr-config", '{"max_decode_steps":3}'])
    assert args._gr_mode is True
    assert args.gr == [{"worker_decision": True}, {"max_decode_steps": 3}]


@pytest.mark.parametrize("prefix", [[], ["bench"]])  # type: ignore[misc]
def test_onerec_adapter_adds_exactly_one_native_bench(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prefix: list[str]
) -> None:
    (tmp_path / "vllm_gr").mkdir()
    (tmp_path / "vllm_gr/__init__.py").touch()
    module = tmp_path / "benchmarks/open_one_rec/one_rec_main.py"
    module.parent.mkdir(parents=True)
    module.touch()
    monkeypatch.setattr(scenarios, "_REPO_ROOT", tmp_path)
    calls = []

    def run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scenarios.subprocess, "run", run)
    scenarios.run_scenario("open-one-rec", [*prefix, "throughput", "--help"])
    assert calls == [
        [
            sys.executable,
            "-P",
            "-m",
            "benchmarks.open_one_rec.one_rec_main",
            "bench",
            "throughput",
            "--help",
        ]
    ]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("scenario", "expected_option"),
    [
        ("beam-performance", "--generated-steps"),
        ("beam-performance-ab", "--base-sha"),
        ("beam-recall", "--catalog-path"),
        ("ab-offline", "--catalog"),
        ("qwen", "--token-id-start"),
    ],
)
def test_module_runner_works_from_package_directory(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    scenario: str,
    expected_option: str,
) -> None:
    monkeypatch.chdir(REPO_ROOT / "vllm_gr")
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    monkeypatch.setenv("PYTHONPATH", ".")
    scenarios.run_scenario(scenario, ["--help"])
    assert expected_option in capfd.readouterr().out


def test_gr_package_does_not_own_native_vllm_entrypoint() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    scripts = config["project"]["scripts"]
    assert "vllm" not in scripts
    assert "vllm-gr" not in scripts


@pytest.mark.parametrize("separator", [[], ["--"]])  # type: ignore[misc]
def test_online_bench_alias_preserves_runner_arguments(
    monkeypatch: pytest.MonkeyPatch, separator: list[str]
) -> None:
    calls = []
    monkeypatch.setattr(bench_cli, "run_scenario", lambda name, argv: calls.append((name, argv)))
    forwarded = ["--model_path", "model", "--option.value=1", "--help"]
    for command in (["serve"], ["open-one-rec", "serve"]):
        selected, argv = cli._mode_arguments(["bench", *command, "--gr", *separator, *forwarded])
        assert selected
        args = bench_cli.parse_bench_args(_parser(), argv)
        bench_cli.GRBenchSubcommand.cmd(args)
    assert calls == [("open-one-rec", ["serve", *forwarded])] * 2
