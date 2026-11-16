# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Registered benchmark commands with adapters for repository-owned runners."""

from __future__ import annotations

import argparse

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_gr.benchmarks.scenarios import SCENARIOS, run_scenario, scenario_names


class WorkloadArgumentParser(argparse.ArgumentParser):
    """Preserve a repository runner's complete argv, including dotted keys.

    These workloads still own their parsers. A dedicated adapter avoids
    interpreting their flags as serving options or changing their ordering.
    Packaged commands can use ordinary subparsers instead of this adapter.
    """

    def parse_known_args(self, args=None, namespace=None):
        result = argparse.Namespace() if namespace is None else namespace
        for name, value in self._defaults.items():
            if not hasattr(result, name):
                setattr(result, name, value)
        values = list(args or [])
        if values[:1] == ["--"]:
            values = values[1:]
        elif result.bench_scenario == "open-one-rec":
            # The qualified form has a native subcommand before its workload options.
            index = 1 if values[:1] == ["bench"] else 0
            if values[index : index + 2] in (["serve", "--"], ["throughput", "--"]):
                del values[index + 1]
        result.scenario_args = values
        return result, []


class GRBenchSubcommand(CLISubcommand):
    name = "bench"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if args.list_scenarios or args.bench_scenario is None:
            print("serve: Alias for open-one-rec serve (online serving benchmark).")
            for name in scenario_names():
                print(f"{name}: {SCENARIOS[name].help}")
            return
        if args.bench_scenario == "serve":
            run_scenario("open-one-rec", ["serve", *args.scenario_args])
        else:
            run_scenario(args.bench_scenario, args.scenario_args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Run a vLLM-GR benchmark workload.",
            description="Run a repository benchmark. Each workload owns its options and --help.",
            usage="%(prog)s <workload> --gr [workload options]",
        )
        parser.add_argument(
            "--list", action="store_true", dest="list_scenarios", help="List workloads."
        )
        children = parser.add_subparsers(dest="bench_scenario", parser_class=WorkloadArgumentParser)
        online = children.add_parser(
            "serve", help="Run the OneRec online serving benchmark.", add_help=False
        )
        online.set_defaults(bench_scenario="serve")
        for name, scenario in SCENARIOS.items():
            child = children.add_parser(name, help=scenario.help, add_help=False)
            child.set_defaults(bench_scenario=name)
        return parser


def parse_bench_args(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """Compatibility helper for callers which construct a FlexibleArgumentParser."""
    return argparse.ArgumentParser.parse_args(parser, argv)


def cmd_init() -> list[CLISubcommand]:
    return [GRBenchSubcommand()]
