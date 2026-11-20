# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys

COMMANDS = {
    "serve": ("vllm_gr.entrypoints.cli.serve", "Launch the GR HTTP or gRPC server."),
    "bench": ("vllm_gr.entrypoints.cli.bench", "Run a repository benchmark workload."),
    "check-env": ("vllm_gr.entrypoints.cli.check_env", "Validate the runtime environment."),
}


def _version() -> str:
    try:
        return importlib.metadata.version("vllm-gr")
    except importlib.metadata.PackageNotFoundError:
        from vllm_gr import __version__

        return __version__


def _mode_arguments(argv: list[str]) -> tuple[bool, list[str]]:
    """Consume only the standalone mode switch before the forwarding delimiter."""
    end = argv.index("--") if "--" in argv else len(argv)
    selected = "--gr" in argv[:end]
    if selected and (not argv or argv[0] not in COMMANDS):
        raise SystemExit("Use 'vllm <serve|bench|check-env> ... --gr'; --gr follows the command.")
    return selected, [arg for index, arg in enumerate(argv) if index >= end or arg != "--gr"]


def dispatch_main() -> None:
    """Handle the --gr mode delegated by the patched vLLM CLI."""
    selected, argv = _mode_arguments(sys.argv[1:])
    if not selected:
        raise SystemExit("The vLLM-GR dispatcher requires the --gr switch.")
    _gr_main(argv)


def _add_mode_option(parser: argparse.ArgumentParser) -> None:
    from vllm_gr.config.parser import common_engine_defaults

    parser.set_defaults(**common_engine_defaults())
    parser.add_argument(
        "--gr", action="store_true", dest="_gr_mode", default=True, help="Use the vLLM-GR command."
    )


def _serving_parser_type():
    # Prefer the shared typed YAML resolver when the installed runtime has it.
    # Propagate dependency failures instead of silently bypassing that resolver.
    try:
        module = importlib.import_module("vllm_gr.entrypoints.cli.serve_config")
    except ModuleNotFoundError as exc:
        if exc.name != "vllm_gr.entrypoints.cli.serve_config":
            raise
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        return FlexibleArgumentParser
    return module.GRArgumentParser


def _gr_main(argv: list[str]) -> None:
    from vllm_gr._env_check import configure_model_runner_default

    configure_model_runner_default()
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()
    command = argv[0] if argv and argv[0] in COMMANDS else None
    # Workload-owned arguments must not undergo serve-specific normalization.
    parser_type = FlexibleArgumentParser
    if command == "serve":
        from vllm_gr.entrypoints.cli.serve_banner import patch_banner

        patch_banner()
        parser_type = _serving_parser_type()
    parser = parser_type(prog="vllm", description="vLLM-GR CLI")
    parser.add_argument("-v", "--version", action="version", version=_version())
    subparsers = parser.add_subparsers(dest="subparser")
    commands = {}
    for name, (module_name, help_text) in COMMANDS.items():
        if name != command:
            subparsers.add_parser(name, help=help_text)
            continue
        module = importlib.import_module(module_name)
        for cmd in module.cmd_init():
            child = cmd.subparser_init(subparsers)
            child.set_defaults(dispatch_function=cmd.cmd)
            _add_mode_option(child)
            commands[cmd.name] = cmd
    if command == "serve":
        argv = [
            arg.replace("--gr.", "--gr-config.", 1) if arg.startswith("--gr.") else arg
            for arg in argv
        ]
    args = (
        argparse.ArgumentParser.parse_args(parser, argv)
        if command == "bench"
        else parser.parse_args(argv)
    )
    if args.subparser in commands:
        commands[args.subparser].validate(args)
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    dispatch_main()
