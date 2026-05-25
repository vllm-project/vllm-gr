# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.metadata

from vllm.logger import init_logger

logger = init_logger(__name__)


def _version() -> str:
    try:
        return importlib.metadata.version("vllm-gr")
    except importlib.metadata.PackageNotFoundError:
        from vllm_gr import __version__

        return __version__


def main() -> None:
    import vllm_gr.entrypoints.cli.serve as gr_serve
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="vLLM-GR CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=_version(),
    )

    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}

    for cmd in gr_serve.cmd_init():
        cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
        cmds[cmd.name] = cmd

    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
