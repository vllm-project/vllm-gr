# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser


class CheckEnvSubcommand(CLISubcommand):
    """Validate the runtime packages and accelerator backend."""

    name = "check-env"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.platforms import current_platform

        from vllm_gr.platforms import activate_backend
        from vllm_gr.platforms.builtins import register_builtin_backends
        from vllm_gr.platforms.loader import load_backend_plugins

        register_builtin_backends()
        load_backend_plugins()
        device_type = args.backend or str(current_platform.device_type)
        report = activate_backend(device_type).check_environment()

        print("vLLM-GR environment check passed")
        for name, value in report.items():
            print(f"  {name}: {value}")

    def subparser_init(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Validate vLLM-GR runtime dependencies.",
            description=(
                "Validate runtime versions and the selected platform through "
                "its registered vLLM-GR backend Provider."
            ),
            usage="vllm-gr check-env [--backend DEVICE_TYPE]",
        )
        parser.add_argument(
            "--backend",
            metavar="DEVICE_TYPE",
            help="Validate this registered backend instead of the detected platform.",
        )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [CheckEnvSubcommand()]
