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
        if args.backend:
            from vllm_gr.platforms import activate_backend_by_device_type

            backend = activate_backend_by_device_type(args.backend)
        else:
            from vllm_gr.platforms import activate_current_backend

            backend = activate_current_backend()
        report = backend.check_environment()

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
            description=("Validate runtime versions and the selected accelerator Provider."),
            usage="vllm-gr check-env [--backend DEVICE_TYPE]",
        )
        parser.add_argument(
            "--backend",
            metavar="DEVICE_TYPE",
            help="Check an explicitly registered backend instead of the detected platform.",
        )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [CheckEnvSubcommand()]
