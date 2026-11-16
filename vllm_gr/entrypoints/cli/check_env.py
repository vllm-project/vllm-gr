# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

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

        from vllm_gr._env_check import check_ascend_environment, check_runtime_versions

        detected_device = str(current_platform.device_type)
        require_npu = bool(args.npu or detected_device == "npu")
        if require_npu:
            report = check_ascend_environment(require_npu_platform=True)
        else:
            report = check_runtime_versions()
            report["platform"] = detected_device

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
                "Validate vLLM/vLLM-Ascend versions, the selected platform, "
                "and the active Triton-Ascend backend."
            ),
            usage="vllm-gr check-env [--npu]",
        )
        parser.add_argument(
            "--npu",
            action="store_true",
            help="Require the Ascend NPU platform even when auto-detection selects another device.",
        )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [CheckEnvSubcommand()]
