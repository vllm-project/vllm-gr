# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.utils.argparse_utils import FlexibleArgumentParser

DESCRIPTION = """Launch a local OpenAI-compatible API server for vllm-gr.

This wraps vLLM's standard OpenAI API server; the vllm-gr beam-search and
catalog-constrained generation patches are loaded automatically via the vLLM
general plugin. Useful for serving OneRec-style checkpoints.
"""


class GRServeSubcommand(ServeSubcommand):
    """The `serve` subcommand for vllm-gr CLI."""

    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm_gr.entrypoints.openai.beam_proc_manager import (
            ensure_beam_proc_config,
        )

        # Prepare BeamProc IPC addresses before vLLM spawns EngineCore so
        # the child process inherits the TX/RX environment variables.
        ensure_beam_proc_config()
        ServeSubcommand.cmd(args)

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            self.name,
            help="Launch a local OpenAI-compatible API server for vllm-gr.",
            description=DESCRIPTION,
            usage="vllm-gr serve [model_tag] [options]",
        )

        serve_parser = make_arg_parser(serve_parser)
        serve_parser.set_defaults(attention_backend="CUSTOM")

        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [GRServeSubcommand()]
