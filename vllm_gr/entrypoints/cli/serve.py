# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

import uvloop
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_gr.entrypoints.api_server import run_server

DESCRIPTION = """Launch a local OpenAI-compatible API server for vllm-GR /recommend serving."""


class GRServeSubcommand(CLISubcommand):
    """The `serve` subcommand for vllm-gr CLI."""

    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        if args.headless:
            raise ValueError("vllm-gr serve does not support --headless")

        if getattr(args, "api_server_count", 1) != 1:
            raise ValueError("vllm-gr serve currently supports api_server_count=1 only")

        uvloop.run(run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            self.name,
            help="Launch a local OpenAI-compatible API server for vllm-GR.",
            description=DESCRIPTION,
            usage="vllm-gr serve [model_tag] [options]",
        )

        serve_parser = make_arg_parser(serve_parser)
        serve_parser.add_argument(
            "--model-family",
            type=str,
            default=None,
            help=(
                "Model family: 'hstu'/'fuxi' for GR recommendation, "
                "'llm'/'pangu' for causal LM passthrough (e.g. Pangu Embedded 1B)."
            ),
        )
        serve_parser.add_argument(
            "--dataset",
            type=str,
            default=None,
            help="Optional dataset name used to apply in-memory config overrides.",
        )
        serve_parser.add_argument(
            "--max-seq-len",
            type=int,
            default=512,
            help="Maximum user history sequence length.",
        )
        serve_parser.add_argument(
            "--max-candidates",
            type=int,
            default=1000,
            help="Maximum number of candidate items.",
        )
        serve_parser.add_argument(
            "--allow-missing-weights",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Allow local model dirs without checkpoint weights (dummy loader).",
        )

        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [GRServeSubcommand()]
