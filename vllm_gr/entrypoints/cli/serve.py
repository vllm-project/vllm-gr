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

    def validate(self, args: argparse.Namespace) -> None:
        # A canonical config owns the attention-backend choice. For legacy
        # serve invocations retain the historical CUSTOM default, but apply it
        # here (after parsing) so it is not mistaken for an explicit legacy
        # CLI override and rejected alongside --vllm-gr-config.
        if (
            getattr(args, "vllm_gr_config", None) is None
            and getattr(args, "attention_backend", None) is None
        ):
            args.attention_backend = "CUSTOM"
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
        serve_parser.set_defaults(attention_backend=None)

        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [GRServeSubcommand()]
