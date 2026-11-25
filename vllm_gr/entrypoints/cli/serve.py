# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import argparse
import json

from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.launchers.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG
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
    def prepare_args(args: argparse.Namespace) -> tuple[argparse.Namespace, str, int | None]:
        from vllm_gr.config.parser import merge_gr_settings
        from vllm_gr.engine.startup import prepare_engine_settings

        gr: dict = {}
        for block in getattr(args, "gr", None) or ():
            if not isinstance(block, dict):
                raise ValueError("--gr-config requires a JSON object")
            gr = merge_gr_settings(gr, block)
        values, host, port = prepare_engine_settings(
            None, vars(args), gr, explicit_fields=getattr(args, "_gr_explicit_native_fields", None)
        )
        return argparse.Namespace(**values), host, port

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args, host, port = GRServeSubcommand.prepare_args(args)
        if getattr(args, "grpc", False):
            from vllm_gr.entrypoints.grpc.serve import patch_grpc_server

            with patch_grpc_server(admin_host=host, admin_port=port):
                ServeSubcommand.cmd(args)
            return
        ServeSubcommand.cmd(args)

    def validate(self, args: argparse.Namespace) -> None:
        # vLLM rejects specifying both native forms of the attention backend.
        attention = getattr(args, "attention_config", None)
        backend = (
            attention.get("backend")
            if isinstance(attention, dict)
            else getattr(attention, "backend", None)
        )
        if (
            getattr(args, "config", None) is None
            and getattr(args, "gr", None) is None
            and getattr(args, "attention_backend", None) is None
            and backend is None
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
            usage="%(prog)s [model_tag] --gr [options]",
        )

        serve_parser = make_arg_parser(serve_parser)
        serve_parser.add_argument(
            "--gr-config",
            dest="gr",
            type=json.loads,
            action="append",
            help="GR settings as JSON, or use --gr.beam_width.",
        )
        serve_parser.set_defaults(attention_backend=None)

        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [GRServeSubcommand()]
