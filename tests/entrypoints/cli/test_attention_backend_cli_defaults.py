# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import unittest
from unittest import mock

import pytest


class TestServeAttentionBackendDefault(unittest.TestCase):
    """Verify --attention-backend defaults to CUSTOM for the serve subcommand,
    and that explicit user values are preserved."""

    def setUp(self) -> None:
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        from vllm_gr.entrypoints.cli.serve import GRServeSubcommand

        self.parser = FlexibleArgumentParser()
        subparsers = self.parser.add_subparsers(dest="subparser")
        self.subcmd = GRServeSubcommand()
        self.subcmd.subparser_init(subparsers)

    def test_no_backend_defaults_to_custom(self) -> None:
        args = self.parser.parse_args(["serve"])
        self.subcmd.validate(args)
        self.assertEqual(args.attention_backend, "CUSTOM")

    def test_canonical_config_does_not_add_legacy_backend_default(self) -> None:
        from vllm.engine.arg_utils import EngineArgs

        from vllm_gr.engine import arg_utils_gr
        from vllm_gr.entrypoints.cli.serve_config import GRArgumentParser

        with mock.patch.object(EngineArgs, "add_cli_args", staticmethod(arg_utils_gr.add_cli_args)):
            parser = GRArgumentParser()
            self.subcmd.subparser_init(parser.add_subparsers(dest="subparser"))
            args = parser.parse_args(["serve", "--gr-config", "{}"])
        self.assertIsNotNone(args.gr)
        native_backend = args.attention_backend
        with mock.patch("vllm_gr.entrypoints.cli.serve.validate_parsed_serve_args"):
            self.subcmd.validate(args)
        self.assertEqual(args.attention_backend, native_backend)

    def test_dash_dash_backend_preserves_value(self) -> None:
        args = self.parser.parse_args(["serve", "--attention-backend", "FLASH_ATTN"])
        self.assertEqual(args.attention_backend, "FLASH_ATTN")

    def test_equals_form_preserves_value(self) -> None:
        args = self.parser.parse_args(["serve", "--attention-backend=FLASH_ATTN"])
        self.assertEqual(args.attention_backend, "FLASH_ATTN")

    def test_custom_explicit_preserves_value(self) -> None:
        args = self.parser.parse_args(["serve", "--attention-backend", "CUSTOM"])
        self.assertEqual(args.attention_backend, "CUSTOM")


pytestmark = pytest.mark.cpu_test
