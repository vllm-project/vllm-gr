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
        args = self.parser.parse_args(["serve", "--vllm-gr-config", "config.yaml"])
        with mock.patch("vllm_gr.entrypoints.cli.serve.validate_parsed_serve_args"):
            self.subcmd.validate(args)
        self.assertIsNone(args.attention_backend)

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
