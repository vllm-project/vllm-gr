# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import sys
import unittest
from unittest import mock


class TestServeAttentionBackendDefault(unittest.TestCase):
    """Verify --attention-backend defaults to CUSTOM for the serve subcommand,
    and that explicit user values are preserved."""

    def setUp(self) -> None:
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        from vllm_gr.entrypoints.cli.serve import GRServeSubcommand

        self.parser = FlexibleArgumentParser()
        subparsers = self.parser.add_subparsers(dest="subparser")
        subcmd = GRServeSubcommand()
        subcmd.subparser_init(subparsers)

    def test_no_backend_defaults_to_custom(self) -> None:
        args = self.parser.parse_args(["serve"])
        self.assertEqual(args.attention_backend, "CUSTOM")

    def test_dash_dash_backend_preserves_value(self) -> None:
        args = self.parser.parse_args(["serve", "--attention-backend", "FLASH_ATTN"])
        self.assertEqual(args.attention_backend, "FLASH_ATTN")

    def test_equals_form_preserves_value(self) -> None:
        args = self.parser.parse_args(["serve", "--attention-backend=FLASH_ATTN"])
        self.assertEqual(args.attention_backend, "FLASH_ATTN")

    def test_custom_explicit_preserves_value(self) -> None:
        args = self.parser.parse_args(["serve", "--attention-backend", "CUSTOM"])
        self.assertEqual(args.attention_backend, "CUSTOM")


class TestNonServeNoMutation(unittest.TestCase):
    """Verify that non-serve commands are not affected by the default."""

    def test_handle_gr_flags_does_not_mutate_argv_for_check_env(self) -> None:
        from vllm_gr.entrypoints.cli.main import _handle_gr_flags

        original_argv = ["vllm-gr", "check-env"]
        with mock.patch.object(sys, "argv", original_argv):
            _handle_gr_flags()
            self.assertNotIn("--attention-backend", sys.argv)
            self.assertEqual(sys.argv, original_argv)

    def test_handle_gr_flags_does_not_mutate_argv_for_unknown_command(self) -> None:
        from vllm_gr.entrypoints.cli.main import _handle_gr_flags

        original_argv = ["vllm-gr", "unknown-cmd"]
        with mock.patch.object(sys, "argv", original_argv):
            _handle_gr_flags()
            self.assertNotIn("--attention-backend", sys.argv)
            self.assertEqual(sys.argv, original_argv)


class TestServeStartup(unittest.TestCase):
    def test_beamproc_config_is_prepared_before_server_start(self) -> None:
        from vllm.entrypoints.cli.serve import ServeSubcommand

        from vllm_gr.entrypoints.cli.serve import GRServeSubcommand

        args = argparse.Namespace()
        calls = mock.Mock()
        with (
            mock.patch(
                "vllm_gr.entrypoints.openai.beam_proc_manager.ensure_beam_proc_config"
            ) as ensure_config,
            mock.patch.object(ServeSubcommand, "cmd") as serve_cmd,
        ):
            calls.attach_mock(ensure_config, "ensure_config")
            calls.attach_mock(serve_cmd, "serve")
            GRServeSubcommand.cmd(args)

        self.assertEqual(
            calls.mock_calls,
            [mock.call.ensure_config(), mock.call.serve(args)],
        )
