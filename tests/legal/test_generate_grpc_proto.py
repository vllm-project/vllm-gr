# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Behaviour of tools/codegen/generate_grpc_proto.py."""

import importlib.util
import tempfile
import types
import unittest
from pathlib import Path

import pytest

from tests.helpers.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "tools/codegen/generate_grpc_proto.py"


def load_generator() -> types.ModuleType:
    """Load the generation script from its source checkout."""
    spec = importlib.util.spec_from_file_location("generate_grpc_proto", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProtoNoticeTests(unittest.TestCase):
    """The binding header is derived from the schema, never hand-maintained."""

    generator: types.ModuleType

    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator()

    def notice(self, text: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            proto = Path(directory) / "sample.proto"
            proto.write_text(text, encoding="utf-8")
            result: str = self.generator.proto_notice(proto)
            return result

    def test_slash_comments_become_hash_comments(self) -> None:
        self.assertEqual(
            self.notice('// SPDX-License-Identifier: Apache-2.0\n\nsyntax = "proto3";\n'),
            "# SPDX-License-Identifier: Apache-2.0\n",
        )

    def test_a_blank_line_inside_the_header_does_not_truncate_it(self) -> None:
        self.assertEqual(
            self.notice('// one\n\n// two\nsyntax = "proto3";\n'),
            "# one\n#\n# two\n",
        )

    def test_a_file_without_a_comment_block_gets_no_notice(self) -> None:
        self.assertEqual(self.notice('syntax = "proto3";\n'), "")


if __name__ == "__main__":
    unittest.main()


pytestmark = pytest.mark.cpu_test
