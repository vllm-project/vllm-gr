# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Behaviour of tools/pre_commit/check_spdx_header.py."""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import pytest

from tests.helpers.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "tools/pre_commit/check_spdx_header.py"
LICENSE = "# SPDX-License-Identifier: Apache-2.0\n"
PROJECT = "# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project\n"
UPSTREAM = "# SPDX-FileCopyrightText: Copyright 2025 The xLLM Authors. All Rights Reserved.\n"
SLASH_LICENSE = "// SPDX-License-Identifier: Apache-2.0\n"
SLASH_PROJECT = "// SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project\n"


class SPDXHook(Protocol):
    """Interface used by tests of the dynamically loaded hook."""

    POLICY_PATH: Path
    REPO_ROOT: Path
    tracked_sources: Callable[[], list[str]]

    def check_all_tracked(self) -> list[str]: ...

    def policy_errors(
        self,
        relative: str,
        licenses: list[str],
        copyrights: list[str],
        policy: dict[str, Any],
    ) -> list[str]: ...


def load_hook() -> SPDXHook:
    """Import the hook by path; tools/pre_commit is not a package."""
    spec = importlib.util.spec_from_file_location("check_spdx_header", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(SPDXHook, module)


class SPDXHeaderTests(unittest.TestCase):
    def run_hook(self, content: str, name: str = "sample.py") -> tuple[int, str]:
        """Run the hook on one file; return its exit code and the file afterwards."""
        code, data = self.run_hook_bytes(content.encode("utf-8"), name)
        return code, data.decode("utf-8")

    def run_hook_bytes(self, content: bytes, name: str = "sample.py") -> tuple[int, bytes]:
        """Same, but byte-exact: keeps a BOM and CRLF visible to the assertions."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_bytes(content)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True
            )
            return result.returncode, path.read_bytes()

    def test_complete_headers_pass_unchanged(self) -> None:
        for content in [
            LICENSE + PROJECT + "x = 1\n",
            # several copyright holders
            LICENSE + PROJECT + UPSTREAM + "x = 1\n",
            # compound license expression
            "# SPDX-License-Identifier: Apache-2.0 AND MIT\n" + PROJECT + "x = 1\n",
        ]:
            with self.subTest(content=content):
                self.assertEqual(self.run_hook(content), (0, content))

    def test_missing_header_is_added_with_project_holder(self) -> None:
        code, text = self.run_hook("x = 1\n")
        self.assertEqual(code, 1)
        self.assertEqual(text, LICENSE + PROJECT + "x = 1\n")

    def test_license_only_file_is_reported_not_given_a_holder(self) -> None:
        # A license expression says nothing about who holds the copyright.
        content = LICENSE + "x = 1\n"
        self.assertEqual(self.run_hook(content), (1, content))

    def test_plain_upstream_copyright_comment_is_not_overridden(self) -> None:
        # An upstream copy may carry an ordinary comment instead of a tag; the
        # hook must not add the project holder on top of it.
        content = LICENSE + "# Copyright (c) 2024 Upstream Authors\n" + "x = 1\n"
        self.assertEqual(self.run_hook(content), (1, content))

    def test_plain_copyright_without_spdx_lines_is_not_adopted(self) -> None:
        # No SPDX line either: completing the header here would claim the file
        # and the next run would pass, so this is the case the policy cannot see.
        content = "# Copyright (c) 2024 Upstream Authors\n# Licensed under MIT.\n" + "x = 1\n"
        self.assertEqual(self.run_hook(content), (1, content))

    def test_completing_a_header_fails_the_run(self) -> None:
        # pre-commit passes only when the hook changed nothing, so a rewrite
        # must fail even though the file is now correct.
        code, text = self.run_hook("x = 1\n")
        self.assertEqual(code, 1)
        self.assertEqual(self.run_hook(text), (0, text))

    def test_tag_outside_the_leading_comment_block_is_not_counted(self) -> None:
        # A fixture string, a heredoc or a tag after code is not a header.
        content = (
            'print(1)\n\nFIXTURE = """\n'
            + LICENSE
            + "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
            + '"""\n'
        )
        code, text = self.run_hook(content)
        self.assertEqual(code, 1)
        self.assertTrue(text.startswith(LICENSE + PROJECT), text[:80])

    def test_bom_and_line_endings_survive_a_rewrite(self) -> None:
        bom = b"\xef\xbb\xbf"
        code, data = self.run_hook_bytes(bom + b"x = 1\r\n")
        self.assertEqual(code, 1)
        self.assertEqual(
            data, bom + (LICENSE + PROJECT).encode("utf-8").replace(b"\n", b"\r\n") + b"x = 1\r\n"
        )

    def test_encoding_cookie_stays_on_the_first_line(self) -> None:
        # PEP 263 only honours a coding cookie on line one or two.
        code, text = self.run_hook("# -*- coding: utf-8 -*-\nx = 1\n")
        self.assertEqual(code, 1)
        self.assertEqual(text, "# -*- coding: utf-8 -*-\n" + LICENSE + PROJECT + "x = 1\n")

    def test_cookie_on_the_second_line_stays_within_the_window(self) -> None:
        # PEP 263 also accepts a cookie on line two, after an ordinary comment.
        # Inserting above it would move it to line four and lose the encoding.
        code, text = self.run_hook("# ordinary comment\n# -*- coding: latin-1 -*-\nx = 1\n")
        self.assertEqual(code, 1)
        self.assertEqual(
            text,
            "# ordinary comment\n# -*- coding: latin-1 -*-\n" + LICENSE + PROJECT + "x = 1\n",
        )

    def test_cookie_after_a_shebang_stays_within_the_window(self) -> None:
        code, text = self.run_hook("#!/usr/bin/env python\n# -*- coding: latin-1 -*-\nx = 1\n")
        self.assertEqual(code, 1)
        self.assertEqual(
            text,
            "#!/usr/bin/env python\n# -*- coding: latin-1 -*-\n" + LICENSE + PROJECT + "x = 1\n",
        )

    def test_non_utf8_file_is_reported_not_crashed_on(self) -> None:
        # A bytes literal keeps this file ASCII while still giving the hook a
        # latin-1 byte that is not valid UTF-8, which check-characters requires.
        content = b"# -*- coding: latin-1 -*-\nprint('\xe9')\n"
        code, data = self.run_hook_bytes(content)
        self.assertEqual(code, 1)
        self.assertEqual(data, content)

    def test_copyright_without_license_is_reported_not_guessed(self) -> None:
        content = UPSTREAM + "x = 1\n"
        self.assertEqual(self.run_hook(content), (1, content))

    def test_empty_values_are_rejected(self) -> None:
        for content in [
            "# SPDX-License-Identifier:\n" + PROJECT + "x = 1\n",
            LICENSE + "# SPDX-FileCopyrightText:\nx = 1\n",
        ]:
            with self.subTest(content=content):
                self.assertEqual(self.run_hook(content), (1, content))

    def test_invalid_license_expression_is_rejected(self) -> None:
        for expression in ["NOT-A-LICENSE", "Apache-2.0 AND", "Apache-2.0 MIT"]:
            content = f"# SPDX-License-Identifier: {expression}\n" + PROJECT + "x = 1\n"
            with self.subTest(expression=expression):
                self.assertEqual(self.run_hook(content), (1, content))

    def test_shebang_is_preserved_when_header_is_added(self) -> None:
        code, text = self.run_hook("#!/usr/bin/env python\nx = 1\n")
        self.assertEqual(code, 1)
        self.assertTrue(text.startswith("#!/usr/bin/env python\n" + LICENSE + PROJECT))

    def test_proto_header_uses_slash_comments(self) -> None:
        # A "#" header would make the schema unparsable.
        code, text = self.run_hook('syntax = "proto3";\n', "sample.proto")
        self.assertEqual(code, 1)
        self.assertEqual(text, SLASH_LICENSE + SLASH_PROJECT + 'syntax = "proto3";\n')

    def test_complete_proto_header_passes_unchanged(self) -> None:
        content = SLASH_LICENSE + SLASH_PROJECT + 'syntax = "proto3";\n'
        self.assertEqual(self.run_hook(content, "sample.proto"), (0, content))

    def test_unknown_file_type_is_reported_not_rewritten(self) -> None:
        content = "some text\n"
        self.assertEqual(self.run_hook(content, "sample.txt"), (1, content))


class AttributionPolicyTests(unittest.TestCase):
    """The --all-tracked pass enforces who may be named on which file."""

    hook: SPDXHook
    policy: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.hook = load_hook()
        cls.policy = {
            "license": "Apache-2.0",
            "copyright": "Copyright contributors to the vLLM-GR project",
            "verbatim_copies": {
                "vendored/copy.py": {
                    "license": "Apache-2.0",
                    "copyrights": ["Copyright contributors to the vLLM project"],
                    "source": "https://github.com/vllm-project/vllm",
                }
            },
        }

    def errors(self, path: str, licenses: list[str], copyrights: list[str]) -> list[str]:
        result: list[str] = self.hook.policy_errors(path, licenses, copyrights, self.policy)
        return result

    def test_project_file_with_project_holder_passes(self) -> None:
        self.assertEqual(
            self.errors("vllm_gr/core.py", ["Apache-2.0"], [self.policy["copyright"]]),
            [],
        )

    def test_project_file_with_only_upstream_holder_fails(self) -> None:
        errors = self.errors(
            "vllm_gr/core.py", ["Apache-2.0"], ["Copyright contributors to the vLLM project"]
        )
        self.assertTrue(any("missing" in error for error in errors))

    def test_project_file_with_unregistered_holder_fails(self) -> None:
        errors = self.errors(
            "vllm_gr/core.py",
            ["Apache-2.0"],
            [self.policy["copyright"], "Copyright someone Else"],
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("not registered", errors[0])

    def test_registered_copy_passes_without_the_project_holder(self) -> None:
        self.assertEqual(
            self.errors(
                "vendored/copy.py",
                ["Apache-2.0"],
                ["Copyright contributors to the vLLM project"],
            ),
            [],
        )

    def test_registered_copy_missing_its_upstream_holder_fails(self) -> None:
        errors = self.errors("vendored/copy.py", ["Apache-2.0"], [])
        self.assertIn("missing registered holder", errors[0])

    def test_registered_copy_that_claims_project_authorship_fails(self) -> None:
        errors = self.errors(
            "vendored/copy.py",
            ["Apache-2.0"],
            ["Copyright contributors to the vLLM project", self.policy["copyright"]],
        )
        self.assertIn("unregistered holder", errors[0])

    def test_registered_copy_with_the_wrong_license_fails(self) -> None:
        errors = self.errors(
            "vendored/copy.py", ["BSD-3-Clause"], ["Copyright contributors to the vLLM project"]
        )
        self.assertIn("license must be", errors[0])

    @unittest.skipUnless(shutil.which("git") is not None, "requires git in PATH")
    def test_the_policy_file_matches_the_repository(self) -> None:
        """Regression guard: every tracked file agrees with spdx_policy.json."""
        self.assertEqual(self.hook.check_all_tracked(), [])


class EmptyFileTests(unittest.TestCase):
    """A zero-byte file is exempt only when it is an empty __init__.py."""

    hook: SPDXHook

    def setUp(self) -> None:
        self.hook = load_hook()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "vendored").mkdir()
        policy = {
            "license": "Apache-2.0",
            "copyright": "Copyright contributors to the vLLM-GR project",
            "verbatim_copies": {
                "vendored/copy.py": {
                    "license": "Apache-2.0",
                    "copyrights": ["Copyright contributors to the vLLM project"],
                    "source": "https://github.com/vllm-project/vllm",
                }
            },
        }
        self.policy_path = self.root / "spdx_policy.json"
        self.policy_path.write_text(json.dumps(policy), encoding="utf-8")
        for name in ["vendored/copy.py", "vendored/__init__.py", "vllm_gr/pkg/__init__.py"]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        self.tracked = ["vendored/copy.py", "vendored/__init__.py", "vllm_gr/pkg/__init__.py"]

    def errors(self) -> list[str]:
        hook = self.hook
        original = hook.POLICY_PATH, hook.REPO_ROOT, hook.tracked_sources
        hook.POLICY_PATH, hook.REPO_ROOT = self.policy_path, self.root
        hook.tracked_sources = lambda: list(self.tracked)
        try:
            return hook.check_all_tracked()
        finally:
            hook.POLICY_PATH, hook.REPO_ROOT, hook.tracked_sources = original

    def test_empty_init_files_are_exempt(self) -> None:
        errors = self.errors()
        self.assertEqual([error for error in errors if "__init__.py" in error], [])

    def test_empty_registered_copy_is_reported(self) -> None:
        errors = self.errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("vendored/copy.py: empty", errors[0])

    def test_empty_source_that_is_not_an_init_is_reported(self) -> None:
        self.tracked.append("vllm_gr/empty.py")
        (self.root / "vllm_gr/empty.py").write_bytes(b"")
        errors = self.errors()
        self.assertIn("vllm_gr/empty.py: empty", errors)


if __name__ == "__main__":
    unittest.main()


# Every case here is a host-only check of the hook and its policy, so the
# whole module is CPU tier and L1 (see docs/contributing/tools/README.md#test-levels-and-ci-selection).
pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
