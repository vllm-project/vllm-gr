# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the SPDX header check tool."""

from __future__ import annotations

import os
import tempfile

from tools.pre_commit.check_spdx_header import (
    SPDXStatus,
    add_header,
    check_spdx_header_status,
)

LICENSE_LINE = "# SPDX-License-Identifier: Apache-2.0"
COPYRIGHT_LINE = "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project"


def _write_tmp(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class TestCheckHeaderStatus:
    def test_complete_header(self) -> None:
        path = _write_tmp(f"{LICENSE_LINE}\n{COPYRIGHT_LINE}\nprint('hi')\n")
        try:
            assert check_spdx_header_status(path) == SPDXStatus.COMPLETE
        finally:
            os.unlink(path)

    def test_empty_file(self) -> None:
        path = _write_tmp("")
        try:
            assert check_spdx_header_status(path) == SPDXStatus.EMPTY
        finally:
            os.unlink(path)

    def test_missing_both(self) -> None:
        path = _write_tmp("print('hello')\n")
        try:
            assert check_spdx_header_status(path) == SPDXStatus.MISSING_BOTH
        finally:
            os.unlink(path)

    def test_missing_copyright(self) -> None:
        path = _write_tmp(f"{LICENSE_LINE}\nprint('hi')\n")
        try:
            assert check_spdx_header_status(path) == SPDXStatus.MISSING_COPYRIGHT
        finally:
            os.unlink(path)

    def test_missing_license(self) -> None:
        path = _write_tmp(f"{COPYRIGHT_LINE}\nprint('hi')\n")
        try:
            assert check_spdx_header_status(path) == SPDXStatus.MISSING_LICENSE
        finally:
            os.unlink(path)

    def test_shebang_with_header(self) -> None:
        path = _write_tmp(f"#!/usr/bin/env python3\n{LICENSE_LINE}\n{COPYRIGHT_LINE}\n")
        try:
            assert check_spdx_header_status(path) == SPDXStatus.COMPLETE
        finally:
            os.unlink(path)

    def test_spdx_mid_file_detected_as_missing(self) -> None:
        """SPDX lines past the header region should not count."""
        lines = [
            "print('start')\n",
            "# comment\n",
            "\n",
            "\n",
            "\n",
            "\n",
            f"{LICENSE_LINE}\n",
            f"{COPYRIGHT_LINE}\n",
        ]
        path = _write_tmp("".join(lines))
        try:
            assert check_spdx_header_status(path) == SPDXStatus.MISSING_BOTH
        finally:
            os.unlink(path)

    def test_spdx_inside_legacy_header_block_counts(self) -> None:
        lines = [
            "# Copyright (c) Meta Platforms, Inc. and affiliates.\n",
            "#\n",
            "# Licensed under the Apache License, Version 2.0 (the \"License\");\n",
            "# you may not use this file except in compliance with the License.\n",
            "#\n",
            f"{LICENSE_LINE}\n",
            f"{COPYRIGHT_LINE}\n",
            "\"\"\"Module docstring.\"\"\"\n",
        ]
        path = _write_tmp("".join(lines))
        try:
            assert check_spdx_header_status(path) == SPDXStatus.COMPLETE
        finally:
            os.unlink(path)

    def test_shebang_encoding_and_header(self) -> None:
        path = _write_tmp(
            "#!/usr/bin/env python3\n"
            "# -*- coding: utf-8 -*-\n"
            "\n"
            f"{LICENSE_LINE}\n"
            f"{COPYRIGHT_LINE}\n"
            "print('hi')\n"
        )
        try:
            assert check_spdx_header_status(path) == SPDXStatus.COMPLETE
        finally:
            os.unlink(path)


class TestAddHeader:
    def test_add_both_to_empty_body(self) -> None:
        path = _write_tmp("print('x')\n")
        try:
            add_header(path, SPDXStatus.MISSING_BOTH)
            assert check_spdx_header_status(path) == SPDXStatus.COMPLETE
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert content.startswith(LICENSE_LINE)
        finally:
            os.unlink(path)

    def test_add_both_preserves_shebang(self) -> None:
        path = _write_tmp("#!/usr/bin/env python3\nprint('x')\n")
        try:
            add_header(path, SPDXStatus.MISSING_BOTH)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert content.startswith("#!/usr/bin/env python3\n")
            assert check_spdx_header_status(path) == SPDXStatus.COMPLETE
        finally:
            os.unlink(path)

    def test_add_copyright_after_license(self) -> None:
        path = _write_tmp(f"{LICENSE_LINE}\nprint('x')\n")
        try:
            add_header(path, SPDXStatus.MISSING_COPYRIGHT)
            assert check_spdx_header_status(path) == SPDXStatus.COMPLETE
        finally:
            os.unlink(path)

    def test_add_license_before_copyright(self) -> None:
        path = _write_tmp(f"{COPYRIGHT_LINE}\nprint('x')\n")
        try:
            add_header(path, SPDXStatus.MISSING_LICENSE)
            assert check_spdx_header_status(path) == SPDXStatus.COMPLETE
        finally:
            os.unlink(path)
