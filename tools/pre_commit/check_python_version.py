#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ensure pre-commit runs under a supported Python (3.11, 3.12, or 3.13)."""

from __future__ import annotations

import sys

SUPPORTED = frozenset({(3, 11), (3, 12), (3, 13)})


def main() -> None:
    version = sys.version_info[:2]
    if version in SUPPORTED:
        return
    major, minor = version
    supported = ", ".join(f"3.{m}" for _, m in sorted(SUPPORTED))
    print(
        f"Unsupported Python {major}.{minor} for pre-commit.\n"
        f"Use one of: {supported}.\n"
        "Create/activate a venv with e.g. `uv venv --python 3.12 .venv`.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
