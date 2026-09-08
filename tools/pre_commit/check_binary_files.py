#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pre-commit hook to prevent binary files from being committed.

This script checks the staged files and fails if any of them is binary
(contains a NUL byte in its initial bytes), unless the file matches one
of the exclusion patterns below.
"""

import fnmatch
import sys
from pathlib import Path

# Paths matching any of these patterns are allowed to be binary.
# Patterns are matched against the full path (fnmatch syntax).
EXCLUDED_PATHS = ["docs/assets/vLLM-GR.png"]

# Number of bytes to inspect to decide whether a file is binary
SAMPLE_SIZE = 8192


def is_excluded(file_path: Path) -> bool:
    """Return True if the file matches one of the exclusion patterns."""
    path_str = str(file_path)
    return any(fnmatch.fnmatch(path_str, pattern) for pattern in EXCLUDED_PATHS)


def is_binary(file_path: Path) -> bool:
    """Return True if the file appears to be binary (contains a NUL byte)."""
    try:
        with open(file_path, "rb") as f:
            sample = f.read(SAMPLE_SIZE)
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False

    return b"\x00" in sample


def main():
    """Main entry point for the pre-commit hook."""
    # Get files from stdin (pre-commit passes filenames via stdin)
    if len(sys.argv) > 1:
        files = [Path(f) for f in sys.argv[1:]]
    else:
        # If no arguments, read from stdin (pre-commit default behavior)
        files = [Path(line.strip()) for line in sys.stdin if line.strip()]

    binary_files = []

    for file_path in files:
        if not file_path.exists():
            continue

        if is_excluded(file_path):
            continue

        if is_binary(file_path):
            binary_files.append(file_path)

    if binary_files:
        print("ERROR: Found binary files:", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Binary files should not be committed. If this file must be added,",
            file=sys.stderr,
        )
        print(
            "add it to the EXCLUDED_PATHS list in tools/pre_commit/check_binary_files.py.",
            file=sys.stderr,
        )
        print("", file=sys.stderr)

        for file_path in binary_files:
            print(f"  {file_path}", file=sys.stderr)

        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
