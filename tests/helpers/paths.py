# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Repository locations for tests.

Tests must not derive the repository root from ``Path(__file__).parents[n]``:
the depth changes whenever a test file moves between directories.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"
