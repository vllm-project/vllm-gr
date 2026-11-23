# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import pytest

from tests.helpers.paths import REPO_ROOT, TESTS_DIR

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_repo_root_is_the_checkout_that_contains_the_package() -> None:
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "vllm_gr" / "__init__.py").is_file()


def test_tests_dir_is_the_directory_that_holds_this_package() -> None:
    assert TESTS_DIR == REPO_ROOT / "tests"
    assert (TESTS_DIR / "helpers" / "paths.py").is_file()
