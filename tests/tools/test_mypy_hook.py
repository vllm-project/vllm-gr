# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Regression checks for mypy target selection and enforcement."""

import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import regex as re
import yaml

from tools.pre_commit import mypy as hook

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_complete_core_directories_include_new_modules_and_stubs() -> None:
    paths = [
        "vllm_gr/v1/beam/new_module.py",
        "vllm_gr/v1/engine/new_module.pyi",
        "vllm_gr/v1/beam_extra/not_core.py",
        "vllm_gr/v1/engine_extra/not_core.py",
    ]
    assert hook.staged_targets(paths) == paths[:2]


def test_directory_matching_includes_nested_python_and_stubs() -> None:
    groups = hook.group_files(
        [
            "vllm_gr/top.py",
            "./vllm_gr/v1/engine/core.py",
            "vllm_gr/a.pyi",
            "vllm_gr/top.py",
            "vllm_gr_extra/no.py",
            "tests/test_one.py",
            "tests_extra/no.py",
            "vllm_gr/data.json",
            "/vllm_gr/no.py",
            "vllm_gr/../no.py",
        ]
    )
    assert groups == {
        "": ["vllm_gr/top.py", "vllm_gr/v1/engine/core.py", "vllm_gr/a.pyi"],
        "tests": ["tests/test_one.py"],
        hook.CUDA_STAGE_GROUP: [],
    }


def test_deferred_modules_are_visible_and_auditable(capsys: pytest.CaptureFixture[str]) -> None:
    paths = ["vllm_gr/v1/beam/geometry.py", "vllm_gr/constraints/startup.py"]
    assert hook.staged_targets(paths) == paths[:1]
    assert paths[1] in capsys.readouterr().out
    assert hook.staged_targets(paths, all_core=True) == paths


def check_failure_propagation(ci: str) -> None:
    target = "vllm_gr/v1/beam/geometry.py"
    with (
        patch.object(sys, "argv", ["mypy.py", ci, "3.12", target]),
        patch.object(hook, "mypy", return_value=1) as run,
    ):
        assert hook.main() == 1
    run.assert_called_once_with([target], "3.12", "skip", cuda_stage=False)


def test_hook_checks_core_and_propagates_mypy_failure() -> None:
    check_failure_propagation("0")
    check_failure_propagation("1")


def test_audit_checks_deferred_modules() -> None:
    target = "vllm_gr/constraints/startup.py"
    with (
        patch.object(sys, "argv", ["mypy.py", "0", "3.12", "--all-core", target]),
        patch.object(hook, "mypy", return_value=1) as run,
    ):
        assert hook.main() == 1
    run.assert_called_once_with([target], "3.12", "skip", cuda_stage=False)


def test_stage_boundary_is_enforced_without_all_core() -> None:
    target = hook.CUDA_STAGE_FILES[0]
    with (
        patch.object(sys, "argv", ["mypy.py", "1", "3.12", target]),
        patch.object(hook, "mypy", return_value=1) as run,
    ):
        assert hook.main() == 1
    run.assert_called_once_with(hook.CUDA_STAGE_FILES, "3.12", "skip", cuda_stage=True)


def test_precommit_keeps_stage_boundary_on_its_pinned_checker() -> None:
    config = yaml.safe_load(Path(".pre-commit-config.yaml").read_text())
    hooks = {item["id"]: item for repo in config["repos"] for item in repo["hooks"]}
    target = "vllm_gr/v1/beam/geometry.py"
    for version in ("3.11", "3.12", "3.13"):
        legacy = hooks[f"mypy-tests-{version}"]
        assert "mypy==1.11.1" in legacy["additional_dependencies"]
        assert all(
            re.search(legacy["files"], name)
            for name in [*hook.CUDA_STAGE_FILES, "tools/pre_commit/mypy.py"]
        )
        # A later refactor can add a worker boundary without updating YAML too.
        added = "vllm_gr/v1/worker/new_stage_boundary.py"
        assert re.search(legacy["files"], added)
        with patch.object(hook, "CUDA_STAGE_FILES", [*hook.CUDA_STAGE_FILES, added]):
            assert hook.group_files([added])[hook.CUDA_STAGE_GROUP] == hook.CUDA_STAGE_FILES
        runtime = hooks[f"mypy-{version}"]
        assert "mypy==2.4.1" in runtime["additional_dependencies"]
        with (
            patch.object(
                sys, "argv", [*shlex.split(runtime["entry"])[1:], target, hook.CUDA_STAGE_FILES[0]]
            ),
            patch.object(hook, "mypy", return_value=0) as run,
        ):
            assert hook.main() == 0
        run.assert_called_once_with([target], version, "skip", cuda_stage=False)
