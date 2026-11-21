# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Tests for the pre-commit mypy wrapper's routing and checker arguments."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tools.pre_commit import mypy as mypy_hook

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize(  # type: ignore[misc]
    "changed_file",
    sorted(
        {
            *mypy_hook.CUDA_STAGE_FILES,
            "tools/pre_commit/mypy.py",
            # Pin moved owners independently of the routing list itself.
            "vllm_gr/v1/worker/model_runner_contracts.py",
            "vllm_gr/v1/worker/gr_gc.py",
            "vllm_gr/v1/worker/gr_context.py",
            "vllm_gr/v1/worker/cuda/v2/gpu_model_runner_v2_patch.py",
            "vllm_gr/v1/worker/cuda/v2/gpu_model_runner_v2_resources.py",
        }
    ),
)
def test_mypy_checks_complete_boundary(changed_file: str) -> None:
    """Changing any boundary member checks its peers and negative cases too."""
    groups = mypy_hook.group_files([changed_file, "tests/v1/worker/test_gr_startup.py"])
    assert groups.pop(mypy_hook.CUDA_STAGE_GROUP) == mypy_hook.CUDA_STAGE_FILES
    assert groups.pop("tests") == ["tests/v1/worker/test_gr_startup.py"]
    assert not any(groups.values())


@pytest.mark.parametrize(  # type: ignore[misc]
    "changed_file",
    sorted(mypy_hook.NPU_WORKER_FILES),
)
def test_npu_worker_mypy_checks_all_callers(changed_file: str) -> None:
    """Changing either NPU helper or caller checks the whole NPU worker boundary."""
    groups = mypy_hook.group_files([changed_file, "tests/test_gr_startup.py"])
    assert groups.pop(mypy_hook.NPU_WORKER_GROUP) == mypy_hook.NPU_WORKER_FILES
    assert groups.pop("tests") == ["tests/test_gr_startup.py"]
    assert not any(groups.values())


def test_mypy_routes_production_files_at_any_depth() -> None:
    paths = [
        "vllm_gr/_env_check.py",
        "vllm_gr/config/models.py",
        "vllm_gr/engine/arg_utils.py",
        "vllm_gr/v1/worker/types.pyi",
    ]
    groups = mypy_hook.group_files(
        [
            *paths,
            paths[0],
            "vllm_gr/entrypoints/grpc/generated/common_pb2.py",
            "tests/config/test_schema.py",
            "tools/codegen/generate_grpc_proto.py",
        ]
    )
    assert groups.pop("") == paths
    assert groups.pop("tests") == ["tests/config/test_schema.py"]
    assert not any(groups.values())


def test_mypy_boundary_avoids_cache_and_requires_negative_errors(monkeypatch: Any) -> None:
    """Pin checker flags without starting a checker subprocess in pytest."""
    commands: list[list[str]] = []

    def run(args: list[str], *, check: bool) -> Any:
        assert check is False
        commands.append(args)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(mypy_hook.subprocess, "run", run)
    assert (
        mypy_hook.mypy(mypy_hook.CUDA_STAGE_FILES, "3.12", "skip", mypy_hook.CUDA_STAGE_GROUP) == 7
    )
    command = commands[0]
    assert "--strict" in command
    assert "--no-incremental" in command
    assert command[command.index("--cache-dir") + 1] == mypy_hook.os.devnull
    assert "--warn-unused-ignores" in command
    assert command[-len(mypy_hook.CUDA_STAGE_FILES) :] == mypy_hook.CUDA_STAGE_FILES


@pytest.mark.parametrize(  # type: ignore[misc]
    "changed_files",
    [
        ["./vllm_gr/config/models.py"],
        [r"vllm_gr\config\models.py"],
        [
            r".\vllm_gr\config\models.py",
            "./vllm_gr/config/models.py",
            "vllm_gr/config/models.py",
            r".\vllm_gr\entrypoints\grpc\generated\common_pb2.py",
        ],
    ],
)
def test_mypy_normalizes_direct_invocation(changed_files: list[str], monkeypatch: Any) -> None:
    checker = Mock(return_value=0)
    monkeypatch.setattr(mypy_hook, "mypy", checker)
    monkeypatch.setattr(mypy_hook.sys, "argv", ["tools/pre_commit/mypy.py", "3.12", *changed_files])

    assert mypy_hook.main() == 0
    checker.assert_called_once_with(["vllm_gr/config/models.py"], "3.12", None, "")
