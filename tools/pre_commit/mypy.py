# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm
"""
Run mypy on changed files.

This script is designed to be used as a pre-commit hook. It runs mypy
on files that have been changed. It groups files into different mypy calls
based on their directory to avoid import following issues.

Usage:
    python tools/pre_commit/mypy.py <python_version> <changed_files...>

Args:
    python_version: Python version to use (e.g., "3.10") or "local" to use
        the local Python version.
    changed_files: List of changed files to check.
"""

import os
import posixpath
import subprocess
import sys

GENERATED_GRPC_PREFIX = "vllm_gr/entrypoints/grpc/generated/"

# Check the interface, implementations and execution/startup bindings together.
# Explicit targets are checked even with --follow-imports skip; native Worker
# internals outside this boundary remain dynamically typed.
CUDA_STAGE_GROUP = "cuda-stage"
CUDA_STAGE_FILES = [
    "vllm_gr/v1/worker/cuda/gpu_stage_hooks.py",
    "vllm_gr/v1/worker/cuda/gpu_beam_stage_runner.py",
    "vllm_gr/v1/worker/cuda/v1/gpu_beam_stage_runner_v1.py",
    "vllm_gr/v1/worker/cuda/v2/gpu_beam_stage_runner_v2.py",
    "vllm_gr/v1/worker/gr_execution.py",
    "vllm_gr/v1/worker/gr_startup.py",
    "vllm_gr/v1/worker/model_runner_common.py",
    "vllm_gr/v1/worker/model_runner_contracts.py",
    "vllm_gr/v1/worker/gr_gc.py",
    "vllm_gr/v1/worker/gr_context.py",
    "vllm_gr/v1/worker/model_runner_patch.py",
    "vllm_gr/v1/worker/cuda/v2/gpu_model_runner_v2_patch.py",
    "vllm_gr/v1/worker/cuda/v2/gpu_model_runner_v2_resources.py",
    "vllm_gr/v1/worker/cuda/gr_inputs.py",
    "vllm_gr/v1/worker/cuda/beam_decode_graph.py",
    "tests/tools/cuda_stage_typing_cases.py",
]

# Check the NPU input/bookkeeping helpers together with both production callers
# so a signature change cannot escape caller-side validation.
NPU_WORKER_GROUP = "npu-worker"
NPU_WORKER_FILES = [
    "vllm_gr/v1/worker/npu/npu_model_runner_patch.py",
    "vllm_gr/v1/worker/npu/npu_beam_stage_runner.py",
    "vllm_gr/v1/worker/npu/npu_model_runner_inputs.py",
    "vllm_gr/v1/worker/npu/npu_model_runner_bookkeeping.py",
]


def group_files(changed_files: list[str]) -> dict[str, list[str]]:
    """Group production and test files and expand runner contract checks."""
    groups: dict[str, list[str]] = {
        "": [],
        "tests": [],
        CUDA_STAGE_GROUP: [],
        NPU_WORKER_GROUP: [],
    }
    normalized_paths = (posixpath.normpath(path.replace("\\", "/")) for path in changed_files)
    for path in dict.fromkeys(normalized_paths):
        if path in CUDA_STAGE_FILES or path == "tools/pre_commit/mypy.py":
            groups[CUDA_STAGE_GROUP] = list(CUDA_STAGE_FILES)
        elif path in NPU_WORKER_FILES:
            groups[NPU_WORKER_GROUP] = list(NPU_WORKER_FILES)
        elif not path.endswith((".py", ".pyi")) or path.startswith(GENERATED_GRPC_PREFIX):
            continue
        elif path.startswith("vllm_gr/"):
            groups[""].append(path)
        elif path.startswith("tests/"):
            groups["tests"].append(path)
    return groups


def mypy(
    targets: list[str],
    python_version: str | None,
    follow_imports: str | None,
    file_group: str,
) -> int:
    """
    Run mypy on the given targets.

    Args:
        targets: List of files or directories to check.
        python_version: Python version to use (e.g., "3.10") or None to use
            the default mypy version.
        follow_imports: Value for the --follow-imports option or None to use
            the default mypy behavior.
        file_group: Group name used to select its checks and report coverage.
    Returns:
        The return code from mypy.
    """
    args = ["mypy"]
    if python_version is not None:
        args += ["--python-version", python_version]
    if follow_imports is not None:
        args += ["--follow-imports", follow_imports]
    if file_group:
        args.append("--strict")
    if file_group in {CUDA_STAGE_GROUP, NPU_WORKER_GROUP}:
        # mypy 1.11.1 cannot serialize some placeholders in the cyclic native
        # imports. Disable cache reads AND writes for this small boundary.
        args += ["--no-incremental", "--cache-dir", os.devnull, "--warn-unused-ignores"]
    print(f"$ {' '.join(args)} ({file_group or 'production'}: {len(targets)} files)")
    return subprocess.run(args + targets, check=False).returncode


def main():
    python_version = sys.argv[1]
    file_groups = group_files(sys.argv[2:])

    if python_version == "local":
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    returncode = 0
    for file_group, changed_files in file_groups.items():
        follow_imports = None if file_group == "" else "skip"
        if changed_files:
            returncode |= mypy(changed_files, python_version, follow_imports, file_group)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
