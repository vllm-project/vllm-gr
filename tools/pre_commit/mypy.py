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
    python tools/pre_commit/mypy.py <ci> <python_version> <changed_files...>

Args:
    ci: "1" if running in CI, "0" otherwise. In CI, follow_imports is set to
        "silent" for the main group of files.
    python_version: Python version to use (e.g., "3.10") or "local" to use
        the local Python version.
    changed_files: List of changed files to check.
"""

import os
import subprocess
import sys

import regex as re

FILES = [
    "vllm_gr/*.py",
]

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

# After fixing errors resulting from changing follow_imports
# from "skip" to "silent", move the following directories to FILES
SEPARATE_GROUPS = [
    "tests",
    # v0 related
    "vllm_gr/attention",
    "vllm_gr/compilation",
    "vllm_gr/lora",
    "vllm_gr/model_executor",
    # v1 related
    "vllm_gr/v1/kv_offload",
    "vllm_gr/v1/spec_decode",
    "vllm_gr/v1/structured_output",
]

# TODO(woosuk): Include the code from Megatron and HuggingFace.
EXCLUDE = [
    "vllm_gr/engine/arg_utils.py",
    "vllm_gr/model_executor/parallel_utils",
    "vllm_gr/model_executor/models",
    "vllm_gr/model_executor/layers/fla/ops",
    # Ignore triton kernels in ops.
    "vllm_gr/attention/ops",
]


def group_files(changed_files: list[str]) -> dict[str, list[str]]:
    """
    Group changed files into different mypy calls.

    Args:
        changed_files: List of changed files.

    Returns:
        A dictionary mapping file group names to lists of changed files.
    """
    exclude_pattern = re.compile(f"^{'|'.join(EXCLUDE)}.*")
    files_pattern = re.compile(f"^({'|'.join(FILES)}).*")
    file_groups = {"": []}
    file_groups.update({k: [] for k in SEPARATE_GROUPS})
    file_groups[CUDA_STAGE_GROUP] = []
    file_groups[NPU_WORKER_GROUP] = []
    for changed_file in changed_files:
        if changed_file in NPU_WORKER_FILES:
            file_groups[NPU_WORKER_GROUP] = list(NPU_WORKER_FILES)
            continue
        if changed_file in CUDA_STAGE_FILES or changed_file == "tools/pre_commit/mypy.py":
            file_groups[CUDA_STAGE_GROUP] = list(CUDA_STAGE_FILES)
            continue
        # Skip files which should be ignored completely
        if exclude_pattern.match(changed_file):
            continue
        # Group files by mypy call
        if files_pattern.match(changed_file):
            file_groups[""].append(changed_file)
            continue
        else:
            for directory in SEPARATE_GROUPS:
                if re.match(f"^{directory}.*", changed_file):
                    file_groups[directory].append(changed_file)
                    break
    return file_groups


def mypy(
    targets: list[str],
    python_version: str | None,
    follow_imports: str | None,
    *,
    cuda_stage: bool = False,
) -> int:
    """
    Run mypy on the given targets.

    Args:
        targets: List of files or directories to check.
        python_version: Python version to use (e.g., "3.10") or None to use
            the default mypy version.
        follow_imports: Value for the --follow-imports option or None to use
            the default mypy behavior.
        cuda_stage: Check the production hooks boundary and expected errors
            without reading or writing mypy caches.
    Returns:
        The return code from mypy.
    """
    args = ["mypy"]
    if python_version is not None:
        args += ["--python-version", python_version]
    if follow_imports is not None:
        args += ["--follow-imports", follow_imports]
    args += ["--ignore-missing-imports"]
    if cuda_stage:
        # mypy 1.11.1 cannot serialize some placeholders in the cyclic native
        # imports. Disable cache reads AND writes for this small boundary.
        args += ["--no-incremental", "--cache-dir", os.devnull, "--warn-unused-ignores"]
    print(f"$ {' '.join(args)} {targets}")
    return subprocess.run(args + targets, check=False).returncode


def main():
    ci = sys.argv[1] == "1"
    python_version = sys.argv[2]
    # Remove duplicate files (e.g. from unmerged git conflicts) to avoid mypy duplicate module errors
    unique_files = list(dict.fromkeys(sys.argv[3:]))
    file_groups = group_files(unique_files)

    if python_version == "local":
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    returncode = 0
    for file_group, changed_files in file_groups.items():
        follow_imports = None if ci and file_group == "" else "skip"
        if changed_files:
            returncode |= mypy(
                changed_files,
                python_version,
                follow_imports,
                cuda_stage=file_group in {CUDA_STAGE_GROUP, NPU_WORKER_GROUP},
            )
    return returncode


if __name__ == "__main__":
    sys.exit(main())
