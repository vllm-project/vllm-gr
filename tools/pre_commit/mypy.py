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
    ci: Compatibility argument ("1" for CI, "0" locally). Both modes use
        follow_imports="skip" and the same staged strict coverage.
    python_version: Python version to use (e.g., "3.10") or "local" to use
        the local Python version.
    changed_files: List of changed files to check. Add --all-core to audit
        runtime files outside the enforced core directories.
"""

import os
import subprocess
import sys
from pathlib import PurePosixPath

# Directory roots, not glob expressions interpreted as regular expressions.
FILES = ("vllm_gr",)
SEPARATE_GROUPS = ("tests",)

# Check the interface, implementations and execution/startup bindings together.
# Explicit targets are checked even with --follow-imports skip; native Worker
# internals outside this boundary remain dynamically typed.
CUDA_STAGE_GROUP = "cuda-stage"
CUDA_STAGE_FILES = [
    "vllm_gr/v1/worker/gpu_stage_hooks.py",
    "vllm_gr/v1/worker/gpu_beam_stage_runner.py",
    "vllm_gr/v1/worker/gpu_beam_stage_runner_v1.py",
    "vllm_gr/v1/worker/gpu_beam_stage_runner_v2.py",
    "vllm_gr/v1/worker/gr_execution.py",
    "vllm_gr/v1/worker/gr_startup.py",
    "vllm_gr/v1/worker/model_runner_common.py",
    "vllm_gr/v1/worker/model_runner_patch.py",
    "vllm_gr/v1/worker/gpu_model_runner_v2_graph.py",
    "tests/tools/cuda_stage_typing_cases.py",
]

# Enforce complete directories, including newly added modules and stubs.
# Other runtime subsystems can be audited with --all-core before onboarding.
CORE_DIRS = ("vllm_gr/v1/beam", "vllm_gr/v1/engine")


def group_files(changed_files: list[str]) -> dict[str, list[str]]:
    """Select Python files using exact path components, including nested files."""
    groups: dict[str, list[str]] = {"": [], "tests": [], CUDA_STAGE_GROUP: []}
    for name in dict.fromkeys(str(PurePosixPath(f)) for f in changed_files):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.suffix not in {".py", ".pyi"}:
            continue
        if name in CUDA_STAGE_FILES or name == "tools/pre_commit/mypy.py":
            groups[CUDA_STAGE_GROUP] = list(CUDA_STAGE_FILES)
            continue
        if path.parts[0] in FILES:
            groups[""].append(name)
        elif path.parts[0] in SEPARATE_GROUPS:
            groups["tests"].append(name)
    return groups


def staged_targets(targets: list[str], *, all_core: bool = False) -> list[str]:
    """Report deferred runtime files instead of silently pretending to check them."""
    selected = []
    deferred = []
    for name in targets:
        if (
            all_core
            or not name.startswith("vllm_gr/")
            or any(PurePosixPath(name).is_relative_to(directory) for directory in CORE_DIRS)
        ):
            selected.append(name)
        else:
            deferred.append(name)
    if deferred:
        print("Deferred runtime files (outside current strict coverage):", flush=True)
        for name in deferred:
            print(f"  {name}", flush=True)
        print("Use --all-core to audit them; see docs/PRE_COMMIT.md.", flush=True)
    return selected


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


def main() -> int:
    # Keep the CI argument for compatibility; both modes use the same coverage.
    python_version = sys.argv[2]
    # Remove duplicate files (e.g. from unmerged git conflicts) to avoid mypy duplicate module errors
    all_core = "--all-core" in sys.argv[3:]
    skip_cuda_stage = "--skip-cuda-stage" in sys.argv[3:]
    unique_files = [
        name for name in sys.argv[3:] if name not in {"--all-core", "--skip-cuda-stage"}
    ]
    file_groups = group_files(unique_files)

    if python_version == "local":
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    returncode = 0
    for file_group, changed_files in file_groups.items():
        if file_group == CUDA_STAGE_GROUP and skip_cuda_stage:
            continue
        if file_group != CUDA_STAGE_GROUP:
            changed_files = staged_targets(changed_files, all_core=all_core)
        follow_imports = "skip"
        if changed_files:
            returncode |= mypy(
                changed_files,
                python_version,
                follow_imports,
                cuda_stage=file_group == CUDA_STAGE_GROUP,
            )
    return returncode


if __name__ == "__main__":
    sys.exit(main())
