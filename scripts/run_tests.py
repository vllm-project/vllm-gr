#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unified test runner for the reduced core vllm-GR test matrix.

Runs core correctness suites plus optional GPU/NPU smoke lanes and examples.

Usage:
    python scripts/run_tests.py
    python scripts/run_tests.py --suite gpu-smoke
    python scripts/run_tests.py --suite npu-smoke
    python scripts/run_tests.py --list-suites
    python scripts/run_tests.py --all
    python scripts/run_tests.py --all --auto-platform
    python scripts/run_tests.py --examples-only
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")


SUITE_DEFINITIONS: dict[str, dict[str, object]] = {
    "unit": {
        "label": "Core Unit Tests",
        "runner": "pytest",
        "files": [
            "test_engine.py",
            "test_gr_schema_contract.py",
            "test_processors.py",
            "test_embedding_collection_patch.py",
            "test_gr_runner_core.py",
            "test_dataset_config_parity.py",
            "test_checkpoint_conversion_contract.py",
            "test_output_contract.py",
            "test_positional_encoder_parity.py",
            "test_run_tests_script.py",
            # vLLM runtime merge Phase 0/1
            "runtime/test_runtime_selection.py",
            "runtime/test_static_descriptors.py",
            # Phase 2 — plugin two-phase split
            "runtime/test_patch_plan.py",
            "runtime/test_activation.py",
            "runtime/test_plugin_register_integration.py",
            # Phase 2 — subprocess isolation tests (§9.2)
            "native/test_import_hygiene.py",
            "native/test_native_protection.py",
            "hstu/test_hstu_patch_guard.py",
            "tools/test_check_spdx_header.py",
            "test_hstu_attention_contract.py",
            "test_hstu_vllm_native_kv_contract.py",
            "test_hstu_prefix_cache_contract.py",
            "test_prefix_cache_benchmark_input.py",
            "test_yuanrong_monkey_patch.py",
            "test_yuanrong_backend.py",
            # Phase 1 — adapter abstraction
            "test_gr_model_adapter_interface.py",
            "test_gr_capabilities.py",
            "test_async_gr_engine.py",
            "test_gr_config_pickle.py",
            "test_batch_recommendation_contract.py",
            "test_output_processor_adapter_e2e.py",
            "test_hstu_adapter_contract.py",
            # Phase 2 — type hierarchy
            "test_inputs_base.py",
            "test_hstu_prompt.py",
            "test_outputs_phase2.py",
            "test_outputs_hierarchy.py",
            "test_reexport.py",
            # Phase 3 — adapter registry
            "test_registry.py",
        ],
    },
    "gpu-smoke": {
        "label": "GPU Smoke Tests",
        "runner": "python",
        "files": [
            "test_gpu_smoke.py",
        ],
    },
    "npu-smoke": {
        "label": "NPU Smoke Tests",
        "runner": "python",
        "files": [
            "test_npu_smoke.py",
        ],
    },
    "platform-gpu": {
        "label": "GPU Platform Lane",
        "includes": ["gpu-smoke"],
    },
    "platform-npu": {
        "label": "NPU Platform Lane",
        "includes": ["npu-smoke"],
    },
    "platform-all": {
        "label": "All Platform Lanes",
        "includes": ["platform-gpu", "platform-npu"],
    },
}


def detect_runtime_backend() -> str:
    """Prefer NPU, then GPU, then CPU for auto-platform selection."""
    try:
        from vllm_gr.utils.npu_adapter import (
            is_cuda_available,
            is_npu_available,
        )
    except Exception as exc:
        print(
            "  [WARN] Failed to import runtime backend detection helpers; "
            f"defaulting to CPU ({exc})."
        )
        return "cpu"

    try:
        if is_npu_available():
            return "npu"
        if is_cuda_available():
            return "gpu"
    except Exception as exc:
        print(
            "  [WARN] Failed while probing runtime backend; "
            f"defaulting to CPU ({exc})."
        )
    return "cpu"


def runtime_backend_to_platform_suite(runtime_backend: str) -> str | None:
    if runtime_backend == "npu":
        return "platform-npu"
    if runtime_backend == "gpu":
        return "platform-gpu"
    return None


def apply_auto_platform_selection(
    selected: list[str],
    runtime_backend: str,
) -> list[str]:
    auto_suite = runtime_backend_to_platform_suite(runtime_backend)
    normalized: list[str] = []
    for suite_name in selected:
        if suite_name != "platform-all":
            normalized.append(suite_name)
            continue
        if auto_suite is not None:
            normalized.append(auto_suite)
    return normalized


def run_command(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    elapsed = time.time() - start

    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {label}  ({elapsed:.1f}s)")
    return result.returncode == 0


def _expand_suite(
    suite_name: str,
    expanded: list[str],
    seen: set[str],
    stack: set[str],
) -> None:
    if suite_name in seen:
        return
    if suite_name in stack:
        raise ValueError(f"Cyclic suite definition detected: {suite_name}")
    if suite_name not in SUITE_DEFINITIONS:
        raise KeyError(f"Unknown suite: {suite_name}")

    stack.add(suite_name)
    suite = SUITE_DEFINITIONS[suite_name]
    for included in suite.get("includes", []):
        _expand_suite(str(included), expanded, seen, stack)
    stack.remove(suite_name)

    if "files" in suite:
        expanded.append(suite_name)
        seen.add(suite_name)


def resolve_suites(selected: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for suite_name in selected:
        _expand_suite(suite_name, expanded, seen, set())
    return expanded


def build_pytest_command(files: list[str], verbose: bool = False) -> list[str]:
    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend(os.path.join(TESTS_DIR, file_name) for file_name in files)
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "-q"])
    return cmd


def build_unittest_command(files: list[str], verbose: bool = False) -> list[str]:
    del verbose
    modules = [
        f"tests.{os.path.splitext(file_name)[0]}"
        for file_name in files
    ]
    return [sys.executable, "-m", "unittest", *modules]


def _pytest_available() -> bool:
    return importlib.util.find_spec("pytest") is not None


def _fallback_unittest_files(files: list[str]) -> list[str]:
    return list(files)


def run_pytest_suite(suite_name: str, verbose: bool = False) -> bool:
    suite = SUITE_DEFINITIONS[suite_name]
    files = suite.get("files", [])
    if not files:
        raise ValueError(f"Suite {suite_name!r} does not define files")

    if not _pytest_available():
        fallback_files = _fallback_unittest_files(list(files))
        if not fallback_files:
            print(
                "  [FAILED] pytest is not installed and no unittest fallback "
                f"is defined for suite {suite_name!r}."
            )
            return False

        print(
            "  [WARN] pytest is not installed; "
            f"falling back to unittest subset for suite {suite_name!r}."
        )
        cmd = build_unittest_command(fallback_files)
        return run_command(
            cmd,
            f"{suite['label']} (unittest fallback)",
        )

    cmd = build_pytest_command(list(files), verbose=verbose)
    return run_command(cmd, str(suite["label"]))


def run_python_suite(suite_name: str) -> bool:
    suite = SUITE_DEFINITIONS[suite_name]
    files = suite.get("files", [])
    if not files:
        raise ValueError(f"Suite {suite_name!r} does not define files")

    results = []
    for file_name in files:
        cmd = [sys.executable, os.path.join(TESTS_DIR, file_name)]
        results.append(run_command(cmd, f"{suite['label']}: {file_name}"))
    return all(results)


def run_example(script: str, extra_args: list[str] | None = None) -> bool:
    path = os.path.join(EXAMPLES_DIR, script)
    if not os.path.exists(path):
        print(f"  [SKIP] {script} not found")
        return True

    cmd = [sys.executable, path]
    if extra_args:
        cmd.extend(extra_args)
    return run_command(cmd, f"Example: {script}")


def run_examples(
    datasets: list[str] | None = None,
    expected_device: str | None = None,
) -> bool:
    results = []

    example_runtime_args: list[str] = []
    if expected_device:
        example_runtime_args.extend(["--expect-device", expected_device])

    results.append(run_example("simple_inference.py", example_runtime_args))

    if datasets:
        for ds in datasets:
            results.append(
                run_example(
                    "dataset_inference.py",
                    [*example_runtime_args, ds],
                )
            )
    else:
        results.append(
            run_example(
                "dataset_inference.py",
                [*example_runtime_args, "debug"],
            )
        )

    return all(results)


def print_suite_catalog() -> None:
    print("=" * 60)
    print("  Available Test Suites")
    print("=" * 60)
    for name, suite in SUITE_DEFINITIONS.items():
        label = suite["label"]
        runner = suite.get("runner", "composite")
        if "files" in suite:
            files = ", ".join(suite["files"])
            print(f"  {name:15s} {label}")
            print(f"  runner: {runner}")
            print(f"  files: {files}")
        else:
            includes = ", ".join(suite.get("includes", []))
            print(f"  {name:15s} {label}")
            print(f"  includes: {includes}")
    print("=" * 60)


def determine_requested_suites(
    args: argparse.Namespace,
) -> tuple[list[str], str | None]:
    runtime_backend = detect_runtime_backend() if args.auto_platform else None

    if args.examples_only:
        if args.suite:
            raise ValueError("--examples-only cannot be combined with --suite")
        return [], runtime_backend

    if args.suite:
        selected = list(args.suite)
    elif args.all:
        selected = ["unit", "platform-all"]
    else:
        selected = ["unit"]

    if runtime_backend is not None:
        selected = apply_auto_platform_selection(selected, runtime_backend)

    return resolve_suites(selected), runtime_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run vllm-GR tests and examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run core unit tests, all platform lanes, and examples",
    )
    parser.add_argument(
        "--auto-platform",
        action="store_true",
        help=(
            "Resolve platform-all to the currently available backend "
            "(NPU preferred over GPU) and assert the same backend in examples"
        ),
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Run example scripts only (skip pytest suites)",
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=sorted(SUITE_DEFINITIONS),
        help=(
            "Run one or more named test suites. "
            "Examples: unit, gpu-smoke, npu-smoke, "
            "platform-gpu, platform-npu, platform-all"
        ),
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="Print all available test suite names and exit",
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        help=(
            "Dataset(s) for dataset_inference.py "
            "(e.g. debug, movielens-1m, kuairand-1k, streaming)"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose pytest output",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_suites:
        print_suite_catalog()
        return

    requested_suites, runtime_backend = determine_requested_suites(args)

    print("=" * 60)
    print("  vllm-GR Test Runner")
    print("=" * 60)

    if args.auto_platform:
        auto_suite = runtime_backend_to_platform_suite(runtime_backend or "cpu")
        if auto_suite is None:
            print(
                "  Auto platform backend: cpu "
                "(platform-all will be skipped)"
            )
        else:
            print(
                "  Auto platform backend: "
                f"{runtime_backend} ({auto_suite})"
            )

    if requested_suites:
        print(f"  Selected suites: {', '.join(requested_suites)}")
    elif args.examples_only:
        print("  Selected suites: examples-only")
    else:
        print("  Selected suites: none")

    results: list[tuple[str, bool]] = []
    start = time.time()

    for suite_name in requested_suites:
        suite_label = str(SUITE_DEFINITIONS[suite_name]["label"])
        runner = SUITE_DEFINITIONS[suite_name].get("runner", "pytest")
        if runner == "python":
            passed = run_python_suite(suite_name)
        else:
            passed = run_pytest_suite(suite_name, args.verbose)
        results.append((suite_label, passed))

    if args.all or args.examples_only:
        results.append((
            "Examples",
            run_examples(
                args.dataset,
                expected_device=runtime_backend if args.auto_platform else None,
            ),
        ))

    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"  Summary  ({elapsed:.1f}s total)")
    print(f"{'=' * 60}")
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {status:8s}  {name}")
        if not passed:
            all_passed = False

    print(f"{'=' * 60}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
