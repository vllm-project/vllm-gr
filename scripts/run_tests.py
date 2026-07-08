#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unified test runner for the vllm-gr test matrix.

Exposes two named suites used by CI and local development:

- ``unit``      — lightweight UT (beam search, catalog, patch helpers,
                  Pangu passthrough routing). No model download, no GPU.
- ``pangu1b-st`` — Pangu 1B system tests (offline + online). Requires
                  the Pangu 1B checkpoint and a GPU/NPU.

PR gate runs ``--all`` which expands to ``unit`` only; ``pangu1b-st`` is
intended for nightly / manual runs.

Usage:
    python scripts/run_tests.py
    python scripts/run_tests.py --suite pangu1b-st
    python scripts/run_tests.py --list-suites
    python scripts/run_tests.py --all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT_DIR, "tests")


SUITE_DEFINITIONS: dict[str, dict[str, object]] = {
    "unit": {
        "label": "Core Unit Tests",
        "runner": "pytest",
        "files": [
            "test_yuanrong_monkey_patch.py",
            "test_yuanrong_backend.py",
            "test_split_qkv_rmsnorm_rope_patch.py",
            # Beam search / catalog (OneRec path)
            "test_catalog.py",
            "test_task_specific_catalog_program.py",
            # Pangu LLM passthrough routing
            "test_pangu_llm_passthrough.py",
        ],
    },
    "pangu1b-st": {
        "label": "Pangu 1B System Tests (offline + online)",
        "runner": "pytest",
        "files": [
            "st/test_pangu1b_accuracy_st.py",
            "st/test_pangu1b_online_st.py",
        ],
    },
}


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


def resolve_suites(selected: list[str]) -> list[str]:
    """Validate suite names and preserve selection order."""
    resolved: list[str] = []
    for suite_name in selected:
        if suite_name not in SUITE_DEFINITIONS:
            raise KeyError(f"Unknown suite: {suite_name}")
        if suite_name not in resolved:
            resolved.append(suite_name)
    return resolved


def build_pytest_command(files: list[str], verbose: bool = False) -> list[str]:
    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend(os.path.join(TESTS_DIR, file_name) for file_name in files)
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "-q"])
    return cmd


def run_pytest_suite(suite_name: str, verbose: bool = False) -> bool:
    suite = SUITE_DEFINITIONS[suite_name]
    files = list(suite.get("files", []))
    if not files:
        raise ValueError(f"Suite {suite_name!r} does not define files")
    cmd = build_pytest_command(files, verbose=verbose)
    return run_command(cmd, str(suite["label"]))


def print_suite_catalog() -> None:
    print("=" * 60)
    print("  Available Test Suites")
    print("=" * 60)
    for name, suite in SUITE_DEFINITIONS.items():
        label = suite["label"]
        runner = suite.get("runner", "pytest")
        files = ", ".join(suite.get("files", []))
        print(f"  {name:15s} {label}")
        print(f"  runner: {runner}")
        print(f"  files: {files}")
    print("=" * 60)


def determine_requested_suites(args: argparse.Namespace) -> list[str]:
    if args.suite:
        selected = list(args.suite)
    elif args.all:
        # PR gate: unit only. pangu1b-st is for nightly / manual runs.
        selected = ["unit"]
    else:
        selected = ["unit"]
    return resolve_suites(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run vllm-gr test suites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the default PR gate suite (currently: unit).",
    )
    parser.add_argument(
        "--auto-platform",
        action="store_true",
        help="(deprecated no-op, kept for old CI) platform suites were removed",
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=sorted(SUITE_DEFINITIONS),
        help=("Run one or more named test suites. Examples: unit, pangu1b-st"),
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="Print all available test suite names and exit",
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

    requested_suites = determine_requested_suites(args)

    print("=" * 60)
    print("  vllm-gr Test Runner")
    print("=" * 60)

    if requested_suites:
        print(f"  Selected suites: {', '.join(requested_suites)}")
    else:
        print("  Selected suites: none")

    results: list[tuple[str, bool]] = []
    start = time.time()

    for suite_name in requested_suites:
        suite_label = str(SUITE_DEFINITIONS[suite_name]["label"])
        runner = SUITE_DEFINITIONS[suite_name].get("runner", "pytest")
        if runner == "python":
            # No suite currently uses the python runner; keep the branch
            # so future standalone-script suites work without rewriting.
            raise ValueError(f"Suite {suite_name!r} uses python runner which is not configured")
        else:
            passed = run_pytest_suite(suite_name, args.verbose)
        results.append((suite_label, passed))

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
