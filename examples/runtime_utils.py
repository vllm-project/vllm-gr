#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Shared helpers for cross-platform example scripts."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def add_repo_root_to_path(script_file: str) -> str:
    """Ensure the repository root is importable for local example execution."""
    script_dir = os.path.dirname(os.path.abspath(script_file))
    repo_root = os.path.dirname(script_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def add_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: Optional[str] = None,
    include_model: bool = True,
) -> argparse.ArgumentParser:
    """Add shared runtime arguments used by example scripts."""
    if include_model:
        parser.add_argument(
            "--model",
            default=default_model,
            help="Model directory to load. Defaults to the example's built-in model path.",
        )
    parser.add_argument(
        "--expect-device",
        choices=["gpu", "npu", "cpu"],
        default=None,
        help=(
            "Optional assertion for the active runtime backend. "
            "Use this to verify the example really ran on GPU or NPU."
        ),
    )
    return parser


def normalize_expected_device(device: Optional[str]) -> Optional[str]:
    """Map user-facing device labels to vLLM platform labels."""
    if device is None:
        return None
    normalized = device.lower()
    if normalized == "gpu":
        return "cuda"
    return normalized


def describe_runtime(expected_device: Optional[str] = None) -> str:
    """Print the active vLLM runtime backend and verify it when requested."""
    from vllm.platforms import current_platform

    actual_device = str(getattr(current_platform, "device_type", "unknown"))

    print("\n[Runtime]")
    print(f"   Active platform: {actual_device}")

    normalized_expected = normalize_expected_device(expected_device)
    if normalized_expected is not None and actual_device != normalized_expected:
        raise RuntimeError(
            "Example runtime backend mismatch: "
            f"expected {normalized_expected}, got {actual_device}. "
            "Check the current environment, installed platform plugins, and base runtime."
        )

    return actual_device
