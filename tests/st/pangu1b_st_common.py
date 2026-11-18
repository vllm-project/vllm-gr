# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Shared helpers for Pangu 1B system tests."""

from __future__ import annotations

import os
from pathlib import Path

SKIP_MODEL = "Set PANGU1B_MODEL_PATH (or PANGU_MODEL_PATH) to openPangu-Embedded-1B-V1.1 weights"
SKIP_VLLM = "vLLM runtime unavailable"


def get_model_path() -> str:
    """Return the configured local checkpoint path, if any."""
    return (
        os.environ.get("PANGU1B_MODEL_PATH") or os.environ.get("PANGU_MODEL_PATH") or ""
    ).strip()


def model_available() -> bool:
    """True when a local Pangu checkpoint with weight files is configured."""
    model_path = get_model_path()
    if not model_path:
        return False

    checkpoint_dir = Path(model_path)
    if not checkpoint_dir.is_dir():
        return False

    for pattern in ("*.safetensors", "*.bin"):
        if list(checkpoint_dir.glob(pattern)):
            return True
    return False


def has_vllm() -> bool:
    try:
        import vllm.engine.arg_utils  # noqa: F401
    except Exception:
        return False
    return True


def require_pangu_st_runtime() -> None:
    """Require Pangu prerequisites, failing instead of skipping in strict CI."""
    import unittest

    strict = os.environ.get("PANGU1B_REQUIRE_RUNTIME", "") == "1"
    if not has_vllm():
        if strict:
            raise RuntimeError(SKIP_VLLM)
        raise unittest.SkipTest(SKIP_VLLM)
    if not model_available():
        if strict:
            raise RuntimeError(SKIP_MODEL)
        raise unittest.SkipTest(SKIP_MODEL)
