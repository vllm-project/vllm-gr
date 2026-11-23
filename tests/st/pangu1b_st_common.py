# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Shared helpers for Pangu 1B system tests."""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, TypedDict

import regex as re

PANGU_CONFIG = Path("recipes/Pangu/pangu1b.yaml")
GSM8K_MINI = Path("tests/resources/pangu1b/gsm8k_mini.json")
_ANSWER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")

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


def require_pangu_st_runtime(*, require_model: bool = True) -> None:
    """Check runtime, retaining the legacy local-model guard for direct callers.

    The pangu_model fixture passes require_model=False and prepares validated
    weights on demand after this runtime check succeeds.
    """
    import unittest

    strict = os.environ.get("PANGU1B_REQUIRE_RUNTIME", "") == "1"
    if not has_vllm():
        if strict:
            raise RuntimeError(SKIP_VLLM)
        raise unittest.SkipTest(SKIP_VLLM)
    if require_model and not model_available():
        if strict:
            raise RuntimeError(SKIP_MODEL)
        raise unittest.SkipTest(SKIP_MODEL)


def _normalize_numeric(value: str) -> str:
    text = value.strip().replace(",", "")
    if "/" in text:
        try:
            return str(Fraction(text).limit_denominator())
        except (ValueError, ZeroDivisionError):
            return text
    if text.endswith(".0"):
        text = text[:-2]
    return text


def extract_final_answer(text: str) -> str | None:
    """Extract a boxed answer or fall back to the last numeric token."""
    text_clean = text.replace(",", "")

    # Priority 1: Try \boxed{} notation first (explicit answer marker)
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        boxed_content = match.group(1)
        numbers = _ANSWER_RE.findall(boxed_content)
        if numbers:
            return _normalize_numeric(numbers[0])

    # Priority 2: Fall back to last numeric token
    matches = _ANSWER_RE.findall(text_clean)
    if not matches:
        return None
    return _normalize_numeric(matches[-1])


def load_dataset(path: Path, num_prompts: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Dataset is empty: {path}")
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset is not valid JSON: {path}") from exc
    if not isinstance(records, list):
        raise ValueError(f"Dataset must be a JSON list: {path}")
    if num_prompts is not None:
        records = records[:num_prompts]
    return records


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


class AnswerScore(TypedDict):
    expected: str
    predicted: str | None
    correct: bool


def score_answer(generated: str, reference: str) -> AnswerScore:
    """Score one completion against its normalized GSM8K reference answer."""
    predicted = extract_final_answer(generated)
    expected = _normalize_numeric(reference)
    return {"expected": expected, "predicted": predicted, "correct": predicted == expected}
