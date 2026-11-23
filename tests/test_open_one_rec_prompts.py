# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""The OpenOneRec judge prompts load, format, and stay attributed."""

from pathlib import Path

import pytest

from benchmarks.open_one_rec.prompts import load_prompt
from tests.helpers.paths import REPO_ROOT

# Host-only checks of the prompt data files: CPU tier, L1.
pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]

PROMPT_DIR = REPO_ROOT / "benchmarks/open_one_rec/prompts"

# How many values each evaluator passes to str.format, read from the call
# sites: tasks/item_understand/utils.py format(text) and format(gt, model),
# tasks/rec_reason/utils.py format(gt, model). A prompt that loses or gains a
# field would otherwise fail only when the benchmark reaches that request.
FORMAT_ARGS = {
    "item_understand_wip_extraction": 1,
    "item_understand_wip_matching": 2,
    "rec_reason_evaluation": 2,
}


def _prompt_files() -> list[Path]:
    """Every prompt data file, read from disk so a new one cannot be missed."""
    return sorted(PROMPT_DIR.glob("*.txt"))


def test_every_prompt_matches_its_evaluator_call() -> None:
    paths = _prompt_files()
    assert {path.stem for path in paths} == FORMAT_ARGS.keys()
    for path in paths:
        text = load_prompt(path.stem)
        assert text.strip(), path.name
        expected = FORMAT_ARGS[path.stem]
        assert text.count("{}") == expected, path.name
        text.format(*range(expected))


def test_every_prompt_has_a_license_file() -> None:
    for path in _prompt_files():
        assert (path.parent / f"{path.name}.license").is_file(), path.name
