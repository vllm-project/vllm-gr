# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Check the configuration hook through its command-line entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
HOOK = Path(__file__).resolve().parents[2] / "tools/pre_commit/validate_config.py"


def _check(tmp_path: Path, source: str, *, name: str = "vllm_gr/config/models.py") -> str:
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(HOOK), name],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert not result.stderr
    return "" if result.returncode == 0 else result.stdout


@pytest.mark.parametrize(  # type: ignore[misc]
    ("declaration", "doc", "message"),
    [
        ('field(default=1, metadata={"gr": ConfigFieldSpec("check")})', '"""Value."""', ""),
        ('field(default=None, metadata={"gr": ConfigFieldSpec("check")})', '"""Value."""', ""),
        (
            'field(default_factory=list, metadata={"gr": ConfigFieldSpec("check")})',
            '"""Value."""',
            "",
        ),
        ('field(metadata={"gr": ConfigFieldSpec("check")})', '"""Value."""', "default value"),
        (
            'field(default=MISSING, metadata={"gr": ConfigFieldSpec("check")})',
            '"""Value."""',
            "default value",
        ),
        (
            'field(default_factory=dataclasses.MISSING, metadata={"gr": ConfigFieldSpec("check")})',
            '"""Value."""',
            "default value",
        ),
        ('field(default=1, metadata={"gr": ConfigFieldSpec("check")})', "", "docstring"),
        ('field(default=1, metadata={"gr": ConfigFieldSpec("check")})', '"""   """', "docstring"),
        ("field(default=1)", '"""Value."""', "ConfigFieldSpec"),
        (
            'field(metadata={"gr": ConfigFieldSpec("check", required=True)})',
            '"""Value."""',
            "Unexpected required",
        ),
    ],
)
def test_model_field_contract(tmp_path: Path, declaration: str, doc: str, message: str) -> None:
    source = (
        f"@dataclass(frozen=True)\nclass GRConfig:\n    value: int = {declaration}\n    {doc}\n"
    )
    output = _check(tmp_path, source)
    if message:
        assert message in output
    else:
        assert output == ""


@pytest.mark.parametrize(  # type: ignore[misc]
    ("default", "required", "message"),
    [
        ("", ", required=True", ""),
        ("default=32, ", ", required=True", "must not have a default"),
        ("default=32, ", "", "must explicitly declare required=True"),
        ("", "", "must explicitly declare required=True"),
    ],
)
def test_required_capacity_is_a_narrow_exception(
    tmp_path: Path, default: str, required: str, message: str
) -> None:
    source = (
        "@dataclass(frozen=True, slots=True)\nclass ConstraintTableConfig:\n"
        f'    max_top_k: int = field({default}metadata={{"gr": ConfigFieldSpec("check"{required})}})\n'
        '    """Required table capacity."""\n'
    )
    output = _check(tmp_path, source)
    if message:
        assert message in output
    else:
        assert output == ""


def test_ordinary_dataclass_is_not_subject_to_config_field_rules(tmp_path: Path) -> None:
    assert _check(tmp_path, "@dataclass\nclass Request:\n    value: int\n") == ""
    assert (
        _check(
            tmp_path,
            "@dataclass\nclass GRConfig:\n    value: int\n",
            name="vllm_gr/other/models.py",
        )
        == ""
    )


def test_existing_decorated_config_still_requires_defaults_and_docs(tmp_path: Path) -> None:
    source = "@config\n@dataclass\nclass ExistingConfig:\n    value: int\n"
    assert "default value" in _check(tmp_path, source, name="tests/example.py")
    source = "@config\n@dataclass\nclass ExistingConfig:\n    value: int = 1\n"
    assert "docstring" in _check(tmp_path, source, name="tests/example.py")


def test_actual_models_are_checked() -> None:
    subprocess.run(
        [sys.executable, str(HOOK), "vllm_gr/config/models.py"],
        cwd=HOOK.parents[2],
        check=True,
    )
