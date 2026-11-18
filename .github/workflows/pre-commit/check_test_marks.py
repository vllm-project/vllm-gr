# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Require hardware helpers for pytest SKU and card-count marks."""

from __future__ import annotations

import ast
import sys
import tokenize
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_TAGS = ("[cuda-sku]", "[npu-sku]", "[hardware-cards]")


def _resource_marks() -> set[str]:
    with (_ROOT / "pyproject.toml").open("rb") as source:
        markers = tomllib.load(source)["tool"]["pytest"]["ini_options"]["markers"]
    return {
        entry.split(":", 1)[0].strip() for entry in markers if any(tag in entry for tag in _TAGS)
    }


def check_file(path: Path, protected: set[str]) -> list[str]:
    with tokenize.open(path) as source:
        tree = ast.parse(source.read(), filename=str(path))
    pytest_names = {"pytest"}
    mark_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pytest_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "pytest"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            mark_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "mark"
            )

    def is_mark(node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id in mark_names
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "mark"
            and isinstance(node.value, ast.Name)
            and node.value.id in pytest_names
        )

    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Attribute, ast.Call)):
            continue
        marker = None
        if isinstance(node, ast.Attribute) and is_mark(node.value):
            marker = node.attr
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and is_mark(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            marker = node.args[1].value
        if marker in protected:
            errors.append(
                f"{path}:{node.lineno}: {marker} must be declared via "
                "hardware_test() or hardware_marks()."
            )
    return errors


def main() -> int:
    protected = _resource_marks()
    paths = {Path(name) for name in sys.argv[1:]}
    if any(path.suffix != ".py" or path.parts[0] in {".github", "tools"} for path in paths):
        paths = set(Path("tests").rglob("*.py"))
    failed = False
    for path in sorted(paths):
        try:
            errors = check_file(path, protected)
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors = [f"{path}: {exc}"]
        for error in errors:
            print(error, file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
