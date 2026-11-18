# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) David Lord
# Adapted from https://davidism.com/attribute-docstrings/
"""
Ensures all fields in a config dataclass have default values
and that each field has a docstring.
"""

import ast
import inspect
import sys
from itertools import pairwise
from pathlib import Path
from typing import NoReturn

import regex as re


def get_attr_docs(cls_node: ast.ClassDef) -> dict[str, str]:
    """
    Get any docstrings placed after attribute assignments in a class body.

    Adapted from https://davidism.com/attribute-docstrings/
    https://davidism.com/mit-license/
    """

    out = {}

    # Consider each pair of nodes.
    for a, b in pairwise(cls_node.body):
        # Must be an assignment then a constant string.
        if (
            not isinstance(a, (ast.Assign, ast.AnnAssign))
            or not isinstance(b, ast.Expr)
            or not isinstance(b.value, ast.Constant)
            or not isinstance(b.value.value, str)
        ):
            continue

        doc = inspect.cleandoc(b.value.value)

        # An assignment can have multiple targets (a = b = v), but an
        # annotated assignment only has one target.
        targets = a.targets if isinstance(a, ast.Assign) else [a.target]

        for target in targets:
            # Must be assigning to a plain name.
            if not isinstance(target, ast.Name):
                continue

            out[target.id] = doc

    return out


class ConfigValidator(ast.NodeVisitor):
    def __init__(self, *, gr_config: bool = False) -> None:
        self.gr_config = gr_config

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Validate class with both @config and @dataclass decorators
        decorators = [
            id
            for d in node.decorator_list
            if (isinstance(d, ast.Name) and ((id := d.id) == "config" or id == "dataclass"))
            or (
                isinstance(d, ast.Call)
                and (isinstance(d.func, ast.Name) and (id := d.func.id) == "dataclass")
            )
        ]

        if self.gr_config and node.name in {"GRConfig", "ConstraintTableConfig"}:
            if "dataclass" not in decorators:
                fail(f"Class {node.name} must be a dataclass.", node)
            validate_class(node, require_metadata=True)
        elif set(decorators) == {"config", "dataclass"}:
            validate_class(node)
        elif set(decorators) == {"config"}:
            fail(f"Class {node.name} with config decorator must be a dataclass.", node)

        self.generic_visit(node)


def _call_name(value: ast.expr) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def _field_kwargs(value: ast.expr | None) -> dict[str, ast.expr] | None:
    if isinstance(value, ast.Call) and _call_name(value.func) == "field":
        return {kw.arg: kw.value for kw in value.keywords if kw.arg is not None}
    return None


def _has_default(value: ast.expr | None) -> bool:
    if value is None or _call_name(value) == "MISSING":
        return False
    kwargs = _field_kwargs(value)
    if kwargs is None:
        return True
    return any(
        name in kwargs and _call_name(kwargs[name]) != "MISSING"
        for name in ("default", "default_factory")
    )


def _field_spec(stmt: ast.AnnAssign) -> ast.Call:
    kwargs = _field_kwargs(stmt.value)
    metadata = kwargs.get("metadata") if kwargs is not None else None
    if isinstance(metadata, ast.Dict):
        for key, value in zip(metadata.keys, metadata.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "gr"
                and isinstance(value, ast.Call)
                and _call_name(value.func) == "ConfigFieldSpec"
            ):
                return value
    fail("Config field must declare ConfigFieldSpec in field metadata['gr'].", stmt)


def validate_class(class_node: ast.ClassDef, *, require_metadata: bool = False) -> None:
    attr_docs = get_attr_docs(class_node)

    for stmt in class_node.body:
        # A field is defined as a class variable that has a type annotation.
        if isinstance(stmt, ast.AnnAssign):
            # Skip ClassVar and InitVar
            # see https://docs.python.org/3/library/dataclasses.html#class-variables
            # and https://docs.python.org/3/library/dataclasses.html#init-only-variables
            if (
                isinstance(stmt.annotation, ast.Subscript)
                and isinstance(stmt.annotation.value, ast.Name)
                and stmt.annotation.value.id in {"ClassVar", "InitVar"}
            ):
                continue

            if isinstance(stmt.target, ast.Name):
                field_name = stmt.target.id
                required = False
                if require_metadata:
                    spec = _field_spec(stmt)
                    required = any(
                        kw.arg == "required"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                        for kw in spec.keywords
                    )
                    if (class_node.name, field_name) == (
                        "ConstraintTableConfig",
                        "max_top_k",
                    ) and not required:
                        fail("Field 'max_top_k' must explicitly declare required=True.", stmt)
                    if required and (class_node.name, field_name) != (
                        "ConstraintTableConfig",
                        "max_top_k",
                    ):
                        fail(f"Unexpected required config field '{field_name}'.", stmt)
                    if required and _has_default(stmt.value):
                        fail(f"Required field '{field_name}' must not have a default.", stmt)
                if not _has_default(stmt.value) and not required:
                    fail(
                        f"Field '{field_name}' in {class_node.name} must have a default value.",
                        stmt,
                    )

                if not attr_docs.get(field_name):
                    fail(
                        f"Field '{field_name}' in {class_node.name} must have a docstring.",
                        stmt,
                    )

                if (
                    isinstance(stmt.annotation, ast.Subscript)
                    and isinstance(stmt.annotation.value, ast.Name)
                    and stmt.annotation.value.id == "Union"
                    and isinstance(stmt.annotation.slice, ast.Tuple)
                ):
                    args = stmt.annotation.slice.elts
                    literal_args = [
                        arg
                        for arg in args
                        if isinstance(arg, ast.Subscript)
                        and isinstance(arg.value, ast.Name)
                        and arg.value.id == "Literal"
                    ]
                    if len(literal_args) > 1:
                        fail(
                            f"Field '{field_name}' in {class_node.name} must "
                            "use a single "
                            "Literal type. Please use 'Literal[Literal1, "
                            "Literal2]' instead of 'Union[Literal1, Literal2]'"
                            ".",
                            stmt,
                        )


def validate_ast(tree: ast.AST, *, gr_config: bool = False) -> None:
    ConfigValidator(gr_config=gr_config).visit(tree)


def _is_gr_config_path(file_path: str) -> bool:
    return Path(file_path).parts[-3:] == ("vllm_gr", "config", "models.py")


def validate_file(file_path: str) -> None:
    try:
        print(f"Validating {file_path} config dataclasses ", end="")
        with open(file_path, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=file_path)
        validate_ast(tree, gr_config=_is_gr_config_path(file_path))
    except ValueError as e:
        print(e)
        raise SystemExit(1) from e
    else:
        print("✅")


def fail(message: str, node: ast.stmt) -> NoReturn:
    raise ValueError(f"❌ line({node.lineno}): {message}")


def main() -> None:
    for filename in sys.argv[1:]:
        if _is_gr_config_path(filename):
            validate_file(filename)
            continue
        # Only run for Python files in vllm/ or tests/
        if not re.match(r"^(vllm|tests)/.*\.py$", filename):
            continue
        # Only run if the file contains @config
        with open(filename, encoding="utf-8") as f:
            if "@config" in f.read():
                validate_file(filename)


if __name__ == "__main__":
    main()
