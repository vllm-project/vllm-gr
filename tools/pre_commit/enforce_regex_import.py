# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm
"""Require third-party regex in Python modules selected by pre-commit."""

import ast
import sys
import tokenize


def main(paths: list[str]) -> int:
    status = 0
    for path in paths:
        try:
            with tokenize.open(path) as source:
                tree = ast.parse(source.read(), filename=path)
        except (OSError, SyntaxError, UnicodeError) as exc:
            print(exc, file=sys.stderr)
            status = 1
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Import)
                and any(alias.name.partition(".")[0] == "re" for alias in node.names)
            ) or (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and (node.module or "").partition(".")[0] == "re"
            ):
                print(f"{path}:{node.lineno}: use 'regex' instead of the standard library 're'")
                status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
