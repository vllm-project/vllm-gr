#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Verify that text in specified files or directories are in English and not in other languages.

If a non-English character is found:
- The test fails.
- Prints the file path, line number, line text.
- Marks with '^' under it the location of the violating character(s).
"""

import argparse
import os
import sys
import unicodedata

# String of non-English characters allowed to bypass the rule
EXCEPT_STRING = "█—▀▄–📚🎉✅❌💡🙏🚀🐛⚠️⚡✗🛠️ 🎨💬🤗→─§：✓‑×…│├└是否"

EXCEPT_PATHS = [
    "benchmarks/open_one_rec",
    "trc",
    "docs/quickstart.md",
    "docs/hot_update_tests.zh-CN.md",
    "docs/weight_update_sidecar_design.zh-CN.md",
    "tests/resources/single_one_rec_prompt.txt",
]


def is_path_exempt(file_path: str, except_paths: list[str] = EXCEPT_PATHS) -> bool:
    """Check if a file path is explicitly exempt or inside an exempt directory."""
    norm_file = os.path.normpath(file_path)
    for except_path in except_paths:
        norm_except = os.path.normpath(except_path)
        if norm_file == norm_except or norm_file.startswith(norm_except + os.sep):
            return True
    return False


# Common binary file extensions to skip
BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".pyc",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".pt",
    ".pth",
    ".safetensors",
    ".onnx",
    ".npy",
    ".npz",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}


def is_english_char(c: str, except_string: str = EXCEPT_STRING) -> bool:
    """
    Check if a character is a standard English / ASCII text character or in except_string.
    English text characters include printable ASCII (0x20..0x7E) and standard whitespace (tab, newline, carriage return).
    Characters in except_string bypass the non-English character rule.
    """
    ord_c = ord(c)
    return (ord_c <= 0x7E) or c in except_string


def is_binary_file(filepath: str, raw_content: bytes) -> bool:
    """Check if file is binary by extension or content."""
    _, ext = os.path.splitext(filepath)
    if ext.lower() in BINARY_EXTENSIONS:
        return True
    # Null byte check in initial chunk
    if b"\x00" in raw_content[:8192]:
        return True
    return False


def check_file_for_non_english(
    filepath: str, except_string: str = EXCEPT_STRING
) -> list[tuple[int, str, str]]:
    """
    Check a single file for non-English characters.
    Returns list of (line_number, line_text, caret_line).
    """
    if not os.path.isfile(filepath):
        return []

    try:
        with open(filepath, "rb") as f:
            raw_content = f.read()
    except OSError:
        return []

    if is_binary_file(filepath, raw_content):
        return []

    try:
        content = raw_content.decode("utf-8")
    except UnicodeDecodeError:
        # Skip files that aren't valid UTF-8 text
        return []

    violations = []
    lines = content.splitlines()

    for line_num, line in enumerate(lines, 1):
        has_violation = any(not is_english_char(c, except_string) for c in line)
        if not has_violation:
            continue

        carets = []
        for c in line:
            width = 2 if unicodedata.east_asian_width(c) in ("F", "W") else 1
            if not is_english_char(c, except_string):
                carets.append("^" * width)
            elif c == "\t":
                carets.append("\t")
            else:
                carets.append(" " * width)

        caret_line = "".join(carets)
        violations.append((line_num, line, caret_line))

    return violations


def collect_files(paths: list[str]) -> list[str]:
    """Collect file paths from a list of files or directories."""
    files = []
    for path in paths:
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            for root, dirs, filenames in os.walk(path):
                if ".git" in dirs:
                    dirs.remove(".git")
                for fn in filenames:
                    files.append(os.path.join(root, fn))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that text in specified files or directories are in English."
    )
    parser.add_argument("paths", nargs="*", help="Path(s) to file(s) or directory(ies) to check.")
    parser.add_argument(
        "--except-string",
        "-e",
        type=str,
        default=EXCEPT_STRING,
        help="String of non-English characters allowed to bypass the rule.",
    )
    args = parser.parse_args()

    files = collect_files(args.paths)

    try:
        for file_path in files:
            if is_path_exempt(file_path):
                continue
            file_violations = check_file_for_non_english(
                file_path,
                except_string=args.except_string,
            )
            if file_violations:
                for line_num, line_text, caret_line in file_violations:
                    print(f"{file_path}:{line_num}:")
                    print(line_text)
                    print(caret_line)
                    print()
                return 1
    except BrokenPipeError:
        # Python flushes standard streams on exit; redirect to devnull to avoid noise
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
