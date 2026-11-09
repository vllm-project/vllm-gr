# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default comparisons between the serving resource and a downloaded candidate."""

from __future__ import annotations

from pathlib import Path

from .contracts import UpdateError


def _files(directory: Path) -> dict[str, int]:
    root = directory.resolve()
    if not root.is_dir():
        raise UpdateError(f"Checkpoint directory does not exist: {directory}")
    files: dict[str, int] = {}
    for path in root.rglob("*"):
        if not path.resolve().is_relative_to(root):
            raise UpdateError(f"Resource file points outside the directory: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.stat().st_size
    return files


def default_ckpt_checker(old_path: Path | None, new_path: Path) -> bool:
    """Require identical relative file names and sizes; weight contents may change."""
    if old_path is None:
        raise UpdateError("Default checkpoint checker requires the current resource_path")
    old_files = _files(old_path)
    return bool(old_files) and old_files == _files(new_path)


def trie_file(path: Path) -> Path:
    """Accept an artifact file or a directory containing exactly one artifact."""
    if path.is_file() and path.suffix == ".safetensors":
        return path
    if path.is_dir():
        artifacts = list(path.glob("*.safetensors"))
        if len(artifacts) == 1 and artifacts[0].is_file():
            if not artifacts[0].resolve().is_relative_to(path.resolve()):
                raise UpdateError("Trie artifact points outside the resource directory")
            return artifacts[0]
    raise UpdateError(f"Expected one Trie safetensors artifact: {path}")


def default_trie_checker(old_path: Path | None, new_path: Path) -> bool:
    """Require the same tensor count/names; values, shapes and file sizes may change."""
    if old_path is None:
        raise UpdateError("Default Trie checker requires the current resource_path")
    from safetensors import safe_open

    with safe_open(trie_file(old_path), framework="numpy") as old:
        old_names = set(old.keys())
    with safe_open(trie_file(new_path), framework="numpy") as new:
        new_names = set(new.keys())
    return bool(old_names) and old_names == new_names
