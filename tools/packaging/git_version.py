# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Resolve a GR checkout version before build dependencies have been installed."""

import subprocess
from pathlib import Path


def checkout_version(repo: Path, base: str) -> str:
    if not (repo / ".git").exists():
        return base

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    # Ignore unrelated milestone tags when finding the nearest packaging release.
    # Git describe selects the nearest reachable match across branches and worktrees.
    tags = {}
    for tag in git("tag", "--list").splitlines():
        release = tag.rsplit("-", 1)[-1].lstrip("vV")
        public, rc, serial = release.partition("rc")
        if all(part.isascii() and part.isdecimal() for part in public.split(".")) and (
            not rc or (serial.isascii() and serial.isdecimal())
        ):
            tags[tag] = release
    patterns = [arg for tag in tags for arg in ("--match", tag)] or ["--exclude", "*"]
    description = git(
        "describe", "--tags", "--long", "--always", "--dirty", "--abbrev=12", *patterns
    )
    dirty = description.endswith("-dirty")
    parts = description.removesuffix("-dirty").rsplit("-", 2)
    if len(parts) == 3:
        tag, distance, node = parts
        release = tags[tag]
        if distance == "0" and not dirty:
            return release
        # Develop toward the next patch (or next RC on a prerelease line).
        prefix, rc, serial = release.partition("rc")
        if rc:
            next_release = f"{prefix}rc{int(serial) + 1}"
        else:
            prefix, dot, patch = release.rpartition(".")
            next_release = f"{prefix}{dot}{int(patch) + 1}"
    else:
        node = "g" + parts[0]
        distance = git("rev-list", "--count", "HEAD")
        next_release = base
    local = f"{node}.dirty" if dirty else node
    return f"{next_release}.dev{distance}+{local}"
