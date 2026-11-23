# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run file-oriented hooks with the same paths pre-commit supplies."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.helpers.paths import REPO_ROOT

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
HOOKS = REPO_ROOT / "tools/pre_commit"


def test_regex_hook_uses_supplied_paths(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    good = tmp_path / "good.py"
    good.write_text("import regex as re\n")
    bad = tmp_path / "renamed module.py"
    bad.write_text("import " + "re\n")
    for name, expected in [(good.name, 0), (bad.name, 1), ("missing.py", 1)]:
        result = subprocess.run(
            [sys.executable, HOOKS / "enforce_regex_import.py", name], cwd=tmp_path
        )
        assert result.returncode == expected, name


def test_cuda_stateless_call_bounds(tmp_path: Path) -> None:
    path = tmp_path / "call.py"
    name = "cuda_device_count_" + "stateless"
    for expr, status in [(f"{name}()", 1), (f"{name}( )", 1), (f"my_{name}()", 0)]:
        path.write_text(f"count = {expr}\n")
        result = subprocess.run([sys.executable, HOOKS / "check_torch_cuda.py", path])
        assert result.returncode == status, expr


def test_png_hook_uses_selected_excalidraw_exports(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "draft.excalidraw.png").write_bytes(b"missing scene")
    (tmp_path / "good image.excalidraw.png").write_bytes(b"excalidraw+json")
    (tmp_path / "icon.png").write_bytes(b"ordinary PNG")
    hook = ["bash", (HOOKS / "png-lint.sh").as_posix()]
    assert (
        subprocess.run([*hook, "icon.png", "good image.excalidraw.png"], cwd=tmp_path).returncode
        == 0
    )
    assert subprocess.run([*hook, "draft.excalidraw.png"], cwd=tmp_path).returncode == 1
    assert subprocess.run(hook, cwd=tmp_path).returncode == 0


def test_shellcheck_hook_uses_selected_or_tracked_scripts(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    checker = tmp_path / "shellcheck"
    checker.write_bytes(
        b"#!/bin/bash\n"
        b"rcfile=\n"
        b"for arg; do\n"
        b"    if [[ $arg == --rcfile=* ]]; then rcfile=${arg#*=}; fi\n"
        b"done\n"
        b"[[ -f $rcfile ]] || exit 1\n"
        b"if [[ ${!#} == --version ]]; then exit 0; fi\n"
        b'printf "%s\\n" "$@" >> checked\n'
        b'[[ "$*" != *reject* ]]\n'
    )
    checker.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    for name in ("good file.sh", "reject.sh", "scratch.sh"):
        (tmp_path / name).touch()
    subprocess.run(["git", "add", "--", "good file.sh", "reject.sh"], cwd=tmp_path, check=True)
    hook = ["bash", (HOOKS / "shellcheck.sh").as_posix()]
    assert subprocess.run([*hook, "good file.sh"], cwd=tmp_path, env=env).returncode == 0
    log = tmp_path / "checked"
    assert "good file.sh" in log.read_text().splitlines()
    assert "reject.sh" not in log.read_text()
    log.unlink()
    assert subprocess.run(hook, cwd=tmp_path, env=env).returncode != 0
    assert "reject.sh" in log.read_text().splitlines()
    assert "scratch.sh" not in log.read_text()


def test_signoff_uses_supplied_message_in_both_checkout_types(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    signoffs = [
        hook for repo in config["repos"] for hook in repo["hooks"] if hook["id"] == "signoff-commit"
    ]
    assert len(signoffs) == 1
    hook = signoffs[0]
    repo = tmp_path / "repository"
    worktree = tmp_path / "linked"
    subprocess.run(["git", "init", "-q", repo], check=True)
    for key, value in [("user.name", "CI (Tools)"), ("user.email", "ci@example.invalid")]:
        subprocess.run(["git", "config", key, value], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "base"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "worktree", "add", "-q", "--detach", worktree], cwd=repo, check=True)
    signature = "Signed-off-by: CI (Tools) <ci@example.invalid>"
    for checkout in (repo, worktree):
        message = checkout / "message file"
        for content in (b"subject\n", f"subject\r\n\r\n{signature}\r\n".encode()):
            message.write_bytes(content)
            for _ in range(2):
                subprocess.run(
                    [hook["entry"], *hook["args"], message.name], cwd=checkout, check=True
                )
            assert message.read_text().splitlines() == ["subject", "", signature]
