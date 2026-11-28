# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Select and expand Buildkite steps before uploading pipelines."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

HARDWARE_PATH = Path(__file__).resolve().parents[1] / "ci_mirror_hardwares.yml"
CI_STEPS_PATH = HARDWARE_PATH.parents[1] / "ci-steps.yml"


def is_documentation(path: bytes) -> bool:
    return (
        (path.startswith(b"docs/") or path.endswith(b".md") or path == b"mkdocs.yml")
        and b"\\" not in path
        and not any(byte < 32 or byte == 127 for byte in path)
    )


def is_docs_only_pr() -> bool:
    if not os.environ.get("BUILDKITE_PULL_REQUEST", "").isdigit():
        return False
    if os.environ.get("BUILDKITE_PULL_REQUEST_BASE_BRANCH") != "main":
        return False

    try:
        # Buildkite can check out GitHub's synthetic merge instead of the PR head.
        head = os.environ.get("BUILDKITE_PULL_REQUEST_HEAD_COMMIT") or "HEAD"
        if head != "HEAD" and (
            len(head) != 40 or any(c not in "0123456789abcdef" for c in head.lower())
        ):
            raise ValueError("invalid PR head SHA")
        git_options = {"check": True, "capture_output": True, "timeout": 60}
        if (
            subprocess.run(
                ["git", "rev-parse", "--is-shallow-repository"], **git_options
            ).stdout.strip()
            != b"false"
        ):
            raise ValueError("shallow checkout")
        subprocess.run(["git", "fetch", "--no-tags", "origin", "refs/heads/main"], **git_options)
        bases = subprocess.run(
            ["git", "merge-base", "--all", "FETCH_HEAD", head], **git_options
        ).stdout.split()
        if len(bases) != 1:
            raise ValueError("no unique PR merge base")
        diff = subprocess.run(
            [
                "git",
                "diff",
                "--no-renames",
                "--ignore-submodules=none",
                "--name-status",
                "-z",
                bases[0].decode("ascii"),
                head,
                "--",
            ],
            **git_options,
        ).stdout
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"Cannot classify PR diff, running normal CI: {error}", file=sys.stderr)
        return False

    # With rename detection disabled, a move must qualify at both paths.
    fields = diff.rstrip(b"\0").split(b"\0")
    return (
        bool(diff)
        and len(fields) % 2 == 0
        and all(
            status in {b"A", b"D", b"M"} and is_documentation(path)
            for status, path in zip(fields[::2], fields[1::2])
        )
    )


def expand_steps(steps: list[Any], presets: dict[str, dict[str, Any]]) -> None:
    for step in steps:
        if not isinstance(step, dict):
            continue
        if "group" in step:
            expand_steps(step["steps"], presets)
        if "mirror_hardwares" not in step:
            continue
        name = step.pop("mirror_hardwares")
        if not isinstance(name, str) or name not in presets:
            raise ValueError(f"Unknown mirror_hardwares preset: {name!r}")
        preset = presets[name]
        conflicts = step.keys() & preset.keys()
        if conflicts:
            raise ValueError(f"Preset {name!r} conflicts with step fields: {sorted(conflicts)}")
        step.update(preset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--upload", action="store_true", help="Upload instead of printing YAML")
    args = parser.parse_args()

    if args.upload and args.pipeline.resolve() == CI_STEPS_PATH and is_docs_only_pr():
        print("Skipping Buildkite CI steps for a documentation-only pull request.", file=sys.stderr)
        return 0

    try:
        presets = yaml.safe_load(HARDWARE_PATH.read_text(encoding="utf-8"))["mirror_hardwares"]
        pipeline = yaml.safe_load(args.pipeline.read_text(encoding="utf-8"))
        expand_steps(pipeline["steps"], presets)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    rendered = yaml.safe_dump(pipeline, sort_keys=False)
    if args.upload:
        return subprocess.run(
            ["buildkite-agent", "pipeline", "upload"],
            input=rendered,
            text=True,
            check=False,
        ).returncode
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
