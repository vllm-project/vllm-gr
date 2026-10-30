# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative CI selection. Unknown changes always retain every device gate."""

import argparse
import json
import os
import subprocess
from pathlib import PurePosixPath


def docs_only(paths: list[str]) -> bool:
    if not paths:
        return False
    for name in paths:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            return False
        documentation = name.startswith("docs/") or name in {"README.md", "README_CN.md"}
        if not documentation or path.suffix.lower() not in {".md", ".png", ".jpg", ".svg"}:
            return False
    return True


def make_plan(event: str, paths: list[str]) -> dict[str, object]:
    # Push only runs CPU checks. The idle-time dispatcher coalesces full branch
    # regressions. Manual dispatch always runs full coverage, even for docs.
    device = event != "push" and not (event == "pull_request" and docs_only(paths))
    return {
        "device-required": str(device).lower(),
        "reason": "full" if device else "push-coalesced" if event == "push" else "docs-only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", "local"))
    parser.add_argument("--base", default=os.environ.get("CI_BASE_SHA", ""))
    args = parser.parse_args()
    paths: list[str] = []
    if args.event == "pull_request" and args.base:
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--no-renames", "-z", args.base, "HEAD", "--"],
                check=True,
                capture_output=True,
            )
            paths = [p for p in result.stdout.decode().split("\0") if p]
        except (subprocess.CalledProcessError, UnicodeError):
            # A missing base or unreadable diff must never become a docs-only pass.
            pass
    plan = make_plan(args.event, paths)
    print(json.dumps(plan, indent=2))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            stream.writelines(f"{key}={value}\n" for key, value in plan.items())


if __name__ == "__main__":
    main()
