# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Expand hardware presets before uploading Buildkite test pipelines."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

HARDWARE_PATH = Path(__file__).resolve().parents[1] / "ci_mirror_hardwares.yml"


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
