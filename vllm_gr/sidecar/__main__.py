# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Load a Python SidecarConfig; the framework owns process execution."""

from __future__ import annotations

import argparse
import asyncio
import logging
import runpy
import signal
import sys
from pathlib import Path

from .config import SidecarConfig
from .controller import Sidecar


def load_config(path: Path) -> SidecarConfig:
    path = path.resolve()
    sys.path.insert(0, str(path.parent))
    try:
        values = runpy.run_path(str(path))
    finally:
        sys.path.pop(0)
    config = values.get("config")
    if not isinstance(config, SidecarConfig):
        raise ValueError("Python configuration must export config: SidecarConfig")
    return config


async def _run(config: SidecarConfig, once: bool) -> None:
    sidecar = Sidecar(config)
    if once:
        async with asyncio.TaskGroup() as group:
            for task in config.tasks:
                group.create_task(sidecar.run_once(task.name))
    else:
        runner = asyncio.create_task(sidecar.run())
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, runner.cancel)
        try:
            await runner
        except asyncio.CancelledError:
            pass
        finally:
            for signum in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(signum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Poll every Task once, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run(load_config(args.config), args.once))


if __name__ == "__main__":
    main()
