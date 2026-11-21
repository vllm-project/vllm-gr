# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run a gate in its own process group with a persistent log and bounded cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Mapping, Sequence


def run_gate_process(
    command: Sequence[str],
    *,
    cwd: Path,
    log: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> int:
    if timeout <= 0:
        raise ValueError("Gate timeout must be positive")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        previous = None
        if os.name == "posix" and threading.current_thread() is threading.main_thread():
            previous = signal.getsignal(signal.SIGTERM)

            def cancelled(signum: int, frame: object) -> None:
                raise KeyboardInterrupt("Gate cancelled")

            signal.signal(signal.SIGTERM, cancelled)
        try:
            return process.wait(timeout=timeout)
        finally:
            # Kill the entire owned group, including children left by a failed driver.
            # A TERM gives legacy EXIT traps and server context managers time to clean up.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            process.wait()
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
