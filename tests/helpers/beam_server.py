# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Manage a Beam driver server and reap it when its enclosing gate is cancelled."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


@contextmanager
def beam_server_process(
    command: list[str], *, env: dict[str, str], cwd: Path, log: TextIO
) -> Iterator[subprocess.Popen[bytes]]:
    gated = env.get("VLLM_GR_BEAM_GATE_GROUP") == "1"
    with subprocess.Popen(
        command, env=env, cwd=cwd, stdout=log, stderr=log, start_new_session=not gated
    ) as process:

        def cancel(signum: int, frame: object) -> None:
            raise SystemExit(128 + signum)

        previous = signal.signal(signal.SIGTERM, cancel) if gated else None
        try:
            yield process
        finally:
            try:
                if process.poll() is None:
                    if gated:
                        process.terminate()
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5 if gated else 30)
                    except subprocess.TimeoutExpired:
                        if gated:
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            finally:
                if previous is not None:
                    signal.signal(signal.SIGTERM, previous)
