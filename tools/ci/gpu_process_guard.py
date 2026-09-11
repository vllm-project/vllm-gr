# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clean only selected EngineCore processes; require no remaining compute apps."""

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

if __package__:
    from . import cuda_devices as devices
else:
    import cuda_devices as devices

ENGINE_NAMES = {"VLLM::EngineCore", "VLLM:EngineCore"}


def is_engine(process: dict[str, Any]) -> bool:
    name = Path(process["name"]).name
    if name in ENGINE_NAMES:
        return True
    # vLLM names data-parallel cores EngineCore_DP<rank>.
    base, _, rank = name.partition("_DP")
    return base in ENGINE_NAMES and rank.isascii() and rank.isdecimal()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise devices.DeviceError(f"Cannot check PID {pid}: {exc}") from exc


def require_empty(processes: list[dict[str, Any]]) -> None:
    if processes:
        raise devices.DeviceError(f"Target GPU still has compute processes: {processes!r}")


def cleanup(gpus: list[dict[str, Any]], processes: list[dict[str, Any]], timeout: float) -> None:
    # Dict preserves order and deduplicates PIDs (including engines using multiple targets).
    pids = dict.fromkeys(proc["pid"] for proc in processes if is_engine(proc))
    signaled = []
    for pid in pids:
        # Recheck GPU membership and name immediately before sending a signal.
        current = devices.compute_processes(gpus)
        if not any(proc["pid"] == pid and is_engine(proc) for proc in current):
            continue
        signaled.append(pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise devices.DeviceError(
                f"Cannot terminate target EngineCore PID {pid}: {exc}"
            ) from exc
    deadline = time.monotonic() + timeout
    while True:
        remaining = [pid for pid in signaled if alive(pid)]
        current = devices.compute_processes(gpus)
        # NVML can briefly retain a process after the OS reports its exit.
        gpu_pids = sorted({proc["pid"] for proc in current if proc["pid"] in signaled})
        if not remaining and not gpu_pids:
            require_empty(current)
            return
        if time.monotonic() >= deadline:
            raise devices.DeviceError(
                f"SIGTERM timed out after {timeout}s; live PIDs: {remaining}; GPU PIDs: {gpu_pids}"
            )
        time.sleep(min(1.0, max(0, deadline - time.monotonic())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cleanup", "check"))
    parser.add_argument("--timeout", type=float, default=30, help="SIGTERM wait timeout in seconds")
    args = parser.parse_args()
    try:
        if not 0 < args.timeout < float("inf"):
            raise devices.DeviceError("Timeout must be finite and positive")
        gpus = devices.selected_devices()
        processes = devices.compute_processes(gpus)
        if args.action == "cleanup":
            cleanup(gpus, processes, args.timeout)
        else:
            require_empty(processes)
        print("PASS: Target GPUs have no compute processes.")
        return 0
    except (devices.DeviceError, OSError) as exc:
        print(f"FAIL: GPU process {args.action}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
