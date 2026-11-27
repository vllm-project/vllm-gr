# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Isolated-runner probes of the production Bash allocator and lock functions."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from tests.helpers.gate_process import run_gate_process
from tests.helpers.paths import REPO_ROOT

pytestmark = pytest.mark.l2

_PREAMBLE = r"""
set -euo pipefail
source tools/devices/resource.sh
pids=""
cleanup() { [[ -z "$pids" ]] || unlock_gpus "$pids" || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
lock_pid_state() {
    local pid=$1 stat rc
    [[ $pid =~ ^[1-9][0-9]*$ ]] || { echo "Invalid lock PID: $pid" >&2; return 1; }
    if stat=$(ps -p "$pid" -o stat=); then
        if [[ $stat =~ ^[[:space:]]*([^[:space:]]+)[[:space:]]*$ ]]; then
            echo "${BASH_REMATCH[1]}"
            return 0
        fi
    else
        rc=$?
        if [[ $rc == 1 && -z $stat ]]; then
            echo gone
            return 0
        fi
    fi
    echo "Could not read STAT for lock PID $pid" >&2
    return 1
}

wait_for_unlock() {
    local pid state previous attempt
    for pid in "$@"; do
        previous=""
        # A container's init may leave an exited lock holder as a zombie.
        # Poll each exact holder for up to five seconds and record state changes.
        for ((attempt=0; attempt<=20; attempt++)); do
            state=$(lock_pid_state "$pid") || return 1
            if [[ $state != "$previous" ]]; then
                echo "PID $pid STAT=$state after unlock"
                previous=$state
            fi
            [[ $state == gone || $state == Z* ]] && break
            if (( attempt == 20 )); then
                echo "FAILED (lock PID $pid remains active with STAT=$state)" >&2
                return 1
            fi
            sleep 0.25
        done
    done
}
threshold=18
if [[ "$npu_state" == true ]]; then threshold=3000; fi
"""

_SCENARIOS = {
    "single": r"""
id=$(get_gpu "$threshold" 15) || exit 77
[[ -n "$id" ]] || exit 77
pids=$(lock_gpu "$id")
[[ "$pids" =~ ^[0-9]+$ && "$pids" -gt 1 ]]
ps -p "$pids"
unlock_gpu "$pids"
wait_for_unlock "$pids"
pids=""
""",
    "pair": r"""
ids=$(get_gpus 2 "$threshold" 15) || exit 77
[[ -n "$ids" ]] || exit 77
IFS=',' read -ra devices <<< "$ids"
[[ ${#devices[@]} -eq 2 && "${devices[0]}" != "${devices[1]}" ]]
pids=$(lock_gpus "$ids")
IFS=',' read -ra processes <<< "$pids"
[[ ${#processes[@]} -eq 2 ]]
for pid in "${processes[@]}"; do
    [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]]
    ps -p "$pid"
done
unlock_gpus "$pids"
wait_for_unlock "${processes[@]}"
pids=""
""",
    "masked": r"""
[[ "$npu_state" != true ]] || exit 77
count=$(nvidia-smi --query-gpu=count --format=csv,noheader | tail -n 1)
[[ "$count" -ge 2 ]] || exit 77
# Select a nonzero physical ID from the runner's allocated, eligible pair.
ids=$(get_gpus 2 "$threshold" 15) || exit 77
IFS=',' read -ra devices <<< "$ids"
[[ ${#devices[@]} -eq 2 ]] || exit 77
mask_id=${devices[0]}
[[ "$mask_id" != 0 ]] || mask_id=${devices[1]}
export CUDA_VISIBLE_DEVICES="$mask_id"
id=$(get_gpu "$threshold" 15) || exit 77
[[ -n "$id" ]] || exit 77
[[ "$id" == "$mask_id" ]]
pids=$(lock_gpu "$id")
[[ "$pids" =~ ^[0-9]+$ && "$pids" -gt 1 ]]
ps -p "$pids"
unlock_gpu "$pids"
wait_for_unlock "$pids"
pids=""
""",
}


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            "single", marks=[pytest.mark.gpu_test, pytest.mark.npu_test, pytest.mark.cards_1]
        ),
        pytest.param(
            "pair", marks=[pytest.mark.gpu_test, pytest.mark.npu_test, pytest.mark.cards_2]
        ),
        pytest.param("masked", marks=[pytest.mark.gpu_test, pytest.mark.cards_2]),
    ],
)
def test_resource_probe(scenario: str, tmp_path: Path) -> None:
    if os.environ.get("VLLM_GR_ISOLATED_RUNNER") != "true":
        pytest.skip("Resource discovery requires VLLM_GR_ISOLATED_RUNNER=true")
    if os.name != "posix" or not shutil.which("bash"):
        pytest.skip("Resource locking requires POSIX Bash")
    if not (shutil.which("npu-smi") or shutil.which("nvidia-smi")):
        pytest.skip("No supported accelerator management tool")
    log = tmp_path / f"resource-{scenario}.log"
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    status = run_gate_process(
        ["bash", "-c", _PREAMBLE + _SCENARIOS[scenario]],
        cwd=REPO_ROOT,
        log=log,
        timeout=90,
        env=env,
    )
    if status == 77:
        pytest.skip(f"Insufficient eligible devices for {scenario}; see {log}")
    assert status == 0, f"Resource probe failed ({status}); see {log}\n{log.read_text()}"
