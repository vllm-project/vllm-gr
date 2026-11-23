# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Device helper regressions without allocating accelerator resources."""

import ctypes
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from uuid import UUID

import pytest

from tests.helpers.paths import REPO_ROOT
from tools.devices import lock_gpu

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def run_shell(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("bash") is None:
        pytest.skip("Bash required")
    return subprocess.run(
        ["bash", "-c", script, "device-test", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    ("pid", "state", "ok"),
    [
        ("101", "Z+", True),
        ("101", "D", False),
        ("102", "S", False),
        ("103", "D", False),
        ("104", "R+", False),
        ("104", "D-then-Z", True),
    ],
)
def test_resource_integration_checks_actual_released_pids(
    tmp_path: Path, pid: str, state: str, ok: bool
) -> None:
    # Exercise the real gate. The mocked ps keeps a released PID visible to
    # model containers whose init does not immediately reap orphaned children.
    resource = tmp_path / "tools/devices/resource.sh"
    resource.parent.mkdir(parents=True)
    resource.write_text(
        """
        get_gpu() { echo "${CUDA_VISIBLE_DEVICES%%,*}"; }
        get_gpus() { echo 3,5; }
        lock_gpu() { if [[ $CUDA_VISIBLE_DEVICES == 3,5 ]]; then echo 101; else echo 104; fi; }
        lock_gpus() { echo 102,103; }
        unlock_gpu() { touch "released.$1"; }
        unlock_gpus() { unlock_gpu "${1%,*}"; unlock_gpu "${1#*,}"; }
        """,
        newline="\n",
    )
    probe = tmp_path / "probe.sh"
    probe.write_text((REPO_ROOT / "tests/development/test_resource.sh").read_text(), newline="\n")
    result = run_shell(
        """
        cd "$1" || exit 1
        target_pid=$2 target_state=$3
        export CUDA_VISIBLE_DEVICES=3,5
        command() {
            if [[ ${1:-} == -v && ${2:-} == npu-smi ]]; then return 1; fi
            builtin command "$@"
        }
        nvidia-smi() { echo 6; }
        python3() { :; }
        sleep() { :; }
        pgrep() { return 1; }
        kill() { :; }
        ps() {
            local queried= arg
            while (( $# )); do
                arg=$1; shift
                if [[ $arg == -p ]]; then queried=$1; shift; fi
            done
            if [[ ! -e released.$queried ]]; then echo ' S'; return; fi
            [[ $queried == "$target_pid" ]] || return 1
            if [[ $target_state == D-then-Z ]]; then
                if [[ ! -f seen-d ]]; then touch seen-d; echo ' D'; else echo ' Z'; fi
            else
                echo " $target_state"
            fi
        }
        source ./probe.sh
        """,
        tmp_path.as_posix(),
        pid,
        state,
    )
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == ok, output
    assert f"PID {pid} STAT={state.split('-')[0]}" in output, output
    if ok:
        assert "All integration tests passed" in output
        assert "CUDA_VISIBLE_DEVICES=3..." in output
    else:
        assert f"FAILED (lock PID {pid}" in output


@pytest.mark.parametrize("plural", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("npu", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("mask", ["0,invalid", "0,", "", "99"])  # type: ignore[misc]
def test_invalid_visibility_returns_no_ids(npu: bool, plural: bool, mask: str) -> None:
    result = run_shell(
        """
        npu-smi() { echo 'Total Count : 4'; }
        source tools/devices/resource.sh
        npu_state=$1
        nvidia-smi() { echo 4; }
        is_available() { return 0; }
        is_available_npu() { return 0; }
        CUDA_VISIBLE_DEVICES=$2 ASCEND_RT_VISIBLE_DEVICES=$2 "$3" 1 1
        """,
        str(npu).lower(),
        mask,
        "get_gpus" if plural else "get_gpu",
    )
    assert result.returncode != 0
    assert result.stdout == ""


@pytest.mark.parametrize("npu", ["true", "false"])  # type: ignore[misc]
def test_ordered_visible_intersection_and_timeout(npu: str) -> None:
    result = run_shell(
        """
        npu-smi() { echo 'Total Count : 4'; }
        source tools/devices/resource.sh
        npu_state=$1
        nvidia-smi() { echo 4; }
        is_available() { return 0; }
        is_available_npu() { return 0; }
        export CUDA_VISIBLE_DEVICES=02,2,0,99 ASCEND_RT_VISIBLE_DEVICES=02,2,0,99
        get_gpus 2 18 1 || exit 1
        unset CUDA_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES
        get_gpu 18 1 || exit 1
        is_available() { return 1; }
        is_available_npu() { return 1; }
        sleep() { command sleep 1; }
        get_gpu 18 1
        """,
        npu,
    )
    assert result.returncode != 0
    assert result.stdout.splitlines()[:2] == ["2,0", "0"]
    assert "Timeout" in result.stdout + result.stderr


@pytest.mark.parametrize(  # type: ignore[misc]
    ("stats", "status", "available"),
    [
        ("0, 4, P8", "0", True),
        ("0, 18, P8", "0", False),
        ("1, 4, P0", "0", False),
        ("0, 4, P8", "1", False),
        ("0, 4", "0", False),
        ("0, -1, P8", "0", False),
        ("0, 0, invalid P8", "0", False),
    ],
)
def test_gpu_requires_usable_statistics(stats: str, status: str, available: bool) -> None:
    result = run_shell(
        """
        npu-smi() { echo 'Total Count : 4'; }
        source tools/devices/resource.sh
        npu_state=false
        mock_stats=$1 rc=$2
        nvidia-smi() { echo "$mock_stats"; return "$rc"; }
        is_available 0 18
        """,
        stats,
        status,
    )
    assert (result.returncode == 0) == available


@pytest.mark.parametrize(  # type: ignore[misc]
    ("capacity", "rate", "status", "available"),
    [
        ("64000", "0", "0", True),
        ("64000", "50", "0", False),
        ("64000", "0", "1", False),
        ("", "0", "0", False),
        ("0", "0", "0", False),
        ("unknown", "0", "0", False),
        ("64000", "", "0", False),
        ("64000", "101", "0", False),
    ],
)
def test_npu_usage_requires_valid_telemetry(
    capacity: str, rate: str, status: str, available: bool
) -> None:
    result = run_shell(
        """
        npu-smi() { echo 'Total Count : 4'; }
        source tools/devices/resource-npu.sh
        cap=$1 rate=$2 rc=$3
        npu-smi() {
            printf 'HBM Capacity(MB) : %s\nHBM Usage Rate(%%) : %s\n' "$cap" "$rate"
            return "$rc"
        }
        is_available_npu 1 16
        """,
        capacity,
        rate,
        status,
    )
    assert (result.returncode == 0) == available


@pytest.mark.parametrize("npu", ["true", "false"])  # type: ignore[misc]
def test_partial_multi_lock_rolls_back_under_errexit(npu: str) -> None:
    result = run_shell(
        """
        set -e
        npu-smi() { echo 'Total Count : 4'; }
        source tools/devices/resource.sh
        npu_state=$1
        lock_gpu() { [[ $1 != 1 ]] || return 1; echo 123; }
        lock_npu() { lock_gpu "$@"; }
        unlock_gpu() { echo "release $1" >&2; }
        unlock_npu() { unlock_gpu "$@"; }
        lock_gpus 0,1
        """,
        npu,
    )
    assert result.returncode != 0
    assert "release 123" in result.stderr


@pytest.mark.parametrize("mask", ["", "3,5", "0,invalid"])  # type: ignore[misc]
def test_mask_rejects_lock_before_allocation(mask: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **kw: pytest.fail("queried after invalid mask")
    )
    monkeypatch.setattr(lock_gpu.ctypes, "CDLL", lambda *_: pytest.fail("unexpected allocation"))
    with pytest.raises((SystemExit, ValueError)):
        lock_gpu.lock_gpu(0, 4)


def test_lock_accepts_python3_without_python_alias() -> None:
    result = run_shell(
        """
        npu-smi() { echo 'Total Count : 4'; }
        source tools/devices/resource.sh
        npu_state=false
        command() {
            if [[ "$*" = "-v python" ]]; then return 1; fi
            builtin command "$@"
        }
        python() { return 127; }
        python3() { echo READY; exec sleep 3; }
        pid=$(lock_gpu 0) || exit 1
        unlock_gpu "$pid"
        """
    )
    assert result.returncode == 0, result.stderr


def test_reordered_cuda_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert lock_gpu._cuda_visible_ordinal(5) == 1
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert lock_gpu._cuda_visible_ordinal(5) == 5


def test_driver_uuid_mismatch_stops_before_context_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    driver = Mock()
    driver.cuInit.return_value = driver.cuDeviceGet.return_value = 0
    driver.cuDeviceGetUuid_v2.return_value = 0  # Returns sixteen zero bytes.
    monkeypatch.setattr(ctypes, "CDLL", lambda _: driver)
    with pytest.raises(SystemExit, match="does not match"):
        lock_gpu._lock_gpu_cuda(1, bytes(range(16)), 4)
    driver.cuCtxCreate_v2.assert_not_called()


@pytest.mark.parametrize("present", [True, False])  # type: ignore[misc]
def test_egl_fallback_matches_physical_uuid_before_opening_display(
    monkeypatch: pytest.MonkeyPatch, present: bool
) -> None:
    wanted = UUID(int=5)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "test-vendor")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: f"GPU-{wanted}")
    monkeypatch.setattr(lock_gpu, "_lock_gpu_cuda", lambda *a: False)
    egl = Mock()
    monkeypatch.setattr(ctypes, "CDLL", lambda _: egl)

    def enumerate_devices(count: int, devices: Any, size: Any) -> int:
        ctypes.cast(size, ctypes.POINTER(ctypes.c_int))[0] = 2
        if count:
            devices[0], devices[1] = 70, 71
        return 1

    def read_uuid(device: int, name: int, length: int, value: Any, size: Any) -> int:
        assert (name, length) == (0x335C, 16)
        value.raw = wanted.bytes if present and device == 71 else bytes(16)
        ctypes.cast(size, ctypes.POINTER(ctypes.c_int))[0] = 16
        return 1

    open_display = Mock(return_value=None)
    procs = {
        b"eglQueryDevicesEXT": enumerate_devices,
        b"eglQueryDeviceBinaryEXT": read_uuid,
        b"eglGetPlatformDisplayEXT": open_display,
    }
    egl.eglGetProcAddress.side_effect = procs.__getitem__
    monkeypatch.setattr(ctypes, "CFUNCTYPE", lambda *args: lambda callback: callback)
    with pytest.raises(SystemExit):
        lock_gpu.lock_gpu(5, 4)
    if present:
        open_display.assert_called_once_with(0x313F, 71, None)
    else:
        open_display.assert_not_called()
