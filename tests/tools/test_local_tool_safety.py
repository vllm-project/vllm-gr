# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise local cleanup and credential handling with mocked tools."""

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.ci import gpu_process_guard as guard

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
ROOT = Path(__file__).resolve().parents[2]
TOKEN = "synthetic-test-value-that-must-not-appear-in-traces"


def test_cleanup_requires_confirmation_before_device_access() -> None:
    with (
        patch.object(sys, "argv", ["guard", "cleanup"]),
        patch.object(guard.devices, "selected_devices") as select,
        patch.object(guard.os, "kill") as kill,
        pytest.raises(SystemExit) as exc,
    ):
        guard.main()
    assert exc.value.code == 2
    select.assert_not_called()
    kill.assert_not_called()


def check_action(action: str) -> None:
    argv = ["guard", action] + (["--yes"] if action == "cleanup" else [])
    with (
        patch.object(sys, "argv", argv),
        patch.object(guard.devices, "selected_devices", return_value=[0]),
        patch.object(guard.devices, "compute_processes", return_value=[]),
        patch.object(guard, "cleanup") as cleanup,
    ):
        assert guard.main() == 0
    if action == "cleanup":
        cleanup.assert_called_once_with([0], [], 30)
    else:
        cleanup.assert_not_called()


def test_explicit_cleanup_and_read_only_check() -> None:
    check_action("check")
    check_action("cleanup")


def setup_mock_act(tmp_path: Path, script: str) -> dict[str, str]:
    devices = tmp_path / "tools" / "devices"
    devices.mkdir(parents=True)
    (devices / "resource.sh").write_text(
        """
get_gpu() { printf '0\\n'; }
lock_gpu() { printf '123\\n'; }
unlock_gpu() { printf '%s\\n' "$1" > unlocked; }
"""
    )
    act = tmp_path / "act"
    act.write_text(script)
    act.chmod(0o700)
    return {
        "PATH": f"{tmp_path}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "HF_TOKEN": TOKEN,
    }


@pytest.mark.parametrize("act_status", [17, 143])  # type: ignore[misc]
def test_run_act_cleans_up_without_tracing_hf_token(tmp_path: Path, act_status: int) -> None:
    env = setup_mock_act(
        tmp_path,
        """#!/bin/bash
set -eu
[[ "$1" == --secret-file ]]
[[ "$(stat -c %a -- "$2")" == 600 ]]
[[ "$(<"$2")" == "HF_TOKEN=$HF_TOKEN" ]]
printf '%s\\n' "$2" > secret-path
exit "$MOCK_ACT_STATUS"
""",
    )
    env["MOCK_ACT_STATUS"] = str(act_status)
    result = subprocess.run(
        ["bash", "-x", str(ROOT / "tools/run_act.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert TOKEN not in result.stdout
    assert TOKEN not in result.stderr
    assert result.returncode == act_status, result.stderr
    assert not Path((tmp_path / "secret-path").read_text().strip()).exists()
    assert (tmp_path / "unlocked").read_text() == "123\n"


def wait_for_mock_act(path: Path, wrapper: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while not path.exists() and wrapper.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), f"act did not reach {path.name}, wrapper exit={wrapper.poll()}"


@pytest.mark.parametrize(  # type: ignore[misc]
    ("signum", "expected_status", "signal_group"),
    [
        pytest.param(signal.SIGINT, 130, False, id="2-130"),
        pytest.param(signal.SIGTERM, 143, False, id="15-143"),
        pytest.param(signal.SIGINT, 130, True, id="foreground-group"),
    ],
)
def test_run_act_forwards_signal_and_waits_for_cleanup(
    tmp_path: Path, signum: signal.Signals, expected_status: int, signal_group: bool
) -> None:
    env = setup_mock_act(
        tmp_path,
        f"#!{sys.executable}\n"
        r"""
import os
import signal
import sys
import time
from pathlib import Path

pid_marker = Path("act-pid.tmp")
pid_marker.write_text(str(os.getpid()))
pid_marker.rename("act-pid")
assert sys.argv[1] == "--secret-file"
assert sys.stdin.readline() == "mock stdin\n"
assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN
secret_file = Path(sys.argv[2])
unlocked = Path("unlocked")
received_signals = []

def handle_signal(signum, _frame):
    assert secret_file.is_file()
    assert not unlocked.exists()
    received_signals.append(signum)
    Path("received-signal").touch()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)
Path("secret-path").write_text(str(secret_file))
Path("running").touch()
while not Path("allow-exit").exists():
    time.sleep(0.01)
assert secret_file.is_file()
assert not unlocked.exists()
Path("handled-signals").write_text("\n".join(map(str, received_signals)))
sys.exit(55)
""",
    )
    act_pid_path = tmp_path / "act-pid"
    with subprocess.Popen(
        ["bash", str(ROOT / "tools/run_act.sh")],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
    ) as wrapper:
        try:
            assert wrapper.stdin is not None
            wrapper.stdin.write("mock stdin\n")
            wrapper.stdin.flush()
            wait_for_mock_act(tmp_path / "running", wrapper)
            secret_file = Path((tmp_path / "secret-path").read_text())
            act_group = os.getpgid(int(act_pid_path.read_text()))
            if signal_group:
                os.killpg(wrapper.pid, signum)
            else:
                wrapper.send_signal(signum)
            wait_for_mock_act(tmp_path / "received-signal", wrapper)
            assert wrapper.poll() is None
            assert secret_file.is_file()
            assert not (tmp_path / "unlocked").exists()
            (tmp_path / "allow-exit").touch()
            stdout, stderr = wrapper.communicate(timeout=5)
        except BaseException:
            if wrapper.poll() is None and act_pid_path.is_file():
                with suppress(ProcessLookupError):
                    os.kill(int(act_pid_path.read_text()), signal.SIGKILL)
                with suppress(subprocess.TimeoutExpired):
                    wrapper.wait(timeout=5)
            if wrapper.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(wrapper.pid, signal.SIGKILL)
                wrapper.wait(timeout=5)
            raise
    assert TOKEN not in stdout
    assert TOKEN not in stderr
    assert wrapper.returncode == expected_status, stderr
    assert (tmp_path / "handled-signals").read_text().splitlines() == [str(int(signum))]
    if signal_group:
        # Linux can coalesce duplicate SIGINTs before either is handled.
        assert act_group != wrapper.pid
    assert not secret_file.exists()
    assert (tmp_path / "unlocked").read_text() == "123\n"
