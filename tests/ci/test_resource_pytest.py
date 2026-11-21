# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Host regressions for resource gate isolation, status handling and process cleanup."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.development import test_resource_probe as probe
from tests.helpers.gate_process import run_gate_process
from tools.devices.lock_gpu import _cuda_visible_ordinal

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture(autouse=True)
def reset_vocab_slice_config() -> None:
    pass


@pytest.mark.parametrize(
    "mask,physical,ordinal", [("1", 1, 0), ("2,0", 2, 0), ("2,0", 0, 1), (None, 1, 1)]
)
def test_visible_ordinal(
    monkeypatch: pytest.MonkeyPatch, mask: str | None, physical: int, ordinal: int
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert _cuda_visible_ordinal(physical) == ordinal


def test_nonisolated_runner_does_not_discover_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("VLLM_GR_ISOLATED_RUNNER", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: pytest.fail("Device discovery attempted"))
    with pytest.raises(pytest.skip.Exception, match="ISOLATED_RUNNER"):
        probe.test_resource_probe("single", tmp_path)


@pytest.mark.parametrize("status", [0, 1, 77])
def test_probe_exit_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int) -> None:
    monkeypatch.setenv("VLLM_GR_ISOLATED_RUNNER", "true")
    # Patch only the resource module's platform view, not pathlib's platform.
    from types import SimpleNamespace

    monkeypatch.setattr(
        probe,
        "os",
        SimpleNamespace(environ={"VLLM_GR_ISOLATED_RUNNER": "true"}, name="posix", pathsep=":"),
    )
    monkeypatch.setattr(shutil, "which", lambda name: name)

    def run(command: list[str], **kwargs: object) -> int:
        log = kwargs["log"]
        assert isinstance(log, Path)
        log.write_text("probe output")
        return status

    monkeypatch.setattr(probe, "run_gate_process", run)
    if status == 77:
        with pytest.raises(pytest.skip.Exception, match="eligible devices"):
            probe.test_resource_probe("single", tmp_path)
    elif status:
        with pytest.raises(AssertionError, match="probe output"):
            probe.test_resource_probe("single", tmp_path)
    else:
        probe.test_resource_probe("single", tmp_path)


def test_process_failure_keeps_log(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    status = run_gate_process(
        [sys.executable, "-c", "print('diagnostic'); raise SystemExit(7)"],
        cwd=tmp_path,
        log=log,
        timeout=10,
    )
    assert status == 7
    assert "diagnostic" in log.read_text()


def test_timeout_terminates_child(tmp_path: Path) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        run_gate_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            log=tmp_path / "timeout.log",
            timeout=0.2,
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
def test_failed_driver_does_not_leave_child(tmp_path: Path) -> None:
    import os
    import time

    marker = tmp_path / "leaked"
    child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
    driver = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {child!r}])"
    assert os.name == "posix"
    assert (
        run_gate_process(
            [sys.executable, "-c", driver], cwd=tmp_path, log=tmp_path / "group.log", timeout=10
        )
        == 0
    )
    time.sleep(1.5)
    assert not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="gate runs in POSIX Bash")
@pytest.mark.parametrize("scenario", ["single", "pair", "masked"])
def test_surviving_holder_fails_probe(tmp_path: Path, scenario: str) -> None:
    resource = tmp_path / "tools/devices/resource.sh"
    resource.parent.mkdir(parents=True)
    resource.write_text("""
npu_state=false
get_gpu() { echo 1; }
get_gpus() { echo 0,1; }
lock_gpu() { echo 100; }
lock_gpus() { echo 100,101; }
unlock_gpu() { :; }
unlock_gpus() { echo "$1" >> cleanup.log; }
ps() { echo S; }
sleep() { :; }
nvidia-smi() { echo 2; }
""")
    result = subprocess.run(
        [
            "bash",
            "-c",
            "unset CUDA_VISIBLE_DEVICES\n" + probe._PREAMBLE + probe._SCENARIOS[scenario],
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "remains active" in result.stderr
    assert (tmp_path / "cleanup.log").read_text().strip()
