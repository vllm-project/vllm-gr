# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Functional adapter failure contracts; no model or accelerator execution."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.st import test_beam_functional_gate as beam
from tests.st import test_pangu1b_accuracy_st as pangu

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture(autouse=True)
def reset_vocab_slice_config() -> None:
    pass


@pytest.mark.parametrize(
    "status,report",
    [(1, {"passed": True}), (0, None), (0, {"passed": False}), (0, {"passed": "true"})],
)
def test_beam_rejects_failed_missing_or_invalid_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: int,
    report: dict[str, object] | None,
) -> None:
    monkeypatch.setenv("VLLM_GR_FUNCTIONAL_GATE", "1")
    monkeypatch.setenv("VLLM_GR_FUNCTIONAL_REPORT_DIR", str(tmp_path))

    def run(command: list[str], **kwargs: object) -> int:
        if report is not None:
            output = Path(command[command.index("--output") + 1])
            (output / "result.json").write_text(json.dumps(report))
        return status

    monkeypatch.setattr(beam, "run_gate_process", run)
    with pytest.raises(AssertionError):
        beam.test_beam_functional("checkpoint", tmp_path)


@pytest.mark.parametrize(
    "accuracy,measure,valid", [(0.1, True, True), (0.1, False, False), (float("nan"), True, False)]
)
def test_pangu_measure_only_preserves_valid_metrics_without_applying_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    accuracy: float,
    measure: bool,
    valid: bool,
) -> None:
    monkeypatch.setenv("PANGU1B_MEASURE_ONLY", "1" if measure else "0")
    monkeypatch.setenv("PANGU1B_NUM_PROMPTS", "8")
    report = tmp_path / "report.json"
    monkeypatch.setenv("PANGU1B_OFFLINE_RESULT_JSON", str(report))
    monkeypatch.setattr(pangu, "DEFAULT_MIN_ACCURACY", "0.5")

    def run(command: list[str], **kwargs: object) -> int:
        assert command[command.index("--min-accuracy") + 1] == ("0" if measure else "0.5")
        Path(command[command.index("--output-json") + 1]).write_text(
            json.dumps({"accuracy": accuracy, "num_samples": 8})
        )
        log = kwargs["log"]
        assert isinstance(log, Path)
        log.write_text("")
        return 0

    monkeypatch.setattr(pangu, "run_gate_process", run)
    case = pangu.TestPangu1bAccuracyST()
    case.model_path = tmp_path
    case.dataset_path = pangu.DEFAULT_DATASET
    if valid:
        case.test_gsm8k_mini_accuracy_cli()
        assert json.loads(report.read_text())["accuracy"] == accuracy
    else:
        with pytest.raises(AssertionError):
            case.test_gsm8k_mini_accuracy_cli()
        assert not report.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process group and socket contract")
@pytest.mark.parametrize("mode", ["timeout", "cancel"])
def test_beam_server_pid_and_port_released(tmp_path: Path, mode: str) -> None:
    from tests.helpers.paths import REPO_ROOT

    listener = tmp_path / "listener.py"
    listener.write_text(
        "import json, os, socket, time\nfrom pathlib import Path\n"
        "s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen()\n"
        "Path('pid.tmp').write_text(json.dumps([os.getpid(), s.getsockname()[1]]))\n"
        "Path('pid.tmp').replace('pid.json')\ntime.sleep(120)\n"
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import os, sys, time\nfrom pathlib import Path\n"
        "from tests.helpers.beam_server import beam_server_process\n"
        "with open('server.log', 'w') as log, beam_server_process("
        "[sys.executable, 'listener.py'], env=dict(os.environ), cwd=Path.cwd(), log=log):\n"
        "    time.sleep(120)\n"
    )
    deadline = 3 if mode == "timeout" else 60
    launch = (
        "import subprocess, sys\nfrom pathlib import Path\n"
        "from tests.helpers.gate_process import run_gate_process\n"
        "try:\n    run_gate_process([sys.executable, 'driver.py'], cwd=Path.cwd(), "
        f"log=Path('driver.log'), timeout={deadline})\n"
        "except subprocess.TimeoutExpired: sys.exit(10)\nexcept KeyboardInterrupt: sys.exit(11)"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "VLLM_GR_BEAM_GATE_GROUP": "1"}
    with subprocess.Popen([sys.executable, "-c", launch], cwd=tmp_path, env=env) as gate:
        try:
            ready = tmp_path / "pid.json"
            until = time.monotonic() + 15
            while not ready.exists() and gate.poll() is None and time.monotonic() < until:
                time.sleep(0.05)
            pid, port = json.loads(ready.read_text())
            if mode == "cancel":
                gate.send_signal(signal.SIGTERM)
            assert gate.wait(timeout=20) == (10 if mode == "timeout" else 11)
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            with socket.socket() as check:
                check.bind(("127.0.0.1", port))
        finally:
            if gate.poll() is None:
                gate.terminate()
                gate.wait(timeout=20)
