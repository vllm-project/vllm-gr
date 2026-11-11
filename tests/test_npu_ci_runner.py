# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Control-plane regressions; these do not establish NPU kernel correctness."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("npu_ci_runner", ROOT / "tools/ci/run_npu_tests.py")
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("device", ["", "-1", "03", "3,4", "3;echo bad", "\uff13"])  # type: ignore[misc]
def test_invalid_device_is_rejected(device: str) -> None:
    with pytest.raises(ValueError):
        runner.device_environment(device, {})


def test_allocated_device_and_vendor_paths_survive() -> None:
    inherited = {"ASCEND_RT_VISIBLE_DEVICES": "3", "PYTHONPATH": "/vendor/acl:/vendor/tbe"}
    env = runner.device_environment("3", inherited)
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "3"
    assert env["PYTHONPATH"].endswith(os.pathsep + inherited["PYTHONPATH"])
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert inherited == {"ASCEND_RT_VISIBLE_DEVICES": "3", "PYTHONPATH": "/vendor/acl:/vendor/tbe"}


@pytest.mark.parametrize(  # type: ignore[misc]
    "inherited",
    [
        {"ASCEND_RT_VISIBLE_DEVICES": "4"},
        {"ASCEND_RT_VISIBLE_DEVICES": ""},
        {"PYTEST_ADDOPTS": "--collect-only"},
        {"PYTEST_PLUGINS": "unexpected_plugin"},
    ],
)
def test_inherited_conflicting_controls_are_rejected(inherited: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        runner.device_environment("3", inherited)


def _observe(tmp_path: Path, body: str, *extra: str) -> tuple[dict[str, Any], int]:
    """Exercise real pytest hooks in a synthetic project with no runtime imports."""
    test = tmp_path / "test_probe.py"
    test.write_text("import pytest\n" + body, encoding="utf-8")
    report = tmp_path / "events.json"
    env = dict(
        os.environ,
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONPATH=str(ROOT / "tools/ci"),
        VLLM_GR_NPU_TEST_REPORT=str(report),
    )
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    command = [sys.executable, "-m", "pytest", "-q", "-p", "npu_pytest_report", *extra, str(test)]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    assert report.is_file(), completed.stdout
    return json.loads(report.read_text(encoding="utf-8")), completed.returncode


@pytest.mark.parametrize(  # type: ignore[misc]
    ("body", "extra", "outcome"),
    [
        ("def test_probe():\n    assert True\n", (), "PASSED"),
        ("def test_probe():\n    pytest.skip('unavailable')\n", (), "SKIPPED"),
        ("@pytest.mark.xfail\ndef test_probe():\n    assert False\n", (), "XFAIL_OR_XPASS"),
        ("@pytest.mark.xfail\ndef test_probe():\n    assert True\n", (), "XFAIL_OR_XPASS"),
        ("def test_probe():\n    assert False\n", (), "FAILED"),
        (
            "@pytest.fixture(autouse=True)\ndef fail():\n    raise RuntimeError('setup')\n"
            "def test_probe():\n    pass\n",
            (),
            "FAILED",
        ),
        (
            "@pytest.fixture(autouse=True)\ndef fail():\n    yield\n    raise RuntimeError('teardown')\n"
            "def test_probe():\n    pass\n",
            (),
            "FAILED",
        ),
        ("def test_probe():\n    pass\n", ("--collect-only",), "NOT_RUN_OR_INCOMPLETE"),
    ],
)
def test_candidate_audits_actual_pytest_phases(
    tmp_path: Path,
    body: str,
    extra: tuple[str, ...],
    outcome: str,
) -> None:
    report, rc = _observe(tmp_path, body, *extra)
    node = report["selected"][0]["nodeid"]
    audit = runner.audit_report(report, [node], rc)
    assert audit["nodes"][node] == outcome
    assert audit["status"] == ("PASSED" if outcome == "PASSED" else "FAILED")


def test_full_reports_existing_skips_without_counting_them_as_passes(tmp_path: Path) -> None:
    report, rc = _observe(
        tmp_path, "def test_ok():\n    pass\ndef test_skip():\n    pytest.skip('optional')\n"
    )
    audit = runner.audit_report(report, None, rc)
    assert audit["status"] == "PASSED"
    assert audit["counts"] == {"PASSED": 1, "SKIPPED": 1}
    assert runner.audit_report(report, ["test_probe.py::missing"], rc)["status"] == "FAILED"


def test_duplicate_or_incomplete_phase_cannot_qualify(tmp_path: Path) -> None:
    report, rc = _observe(tmp_path, "def test_probe():\n    pass\n")
    node = report["selected"][0]["nodeid"]
    report["reports"].append(report["reports"][1])
    assert runner.audit_report(report, [node], rc)["status"] == "FAILED"


def test_stale_output_is_never_reused(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    old = output / "result.json"
    old.write_text('{"status": "PASSED"}', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/ci/run_npu_tests.py"),
            "--device",
            "3",
            "--output-dir",
            str(output),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    assert result.returncode != 0
    assert old.read_text(encoding="utf-8") == '{"status": "PASSED"}'


def test_candidate_manifest_has_exact_existing_unique_nodes() -> None:
    manifest = json.loads((ROOT / "tools/ci/npu_test_suites.json").read_text(encoding="utf-8"))
    assert len(runner.candidate_nodes(manifest)) == 17
    tier = manifest["tiers"]["p0-candidate"]
    tier["groups"]["kernels"]["nodes"][1] = tier["groups"]["kernels"]["nodes"][0]
    with pytest.raises(ValueError, match="duplicated"):
        runner.candidate_nodes(manifest)


def test_timeout_preserves_failure_and_captured_log(tmp_path: Path) -> None:
    command = [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(30)"]
    assert runner.run_command(command, dict(os.environ), tmp_path / "child.log", 1) == 124
    assert "started" in (tmp_path / "child.log").read_text(encoding="utf-8")


def test_post_execution_inspection_error_cannot_leave_passed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "new-run"
    monkeypatch.setattr(sys, "argv", ["runner", "--device", "3", "--output-dir", str(output)])
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("PYTEST_PLUGINS", raising=False)
    inspections = iter([{"content_sha256": "initial"}, OSError("source unavailable")])

    def identity() -> dict[str, Any]:
        value = next(inspections)
        if isinstance(value, OSError):
            raise value
        assert isinstance(value, dict)
        return value

    def command(argv: list[str], env: dict[str, str], log: Path, timeout: int | None) -> int:
        if "-m" in argv:
            assert timeout is None  # Full retains the surrounding job's deadline.
            node = "tests/test_probe.py::test_ok"
            phases = [
                {"nodeid": node, "phase": phase, "outcome": "passed"}
                for phase in ("setup", "call", "teardown")
            ]
            runner.write_json(
                output / "pytest-events.json",
                {
                    "selected": [{"nodeid": node}],
                    "reports": phases,
                    "exitstatus": 0,
                },
            )
            (output / "junit.xml").write_text("<testsuites/>", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "source_identity", identity)
    monkeypatch.setattr(runner, "run_command", command)
    assert runner.main() == 1
    assert json.loads((output / "result.json").read_text(encoding="utf-8"))["status"] == "FAILED"


@pytest.mark.skipif(os.name != "posix", reason="Ascend child-session cleanup uses Linux /proc")  # type: ignore[misc]
def test_timeout_cleans_tagged_child_in_separate_session(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    child = (
        "import os,time; from pathlib import Path; Path(%r).write_text(str(os.getpid())); time.sleep(60)"
        % str(pidfile)
    )
    parent = (
        "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r],start_new_session=True); time.sleep(60)"
        % child
    )
    log = tmp_path / "timeout.log"
    assert runner.run_command([sys.executable, "-c", parent], dict(os.environ), log, 2) == 124
    receipt = json.loads(log.with_suffix(".execution.json").read_text(encoding="utf-8"))
    assert int(pidfile.read_text()) in receipt["cleaned_owned_pids"]
    assert receipt["remaining_owned_pids"] == []
    assert receipt["output_reader_finished"]


def test_collection_skip_keeps_reason_and_does_not_qualify(tmp_path: Path) -> None:
    report, rc = _observe(tmp_path, "pytest.skip('missing dependency', allow_module_level=True)\n")
    assert "missing dependency" in report["collection_skips"][0]["detail"]
    assert runner.audit_report(report, ["test_probe.py::test_probe"], rc)["status"] == "FAILED"


def test_full_rejects_corrupt_phase_even_with_another_pass(tmp_path: Path) -> None:
    report, rc = _observe(tmp_path, "def test_probe():\n    pass\n")
    report["selected"].append({"nodeid": "test_probe.py::corrupt"})
    report["reports"].append(
        {
            "nodeid": "test_probe.py::corrupt",
            "phase": "unknown",
            "outcome": "skipped",
        }
    )
    assert runner.audit_report(report, None, rc)["status"] == "FAILED"
