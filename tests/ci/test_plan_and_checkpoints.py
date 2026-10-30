# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU integration coverage for conservative planning and exact-runtime retries."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci import run_stage
from tools.ci.plan import docs_only, make_plan


@pytest.mark.parametrize(  # type: ignore[misc]
    "paths,expected",
    [
        ([], False),
        (["docs/guide.md", "README.md"], True),
        (["docs/diagram.svg"], True),
        (["docs/../setup.py"], False),
        (["/docs/guide.md"], False),
        (["docs/conf.py"], False),
        (["examples/run.py"], False),
        ([".github/workflows/pre-commit.yml"], False),
        (["README.md", "vllm_gr/__init__.py"], False),
    ],
)
def test_conservative_coverage(paths: list[str], expected: bool) -> None:
    assert docs_only(paths) is expected
    assert make_plan("pull_request", paths)["device-required"] == str(not expected).lower()


def test_push_coalesces_and_manual_defaults_to_full() -> None:
    assert make_plan("push", ["vllm_gr/engine.py"])["reason"] == "push-coalesced"
    assert make_plan("workflow_dispatch", ["README.md"])["device-required"] == "true"


def test_diff_failure_does_not_skip_device(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "tools/ci/plan.py"
    result = subprocess.run(
        [sys.executable, str(script), "--event", "pull_request", "--base", "absent"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout)["device-required"] == "true"


def test_stage_reuse_and_failed_retry_invalidate_previous_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; p=Path('calls'); p.write_text(p.read_text()+'x' if p.exists() else 'x')",
    ]
    cache = tmp_path / "cache"
    assert run_stage.execute("recall", command, cache, "same", False) == 0
    assert run_stage.execute("recall", command, cache, "same", True) == 0
    assert Path("calls").read_text() == "x"
    assert json.loads(Path("ci-results/recall.json").read_text())["reused"] is True
    assert (
        run_stage.execute(
            "recall", [sys.executable, "-c", "raise SystemExit(7)"], cache, "same", False
        )
        == 7
    )
    assert run_stage.execute("recall", command, cache, "same", True) == 0
    assert Path("calls").read_text() == "xx"


def test_interruption_removes_previous_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    run_stage.execute("recall", [sys.executable, "-c", "pass"], cache, "same", False)

    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_stage.execute("recall", ["fake"], cache, "same", False)
    assert not (cache / "same.json").exists()


@pytest.mark.parametrize("age,expected", [(0, True), (21599, True), (21600, False), (-1, False)])  # type: ignore[misc]
def test_checkpoint_expiry(age: float, expected: bool) -> None:
    record = {"fingerprint": "key", "returncode": 0, "completed_at": 30000 - age}
    assert run_stage.reusable(record, "key", 30000) is expected
    assert not run_stage.reusable(record, "another", 30000)


def test_fingerprint_invalidates_snapshot_runtime_policy_and_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for key in ("CI_HEAD_SHA", "CI_BASE_SHA", "CI_MERGE_SHA", "RUNNER_NAME"):
        monkeypatch.setenv(key, key)
    monkeypatch.setenv("CI_CUDA_MAPPING_FILE", str(tmp_path / "mapping.json"))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    Path("mapping.json").write_text('{"uuid": "GPU-one"}')
    Path("resolved-benchmark-policy.env").write_text("threshold=1\n")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "GPU-one, 570\n")
    monkeypatch.setattr(run_stage.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(run_stage.importlib.metadata, "distributions", list)
    key = run_stage.fingerprint("recall", ["bash", "recall.sh"])
    assert key == run_stage.fingerprint("recall", ["bash", "recall.sh"])
    for name in (
        "CI_HEAD_SHA",
        "CI_BASE_SHA",
        "CI_MERGE_SHA",
        "RUNNER_NAME",
        "CI_RUNTIME_REVISION",
        "V1_RECALL_LIMIT",
    ):
        previous = os.environ.get(name)
        monkeypatch.setenv(name, "changed")
        assert key != run_stage.fingerprint("recall", ["bash", "recall.sh"])
        if previous is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, previous)
    assert key != run_stage.fingerprint("performance", ["bash", "recall.sh"])
    assert key != run_stage.fingerprint("recall", ["bash", "other.sh"])
    Path("OpenOneRec").mkdir()
    Path("OpenOneRec/data.json").write_text("{}")
    assert key != run_stage.fingerprint("recall", ["bash", "recall.sh"])
    Path("OpenOneRec/data.json").unlink()
    Path("mapping.json").write_text('{"uuid": "GPU-two"}')
    assert key != run_stage.fingerprint("recall", ["bash", "recall.sh"])
    Path("mapping.json").write_text('{"uuid": "GPU-one"}')
    Path("resolved-benchmark-policy.env").write_text("threshold=2\n")
    assert key != run_stage.fingerprint("recall", ["bash", "recall.sh"])


def test_incomplete_identity_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI_HEAD_SHA", raising=False)
    with pytest.raises(ValueError, match="complete snapshot"):
        run_stage.fingerprint("recall", ["true"])


def test_regenerated_assets_reuse_a_real_fingerprint_but_changed_bytes_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("CI_HEAD_SHA", "CI_BASE_SHA", "CI_MERGE_SHA", "RUNNER_NAME"):
        monkeypatch.setenv(name, name)
    monkeypatch.setenv("CI_CUDA_MAPPING_FILE", str(tmp_path / "mapping.json"))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("CI_CHECKPOINT_ROOT", str(tmp_path / "cache"))
    Path("mapping.json").write_text('{"uuid": "GPU-one"}')
    Path("resolved-benchmark-policy.env").write_text("threshold=1\n")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "GPU-one, 570\n")
    monkeypatch.setattr(run_stage.importlib.metadata, "distributions", list)
    asset = Path("generated_data/video_constraint_triples.json")
    asset.parent.mkdir()
    asset.write_text('[{"tokens": [1, 2, 3]}]')
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; p=Path('calls'); p.write_text(p.read_text()+'x' if p.exists() else 'x')",
    ]
    cache = tmp_path / "cache"
    key = run_stage.fingerprint("performance", command)
    assert run_stage.execute("performance", command, cache, key, False) == 0
    previous = asset.stat()
    content = asset.read_bytes()
    asset.unlink()
    asset.write_bytes(content)
    os.utime(asset, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1000000000))
    regenerated = run_stage.fingerprint("performance", command)
    assert regenerated == key
    assert run_stage.execute("performance", command, cache, regenerated, True) == 0
    assert Path("calls").read_text() == "x"
    assert json.loads(Path("ci-results/performance.json").read_text())["reused"] is True

    previous = asset.stat()
    asset.write_bytes(content.replace(b"3", b"4"))
    os.utime(asset, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert asset.stat().st_size == previous.st_size
    changed = run_stage.fingerprint("performance", command)
    assert changed != key  # Same size and restored mtime cannot hide changed inputs.
    assert run_stage.execute("performance", command, cache, changed, True) == 0
    assert Path("calls").read_text() == "xx"


def test_later_generated_catalogs_do_not_invalidate_device_suite_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    before = run_stage.asset_inventory("device-tests")
    Path("generated_data").mkdir()
    Path("generated_data/video_catalog.json").write_text("{}")
    assert run_stage.asset_inventory("device-tests") == before
    assert run_stage.asset_inventory("performance") != before
