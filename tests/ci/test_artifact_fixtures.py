# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-def,no-any-return"
"""Exercise lazy input fixtures and the real Pangu unittest fixture lifecycle."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
pytest_plugins = ("pytester",)


@pytest.fixture(autouse=True)
def reset_vocab_slice_config() -> Generator[None, None, None]:
    yield


@pytest.fixture(autouse=True)
def isolated_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLLM_GR_TEST_CACHE", str(tmp_path / "cache"))
    for name in (
        "PANGU1B_MODEL_PATH",
        "PANGU_MODEL_PATH",
        "PANGU1B_DATASET_PATH",
        "PANGU1B_REQUIRE_RUNTIME",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


def _fixtures(pytester: pytest.Pytester, *, fail: bool = False, runtime: bool = True) -> Path:
    log = pytester.path / "events.jsonl"
    pytester.makeconftest(
        f"""
import json
from pathlib import Path
import pytest
from tests.helpers.artifact_fixtures import pangu_model, gsm8k_mini
from tests.helpers import artifacts
from tests.st import pangu1b_st_common
patcher = pytest.MonkeyPatch()
log = Path({str(log)!r})
def download(spec, output):
    with log.open('a') as stream:
        stream.write(json.dumps({{'event': 'download', 'name': spec.name}}) + '\\n')
    if {fail!r}:
        raise artifacts.ArtifactError('download unavailable')
    for name in spec.required_files:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.safetensors':
            from safetensors.numpy import save_file
            import numpy as np
            save_file({{'weight': np.zeros(1)}}, str(path))
            continue
        path.write_text('{{}}' if path.suffix == '.json' else 'fixture bytes')
def pytest_configure(config):
    patcher.setattr(artifacts, '_download', download)
    patcher.setattr(pangu1b_st_common, 'has_vllm', lambda: {runtime!r})
def pytest_unconfigure(config):
    patcher.undo()
"""
    )
    return log


def test_collect_and_deselection_do_not_download(pytester: pytest.Pytester) -> None:
    log = _fixtures(pytester)
    pytester.makepyfile("def test_model(pangu_model): pass\ndef test_host(): pass\n")
    assert pytester.runpytest("--collect-only").ret == 0
    assert not log.exists()
    pytester.runpytest("-k", "test_host").assert_outcomes(passed=1, deselected=1)
    assert not log.exists()


def test_selected_cases_share_download_and_record_identity(pytester: pytest.Pytester) -> None:
    log = _fixtures(pytester)
    pytester.makepyfile(
        "def test_first(pangu_model, gsm8k_mini):\n"
        "    assert (pangu_model / 'config.json').is_file()\n"
        "    assert gsm8k_mini.is_file()\n"
        "def test_second(pangu_model):\n"
        "    assert (pangu_model / 'model.safetensors').is_file()\n"
    )
    report = pytester.path / "report.xml"
    pytester.runpytest(f"--junitxml={report}").assert_outcomes(passed=2)
    assert len(log.read_text().splitlines()) == 1
    text = report.read_text("utf-8")
    assert "pangu_model.identity" in text
    assert "gsm8k_mini.canonical_sha256" in text


def test_missing_model_download_failure_is_error_not_skip(pytester: pytest.Pytester) -> None:
    log = _fixtures(pytester, fail=True)
    pytester.makepyfile("def test_model(pangu_model): pass")
    result = pytester.runpytest()
    result.assert_outcomes(errors=1)
    assert "download unavailable" in result.stdout.str()
    assert len(log.read_text().splitlines()) == 1


def test_missing_runtime_does_not_download(pytester: pytest.Pytester) -> None:
    log = _fixtures(pytester, runtime=False)
    pytester.makepyfile("def test_model(pangu_model): pass")
    pytester.runpytest().assert_outcomes(skipped=1)
    assert not log.exists()


def test_strict_missing_runtime_is_error(pytester: pytest.Pytester, monkeypatch) -> None:
    log = _fixtures(pytester, runtime=False)
    monkeypatch.setenv("PANGU1B_REQUIRE_RUNTIME", "1")
    pytester.makepyfile("def test_model(pangu_model): pass")
    pytester.runpytest().assert_outcomes(errors=1)
    assert not log.exists()


def test_invalid_explicit_model_does_not_fallback_to_download(
    pytester: pytest.Pytester, monkeypatch
) -> None:
    log = _fixtures(pytester)
    monkeypatch.setenv("PANGU1B_MODEL_PATH", str(pytester.path / "absent"))
    pytester.makepyfile("def test_model(pangu_model): pass")
    result = pytester.runpytest()
    result.assert_outcomes(errors=1)
    assert "does not exist" in result.stdout.str()
    assert not log.exists()


def test_invalid_dataset_is_error(pytester: pytest.Pytester, monkeypatch) -> None:
    _fixtures(pytester)
    bad = pytester.path / "bad.json"
    bad.write_text('[{"question": "missing answer"}]', "utf-8")
    monkeypatch.setenv("PANGU1B_DATASET_PATH", str(bad))
    pytester.makepyfile("def test_data(gsm8k_mini): pass")
    result = pytester.runpytest()
    result.assert_outcomes(errors=1)
    assert "question/answer" in result.stdout.str()


def test_real_pangu_offline_class_uses_prepared_inputs(pytester: pytest.Pytester) -> None:
    log = _fixtures(pytester)
    pytester.makepyfile(
        """
import json
from pathlib import Path
from types import SimpleNamespace
from tests.st import test_pangu1b_accuracy_st as offline
def run(command, **kwargs):
    assert Path(command[command.index('--model') + 1]).is_dir()
    assert Path(command[command.index('--dataset-path') + 1]).is_file()
    Path(command[command.index('--output-json') + 1]).write_text(
        json.dumps({'accuracy': 1.0, 'num_samples': 8}))
    return SimpleNamespace(returncode=0, stdout='', stderr='')
patcher = offline.mock.patch.object(offline, 'subprocess', SimpleNamespace(run=run))
patcher.start()
TestPangu1bAccuracyST = offline.TestPangu1bAccuracyST
def teardown_module():
    patcher.stop()
"""
    )
    pytester.runpytest().assert_outcomes(passed=1)
    assert len(log.read_text().splitlines()) == 1


def test_real_pangu_online_class_prepares_before_server_and_cleans_up(
    pytester: pytest.Pytester,
) -> None:
    log = _fixtures(pytester)
    pytester.makepyfile(
        f"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from tests.st import test_pangu1b_online_st as online
log = Path({str(log)!r})
@contextmanager
def server(model):
    assert Path(model).is_dir()
    with log.open('a') as stream: stream.write('{{"event": "start"}}\\n')
    try:
        yield 'http://127.0.0.1:1'
    finally:
        with log.open('a') as stream: stream.write('{{"event": "stop"}}\\n')
patcher = online.mock.patch.object(online, '_pangu_serve_server', server)
patcher.start()
def get(url, **kwargs):
    return SimpleNamespace(status_code=200, json=lambda: {{'data': [{{'id': online.SERVED_MODEL_NAME}}]}})
client = online.mock.patch.object(online, 'requests', SimpleNamespace(get=get))
client.start()
TestPangu1bOnlineServeST = online.TestPangu1bOnlineServeST
def teardown_module():
    client.stop()
    patcher.stop()
"""
    )
    pytester.runpytest("-k", "health_endpoint or v1_models").assert_outcomes(passed=2, deselected=3)
    events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
    assert events == ["download", "start", "stop"]


def test_short_online_override_fails_before_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.st import test_pangu1b_online_st as online

    case = online.TestPangu1bOnlineServeST()
    case.dataset_path = tmp_path / "short.json"
    case.dataset_path.write_text('[{"question":"q","answer":"a"}]')
    monkeypatch.setattr(
        online, "_chat_completion", lambda *a, **kw: pytest.fail("unexpected request")
    )
    with pytest.raises(AssertionError, match="eight prompts"):
        case.test_gsm8k_mini_online_accuracy()
