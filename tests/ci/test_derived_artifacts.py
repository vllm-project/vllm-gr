# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-def,no-any-return"
"""Derived input preparation, including cross-process locking and OneRec recipes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ci.test_artifact_preparation import _derived
from tests.helpers.artifacts import ArtifactError
from tests.helpers.paths import REPO_ROOT

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
pytest_plugins = ("pytester",)


def test_model_fixture_is_lazy_and_prepares_once_per_session(pytester, monkeypatch, tmp_path):
    from tests.helpers import artifact_fixtures

    calls = []
    monkeypatch.delenv("VLLM_GR_TEST_MODEL", raising=False)

    def prepare(spec, **kwargs):
        calls.append(spec.name)
        return tmp_path

    monkeypatch.setattr(artifact_fixtures, "ensure_model", prepare)
    pytester.makeconftest("from tests.helpers.artifact_fixtures import onerec_model")
    pytester.makepyfile("""
        def test_host(): pass
        def test_model_a(onerec_model): assert onerec_model.is_dir()
        def test_model_b(onerec_model): assert onerec_model.is_dir()
    """)
    pytester.runpytest("--collect-only").assert_outcomes()
    pytester.runpytest("-k", "host").assert_outcomes(passed=1, deselected=2)
    assert calls == []
    pytester.runpytest("-k", "model_a or model_b").assert_outcomes(passed=2, deselected=1)
    assert calls == ["onerec1b7"]


def test_mutating_derived_input_is_not_published(tmp_path) -> None:
    source = tmp_path / "input.json"
    source.write_text("[1]", "utf-8")

    def mutate(output):
        source.write_text("[2]", "utf-8")
        (output / "result.json").write_text("[1]", "utf-8")

    with pytest.raises(ArtifactError, match="input changed"):
        _derived(source, tmp_path / "cache", [], builder=mutate)
    assert not list((tmp_path / "cache").glob("objects/*/.gr-artifact.json"))


def test_missing_derived_output_is_not_published(tmp_path) -> None:
    source = tmp_path / "input.json"
    source.write_text("[]", "utf-8")
    with pytest.raises(ArtifactError, match="Missing or empty"):
        _derived(source, tmp_path / "cache", [], builder=lambda output: None)


def test_invalid_derived_json_is_not_published(tmp_path) -> None:
    source = tmp_path / "input.json"
    source.write_text("[]", "utf-8")

    def invalid(output):
        (output / "result.json").write_text("{", "utf-8")

    with pytest.raises(ArtifactError, match="Invalid derived JSON"):
        _derived(source, tmp_path / "cache", [], builder=invalid)
    assert not list((tmp_path / "cache").glob("objects/*/.gr-artifact.json"))


def test_process_contention_builds_once(tmp_path) -> None:
    source = tmp_path / "input.json"
    source.write_text("[1]", "utf-8")
    script = tmp_path / "prepare.py"
    script.write_text(
        """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from tests.helpers.artifacts import ensure_derived
root = Path(sys.argv[1])
def build(output):
    with (root / 'calls').open('a') as stream:
        stream.write('build\\n')
    time.sleep(0.2)
    (output / 'out.json').write_text('[1]')
ensure_derived(name='process-test', inputs={'source': root / 'input.json'}, parameters={},
               builder_version='1', required_files=('out.json',), builder=build, cache_dir=root / 'cache')
""",
        "utf-8",
    )
    processes = [
        subprocess.Popen(
            [sys.executable, str(script), str(tmp_path), str(REPO_ROOT)],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert (tmp_path / "calls").read_text().splitlines() == ["build"]


@pytest.mark.parametrize("mapping", ["sid2pid", "sid2iid"])
def test_onerec_mapping_recipe_uses_existing_converter_and_caches_outputs(
    tmp_path, monkeypatch, mapping
) -> None:
    from tests.helpers import onerec_artifacts

    data = tmp_path / "benchmark_data"
    data.mkdir()
    (data / f"{mapping}.json").write_text('{"0": "item"}', "utf-8")
    calls = []
    real_run = subprocess.run

    def run(command, **kwargs):
        calls.append(command)
        if "--mapping-file" in command:
            # Run the real, import-light mapping converter. The compiler stage
            # requires the prepared Torch runtime and is simulated in this test.
            return real_run(command, **kwargs)
        catalog = Path(command[command.index("--catalog") + 1])
        assert json.loads(catalog.read_text("utf-8")) == [
            ["<|sid_begin|>", "<s_a_0>", "<s_b_0>", "<s_c_0>", "<|sid_end|>"]
        ]
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"triples": [["<s_a_0>", "<s_b_0>", "<s_c_0>"]]}', "utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        onerec_artifacts,
        "subprocess",
        SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired),
    )
    path = onerec_artifacts.ensure_onerec_constraints(data, mapping=mapping, cache_dir=tmp_path)
    assert (path / "catalog.json").is_file()
    assert (path / "triples.json").is_file()
    assert (
        onerec_artifacts.ensure_onerec_constraints(data, mapping=mapping, cache_dir=tmp_path)
        == path
    )
    assert len(calls) == 2


def test_onerec_recipe_rejects_unknown_mapping(tmp_path) -> None:
    from tests.helpers.onerec_artifacts import ensure_onerec_constraints

    with pytest.raises(ValueError, match="sid2pid or sid2iid"):
        ensure_onerec_constraints(tmp_path, mapping="video")
