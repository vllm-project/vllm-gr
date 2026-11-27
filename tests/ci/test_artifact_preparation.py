# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-def,no-any-return"
"""Host regressions for required inputs; no model runtime or network needed."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.helpers import artifacts
from tests.helpers.artifacts import (
    ArtifactError,
    SnapshotSpec,
    ensure_dataset,
    ensure_derived,
    ensure_model,
    snapshot_spec,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture(autouse=True)
def isolated_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLLM_GR_TEST_CACHE", str(tmp_path / "default-cache"))
    for name in (
        "VLLM_GR_TEST_MODEL",
        "VLLM_GR_TEST_DATASET",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def model_spec() -> SnapshotSpec:
    return SnapshotSpec(
        "tiny",
        "tests/tiny",
        "a" * 40,
        "model",
        ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"),
    )


def _model(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (output / name).write_text("{}", "utf-8")
    import numpy as np
    from safetensors.numpy import save_file

    save_file({"weight": np.zeros(1)}, str(output / "model.safetensors"))


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> list[SnapshotSpec]:
    calls = []

    def download(spec: SnapshotSpec, output: Path) -> None:
        calls.append(spec)
        _model(output)

    monkeypatch.setattr(artifacts, "_download", download)
    return calls


def test_cold_warm_and_different_machine_caches(model_spec, downloads, tmp_path) -> None:
    first = ensure_model(model_spec, cache_dir=tmp_path / "machine-a")
    assert ensure_model(model_spec, cache_dir=tmp_path / "machine-a") == first
    second = ensure_model(model_spec, cache_dir=tmp_path / "machine-b")
    assert len(downloads) == 2
    assert first != second
    assert artifacts._files(first) == artifacts._files(second)
    manifest = json.loads((first / ".gr-artifact.json").read_text("utf-8"))
    assert manifest["input"]["revision"] == model_spec.revision


@pytest.mark.parametrize("damage", ["missing", "empty", "changed", "bad-json", "manifest"])
def test_damaged_cache_is_rebuilt(model_spec, downloads, tmp_path, damage) -> None:
    path = ensure_model(model_spec, cache_dir=tmp_path)
    if damage == "missing":
        (path / "model.safetensors").unlink()
    elif damage == "empty":
        (path / "model.safetensors").write_bytes(b"")
    elif damage == "changed":
        (path / "model.safetensors").write_bytes(b"other weight bytes")
    elif damage == "bad-json":
        (path / "config.json").write_text("{", "utf-8")
    else:
        (path / ".gr-artifact.json").unlink()
    assert ensure_model(model_spec, cache_dir=tmp_path) == path
    assert len(downloads) == 2


def test_revision_changes_cache_key(model_spec, downloads, tmp_path) -> None:
    first = ensure_model(model_spec, cache_dir=tmp_path)
    second = ensure_model(replace(model_spec, revision="b" * 40), cache_dir=tmp_path)
    assert first != second
    assert len(downloads) == 2


def test_explicit_local_override_is_validated_not_modified(model_spec, downloads, tmp_path) -> None:
    local = tmp_path / "manual-model"
    _model(local)
    before = artifacts._files(local)
    assert ensure_model(model_spec, local_override=local) == local.resolve()
    assert not downloads
    assert artifacts._files(local) == before
    assert not (local / ".gr-artifact.json").exists()
    (local / "tokenizer.json").unlink()
    with pytest.raises(ArtifactError, match="Missing or empty"):
        ensure_model(model_spec, local_override=local)
    assert not downloads
    with pytest.raises(ArtifactError, match="does not exist"):
        ensure_model(model_spec, local_override=tmp_path / "absent")


def test_local_checkpoint_can_use_an_alternative_weight_layout(
    model_spec, downloads, tmp_path
) -> None:
    local = tmp_path / "converted-model"
    _model(local)
    (local / "model.safetensors").rename(local / "pytorch_model.bin")
    assert ensure_model(model_spec, local_override=local) == local.resolve()
    assert not downloads


def test_local_custom_code_must_be_present(model_spec, tmp_path) -> None:
    local = tmp_path / "model"
    _model(local)
    (local / "config.json").write_text(
        json.dumps({"auto_map": {"AutoModel": "modeling_custom.CustomModel"}}), "utf-8"
    )
    with pytest.raises(ArtifactError, match="modeling_custom.py"):
        ensure_model(model_spec, local_override=local)
    (local / "modeling_custom.py").write_text("# custom model\n", "utf-8")
    assert ensure_model(model_spec, local_override=local) == local.resolve()


def test_missing_indexed_weight_shard_is_rejected(model_spec, tmp_path) -> None:
    local = tmp_path / "model"
    _model(local)
    (local / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "missing.safetensors"}}), "utf-8"
    )
    with pytest.raises(ArtifactError, match="missing.safetensors"):
        ensure_model(model_spec, local_override=local)
    (local / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "../outside.safetensors"}}), "utf-8"
    )
    with pytest.raises(ValueError, match="Unsafe"):
        ensure_model(model_spec, local_override=local)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "C:/outside", "a\\b", ".", "", ".gr-artifact.json"]
)
def test_unsafe_spec_paths_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        SnapshotSpec("test", "tests/model", "a" * 40, "model", (name,))


@pytest.mark.parametrize("revision", ["main", "latest", "v1.0", "a" * 39])
def test_floating_or_invalid_revisions_rejected(model_spec, revision) -> None:
    with pytest.raises(ValueError, match="full commit SHA"):
        replace(model_spec, revision=revision)


@pytest.mark.parametrize("failure", [RuntimeError("offline"), KeyboardInterrupt()])
def test_interrupted_download_is_not_published(model_spec, monkeypatch, tmp_path, failure) -> None:
    def fail(spec, output):
        _model(output)
        raise failure

    monkeypatch.setattr(artifacts, "_download", fail)
    with pytest.raises(type(failure)):
        ensure_model(model_spec, cache_dir=tmp_path)
    assert not list((tmp_path / "objects").glob("*/.gr-artifact.json"))
    assert not [p for p in (tmp_path / "objects").iterdir() if p.is_dir()]


def test_thread_contention_downloads_once(model_spec, monkeypatch, tmp_path) -> None:
    calls = []

    def download(spec, output):
        calls.append(spec)
        time.sleep(0.1)
        _model(output)

    monkeypatch.setattr(artifacts, "_download", download)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ensure_model(model_spec, cache_dir=tmp_path), range(2)))
    assert results[0] == results[1]
    assert len(calls) == 1


def test_dataset_validation_and_api_kind(tmp_path, monkeypatch, model_spec) -> None:
    spec = SnapshotSpec("data", "tests/data", "a" * 40, "dataset", ("rows.parquet", "map.json"))

    def download(spec, output):
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table({"row": [1]}), output / "rows.parquet")
        (output / "map.json").write_text('{"0": "item"}', "utf-8")

    monkeypatch.setattr(artifacts, "_download", download)
    path = ensure_dataset(spec, cache_dir=tmp_path)
    assert (path / "rows.parquet").is_file()
    (path / "rows.parquet").write_bytes(b"PAR1payloadPAR1")
    with pytest.raises(ArtifactError, match="Parquet"):
        ensure_dataset(spec, local_override=path)
    with pytest.raises(ValueError, match="model spec"):
        ensure_model(spec)
    with pytest.raises(ValueError, match="dataset spec"):
        ensure_dataset(model_spec)


def test_sdk_receives_pin_subset_and_uses_default_authentication(
    model_spec, monkeypatch, tmp_path
) -> None:
    calls = []
    monkeypatch.setenv("HF_TOKEN", "unit-test-secret")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kw: calls.append(kw)),
    )
    artifacts._download(model_spec, tmp_path)
    assert calls == [
        {
            "repo_id": model_spec.repo_id,
            "repo_type": "model",
            "revision": model_spec.revision,
            "local_dir": str(tmp_path),
            "allow_patterns": list(model_spec.required_files),
            "etag_timeout": 60,
        }
    ]


@pytest.mark.parametrize("status", [403, 404, 503])
def test_download_failure_preserves_identity_without_credentials(
    model_spec, monkeypatch, tmp_path, status
) -> None:
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        error = RuntimeError("unit-test-secret")
        monkeypatch.setattr(error, "response", SimpleNamespace(status_code=status), raising=False)
        raise error

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fail))
    with pytest.raises(ArtifactError) as error:
        artifacts._download(model_spec, tmp_path)
    assert len(calls) == 1
    assert "unit-test-secret" not in str(error.value)


def _derived(source: Path, cache: Path, calls: list[Path], **changes: Any) -> Path:
    def build(output: Path) -> None:
        calls.append(output)
        (output / "result.json").write_text(source.read_text("utf-8"), "utf-8")

    options: dict[str, Any] = dict(
        name="test-derived",
        inputs={"source": source},
        parameters={"mode": "a"},
        builder_version="v1",
        required_files=("result.json",),
        builder=build,
        cache_dir=cache,
    )
    options.update(changes)
    return ensure_derived(**options)


def test_derived_cache_tracks_input_parameters_builder_and_outputs(tmp_path) -> None:
    source = tmp_path / "input.json"
    source.write_text("[1]", "utf-8")
    calls: list[Path] = []
    path = _derived(source, tmp_path / "cache", calls)
    assert _derived(source, tmp_path / "cache", calls) == path
    assert len(calls) == 1
    (path / "result.json").write_text("corrupt", "utf-8")
    assert _derived(source, tmp_path / "cache", calls) == path
    assert len(calls) == 2
    assert _derived(source, tmp_path / "cache", calls, parameters={"mode": "b"}) != path
    assert _derived(source, tmp_path / "cache", calls, builder_version="v2") != path
    source.write_text("[2]", "utf-8")
    assert _derived(source, tmp_path / "cache", calls) != path


def test_reviewed_registry_has_immutable_models_and_dataset() -> None:
    for name in ("onerec1b7", "onerec_benchmark"):
        spec = snapshot_spec(name)
        assert len(spec.revision) == 40
        assert len(spec.required_files) >= 8
    from tools.AB_Tests.download_onerec_dataset import DEFAULT_REVISION

    assert snapshot_spec("onerec_benchmark").revision == DEFAULT_REVISION


def test_warm_derived_checks_inputs_after_waiting(tmp_path, monkeypatch) -> None:
    from filelock import FileLock

    source = tmp_path / "input.json"
    source.write_text("[1]")
    _derived(source, tmp_path, [])
    acquire = FileLock.acquire

    def change_on_acquire(lock, *args, **kwargs):
        source.write_text("[2]")
        return acquire(lock, *args, **kwargs)

    monkeypatch.setattr(FileLock, "acquire", change_on_acquire)
    with pytest.raises(ArtifactError, match="input changed"):
        _derived(source, tmp_path, [])
