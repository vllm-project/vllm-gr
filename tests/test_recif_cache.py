# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tools.recif_cache import (
    checkpoint_identity,
    converter_sha,
    prepare_runtime_checkpoint,
)


def _write_source_checkpoint(source: Path) -> None:
    source.mkdir()
    (source / "config.json").write_text(
        '{"architectures": ["Qwen3MoeForCausalLM"]}',
        encoding="utf-8",
    )
    (source / "_model_rank0.pt").write_bytes(b"backbone")
    (source / "external_rank0.pt").write_bytes(b"external")


def _write_runtime_files(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        '{"architectures": ["RecifForCausalLM"]}',
        encoding="utf-8",
    )
    (output / "model.safetensors").write_bytes(b"runtime-weights")


def _read_manifest(runtime_dir: Path) -> dict[str, Any]:
    manifest_path = runtime_dir / "conversion_manifest.json"
    assert manifest_path.is_file(), "converted cache must publish a manifest"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_prepare_runtime_checkpoint_reuses_validated_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_hf"
    _write_source_checkpoint(source)
    calls: list[Path] = []

    def fake_convert(input_dir: Path, output_dir: Path, **_kwargs: object) -> None:
        calls.append(input_dir)
        _write_runtime_files(output_dir)

    monkeypatch.setattr("tools.recif_cache.convert_checkpoint", fake_convert)

    first = prepare_runtime_checkpoint(source, output)
    assert first == output
    assert len(calls) == 1
    manifest = _read_manifest(output)
    assert manifest["version"] == 1
    assert manifest["source_identity"] == checkpoint_identity(source)
    assert manifest["converter_sha"] == converter_sha()
    assert set(manifest["files"]) == {"config.json", "model.safetensors"}

    # A second call must hit the validated manifest without reconverting.
    second = prepare_runtime_checkpoint(source, output)
    assert second == output
    assert len(calls) == 1


def test_prepare_runtime_checkpoint_reconverts_unmanifested_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_hf"
    _write_source_checkpoint(source)
    # Stale artifacts from an older converter (or an interrupted run) must
    # not be trusted just because the two filenames exist.
    _write_runtime_files(output)
    calls: list[Path] = []

    def fake_convert(input_dir: Path, output_dir: Path, **_kwargs: object) -> None:
        calls.append(input_dir)
        _write_runtime_files(output_dir)

    monkeypatch.setattr("tools.recif_cache.convert_checkpoint", fake_convert)
    prepare_runtime_checkpoint(source, output)
    assert len(calls) == 1
    _read_manifest(output)


def test_prepare_runtime_checkpoint_reconverts_on_identity_or_converter_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_hf"
    _write_source_checkpoint(source)
    calls: list[Path] = []

    def fake_convert(input_dir: Path, output_dir: Path, **_kwargs: object) -> None:
        calls.append(input_dir)
        _write_runtime_files(output_dir)

    monkeypatch.setattr("tools.recif_cache.convert_checkpoint", fake_convert)

    prepare_runtime_checkpoint(source, output)
    assert len(calls) == 1

    # A different source checkpoint must invalidate the cache entry.
    (source / "config.json").write_text(
        '{"architectures": ["Qwen3MoeForCausalLM"], "changed": true}',
        encoding="utf-8",
    )
    prepare_runtime_checkpoint(source, output)
    assert len(calls) == 2
    assert _read_manifest(output)["source_identity"] == checkpoint_identity(source)

    # A manifest stamped by an older converter SHA must also be rejected.
    manifest = _read_manifest(output)
    manifest["converter_sha"] = "outdated-converter"
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    prepare_runtime_checkpoint(source, output)
    assert len(calls) == 3


def test_prepare_runtime_checkpoint_reconverts_corrupted_runtime_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_hf"
    _write_source_checkpoint(source)
    calls: list[Path] = []

    def fake_convert(input_dir: Path, output_dir: Path, **_kwargs: object) -> None:
        calls.append(input_dir)
        _write_runtime_files(output_dir)

    monkeypatch.setattr("tools.recif_cache.convert_checkpoint", fake_convert)
    prepare_runtime_checkpoint(source, output)
    assert len(calls) == 1

    (output / "model.safetensors").write_bytes(b"tampered")
    prepare_runtime_checkpoint(source, output)
    assert len(calls) == 2
    assert (output / "model.safetensors").read_bytes() == b"runtime-weights"


def test_failed_initial_conversion_leaves_no_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_hf"
    _write_source_checkpoint(source)

    def bad_convert(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("conversion failed")

    monkeypatch.setattr("tools.recif_cache.convert_checkpoint", bad_convert)
    with pytest.raises(RuntimeError, match="conversion failed"):
        prepare_runtime_checkpoint(source, output)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))


def test_failed_reconversion_preserves_previously_published_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_hf"
    _write_source_checkpoint(source)

    def good_convert(input_dir: Path, output_dir: Path, **_kwargs: object) -> None:
        _write_runtime_files(output_dir)

    monkeypatch.setattr("tools.recif_cache.convert_checkpoint", good_convert)
    prepare_runtime_checkpoint(source, output)

    # Invalidate the entry so the next call attempts a reconversion, then
    # make that conversion fail: the published directory must stay intact.
    manifest = _read_manifest(output)
    manifest["converter_sha"] = "stale-converter"
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    def bad_convert(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("conversion failed")

    monkeypatch.setattr("tools.recif_cache.convert_checkpoint", bad_convert)
    with pytest.raises(RuntimeError, match="conversion failed"):
        prepare_runtime_checkpoint(source, output)

    assert (output / "config.json").is_file()
    assert (output / "model.safetensors").read_bytes() == b"runtime-weights"
    assert (output / "conversion_manifest.json").is_file()
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))


def test_concurrent_prepare_is_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_hf"
    _write_source_checkpoint(source)
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def fake_convert(input_dir: Path, output_dir: Path, **_kwargs: object) -> None:
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.15)
            _write_runtime_files(output_dir)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr("tools.recif_cache.convert_checkpoint", fake_convert)
    results: list[Path] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(prepare_runtime_checkpoint(source, output))
        except Exception as exc:  # noqa: BLE001 - failure only on bug
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert all(result == output for result in results)
    assert max_active == 1
    _read_manifest(output)
    assert (output / "model.safetensors").read_bytes() == b"runtime-weights"
