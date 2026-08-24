# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save

from vllm_gr.constraints import artifact as artifact_module
from vllm_gr.constraints.artifact import (
    load_constraint_table_artifact,
    serialize_constraint_table_artifact,
    sha256_bytes,
    write_constraint_table_artifact,
)
from vllm_gr.constraints.table import TABLE_NAMES

from .helpers import TOKENIZER_DIGEST, make_table


def _write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return sha256_bytes(data)


def _raw_artifact(
    *,
    tensors: dict[str, np.ndarray] | None = None,
    metadata: dict[str, str] | None = None,
) -> bytes:
    table = make_table()
    values = {name: np.array(getattr(table, name), copy=True) for name in TABLE_NAMES}
    if tensors:
        values.update(tensors)
    meta = {
        "format": table.info.format,
        "tokenizer_digest": table.info.tokenizer_digest,
        "vocab_size": str(table.info.vocab_size),
        "max_prefix1_degree": str(table.info.max_prefix1_degree),
        "max_prefix2_degree": str(table.info.max_prefix2_degree),
    }
    if metadata:
        meta.update(metadata)
    return save(values, metadata=meta)


def _load(path: Path, digest: str, **overrides: object):
    kwargs = {
        "expected_digest": digest,
        "expected_tokenizer_digest": TOKENIZER_DIGEST,
        "expected_vocab_size": 8,
    }
    kwargs.update(overrides)
    return load_constraint_table_artifact(path, **kwargs)


def test_round_trip_uses_whole_file_digest(tmp_path: Path) -> None:
    data = serialize_constraint_table_artifact(make_table())
    path = tmp_path / "table.safetensors"
    digest = _write(path, data)

    table = _load(path, digest)

    assert table.info.artifact_digest == digest
    assert table.info.tokenizer_digest == TOKENIZER_DIGEST
    assert table.info.vocab_size == 8
    for name in TABLE_NAMES:
        assert np.array_equal(getattr(table, name), getattr(make_table(), name))


def test_tamper_fails_before_parse(tmp_path: Path) -> None:
    data = bytearray(serialize_constraint_table_artifact(make_table()))
    digest = sha256_bytes(bytes(data))
    data[-1] ^= 1
    path = tmp_path / "tampered.safetensors"
    path.write_bytes(data)

    with pytest.raises(ValueError, match="digest mismatch"):
        _load(path, digest)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_tokenizer_digest": "sha256:" + "b" * 64}, "tokenizer digest"),
        ({"expected_vocab_size": 9}, "vocab_size"),
        ({"expected_format": "constraint_table_v2"}, "format mismatch"),
    ],
)
def test_metadata_mismatch_fails_startup(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    data = _raw_artifact()
    path = tmp_path / "table.safetensors"
    digest = _write(path, data)
    with pytest.raises(ValueError, match=message):
        _load(path, digest, **overrides)


def test_wrong_dtype_fails_startup(tmp_path: Path) -> None:
    data = _raw_artifact(tensors={"first_token_ids": np.asarray([1, 2], dtype=np.int64)})
    path = tmp_path / "bad-dtype.safetensors"
    digest = _write(path, data)
    with pytest.raises(ValueError, match="first_token_ids.dtype"):
        _load(path, digest)


def test_bad_offset_fails_startup(tmp_path: Path) -> None:
    data = _raw_artifact(
        tensors={
            "prefix1_offsets": np.asarray(
                [0, 0, 2, 4, 4, 4, 4, 4, 4],
                dtype=np.int32,
            )
        }
    )
    path = tmp_path / "bad-offset.safetensors"
    digest = _write(path, data)
    with pytest.raises(ValueError, match="prefix1_offsets"):
        _load(path, digest)


def test_out_of_range_token_fails_startup(tmp_path: Path) -> None:
    data = _raw_artifact(tensors={"prefix2_values": np.asarray([5, 6, 8], dtype=np.int32)})
    path = tmp_path / "bad-token.safetensors"
    digest = _write(path, data)
    with pytest.raises(ValueError, match="out-of-range"):
        _load(path, digest)


def test_missing_or_extra_table_fails_startup(tmp_path: Path) -> None:
    table = make_table()
    values = {
        name: np.array(getattr(table, name), copy=True)
        for name in TABLE_NAMES
        if name != "prefix2_values"
    }
    values["unexpected"] = np.asarray([1], dtype=np.int32)
    metadata = {
        "format": table.info.format,
        "tokenizer_digest": table.info.tokenizer_digest,
        "vocab_size": "8",
        "max_prefix1_degree": "2",
        "max_prefix2_degree": "1",
    }
    data = save(values, metadata=metadata)
    path = tmp_path / "bad-tables.safetensors"
    digest = _write(path, data)
    with pytest.raises(ValueError, match="table mismatch"):
        _load(path, digest)


def test_artifact_is_opened_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = _raw_artifact()
    path = tmp_path / "table.safetensors"
    digest = _write(path, data)
    calls = 0
    original = Path.open

    def counted(self: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    _load(path, digest)
    assert calls == 1


def test_oversized_artifact_fails_before_payload_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "oversized.safetensors"
    path.write_bytes(b"too large")
    monkeypatch.setattr(artifact_module, "_MAX_ARTIFACT_BYTES", 4)

    with pytest.raises(ValueError, match=r"size 9 exceeds 4 byte limit"):
        _load(path, sha256_bytes(b"too large"))


def test_atomic_write_replaces_existing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "table.safetensors"
    path.write_bytes(b"old artifact")

    digest = write_constraint_table_artifact(make_table(), path)

    assert sha256_bytes(path.read_bytes()) == digest
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_atomic_write_failure_preserves_existing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "table.safetensors"
    old_data = b"old artifact"
    path.write_bytes(old_data)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(artifact_module.os, "replace", fail_replace)
    with pytest.raises(ValueError, match="cannot write constraint Artifact"):
        write_constraint_table_artifact(make_table(), path)

    assert path.read_bytes() == old_data
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_atomic_write_fsync_failure_preserves_existing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "table.safetensors"
    old_data = b"old artifact"
    path.write_bytes(old_data)

    def fail_fsync(_file_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(artifact_module.os, "fsync", fail_fsync)
    with pytest.raises(ValueError, match="cannot write constraint Artifact"):
        write_constraint_table_artifact(make_table(), path)

    assert path.read_bytes() == old_data
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))
