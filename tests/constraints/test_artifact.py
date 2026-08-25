# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save

from tests.constraints.test_table import make_table
from vllm_gr.constraints.artifact import (
    load_constraint_table_artifact,
    save_constraint_table_artifact,
    serialize_constraint_table_artifact,
)
from vllm_gr.constraints.table import CONSTRAINT_TABLE_TENSOR_NAMES, ConstraintTable

_TOKENIZER_DIGEST = "sha256:" + "b" * 64


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _load(path: Path, digest: str, *, vocab_size: int = 6) -> ConstraintTable:
    return load_constraint_table_artifact(
        path,
        expected_digest=digest,
        expected_tokenizer_digest=_TOKENIZER_DIGEST,
        expected_vocab_size=vocab_size,
    )


def _metadata() -> dict[str, str]:
    return {
        "format": "constraint_table_v1",
        "tokenizer_digest": _TOKENIZER_DIGEST,
        "vocab_size": "6",
        "max_prefix1_degree": "2",
        "max_prefix2_degree": "2",
    }


def test_artifact_round_trip_and_reproducible_bytes(tmp_path: Path) -> None:
    table = make_table()
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"

    first_digest = save_constraint_table_artifact(table, first)
    second_digest = save_constraint_table_artifact(table, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_digest == second_digest
    loaded = _load(first, first_digest)
    assert loaded.metadata.artifact_digest == first_digest
    for name in CONSTRAINT_TABLE_TENSOR_NAMES:
        assert np.array_equal(getattr(loaded, name), getattr(table, name))
        assert getattr(loaded, name).flags.writeable is False


def test_artifact_digest_is_not_self_referential_metadata() -> None:
    data = serialize_constraint_table_artifact(make_table())
    header_size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + header_size])
    assert "artifact_digest" not in header["__metadata__"]


def test_tampering_fails_before_payload_validation(tmp_path: Path) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)

    with pytest.raises(ValueError, match="digest mismatch"):
        _load(path, digest)


def test_rejects_tokenizer_and_vocab_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)

    with pytest.raises(ValueError, match="tokenizer digest mismatch"):
        load_constraint_table_artifact(
            path,
            expected_digest=digest,
            expected_tokenizer_digest="sha256:" + "c" * 64,
            expected_vocab_size=6,
        )
    with pytest.raises(ValueError, match="vocab_size mismatch"):
        _load(path, digest, vocab_size=7)


def test_rejects_extra_or_missing_tensor(tmp_path: Path) -> None:
    table = make_table()
    tensors = {name: np.asarray(getattr(table, name)) for name in CONSTRAINT_TABLE_TENSOR_NAMES}
    tensors["extra"] = np.asarray([1], dtype=np.int32)
    data = save(tensors, metadata=_metadata())
    path = tmp_path / "extra.safetensors"
    path.write_bytes(data)

    with pytest.raises(ValueError, match="tensor mismatch"):
        _load(path, _digest(data))

    tensors.pop("extra")
    tensors.pop("first_token_ids")
    data = save(tensors, metadata=_metadata())
    path.write_bytes(data)
    with pytest.raises(ValueError, match="tensor mismatch"):
        _load(path, _digest(data))


def test_rejects_shape_and_dtype_after_verified_digest(tmp_path: Path) -> None:
    table = make_table()
    tensors = {name: np.asarray(getattr(table, name)) for name in CONSTRAINT_TABLE_TENSOR_NAMES}
    tensors["first_token_ids"] = tensors["first_token_ids"].reshape(1, 2)
    data = save(tensors, metadata=_metadata())
    path = tmp_path / "bad.safetensors"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="1-D"):
        _load(path, _digest(data))

    tensors["first_token_ids"] = np.asarray([1, 3], dtype=np.int64)
    data = save(tensors, metadata=_metadata())
    path.write_bytes(data)
    with pytest.raises(ValueError, match="dtype"):
        _load(path, _digest(data))


def test_atomic_save_leaves_no_temporary_file(tmp_path: Path) -> None:
    output = tmp_path / "table.safetensors"
    save_constraint_table_artifact(make_table(), output)
    assert [path.name for path in tmp_path.iterdir()] == [output.name]
