# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-read safetensors Artifact persistence and startup validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from vllm_gr.config import CONSTRAINT_TABLE_FORMAT
from vllm_gr.constraints.table import (
    MAX_TABLE_BYTES,
    TABLE_NAMES,
    ConstraintTable,
    ConstraintTableInfo,
    validate_constraint_table,
)

_METADATA_KEYS = frozenset(
    {
        "format",
        "tokenizer_digest",
        "vocab_size",
        "max_prefix1_degree",
        "max_prefix2_degree",
    }
)
_MAX_HEADER_BYTES = 1024 * 1024
_MAX_ARTIFACT_BYTES = MAX_TABLE_BYTES + _MAX_HEADER_BYTES + 8


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_artifact_bytes(artifact_path: Path) -> bytes:
    """Open once and reject oversized Artifacts before allocating their payload."""

    try:
        with artifact_path.open("rb") as artifact_file:
            artifact_size = os.fstat(artifact_file.fileno()).st_size
            if artifact_size > _MAX_ARTIFACT_BYTES:
                raise ValueError(
                    f"constraint Artifact size {artifact_size} exceeds "
                    f"{_MAX_ARTIFACT_BYTES} byte limit"
                )
            data = artifact_file.read(artifact_size + 1)
    except OSError as exc:
        raise ValueError(f"cannot read constraint Artifact {str(artifact_path)!r}: {exc}") from exc

    if len(data) != artifact_size:
        raise ValueError(
            "constraint Artifact size changed while being read: "
            f"expected {artifact_size} bytes, got {len(data)}"
        )
    return data


def _artifact_metadata(data: bytes) -> dict[str, str]:
    if len(data) < 8:
        raise ValueError("constraint Artifact is too short to contain a safetensors header")
    header_size = int.from_bytes(data[:8], byteorder="little", signed=False)
    if header_size <= 0 or header_size > _MAX_HEADER_BYTES or 8 + header_size > len(data):
        raise ValueError(f"invalid safetensors header size {header_size}")
    try:
        header = json.loads(data[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header: {exc}") from exc
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError("constraint Artifact must contain string safetensors metadata")
    missing = sorted(_METADATA_KEYS - set(metadata))
    unknown = sorted(set(metadata) - _METADATA_KEYS)
    if missing or unknown:
        raise ValueError(
            f"constraint Artifact metadata mismatch: missing={missing}, unknown={unknown}"
        )
    return metadata


def _metadata_int(metadata: dict[str, str], name: str) -> int:
    try:
        value = int(metadata[name])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"constraint Artifact metadata {name!r} must be an integer") from exc
    if value < 0:
        raise ValueError(f"constraint Artifact metadata {name!r} must be non-negative")
    return value


def load_constraint_table_artifact(
    path: str | Path,
    *,
    expected_digest: str,
    expected_tokenizer_digest: str,
    expected_vocab_size: int,
    expected_format: str = CONSTRAINT_TABLE_FORMAT,
) -> ConstraintTable:
    """Read once, then hash, parse, and fully validate from the same bytes."""

    artifact_path = Path(path)
    data = _read_artifact_bytes(artifact_path)

    actual_digest = sha256_bytes(data)
    if actual_digest != expected_digest:
        raise ValueError(
            f"constraint Artifact digest mismatch: expected {expected_digest}, got {actual_digest}"
        )
    metadata = _artifact_metadata(data)
    if metadata["format"] != expected_format:
        raise ValueError(
            f"constraint Artifact format mismatch: expected {expected_format!r}, "
            f"got {metadata['format']!r}"
        )
    if metadata["tokenizer_digest"] != expected_tokenizer_digest:
        raise ValueError(
            "constraint Artifact tokenizer digest mismatch: expected "
            f"{expected_tokenizer_digest}, got {metadata['tokenizer_digest']}"
        )
    vocab_size = _metadata_int(metadata, "vocab_size")
    if vocab_size != expected_vocab_size:
        raise ValueError(
            f"constraint Artifact vocab_size mismatch: expected {expected_vocab_size}, "
            f"got {vocab_size}"
        )

    try:
        from safetensors.numpy import load

        tensors: dict[str, NDArray[Any]] = load(data)
    except Exception as exc:
        raise ValueError(f"cannot parse constraint Artifact safetensors payload: {exc}") from exc
    missing_tables = sorted(set(TABLE_NAMES) - set(tensors))
    extra_tables = sorted(set(tensors) - set(TABLE_NAMES))
    if missing_tables or extra_tables:
        raise ValueError(
            f"constraint Artifact table mismatch: missing={missing_tables}, extra={extra_tables}"
        )

    total_bytes = sum(int(tensors[name].nbytes) for name in TABLE_NAMES)
    info = ConstraintTableInfo(
        format=metadata["format"],
        artifact_digest=actual_digest,
        tokenizer_digest=metadata["tokenizer_digest"],
        vocab_size=vocab_size,
        max_prefix1_degree=_metadata_int(metadata, "max_prefix1_degree"),
        max_prefix2_degree=_metadata_int(metadata, "max_prefix2_degree"),
        total_bytes=total_bytes,
    )
    table = ConstraintTable(
        **{name: tensors[name] for name in TABLE_NAMES},
        info=info,
    )
    # ConstraintTable owns immutable copies. Release the serialized payload and
    # Safetensors backing arrays before semantic validation allocates temporary
    # NumPy buffers for offsets, keys, and degree checks.
    del data, tensors
    validate_constraint_table(table)
    return table


def serialize_constraint_table_artifact(table: ConstraintTable) -> bytes:
    """Serialize a compiler-produced Host table without touching the filesystem."""

    validate_constraint_table(table)
    metadata = {
        "format": table.info.format,
        "tokenizer_digest": table.info.tokenizer_digest,
        "vocab_size": str(table.info.vocab_size),
        "max_prefix1_degree": str(table.info.max_prefix1_degree),
        "max_prefix2_degree": str(table.info.max_prefix2_degree),
    }
    tensors: dict[str, Any] = {
        name: np.ascontiguousarray(getattr(table, name)) for name in TABLE_NAMES
    }
    from safetensors.numpy import save

    return cast(bytes, save(tensors, metadata=metadata))


def write_constraint_table_artifact(table: ConstraintTable, path: str | Path) -> str:
    """Atomically write an Artifact and return its whole-file SHA-256."""

    data = serialize_constraint_table_artifact(table)
    output = Path(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            written = temporary_file.write(data)
            if written != len(data):
                raise OSError(
                    f"short write for temporary Artifact: expected {len(data)} bytes, "
                    f"wrote {written}"
                )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, output)
        temporary_path = None

        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(output.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ValueError(f"cannot write constraint Artifact {str(output)!r}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return sha256_bytes(data)
