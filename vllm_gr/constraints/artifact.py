# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Safetensors persistence for the canonical constraint-table artifact."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vllm_gr.config.models import CONSTRAINT_TABLE_FORMAT
from vllm_gr.config.validation import (
    validate_artifact_digest,
    validate_constraint_table_format,
    validate_tokenizer_digest,
)
from vllm_gr.constraints.table import (
    CONSTRAINT_TABLE_TENSOR_NAMES,
    MAX_CONSTRAINT_TABLE_BYTES,
    ConstraintTable,
    ConstraintTableMetadata,
    validate_constraint_table,
)

_ARTIFACT_METADATA_KEYS = frozenset(
    {
        "format",
        "tokenizer_digest",
        "vocab_size",
        "max_prefix1_degree",
        "max_prefix2_degree",
    }
)
_MAX_SAFETENSORS_HEADER_BYTES = 1024 * 1024
_MAX_ARTIFACT_BYTES = MAX_CONSTRAINT_TABLE_BYTES + _MAX_SAFETENSORS_HEADER_BYTES + 8


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_artifact(path: Path) -> bytes:
    """Read one opened file once, enforcing the size limit before allocation."""

    try:
        with path.open("rb") as artifact_file:
            artifact_size = os.fstat(artifact_file.fileno()).st_size
            if artifact_size > _MAX_ARTIFACT_BYTES:
                raise ValueError(
                    f"constraint table artifact size {artifact_size} exceeds "
                    f"the {_MAX_ARTIFACT_BYTES} byte limit"
                )
            data = artifact_file.read(artifact_size + 1)
    except OSError as exc:
        raise ValueError(f"cannot read constraint table artifact {str(path)!r}: {exc}") from exc
    if len(data) != artifact_size:
        raise ValueError(
            "constraint table artifact size changed while reading: "
            f"expected {artifact_size}, got {len(data)}"
        )
    return data


def _load_artifact_metadata(data: bytes) -> dict[str, str]:
    if len(data) < 8:
        raise ValueError("constraint table artifact is too short for a safetensors header")
    header_size = int.from_bytes(data[:8], "little", signed=False)
    if (
        header_size <= 0
        or header_size > _MAX_SAFETENSORS_HEADER_BYTES
        or 8 + header_size > len(data)
    ):
        raise ValueError(f"invalid safetensors header size {header_size}")
    try:
        header = json.loads(data[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError("constraint table artifact must contain string safetensors metadata")
    missing = sorted(_ARTIFACT_METADATA_KEYS - set(metadata))
    unknown = sorted(set(metadata) - _ARTIFACT_METADATA_KEYS)
    if missing or unknown:
        raise ValueError(
            f"constraint table artifact metadata mismatch: missing={missing}, unknown={unknown}"
        )
    return metadata


def _metadata_int(metadata: dict[str, str], name: str, *, minimum: int) -> int:
    raw = metadata.get(name)
    if raw is None or not raw.isdecimal():
        raise ValueError(f"constraint table artifact metadata {name!r} must be an integer")
    value = int(raw)
    if value < minimum:
        raise ValueError(f"constraint table artifact metadata {name!r} must be >= {minimum}")
    return value


def load_constraint_table_artifact(
    path: str | Path,
    *,
    expected_digest: str,
    expected_tokenizer_digest: str,
    expected_vocab_size: int,
    expected_format: str = CONSTRAINT_TABLE_FORMAT,
) -> ConstraintTable:
    """Read once, hash, inspect metadata, and validate the complete host ABI."""

    normalized_digest = validate_artifact_digest(expected_digest)
    normalized_tokenizer_digest = validate_tokenizer_digest(expected_tokenizer_digest)
    normalized_format = validate_constraint_table_format(expected_format)
    if normalized_digest is None or normalized_tokenizer_digest is None:
        raise ValueError("expected artifact and tokenizer digests are required")
    if type(expected_vocab_size) is not int or expected_vocab_size <= 0:
        raise ValueError("expected_vocab_size must be a positive integer")

    artifact_path = Path(path)
    data = _read_artifact(artifact_path)
    actual_digest = _sha256_bytes(data)
    if actual_digest != normalized_digest:
        raise ValueError(
            "constraint table artifact digest mismatch: "
            f"expected {normalized_digest}, got {actual_digest}"
        )

    metadata = _load_artifact_metadata(data)
    if metadata["format"] != normalized_format:
        raise ValueError(
            "constraint table artifact format mismatch: "
            f"expected {normalized_format!r}, got {metadata['format']!r}"
        )
    if metadata["tokenizer_digest"] != normalized_tokenizer_digest:
        raise ValueError(
            "constraint table artifact tokenizer digest mismatch: "
            f"expected {normalized_tokenizer_digest}, "
            f"got {metadata['tokenizer_digest']}"
        )
    vocab_size = _metadata_int(metadata, "vocab_size", minimum=1)
    if vocab_size != expected_vocab_size:
        raise ValueError(
            "constraint table artifact vocab_size mismatch: "
            f"expected {expected_vocab_size}, got {vocab_size}"
        )

    try:
        from safetensors.numpy import load

        tensors: dict[str, NDArray[Any]] = load(data)
    except Exception as exc:
        raise ValueError(f"cannot parse constraint table safetensors payload: {exc}") from exc
    missing_tensors = sorted(set(CONSTRAINT_TABLE_TENSOR_NAMES) - set(tensors))
    extra_tensors = sorted(set(tensors) - set(CONSTRAINT_TABLE_TENSOR_NAMES))
    if missing_tensors or extra_tensors:
        raise ValueError(
            "constraint table artifact tensor mismatch: "
            f"missing={missing_tensors}, extra={extra_tensors}"
        )

    total_bytes = sum(int(tensors[name].nbytes) for name in CONSTRAINT_TABLE_TENSOR_NAMES)
    table = ConstraintTable(
        first_token_ids=tensors["first_token_ids"],
        prefix1_offsets=tensors["prefix1_offsets"],
        prefix1_values=tensors["prefix1_values"],
        prefix1_pair_keys=tensors["prefix1_pair_keys"],
        prefix2_value_offsets=tensors["prefix2_value_offsets"],
        prefix2_values=tensors["prefix2_values"],
        metadata=ConstraintTableMetadata(
            format=metadata["format"],
            artifact_digest=actual_digest,
            tokenizer_digest=metadata["tokenizer_digest"],
            vocab_size=vocab_size,
            max_prefix1_degree=_metadata_int(metadata, "max_prefix1_degree", minimum=0),
            max_prefix2_degree=_metadata_int(metadata, "max_prefix2_degree", minimum=0),
            total_bytes=total_bytes,
        ),
    )
    del data, tensors
    validate_constraint_table(table)
    return table


def serialize_constraint_table_artifact(table: ConstraintTable) -> bytes:
    """Serialize a validated table without adding self-referential metadata."""

    validate_constraint_table(table)
    metadata = {
        "format": table.metadata.format,
        "tokenizer_digest": table.metadata.tokenizer_digest,
        "vocab_size": str(table.metadata.vocab_size),
        "max_prefix1_degree": str(table.metadata.max_prefix1_degree),
        "max_prefix2_degree": str(table.metadata.max_prefix2_degree),
    }
    tensors: dict[str, Any] = {
        name: np.ascontiguousarray(getattr(table, name)) for name in CONSTRAINT_TABLE_TENSOR_NAMES
    }
    # safetensors.numpy.save currently routes metadata through a hash map, so
    # repeated calls can emit a different JSON-key order and therefore a
    # different whole-file digest. Build the small safetensors header directly
    # in ABI order; the payload remains standard safetensors and is parsed by
    # the official loader. Header length is padded to an 8-byte boundary as
    # required by the format.
    dtype_names = {
        np.dtype(np.int32): "I32",
        np.dtype(np.int64): "I64",
    }
    header: dict[str, object] = {"__metadata__": metadata}
    payload_parts: list[bytes] = []
    offset = 0
    for name in CONSTRAINT_TABLE_TENSOR_NAMES:
        tensor = tensors[name]
        data = tensor.tobytes(order="C")
        header[name] = {
            "dtype": dtype_names[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        payload_parts.append(data)
        offset += len(data)
    header_bytes = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    return (
        len(header_bytes).to_bytes(8, "little", signed=False)
        + header_bytes
        + b"".join(payload_parts)
    )


def save_constraint_table_artifact(
    table: ConstraintTable,
    path: str | Path,
) -> str:
    """Atomically save an artifact and return its whole-file SHA-256 digest."""

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
                raise OSError(f"short artifact write: expected {len(data)}, wrote {written}")
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
        raise ValueError(f"cannot save constraint table artifact {str(output)!r}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return _sha256_bytes(data)
