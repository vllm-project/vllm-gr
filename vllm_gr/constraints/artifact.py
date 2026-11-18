# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Safetensors persistence for the canonical constraint-table artifact."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
from numpy.typing import NDArray

from vllm_gr.config.models import (
    CONSTRAINT_TABLE_FORMAT,
    MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
)
from vllm_gr.config.validation import (
    validate_artifact_digest,
    validate_constraint_table_format,
    validate_tokenizer_digest,
)
from vllm_gr.constraints.table import (
    MAX_CONSTRAINT_TABLE_BYTES,
    ConstraintTable,
    ConstraintTableMetadata,
    constraint_table_tensor_names,
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
_TENSOR_DTYPES: dict[str, tuple[str, np.dtype[Any]]] = {
    "first_token_offsets": ("I32", np.dtype(np.int32)),
    "first_token_ids": ("I32", np.dtype(np.int32)),
    "prefix1_offsets": ("I32", np.dtype(np.int32)),
    "prefix1_values": ("I32", np.dtype(np.int32)),
    "prefix1_pair_keys": ("I64", np.dtype(np.int64)),
    "prefix2_value_offsets": ("I32", np.dtype(np.int32)),
    "prefix2_values": ("I32", np.dtype(np.int32)),
}


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


class _DuplicateHeaderKeyError(ValueError):
    pass


def _reject_duplicate_header_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateHeaderKeyError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _load_artifact_header(data: bytes) -> tuple[dict[str, str], dict[str, object], int]:
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
        header = json.loads(
            data[8 : 8 + header_size].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_header_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateHeaderKeyError) as exc:
        raise ValueError(f"invalid safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError("constraint table artifact must contain string safetensors metadata")
    expected_keys = _ARTIFACT_METADATA_KEYS
    if metadata.get("format") == MULTI_TRIE_CONSTRAINT_TABLE_FORMAT:
        expected_keys = expected_keys | {"trie_count"}
    missing = sorted(expected_keys - set(metadata))
    unknown = sorted(set(metadata) - expected_keys)
    if missing or unknown:
        raise ValueError(
            f"constraint table artifact metadata mismatch: missing={missing}, unknown={unknown}"
        )
    return cast(dict[str, str], metadata), header, 8 + header_size


def _load_immutable_tensors(
    data: bytes,
    header: dict[str, object],
    payload_offset: int,
) -> tuple[dict[str, NDArray[Any]], int]:
    """Validate tensor descriptors and return zero-copy views over ``data``."""

    tensor_names = set(header) - {"__metadata__"}
    names = constraint_table_tensor_names(cast(dict[str, str], header["__metadata__"])["format"])
    missing_tensors = sorted(set(names) - tensor_names)
    extra_tensors = sorted(tensor_names - set(names))
    if missing_tensors or extra_tensors:
        raise ValueError(
            "constraint table artifact tensor mismatch: "
            f"missing={missing_tensors}, extra={extra_tensors}"
        )

    payload_size = len(data) - payload_offset
    descriptors: dict[str, tuple[np.dtype[Any], int, int, int]] = {}
    intervals: list[tuple[int, int, str]] = []
    total_bytes = 0
    for name in names:
        descriptor = header[name]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise ValueError(f"invalid safetensors descriptor for tensor {name!r}")

        dtype_name = descriptor["dtype"]
        expected_dtype_name, dtype = _TENSOR_DTYPES[name]
        if dtype_name != expected_dtype_name:
            raise ValueError(
                f"constraint table tensor {name}.dtype must be {dtype}, got {dtype_name!r}"
            )

        shape = descriptor["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 1
            or type(shape[0]) is not int
            or shape[0] < 0
        ):
            raise ValueError(
                f"constraint table tensor {name} must be 1-D with a non-negative shape"
            )
        element_count = shape[0]

        offsets = descriptor["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int or value < 0 for value in offsets)
        ):
            raise ValueError(f"invalid data_offsets for tensor {name!r}")
        start, end = offsets
        if start > end or end > payload_size:
            raise ValueError(f"data_offsets for tensor {name!r} are outside the payload")
        expected_bytes = element_count * dtype.itemsize
        if end - start != expected_bytes:
            raise ValueError(
                f"tensor {name!r} byte length {end - start} does not match "
                f"shape {shape} and dtype {dtype_name}"
            )
        descriptors[name] = (dtype, element_count, start, end)
        intervals.append((start, end, name))
        total_bytes += expected_bytes

    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise ValueError(
                f"safetensors payload must be contiguous and non-overlapping near tensor {name!r}"
            )
        cursor = end
    if cursor != payload_size:
        raise ValueError(
            f"safetensors payload contains {payload_size - cursor} unclaimed trailing bytes"
        )
    if total_bytes > MAX_CONSTRAINT_TABLE_BYTES:
        raise ValueError(
            f"constraint table total_bytes {total_bytes} exceeds "
            f"the {MAX_CONSTRAINT_TABLE_BYTES} byte limit"
        )

    tensors: dict[str, NDArray[Any]] = {}
    for name in names:
        dtype, element_count, start, _ = descriptors[name]
        tensors[name] = np.frombuffer(
            data,
            dtype=dtype,
            count=element_count,
            offset=payload_offset + start,
        )
    return tensors, total_bytes


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

    metadata, header, payload_offset = _load_artifact_header(data)
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

    tensors, total_bytes = _load_immutable_tensors(data, header, payload_offset)
    table = ConstraintTable._from_immutable_backing(
        backing=data,
        first_token_ids=tensors["first_token_ids"],
        prefix1_offsets=tensors["prefix1_offsets"],
        prefix1_values=tensors["prefix1_values"],
        prefix1_pair_keys=tensors["prefix1_pair_keys"],
        prefix2_value_offsets=tensors["prefix2_value_offsets"],
        prefix2_values=tensors["prefix2_values"],
        first_token_offsets=tensors.get("first_token_offsets"),
        metadata=ConstraintTableMetadata(
            format=metadata["format"],
            artifact_digest=actual_digest,
            tokenizer_digest=metadata["tokenizer_digest"],
            vocab_size=vocab_size,
            max_prefix1_degree=_metadata_int(metadata, "max_prefix1_degree", minimum=0),
            max_prefix2_degree=_metadata_int(metadata, "max_prefix2_degree", minimum=0),
            total_bytes=total_bytes,
            trie_count=(
                _metadata_int(metadata, "trie_count", minimum=1)
                if metadata["format"] == MULTI_TRIE_CONSTRAINT_TABLE_FORMAT
                else 1
            ),
        ),
    )
    validate_constraint_table(table)
    return table


def _build_artifact_header(table: ConstraintTable) -> bytes:
    metadata = {
        "format": table.metadata.format,
        "tokenizer_digest": table.metadata.tokenizer_digest,
        "vocab_size": str(table.metadata.vocab_size),
        "max_prefix1_degree": str(table.metadata.max_prefix1_degree),
        "max_prefix2_degree": str(table.metadata.max_prefix2_degree),
    }
    if table.metadata.format == MULTI_TRIE_CONSTRAINT_TABLE_FORMAT:
        metadata["trie_count"] = str(table.metadata.trie_count)
    # safetensors.numpy.save currently routes metadata through a hash map, so
    # repeated calls can emit a different JSON-key order and therefore a
    # different whole-file digest. Build the small safetensors header directly
    # in ABI order. The payload remains standard safetensors; header length is
    # padded to an 8-byte boundary as required by the format.
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for name in table.tensor_names:
        tensor = getattr(table, name)
        dtype_name, expected_dtype = _TENSOR_DTYPES[name]
        if tensor.dtype != expected_dtype or not tensor.flags.c_contiguous:
            raise ValueError(f"constraint table tensor {name} is not ABI-contiguous")
        header[name] = {
            "dtype": dtype_name,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes
    header_bytes = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    if len(header_bytes) > _MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(
            f"safetensors header size {len(header_bytes)} exceeds "
            f"the {_MAX_SAFETENSORS_HEADER_BYTES} byte limit"
        )
    return header_bytes


def _artifact_chunks(table: ConstraintTable, header_bytes: bytes) -> Iterator[bytes | memoryview]:
    yield len(header_bytes).to_bytes(8, "little", signed=False)
    yield header_bytes
    for name in table.tensor_names:
        yield memoryview(getattr(table, name)).cast("B")


def serialize_constraint_table_artifact(table: ConstraintTable) -> bytes:
    """Serialize a validated table with no per-tensor payload copies."""

    validate_constraint_table(table)
    header_bytes = _build_artifact_header(table)
    output = io.BytesIO()
    for chunk in _artifact_chunks(table, header_bytes):
        output.write(chunk)
    return output.getvalue()


def _write_all(output: BinaryIO, data: bytes | memoryview) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = output.write(view[written:])
        if count is None or count <= 0:
            raise OSError(f"short artifact write after {written} of {len(view)} bytes")
        written += count


def save_constraint_table_artifact(
    table: ConstraintTable,
    path: str | Path,
) -> str:
    """Atomically save an artifact and return its whole-file SHA-256 digest."""

    validate_constraint_table(table)
    header_bytes = _build_artifact_header(table)
    output = Path(path)
    temporary_path: Path | None = None
    hasher = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for chunk in _artifact_chunks(table, header_bytes):
                _write_all(cast(BinaryIO, temporary_file), chunk)
                hasher.update(chunk)
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
    return "sha256:" + hasher.hexdigest()
