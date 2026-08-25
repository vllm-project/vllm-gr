# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Immutable host representation of the canonical six-table ABI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from vllm_gr.config.models import CONSTRAINT_TABLE_FORMAT

CONSTRAINT_TABLE_TENSOR_NAMES = (
    "first_token_ids",
    "prefix1_offsets",
    "prefix1_values",
    "prefix1_pair_keys",
    "prefix2_value_offsets",
    "prefix2_values",
)
MAX_CONSTRAINT_TABLE_BYTES = 512 * 1024 * 1024
_VALIDATION_CHUNK_SIZE = 1024 * 1024
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
IntArray: TypeAlias = NDArray[np.signedinteger[Any]]


@dataclass(frozen=True, slots=True)
class ConstraintTableMetadata:
    """Small metadata values required to validate and identify a table."""

    format: str
    artifact_digest: str | None
    tokenizer_digest: str
    vocab_size: int
    max_prefix1_degree: int
    max_prefix2_degree: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class ConstraintTable:
    """Six CPU-resident NumPy tensors and their immutable metadata."""

    first_token_ids: NDArray[np.int32]
    prefix1_offsets: NDArray[np.int32]
    prefix1_values: NDArray[np.int32]
    prefix1_pair_keys: NDArray[np.int64]
    prefix2_value_offsets: NDArray[np.int32]
    prefix2_values: NDArray[np.int32]
    metadata: ConstraintTableMetadata

    def __post_init__(self) -> None:
        # A read-only flag alone can be re-enabled when an ndarray owns its
        # memory. Rebuilding over immutable bytes makes the returned tensors
        # permanently non-writable, matching the legacy resource guarantee.
        for name in CONSTRAINT_TABLE_TENSOR_NAMES:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise ValueError(f"constraint table tensor {name} must be a numpy.ndarray")
            frozen = np.frombuffer(
                np.ascontiguousarray(value).tobytes(), dtype=value.dtype
            ).reshape(value.shape)
            object.__setattr__(self, name, frozen)

    @property
    def total_elements(self) -> int:
        return sum(int(getattr(self, name).size) for name in CONSTRAINT_TABLE_TENSOR_NAMES)

    @property
    def total_bytes(self) -> int:
        return sum(int(getattr(self, name).nbytes) for name in CONSTRAINT_TABLE_TENSOR_NAMES)


def _validate_int(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"constraint table {name} must be a {qualifier} integer")
    return value


def validate_constraint_table_metadata(
    metadata: ConstraintTableMetadata,
    *,
    require_artifact_digest: bool = False,
) -> None:
    if not isinstance(metadata, ConstraintTableMetadata):
        raise ValueError("constraint table metadata must be ConstraintTableMetadata")
    if metadata.format != CONSTRAINT_TABLE_FORMAT:
        raise ValueError(
            f"constraint table format must be {CONSTRAINT_TABLE_FORMAT!r}, got {metadata.format!r}"
        )
    _validate_int(metadata.vocab_size, "vocab_size", minimum=1)
    _validate_int(metadata.max_prefix1_degree, "max_prefix1_degree", minimum=0)
    _validate_int(metadata.max_prefix2_degree, "max_prefix2_degree", minimum=0)
    _validate_int(metadata.total_bytes, "total_bytes", minimum=0)
    if metadata.artifact_digest is None:
        if require_artifact_digest:
            raise ValueError("persisted constraint table requires an artifact_digest")
    elif (
        not isinstance(metadata.artifact_digest, str)
        or _SHA256_RE.fullmatch(metadata.artifact_digest) is None
    ):
        raise ValueError("constraint table artifact_digest must be a SHA-256 digest or None")
    if (
        not isinstance(metadata.tokenizer_digest, str)
        or _SHA256_RE.fullmatch(metadata.tokenizer_digest) is None
    ):
        raise ValueError("constraint table tokenizer_digest must be a SHA-256 digest")


def _validate_offsets(
    offsets: NDArray[np.int32],
    name: str,
    *,
    require_positive: bool,
) -> int:
    max_degree = 0
    pair_count = max(0, offsets.size - 1)
    for start in range(0, pair_count, _VALIDATION_CHUNK_SIZE):
        stop = min(pair_count, start + _VALIDATION_CHUNK_SIZE)
        degrees = np.subtract(
            offsets[start + 1 : stop + 1],
            offsets[start:stop],
            dtype=np.int64,
        )
        if np.any(degrees < 0):
            raise ValueError(f"{name} must be non-decreasing")
        if require_positive and np.any(degrees <= 0):
            raise ValueError("every prefix1 pair must have at least one prefix2 value")
        if degrees.size:
            max_degree = max(max_degree, int(degrees.max()))
    return max_degree


def derive_constraint_table_degrees(
    prefix1_offsets: NDArray[np.int32],
    prefix2_value_offsets: NDArray[np.int32],
) -> tuple[int, int]:
    return (
        _validate_offsets(
            prefix1_offsets,
            "prefix1_offsets",
            require_positive=False,
        ),
        _validate_offsets(
            prefix2_value_offsets,
            "prefix2_value_offsets",
            require_positive=False,
        ),
    )


def _is_strictly_increasing(values: IntArray) -> bool:
    for start in range(1, values.size, _VALIDATION_CHUNK_SIZE):
        stop = min(values.size, start + _VALIDATION_CHUNK_SIZE)
        if np.any(values[start:stop] <= values[start - 1 : stop - 1]):
            return False
    return True


def _values_increase_within_buckets(
    values: NDArray[np.int32],
    offsets: NDArray[np.int32],
) -> bool:
    for start in range(1, values.size, _VALIDATION_CHUNK_SIZE):
        stop = min(values.size, start + _VALIDATION_CHUNK_SIZE)
        invalid = values[start:stop] <= values[start - 1 : stop - 1]
        boundary_start = int(np.searchsorted(offsets, start, side="left"))
        boundary_stop = int(np.searchsorted(offsets, stop, side="left"))
        boundaries = offsets[boundary_start:boundary_stop]
        if boundaries.size:
            invalid[boundaries - start] = False
        if np.any(invalid):
            return False
    return True


def _active_tokens_match(first: NDArray[np.int32], offsets: NDArray[np.int32]) -> bool:
    cursor = 0
    vocab_size = offsets.size - 1
    for start in range(0, vocab_size, _VALIDATION_CHUNK_SIZE):
        stop = min(vocab_size, start + _VALIDATION_CHUNK_SIZE)
        degrees = np.subtract(
            offsets[start + 1 : stop + 1],
            offsets[start:stop],
            dtype=np.int64,
        )
        active = np.flatnonzero(degrees > 0).astype(np.int64, copy=False)
        active += start
        count = int(active.size)
        if cursor + count > first.size or not np.array_equal(
            first[cursor : cursor + count], active
        ):
            return False
        cursor += count
    return cursor == first.size


def _pair_keys_match(
    first: NDArray[np.int32],
    offsets: NDArray[np.int32],
    values: NDArray[np.int32],
    pair_keys: NDArray[np.int64],
    vocab_size: int,
) -> bool:
    for token_value in first:
        token_id = int(token_value)
        begin = int(offsets[token_id])
        end = int(offsets[token_id + 1])
        base = token_id * vocab_size
        for start in range(begin, end, _VALIDATION_CHUNK_SIZE):
            stop = min(end, start + _VALIDATION_CHUNK_SIZE)
            expected = values[start:stop].astype(np.int64)
            expected += base
            if not np.array_equal(expected, pair_keys[start:stop]):
                return False
    return True


def validate_constraint_table(table: ConstraintTable) -> None:
    """Validate shape, dtype, CSR structure, semantics, metadata, and size."""

    if not isinstance(table, ConstraintTable):
        raise ValueError("constraint table must be a ConstraintTable")
    metadata = table.metadata
    validate_constraint_table_metadata(metadata)

    expected_dtypes = {
        "first_token_ids": np.dtype(np.int32),
        "prefix1_offsets": np.dtype(np.int32),
        "prefix1_values": np.dtype(np.int32),
        "prefix1_pair_keys": np.dtype(np.int64),
        "prefix2_value_offsets": np.dtype(np.int32),
        "prefix2_values": np.dtype(np.int32),
    }
    for name, expected_dtype in expected_dtypes.items():
        tensor = getattr(table, name)
        if tensor.ndim != 1:
            raise ValueError(
                f"constraint table tensor {name} must be 1-D, got shape {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"constraint table tensor {name}.dtype must be {expected_dtype}, got {tensor.dtype}"
            )

    if table.total_bytes != metadata.total_bytes:
        raise ValueError(
            f"constraint table total_bytes metadata {metadata.total_bytes} "
            f"does not match actual {table.total_bytes}"
        )
    if table.total_bytes > MAX_CONSTRAINT_TABLE_BYTES:
        raise ValueError(
            f"constraint table total_bytes {table.total_bytes} exceeds "
            f"the {MAX_CONSTRAINT_TABLE_BYTES} byte limit"
        )

    vocab_size = metadata.vocab_size
    first = table.first_token_ids
    offsets1 = table.prefix1_offsets
    values1 = table.prefix1_values
    pair_keys = table.prefix1_pair_keys
    offsets2 = table.prefix2_value_offsets
    values2 = table.prefix2_values

    if offsets1.shape != (vocab_size + 1,):
        raise ValueError(f"prefix1_offsets must have shape (vocab_size + 1,), got {offsets1.shape}")
    if int(offsets1[0]) != 0 or int(offsets1[-1]) != len(values1):
        raise ValueError("prefix1_offsets must start at zero and end at prefix1_values size")
    if offsets2.shape != (len(values1) + 1,):
        raise ValueError("prefix2_value_offsets must have one entry per prefix1 value plus the end")
    if int(offsets2[0]) != 0 or int(offsets2[-1]) != len(values2):
        raise ValueError("prefix2_value_offsets must start at zero and end at prefix2_values size")
    if len(pair_keys) != len(values1):
        raise ValueError("prefix1_pair_keys length must equal prefix1_values length")

    derived1 = _validate_offsets(offsets1, "prefix1_offsets", require_positive=False)
    derived2 = _validate_offsets(offsets2, "prefix2_value_offsets", require_positive=True)
    if metadata.max_prefix1_degree != derived1:
        raise ValueError(
            f"max_prefix1_degree {metadata.max_prefix1_degree} does not "
            f"match derived degree {derived1}"
        )
    if metadata.max_prefix2_degree != derived2:
        raise ValueError(
            f"max_prefix2_degree {metadata.max_prefix2_degree} does not "
            f"match derived degree {derived2}"
        )

    for name, values in (
        ("first_token_ids", first),
        ("prefix1_values", values1),
        ("prefix2_values", values2),
    ):
        if values.size and (int(values.min()) < 0 or int(values.max()) >= vocab_size):
            raise ValueError(f"{name} contains an out-of-range token ID")

    if first.size == 0:
        raise ValueError("first_token_ids must be non-empty")
    if not _is_strictly_increasing(first):
        raise ValueError("first_token_ids must be strictly increasing")
    if not _active_tokens_match(first, offsets1):
        raise ValueError(
            "first_token_ids must exactly match prefix1_offsets tokens with positive degree"
        )
    if values1.size and not _values_increase_within_buckets(values1, offsets1):
        raise ValueError("prefix1_values must be strictly increasing within each bucket")
    if pair_keys.size > 1 and not _is_strictly_increasing(pair_keys):
        raise ValueError("prefix1_pair_keys must be strictly increasing")
    if values1.size and not _pair_keys_match(first, offsets1, values1, pair_keys, vocab_size):
        raise ValueError(
            "prefix1_pair_keys must equal a_id * vocab_size + b_id for each CSR bucket"
        )
    if values2.size and not _values_increase_within_buckets(values2, offsets2):
        raise ValueError("prefix2_values must be strictly increasing within each bucket")
