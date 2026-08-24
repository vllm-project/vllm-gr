# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Immutable Host representation and validation for the six-table ABI."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from vllm_gr.config import CONSTRAINT_TABLE_FORMAT

TABLE_NAMES = (
    "first_token_ids",
    "prefix1_offsets",
    "prefix1_values",
    "prefix1_pair_keys",
    "prefix2_value_offsets",
    "prefix2_values",
)
MAX_TABLE_BYTES = 512 * 1024 * 1024
VALIDATION_CHUNK_ELEMENTS = 1024 * 1024
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ConstraintTableInfo:
    format: str
    artifact_digest: str | None
    tokenizer_digest: str
    vocab_size: int
    max_prefix1_degree: int
    max_prefix2_degree: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class ConstraintTable:
    """Six CPU-resident NumPy tables and their small immutable metadata."""

    first_token_ids: np.ndarray
    prefix1_offsets: np.ndarray
    prefix1_values: np.ndarray
    prefix1_pair_keys: np.ndarray
    prefix2_value_offsets: np.ndarray
    prefix2_values: np.ndarray
    info: ConstraintTableInfo

    def __post_init__(self) -> None:
        for name in TABLE_NAMES:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise ValueError(f"constraint table {name} must be a numpy.ndarray")
            frozen = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            object.__setattr__(self, name, frozen)

    @property
    def total_elements(self) -> int:
        return sum(int(getattr(self, name).size) for name in TABLE_NAMES)

    @property
    def total_bytes(self) -> int:
        return sum(int(getattr(self, name).nbytes) for name in TABLE_NAMES)


def _validate_integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"constraint table {name} must be a {qualifier} integer")
    return value


def validate_constraint_table_info(
    info: ConstraintTableInfo,
    *,
    require_artifact_digest: bool = False,
) -> None:
    if not isinstance(info, ConstraintTableInfo):
        raise ValueError("constraint table info must be ConstraintTableInfo")
    if info.format != CONSTRAINT_TABLE_FORMAT:
        raise ValueError(
            f"constraint table format must be {CONSTRAINT_TABLE_FORMAT!r}, got {info.format!r}"
        )
    _validate_integer(info.vocab_size, "vocab_size", minimum=1)
    _validate_integer(info.max_prefix1_degree, "max_prefix1_degree", minimum=0)
    _validate_integer(info.max_prefix2_degree, "max_prefix2_degree", minimum=0)
    _validate_integer(info.total_bytes, "total_bytes", minimum=0)
    if info.artifact_digest is None:
        if require_artifact_digest:
            raise ValueError("constraint Backend requires a persisted Artifact digest")
    elif (
        not isinstance(info.artifact_digest, str)
        or _SHA256_RE.fullmatch(info.artifact_digest) is None
    ):
        raise ValueError("constraint table artifact_digest must be a SHA-256 digest or None")
    if (
        not isinstance(info.tokenizer_digest, str)
        or _SHA256_RE.fullmatch(info.tokenizer_digest) is None
    ):
        raise ValueError("constraint table tokenizer_digest must be a SHA-256 digest")


def _offset_degree_summary(
    offsets: np.ndarray,
    name: str,
    *,
    require_positive: bool,
) -> int:
    max_degree = 0
    for start in range(0, max(0, offsets.size - 1), VALIDATION_CHUNK_ELEMENTS):
        end = min(offsets.size - 1, start + VALIDATION_CHUNK_ELEMENTS)
        degrees = np.subtract(
            offsets[start + 1 : end + 1],
            offsets[start:end],
            dtype=np.int64,
        )
        if np.any(degrees < 0):
            raise ValueError(f"{name} must be non-decreasing")
        if require_positive and np.any(degrees <= 0):
            raise ValueError("every prefix1 pair must have at least one prefix2 value")
        if degrees.size:
            max_degree = max(max_degree, int(degrees.max()))
    return max_degree


def derive_degrees(
    prefix1_offsets: np.ndarray,
    prefix2_value_offsets: np.ndarray,
) -> tuple[int, int]:
    return (
        _offset_degree_summary(
            prefix1_offsets,
            "prefix1_offsets",
            require_positive=False,
        ),
        _offset_degree_summary(
            prefix2_value_offsets,
            "prefix2_value_offsets",
            require_positive=False,
        ),
    )


def _is_strictly_increasing(values: np.ndarray) -> bool:
    for start in range(1, values.size, VALIDATION_CHUNK_ELEMENTS):
        end = min(values.size, start + VALIDATION_CHUNK_ELEMENTS)
        if np.any(values[start:end] <= values[start - 1 : end - 1]):
            return False
    return True


def _active_tokens_match(first: np.ndarray, offsets: np.ndarray) -> bool:
    cursor = 0
    vocab_size = offsets.size - 1
    for start in range(0, vocab_size, VALIDATION_CHUNK_ELEMENTS):
        end = min(vocab_size, start + VALIDATION_CHUNK_ELEMENTS)
        degrees = np.subtract(
            offsets[start + 1 : end + 1],
            offsets[start:end],
            dtype=np.int64,
        )
        active = np.flatnonzero(degrees > 0).astype(np.int64, copy=False) + start
        count = active.size
        if cursor + count > first.size or not np.array_equal(
            first[cursor : cursor + count], active
        ):
            return False
        cursor += count
    return cursor == first.size


def _pair_keys_match(
    first: np.ndarray,
    offsets: np.ndarray,
    values: np.ndarray,
    pair_keys: np.ndarray,
    vocab_size: int,
) -> bool:
    for token_value in first:
        token = int(token_value)
        begin = int(offsets[token])
        end = int(offsets[token + 1])
        base = token * vocab_size
        for start in range(begin, end, VALIDATION_CHUNK_ELEMENTS):
            stop = min(end, start + VALIDATION_CHUNK_ELEMENTS)
            expected = values[start:stop].astype(np.int64)
            expected += base
            if not np.array_equal(expected, pair_keys[start:stop]):
                return False
    return True


def _values_increase_within_buckets(values: np.ndarray, offsets: np.ndarray) -> bool:
    for start in range(1, values.size, VALIDATION_CHUNK_ELEMENTS):
        end = min(values.size, start + VALIDATION_CHUNK_ELEMENTS)
        invalid = values[start:end] <= values[start - 1 : end - 1]
        boundary_start = int(np.searchsorted(offsets, start, side="left"))
        boundary_end = int(np.searchsorted(offsets, end, side="left"))
        boundaries = offsets[boundary_start:boundary_end]
        if boundaries.size:
            invalid[boundaries - start] = False
        if np.any(invalid):
            return False
    return True


def validate_constraint_table(table: ConstraintTable) -> None:
    """Validate dtype, shape, CSR, semantic relations, ranges, and size."""

    info = table.info
    validate_constraint_table_info(info)

    expected_dtypes = {
        "first_token_ids": np.dtype(np.int32),
        "prefix1_offsets": np.dtype(np.int32),
        "prefix1_values": np.dtype(np.int32),
        "prefix1_pair_keys": np.dtype(np.int64),
        "prefix2_value_offsets": np.dtype(np.int32),
        "prefix2_values": np.dtype(np.int32),
    }
    for name, expected in expected_dtypes.items():
        array = getattr(table, name)
        if array.ndim != 1:
            raise ValueError(f"table {name} must be 1-D (NPU ABI), got shape {tuple(array.shape)}")
        if array.dtype != expected:
            raise ValueError(f"table {name}.dtype must be {expected}, got {array.dtype}")

    actual_bytes = table.total_bytes
    if actual_bytes != info.total_bytes:
        raise ValueError(
            f"constraint table total_bytes metadata {info.total_bytes} != actual {actual_bytes}"
        )
    if actual_bytes > MAX_TABLE_BYTES:
        raise ValueError(
            f"constraint table total_bytes {actual_bytes} exceeds {MAX_TABLE_BYTES} byte limit"
        )

    vocab_size = info.vocab_size
    offsets1 = table.prefix1_offsets
    values1 = table.prefix1_values
    pair_keys = table.prefix1_pair_keys
    offsets2 = table.prefix2_value_offsets
    values2 = table.prefix2_values

    if offsets1.shape != (vocab_size + 1,):
        raise ValueError(
            "prefix1_offsets must have shape (vocab_size+1,), "
            f"got {offsets1.shape} for vocab_size={vocab_size}"
        )
    if int(offsets1[0]) != 0 or int(offsets1[-1]) != len(values1):
        raise ValueError("prefix1_offsets must start at zero and end at prefix1_values size")

    if offsets2.shape != (len(values1) + 1,):
        raise ValueError("prefix2_value_offsets must have one entry per prefix1 value plus the end")
    if int(offsets2[0]) != 0 or int(offsets2[-1]) != len(values2):
        raise ValueError("prefix2_value_offsets must start at zero and end at prefix2_values size")

    if len(pair_keys) != len(values1):
        raise ValueError("prefix1_pair_keys length must equal prefix1_values length")
    if len(pair_keys) > 1 and not _is_strictly_increasing(pair_keys):
        raise ValueError("prefix1_pair_keys must be strictly increasing")

    derived1 = _offset_degree_summary(
        offsets1,
        "prefix1_offsets",
        require_positive=False,
    )
    derived2 = _offset_degree_summary(
        offsets2,
        "prefix2_value_offsets",
        require_positive=True,
    )
    if info.max_prefix1_degree != derived1:
        raise ValueError(
            f"max_prefix1_degree {info.max_prefix1_degree} does not match derived {derived1}"
        )
    if info.max_prefix2_degree != derived2:
        raise ValueError(
            f"max_prefix2_degree {info.max_prefix2_degree} does not match derived {derived2}"
        )

    for name, values in (
        ("first_token_ids", table.first_token_ids),
        ("prefix1_values", values1),
        ("prefix2_values", values2),
    ):
        if values.size and (int(values.min()) < 0 or int(values.max()) >= vocab_size):
            raise ValueError(f"{name} contains an out-of-range token ID")

    first = table.first_token_ids
    if first.size == 0:
        raise ValueError("first_token_ids must be non-empty")
    if first.size and not _is_strictly_increasing(first):
        raise ValueError("first_token_ids must be strictly increasing and unique")
    if not _active_tokens_match(first, offsets1):
        raise ValueError(
            "first_token_ids must exactly match prefix1_offsets tokens with positive degree"
        )

    if values1.size and not _pair_keys_match(
        first,
        offsets1,
        values1,
        pair_keys,
        vocab_size,
    ):
        raise ValueError(
            "prefix1_pair_keys must equal a_id * vocab_size + b_id for each CSR bucket"
        )

    if values2.size > 1 and not _values_increase_within_buckets(values2, offsets2):
        raise ValueError("prefix2_values must be strictly increasing and unique within each pair")
