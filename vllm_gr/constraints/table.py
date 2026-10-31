# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Immutable canonical tables: six persisted tensors in v1, seven in v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import regex as re
from numpy.typing import NDArray

from vllm_gr.config.models import (
    CONSTRAINT_TABLE_FORMAT,
    MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
)

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
    trie_count: int = 1


@dataclass(frozen=True, slots=True)
class ConstraintTable:
    """CPU tables with normalized root boundaries, including implicit v1 root 0."""

    first_token_ids: NDArray[np.int32]
    prefix1_offsets: NDArray[np.int32]
    prefix1_values: NDArray[np.int32]
    prefix1_pair_keys: NDArray[np.int64]
    prefix2_value_offsets: NDArray[np.int32]
    prefix2_values: NDArray[np.int32]
    metadata: ConstraintTableMetadata
    first_token_offsets: NDArray[np.int32] | None = None

    @classmethod
    def _from_immutable_backing(
        cls,
        *,
        backing: bytes,
        first_token_ids: NDArray[np.int32],
        prefix1_offsets: NDArray[np.int32],
        prefix1_values: NDArray[np.int32],
        prefix1_pair_keys: NDArray[np.int64],
        prefix2_value_offsets: NDArray[np.int32],
        prefix2_values: NDArray[np.int32],
        metadata: ConstraintTableMetadata,
        first_token_offsets: NDArray[np.int32] | None = None,
    ) -> ConstraintTable:
        """Build loader-owned views over one proven-immutable bytes buffer.

        The normal constructor deliberately copies caller-owned arrays so they
        cannot be mutated behind the frozen dataclass.  The artifact loader has
        a stronger guarantee: every array is a view over the same immutable
        ``bytes`` object.  Re-copying those views would only multiply startup
        memory, so this private constructor verifies that guarantee and then
        installs the views without running :meth:`__post_init__`.
        """

        values = {
            "first_token_ids": first_token_ids,
            "prefix1_offsets": prefix1_offsets,
            "prefix1_values": prefix1_values,
            "prefix1_pair_keys": prefix1_pair_keys,
            "prefix2_value_offsets": prefix2_value_offsets,
            "prefix2_values": prefix2_values,
        }
        if first_token_offsets is not None:
            values["first_token_offsets"] = first_token_offsets
        table = object.__new__(cls)
        for name, value in values.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(f"constraint table tensor {name} must be a numpy.ndarray")
            if value.flags.writeable:
                raise ValueError(
                    f"constraint table tensor {name} must be read-only when "
                    "constructed from immutable backing"
                )

            owner: object = value
            seen: set[int] = set()
            while id(owner) not in seen:
                seen.add(id(owner))
                if isinstance(owner, np.ndarray) and owner.base is not None:
                    owner = owner.base
                elif isinstance(owner, memoryview):
                    owner = owner.obj
                else:
                    break
            if owner is not backing:
                raise ValueError(
                    f"constraint table tensor {name} does not reference the "
                    "loader's immutable backing"
                )
            object.__setattr__(table, name, value)
        object.__setattr__(table, "metadata", metadata)
        if first_token_offsets is None:
            object.__setattr__(table, "first_token_offsets", _single_root_offsets(first_token_ids))
        return table

    def __post_init__(self) -> None:
        # A read-only flag alone can be re-enabled when an ndarray owns its
        # memory. Rebuilding over immutable bytes makes the returned tensors
        # permanently non-writable, matching the legacy resource guarantee.
        if self.first_token_offsets is None:
            object.__setattr__(
                self, "first_token_offsets", _single_root_offsets(self.first_token_ids)
            )
        for name in (*CONSTRAINT_TABLE_TENSOR_NAMES, "first_token_offsets"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise ValueError(f"constraint table tensor {name} must be a numpy.ndarray")
            frozen = np.frombuffer(
                np.ascontiguousarray(value).tobytes(), dtype=value.dtype
            ).reshape(value.shape)
            object.__setattr__(self, name, frozen)

    @property
    def tensor_names(self) -> tuple[str, ...]:
        """Persisted tensors; legacy v1 omits the implicit single-root boundaries."""
        return constraint_table_tensor_names(self.metadata.format)

    @property
    def root_offsets(self) -> NDArray[np.int32]:
        assert self.first_token_offsets is not None
        return self.first_token_offsets

    @property
    def max_first_degree(self) -> int:
        return int(np.diff(self.root_offsets.astype(np.int64)).max(initial=0))

    @property
    def total_elements(self) -> int:
        return sum(int(getattr(self, name).size) for name in self.tensor_names)

    @property
    def total_bytes(self) -> int:
        return sum(int(getattr(self, name).nbytes) for name in self.tensor_names)


def constraint_table_tensor_names(format: str) -> tuple[str, ...]:
    return CONSTRAINT_TABLE_TENSOR_NAMES + (
        ("first_token_offsets",) if format == MULTI_TRIE_CONSTRAINT_TABLE_FORMAT else ()
    )


def _single_root_offsets(first: NDArray[np.int32]) -> NDArray[np.int32]:
    return np.frombuffer(np.asarray([0, first.size], dtype=np.int32).tobytes(), dtype=np.int32)


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
    if metadata.format not in (CONSTRAINT_TABLE_FORMAT, MULTI_TRIE_CONSTRAINT_TABLE_FORMAT):
        raise ValueError(
            f"constraint table format must be {CONSTRAINT_TABLE_FORMAT!r}, got {metadata.format!r}"
        )
    _validate_int(metadata.vocab_size, "vocab_size", minimum=1)
    _validate_int(metadata.trie_count, "trie_count", minimum=1)
    if metadata.format == CONSTRAINT_TABLE_FORMAT and metadata.trie_count != 1:
        raise ValueError("constraint_table_v1 requires trie_count=1")
    if metadata.trie_count * metadata.vocab_size**2 - 1 > np.iinfo(np.int64).max:
        raise ValueError("constraint table root/SID keys exceed int64 capacity")
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
    trie_id: int = 0,
) -> bool:
    for token_value in first:
        token_id = trie_id * vocab_size + int(token_value)
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
        "first_token_offsets": np.dtype(np.int32),
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
    root_offsets = table.root_offsets
    if root_offsets.shape != (metadata.trie_count + 1,):
        raise ValueError("first_token_offsets must have trie_count + 1 entries")
    if int(root_offsets[0]) != 0 or int(root_offsets[-1]) != first.size:
        raise ValueError("first_token_offsets must cover first_token_ids")
    _validate_offsets(root_offsets, "first_token_offsets", require_positive=False)

    if offsets1.shape != (metadata.trie_count * vocab_size + 1,):
        raise ValueError(
            f"prefix1_offsets must have shape (trie_count * vocab_size + 1,), got {offsets1.shape}"
        )
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
    if values1.size and not _values_increase_within_buckets(values1, offsets1):
        raise ValueError("prefix1_values must be strictly increasing within each bucket")
    for trie_id in range(metadata.trie_count):
        root_first = first[root_offsets[trie_id] : root_offsets[trie_id + 1]]
        if root_first.size == 0 or not _is_strictly_increasing(root_first):
            raise ValueError("first_token_ids must be non-empty and strictly increasing per root")
        root_rows = offsets1[trie_id * vocab_size : (trie_id + 1) * vocab_size + 1]
        if not _active_tokens_match(root_first, root_rows):
            raise ValueError(
                "first_token_ids must exactly match prefix1_offsets tokens with positive degree"
            )
        if not _pair_keys_match(root_first, offsets1, values1, pair_keys, vocab_size, trie_id):
            raise ValueError(
                "prefix1_pair_keys must equal trie_id * vocab_size**2 + a_id * vocab_size + b_id"
            )
    if pair_keys.size > 1 and not _is_strictly_increasing(pair_keys):
        raise ValueError("prefix1_pair_keys must be strictly increasing")
    if values2.size and not _values_increase_within_buckets(values2, offsets2):
        raise ValueError("prefix2_values must be strictly increasing within each bucket")
