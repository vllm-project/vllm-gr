# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Immutable Host representation and validation for the six-table ABI."""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class ConstraintTableInfo:
    format: str
    artifact_digest: str
    tokenizer_digest: str
    vocab_size: int
    max_prefix1_degree: int
    max_prefix2_degree: int
    total_bytes: int
    max_top_k: int = 256
    device_memory_bytes: int = 0


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


def derive_degrees(
    prefix1_offsets: np.ndarray,
    prefix2_value_offsets: np.ndarray,
) -> tuple[int, int]:
    prefix1_degree = int(np.diff(prefix1_offsets).max()) if prefix1_offsets.size > 1 else 0
    prefix2_degree = (
        int(np.diff(prefix2_value_offsets).max()) if prefix2_value_offsets.size > 1 else 0
    )
    return prefix1_degree, prefix2_degree


def validate_constraint_table(table: ConstraintTable) -> None:
    """Validate dtype, shape, CSR, semantic relations, ranges, and size."""

    info = table.info
    if info.format != CONSTRAINT_TABLE_FORMAT:
        raise ValueError(
            f"constraint table format must be {CONSTRAINT_TABLE_FORMAT!r}, got {info.format!r}"
        )
    if isinstance(info.vocab_size, bool) or info.vocab_size <= 0:
        raise ValueError("constraint table vocab_size must be a positive integer")

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
            raise ValueError(
                f"table {name} must be 1-D (NPU ABI), got shape {tuple(array.shape)}"
            )
        if array.dtype != expected:
            raise ValueError(f"table {name}.dtype must be {expected}, got {array.dtype}")

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
    if np.any(np.diff(offsets1) < 0):
        raise ValueError("prefix1_offsets must be non-decreasing")

    if offsets2.shape != (len(values1) + 1,):
        raise ValueError(
            "prefix2_value_offsets must have one entry per prefix1 value plus the end"
        )
    if int(offsets2[0]) != 0 or int(offsets2[-1]) != len(values2):
        raise ValueError(
            "prefix2_value_offsets must start at zero and end at prefix2_values size"
        )
    if np.any(np.diff(offsets2) < 0):
        raise ValueError("prefix2_value_offsets must be non-decreasing")

    if len(pair_keys) != len(values1):
        raise ValueError("prefix1_pair_keys length must equal prefix1_values length")
    if len(pair_keys) > 1 and np.any(np.diff(pair_keys) <= 0):
        raise ValueError("prefix1_pair_keys must be strictly increasing")

    derived1, derived2 = derive_degrees(offsets1, offsets2)
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
    if first.size and np.any(np.diff(first) <= 0):
        raise ValueError("first_token_ids must be strictly increasing and unique")
    active_tokens = np.flatnonzero(np.diff(offsets1) > 0)
    if not np.array_equal(first, active_tokens):
        raise ValueError(
            "first_token_ids must exactly match prefix1_offsets tokens with positive degree"
        )

    if values1.size:
        bucket_ids = np.repeat(
            np.arange(vocab_size, dtype=np.int64),
            np.diff(offsets1).astype(np.int64),
        )
        expected_keys = bucket_ids * vocab_size + values1.astype(np.int64)
        if not np.array_equal(expected_keys, pair_keys):
            raise ValueError(
                "prefix1_pair_keys must equal a_id * vocab_size + b_id for each CSR bucket"
            )

    actual_bytes = table.total_bytes
    if actual_bytes != info.total_bytes:
        raise ValueError(
            f"constraint table total_bytes metadata {info.total_bytes} != actual {actual_bytes}"
        )
    if actual_bytes > MAX_TABLE_BYTES:
        raise ValueError(
            f"constraint table total_bytes {actual_bytes} exceeds {MAX_TABLE_BYTES} byte limit"
        )
