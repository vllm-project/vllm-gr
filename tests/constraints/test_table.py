# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from vllm_gr.config import CONSTRAINT_TABLE_FORMAT
from vllm_gr.constraints.table import (
    CONSTRAINT_TABLE_TENSOR_NAMES,
    ConstraintTable,
    ConstraintTableMetadata,
    validate_constraint_table,
)

_TOKENIZER_DIGEST = "sha256:" + "b" * 64


def make_table(**overrides: object) -> ConstraintTable:
    tensors: dict[str, object] = {
        "first_token_ids": np.asarray([1, 3], dtype=np.int32),
        "prefix1_offsets": np.asarray([0, 0, 2, 2, 3, 3, 3], dtype=np.int32),
        "prefix1_values": np.asarray([2, 4, 5], dtype=np.int32),
        "prefix1_pair_keys": np.asarray([8, 10, 23], dtype=np.int64),
        "prefix2_value_offsets": np.asarray([0, 2, 3, 5], dtype=np.int32),
        "prefix2_values": np.asarray([0, 1, 3, 1, 4], dtype=np.int32),
    }
    tensors.update(overrides)
    total_bytes = sum(
        int(cast(NDArray[Any], tensors[name]).nbytes) for name in CONSTRAINT_TABLE_TENSOR_NAMES
    )
    metadata = tensors.pop(
        "metadata",
        ConstraintTableMetadata(
            format=CONSTRAINT_TABLE_FORMAT,
            artifact_digest=None,
            tokenizer_digest=_TOKENIZER_DIGEST,
            vocab_size=6,
            max_prefix1_degree=2,
            max_prefix2_degree=2,
            total_bytes=total_bytes,
        ),
    )
    return ConstraintTable(
        first_token_ids=cast(NDArray[np.int32], tensors["first_token_ids"]),
        prefix1_offsets=cast(NDArray[np.int32], tensors["prefix1_offsets"]),
        prefix1_values=cast(NDArray[np.int32], tensors["prefix1_values"]),
        prefix1_pair_keys=cast(NDArray[np.int64], tensors["prefix1_pair_keys"]),
        prefix2_value_offsets=cast(NDArray[np.int32], tensors["prefix2_value_offsets"]),
        prefix2_values=cast(NDArray[np.int32], tensors["prefix2_values"]),
        metadata=cast(ConstraintTableMetadata, metadata),
    )


def test_valid_table_is_immutable() -> None:
    table = make_table()
    validate_constraint_table(table)

    for name in CONSTRAINT_TABLE_TENSOR_NAMES:
        tensor = getattr(table, name)
        assert tensor.flags.writeable is False
        with pytest.raises(ValueError):
            tensor.setflags(write=True)


def test_rejects_wrong_shape_and_dtype() -> None:
    with pytest.raises(ValueError, match="1-D"):
        validate_constraint_table(make_table(first_token_ids=np.asarray([[1, 3]], dtype=np.int32)))
    with pytest.raises(ValueError, match="dtype"):
        validate_constraint_table(
            make_table(prefix1_pair_keys=np.asarray([8, 10, 23], dtype=np.int32))
        )


def test_rejects_invalid_csr_offsets() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_constraint_table(
            make_table(prefix1_offsets=np.asarray([0, 0, 2, 1, 3, 3, 3], dtype=np.int32))
        )
    with pytest.raises(ValueError, match="at least one prefix2"):
        validate_constraint_table(
            make_table(prefix2_value_offsets=np.asarray([0, 2, 2, 5], dtype=np.int32))
        )


def test_rejects_pair_key_mismatch() -> None:
    with pytest.raises(ValueError, match=r"a_id \* vocab_size"):
        validate_constraint_table(
            make_table(prefix1_pair_keys=np.asarray([8, 10, 24], dtype=np.int64))
        )


def test_rejects_unsorted_or_duplicate_bucket_values() -> None:
    with pytest.raises(ValueError, match="prefix1_values.*strictly"):
        validate_constraint_table(make_table(prefix1_values=np.asarray([4, 2, 5], dtype=np.int32)))
    with pytest.raises(ValueError, match="prefix2_values.*strictly"):
        validate_constraint_table(
            make_table(prefix2_values=np.asarray([1, 1, 3, 1, 4], dtype=np.int32))
        )


def test_rejects_token_out_of_vocab() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        validate_constraint_table(
            make_table(prefix2_values=np.asarray([0, 6, 3, 1, 4], dtype=np.int32))
        )


def test_rejects_metadata_mismatch() -> None:
    table = make_table()
    with pytest.raises(ValueError, match="max_prefix1_degree"):
        validate_constraint_table(
            make_table(metadata=replace(table.metadata, max_prefix1_degree=3))
        )
    with pytest.raises(ValueError, match="total_bytes"):
        validate_constraint_table(make_table(metadata=replace(table.metadata, total_bytes=1)))


def test_validation_does_not_use_full_array_diff_repeat_or_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("full-array temporary helper was used")

    monkeypatch.setattr(np, "diff", forbidden)
    monkeypatch.setattr(np, "repeat", forbidden)
    monkeypatch.setattr(np, "unique", forbidden)

    validate_constraint_table(make_table())


pytestmark = pytest.mark.cpu_test
