# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import vllm_gr.constraints.table as table_module
from vllm_gr.constraints.backend import ConstraintBackendInfo
from vllm_gr.constraints.table import (
    TABLE_NAMES,
    ConstraintTable,
    validate_constraint_table,
    validate_constraint_table_info,
)

from .helpers import make_table


@pytest.mark.parametrize(
    "field",
    ["vocab_size", "max_prefix1_degree", "max_prefix2_degree", "total_bytes"],
)
def test_table_info_rejects_bool_integer_metadata(field: str) -> None:
    info = replace(make_table().info, **{field: True})
    with pytest.raises(ValueError, match=field):
        validate_constraint_table_info(info)


def test_none_digest_is_only_allowed_before_backend_creation() -> None:
    info = replace(make_table().info, artifact_digest=None)
    validate_constraint_table_info(info)

    with pytest.raises(ValueError, match="persisted Artifact digest"):
        ConstraintBackendInfo(
            table=info,
            max_top_k=4,
            device_memory_bytes=0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_top_k", True), ("device_memory_bytes", True)],
)
def test_backend_info_rejects_bool_integer_metadata(field: str, value: object) -> None:
    values = {
        "table": make_table().info,
        "max_top_k": 4,
        "device_memory_bytes": 0,
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        ConstraintBackendInfo(**values)  # type: ignore[arg-type]


def _wide_prefix2_table(*, values: list[int]) -> ConstraintTable:
    source = make_table()
    arrays = {name: np.array(getattr(source, name), copy=True) for name in TABLE_NAMES}
    arrays["prefix2_value_offsets"] = np.asarray([0, 3, 4, 5], dtype=np.int32)
    arrays["prefix2_values"] = np.asarray(values, dtype=np.int32)
    info = replace(
        source.info,
        max_prefix2_degree=3,
        total_bytes=sum(int(array.nbytes) for array in arrays.values()),
    )
    return ConstraintTable(**arrays, info=info)


def test_large_relationship_validation_uses_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _wide_prefix2_table(values=[3, 4, 5, 6, 7])
    monkeypatch.setattr(table_module, "VALIDATION_CHUNK_ELEMENTS", 1)
    monkeypatch.setattr(
        table_module.np,
        "repeat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("np.repeat is forbidden")),
    )
    monkeypatch.setattr(
        table_module.np,
        "diff",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("np.diff is forbidden")),
    )

    validate_constraint_table(table)


def test_prefix2_order_violation_across_chunk_boundary_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _wide_prefix2_table(values=[3, 2, 5, 6, 7])
    monkeypatch.setattr(table_module, "VALIDATION_CHUNK_ELEMENTS", 1)

    with pytest.raises(ValueError, match="prefix2_values must be strictly increasing"):
        validate_constraint_table(table)
