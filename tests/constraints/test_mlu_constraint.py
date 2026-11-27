# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CPU semantic checks for the tensor code used by the MLU constraint backend."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from tests.constraints.test_table import make_table
from tests.helpers.paths import REPO_ROOT
from vllm_gr.config import ConstraintBackendKind
from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintBackendInfo,
    ConstraintMetadata,
)
from vllm_gr.constraints.compiler import ConstraintTableResource
from vllm_gr.constraints.cpu_backend import CPUConstraintBackend
from vllm_gr.constraints.table import (
    CONSTRAINT_TABLE_TENSOR_NAMES,
    validate_constraint_table,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture  # type: ignore[misc]
def plugin(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "plugins" / "mlu" / "src"))
    return import_module("vllm_gr_mlu.constraints.backend")


@pytest.fixture  # type: ignore[misc]
def table():
    artifact = make_table(
        first_token_ids=np.array([1, 3, 5], dtype=np.int32),
        prefix1_offsets=np.array([0, 0, 2, 2, 3, 3, 4], dtype=np.int32),
        prefix1_values=np.array([2, 4, 5, 0], dtype=np.int32),
        prefix1_pair_keys=np.array([8, 10, 23, 30], dtype=np.int64),
        prefix2_value_offsets=np.array([0, 2, 3, 5, 6], dtype=np.int32),
        prefix2_values=np.array([0, 1, 3, 1, 4, 2], dtype=np.int32),
    )
    validate_constraint_table(artifact)
    return artifact


@pytest.mark.parametrize(  # type: ignore[misc]
    ("step", "prefixes"),
    [
        (0, [[0, 0, 0], [0, 0, 0]]),
        (1, [[1, 0, 0], [3, 0, 0], [5, 0, 0], [2, 0, 0]]),
        (2, [[1, 2, 0], [5, 0, 0], [1, 5, 0], [-1, 14, 0], [0, 8, 0]]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_fixed_shape_selection_matches_cpu_reference(
    plugin: ModuleType, table: Any, step: int, prefixes: list[list[int]], dtype: torch.dtype
) -> None:
    info = ConstraintBackendInfo(
        table.metadata, ConstraintBackendKind.MLU, 4, 8, 0, table.total_bytes
    )
    storage = plugin.DeviceConstraintTable.upload(table, torch.device("cpu"))
    backend = plugin.MLUConstraintBackend(storage, info, dtype)
    logits = torch.tensor([0.75, -2, 0, 2, 1, 4], dtype=dtype).repeat(len(prefixes), 1)
    meta = ConstraintMetadata(
        torch.tensor(prefixes, dtype=torch.int32),
        torch.arange(len(prefixes), dtype=torch.float32),
        step,
        4,
    )
    actual = backend.select(logits, meta)
    expected = CPUConstraintBackend(table, info).select(logits, meta)
    assert torch.equal(actual.valid_mask, expected.valid_mask)
    mask = actual.valid_mask
    assert torch.equal(actual.token_ids[mask], expected.token_ids[mask])
    torch.testing.assert_close(actual.logprobs, expected.logprobs)
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.select(logits, meta)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("value", "is_nan"), [(float("nan"), True), (float("inf"), True), (float("-inf"), False)]
)
def test_nonfinite_scores_do_not_change_structural_validity(
    plugin: ModuleType, table: Any, value: float, is_nan: bool
) -> None:
    storage = plugin.DeviceConstraintTable.upload(table, torch.device("cpu"))
    logits = torch.full((1, 6), value)
    out = storage.select(logits, torch.tensor([[5, 0, 0]], dtype=torch.int32), torch.ones(1), 2, 4)
    assert out.valid_mask.tolist() == [[True, False, False, False]]
    assert int(out.token_ids[0, 0]) == 2
    score = out.logprobs[0, 0]
    assert bool(score.isnan() if is_nan else score.isneginf())
    assert torch.all(out.logprobs[0, 1:] == INVALID_LOGIT)


def test_local_cutoff_ties_are_deterministic_and_legal(plugin: ModuleType, table: Any) -> None:
    storage = plugin.DeviceConstraintTable.upload(table, torch.device("cpu"))
    logits = torch.zeros((1, 6), dtype=torch.float16)
    prefixes = torch.zeros((1, 3), dtype=torch.int32)
    temperatures = torch.ones(1)
    first = storage.select(logits, prefixes, temperatures, 0, 2)
    second = storage.select(logits, prefixes, temperatures, 0, 2)
    assert first.valid_mask.all()
    assert torch.equal(first.token_ids, second.token_ids)
    selected = set(first.token_ids[0].tolist())
    assert len(selected) == 2 and selected.issubset(table.first_token_ids)
    torch.testing.assert_close(first.logprobs, torch.full((1, 2), -np.log(6), dtype=torch.float32))


def test_padding_sentinel_is_a_valid_score_when_mask_says_so(
    plugin: ModuleType, table: Any
) -> None:
    storage = plugin.DeviceConstraintTable.upload(table, torch.device("cpu"))
    logits = torch.tensor([[0, 0, -1, 0, 0, 0]], dtype=torch.float16)
    out = storage.select(
        logits, torch.tensor([[5, 0, 0]], dtype=torch.int32), torch.tensor([1e-20]), 2, 2
    )
    assert out.valid_mask.tolist() == [[True, False]]
    assert torch.all(out.logprobs == INVALID_LOGIT)


def test_legacy_digest_reuses_or_rejects_until_reset(
    plugin: ModuleType, table: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = import_module("vllm_gr_mlu.constraints.execution")
    adapter = execution.MLUConstrainedTopKExecution()
    monkeypatch.setattr(adapter, "_requested_device", torch.device("cpu"))
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(execution, "require_mlu_device", lambda _: None)

    def resource(artifact: Any) -> ConstraintTableResource:
        return ConstraintTableResource(
            **{name: getattr(artifact, name) for name in CONSTRAINT_TABLE_TENSOR_NAMES},
            vocab_size=artifact.metadata.vocab_size,
            max_prefix1_degree=artifact.metadata.max_prefix1_degree,
            max_prefix2_degree=artifact.metadata.max_prefix2_degree,
        )

    first, second = resource(table), resource(make_table())
    adapter.load_tables(first)
    loaded = adapter._table
    adapter.load_tables(first)
    assert adapter._table is loaded
    with pytest.raises(ValueError, match="reset"):
        adapter.load_tables(second)
    out = adapter.run(
        torch.arange(6, dtype=torch.float16).reshape(1, 6),
        torch.zeros((1, 1, 1), dtype=torch.int32),
        torch.empty(0),
        0,
        4,
    )
    assert out.top_tokens[0, out.valid_mask[0]].tolist() == [5, 3, 1]
    adapter.reset()
    adapter.load_tables(second)
    assert adapter._table is not loaded
    adapter.close()
    adapter.close()
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.run(torch.empty(0), torch.empty(0), torch.empty(0), 0, 4)
