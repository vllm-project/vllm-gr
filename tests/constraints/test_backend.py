# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintBackendKind
from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintBackendInfo,
    ConstraintMetadata,
    build_constraint_output,
    validate_select_inputs,
)


def _info(**overrides: object) -> ConstraintBackendInfo:
    values: dict[str, object] = {
        "table_metadata": make_table().metadata,
        "backend": ConstraintBackendKind.CPU,
        "max_top_k": 4,
        "max_rows": 2,
        "host_memory_bytes": make_table().total_bytes,
        "device_memory_bytes": 0,
    }
    values.update(overrides)
    return ConstraintBackendInfo(**values)


def _metadata(rows: int = 2, **overrides: object) -> ConstraintMetadata:
    values: dict[str, object] = {
        "prefix_token_ids": torch.zeros(rows, 3, dtype=torch.int32),
        "temperatures": torch.ones(1, dtype=torch.float32),
        "current_step": 0,
        "top_k": 2,
    }
    values.update(overrides)
    return ConstraintMetadata(**values)


def test_public_backend_dataclasses_are_frozen_and_slotted() -> None:
    info = _info()
    with pytest.raises((AttributeError, TypeError)):
        info.max_rows = 3
    assert not hasattr(info, "__dict__")
    assert INVALID_LOGIT == -1.0e20


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_common_validation_accepts_supported_logits(dtype: torch.dtype) -> None:
    validate_select_inputs(torch.zeros(2, 6, dtype=dtype), _metadata(), _info())


def test_common_validation_rejects_rows_shapes_dtypes_devices_and_topk() -> None:
    logits = torch.zeros(2, 6)
    with pytest.raises(ValueError, match="rows"):
        validate_select_inputs(torch.zeros(3, 6), _metadata(3), _info())
    with pytest.raises(ValueError, match="contiguous"):
        validate_select_inputs(torch.zeros(6, 2).t(), _metadata(), _info())
    with pytest.raises(ValueError, match=r"\[2, 3\]"):
        validate_select_inputs(
            logits,
            _metadata(prefix_token_ids=torch.zeros(2, 2, dtype=torch.int32)),
            _info(),
        )
    with pytest.raises(ValueError, match="int32"):
        validate_select_inputs(
            logits,
            _metadata(prefix_token_ids=torch.zeros(2, 3, dtype=torch.int64)),
            _info(),
        )
    with pytest.raises(ValueError, match="top_k"):
        validate_select_inputs(logits, _metadata(top_k=5), _info())
    with pytest.raises(ValueError, match="current_step"):
        validate_select_inputs(logits, _metadata(current_step=True), _info())


def test_output_builder_rejects_instead_of_repairing_operator_abi() -> None:
    logits = torch.zeros(2, 6)
    good_tokens = torch.zeros(2, 2, dtype=torch.int32)
    good_logprobs = torch.zeros(2, 2, dtype=torch.float32)
    good_mask = torch.ones(2, 2, dtype=torch.bool)
    output = build_constraint_output(good_tokens, good_logprobs, good_mask, logits=logits, top_k=2)
    assert output.valid_mask.dtype == torch.bool

    with pytest.raises(RuntimeError, match="int32"):
        build_constraint_output(
            good_tokens.long(), good_logprobs, good_mask, logits=logits, top_k=2
        )
    with pytest.raises(RuntimeError, match="float32"):
        build_constraint_output(
            good_tokens, good_logprobs.half(), good_mask, logits=logits, top_k=2
        )
    with pytest.raises(RuntimeError, match="contiguous"):
        build_constraint_output(
            good_tokens,
            torch.zeros(2, 4)[:, ::2],
            good_mask,
            logits=logits,
            top_k=2,
        )


def test_output_builder_preserves_explicit_validity_at_sentinel_score() -> None:
    logits = torch.zeros(1, 2)
    logprobs = torch.tensor([[0.0, INVALID_LOGIT]])
    valid_mask = torch.tensor([[True, True]])

    output = build_constraint_output(
        torch.tensor([[1, 0]], dtype=torch.int32),
        logprobs,
        valid_mask,
        logits=logits,
        top_k=2,
    )

    assert output.valid_mask.tolist() == [[True, True]]


def test_info_remains_a_diagnostic_value_after_table_metadata_replacement() -> None:
    info = _info()
    changed = replace(info, max_rows=7)
    assert info.max_rows == 2
    assert changed.max_rows == 7


pytestmark = pytest.mark.cpu_test
