# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintMetadata,
    build_constraint_output,
    effective_temperatures,
    validate_select_inputs,
)

from .helpers import make_backend_info, make_table


def _metadata(*, rows: int = 1, step: int = 0, top_k: int = 2) -> ConstraintMetadata:
    return ConstraintMetadata(
        prefix_token_ids=torch.zeros((rows, 2), dtype=torch.int32),
        temperatures=torch.ones(1, dtype=torch.float32),
        current_step=step,
        top_k=top_k,
    )


def test_metadata_is_frozen_and_slotted() -> None:
    metadata = _metadata()
    with pytest.raises((AttributeError, TypeError)):
        metadata.top_k = 3  # type: ignore[misc]
    assert not hasattr(metadata, "__dict__")


@pytest.mark.parametrize("step", [-1, 3])
def test_rejects_invalid_step(step: int) -> None:
    with pytest.raises(ValueError, match="current_step"):
        validate_select_inputs(
            torch.zeros((1, 8)),
            _metadata(step=step),
            make_backend_info(make_table()),
        )


def test_rejects_top_k_above_configured_limit() -> None:
    with pytest.raises(ValueError, match="top_k"):
        validate_select_inputs(
            torch.zeros((1, 8)),
            _metadata(top_k=5),
            make_backend_info(make_table(), max_top_k=4),
        )


def test_hot_input_validation_does_not_reduce_temperature_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_isfinite(_tensor: torch.Tensor) -> torch.Tensor:
        raise AssertionError("hot validation must not inspect Device values")

    monkeypatch.setattr(torch, "isfinite", fail_isfinite)
    metadata = ConstraintMetadata(
        prefix_token_ids=torch.zeros((1, 2), dtype=torch.int32),
        temperatures=torch.tensor([float("nan")], dtype=torch.float32),
        current_step=0,
        top_k=1,
    )
    validate_select_inputs(
        torch.zeros((1, 8)),
        metadata,
        make_backend_info(make_table()),
    )


def test_effective_temperatures_maps_greedy_zero_without_host_branch() -> None:
    temperatures = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    assert effective_temperatures(temperatures).tolist() == [1.0, 0.5, 1.0]


def test_valid_mask_is_derived_only_from_invalid_logit_threshold() -> None:
    logits = torch.zeros((1, 8))
    output = build_constraint_output(
        torch.tensor([[3, 0]], dtype=torch.int32),
        torch.tensor([[-0.1, INVALID_LOGIT]], dtype=torch.float32),
        logits=logits,
        top_k=2,
    )
    assert output.token_ids.dtype == torch.int32
    assert output.logprobs.dtype == torch.float32
    assert output.valid_mask.dtype == torch.bool
    assert output.valid_mask.tolist() == [[True, False]]


def test_output_contract_rejects_dtype_coercion() -> None:
    with pytest.raises(RuntimeError, match="token_ids dtype"):
        build_constraint_output(
            torch.tensor([[3]], dtype=torch.int64),
            torch.tensor([[-0.1]], dtype=torch.float32),
            logits=torch.zeros((1, 8)),
            top_k=1,
        )
