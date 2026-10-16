# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler metadata construction for Constraint Table requests (N2).

``build_beam_request_metadata`` is a pure function (no vLLM imports), so these
tests run locally.  The constraint_step contract: prefill is always 0; a
constrained decode MUST carry a valid ``constraint_step`` — a missing/invalid
value is a protocol error, never silently defaulted to prefill.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_gr.v1.engine.scheduler_metadata import build_beam_request_metadata


def _req(**attrs: Any) -> SimpleNamespace:
    defaults: dict[str, Any] = dict(
        request_id="req-0",
        num_computed_tokens=4,
        is_beam_decode=False,
        is_constraint_prefill=False,
        is_constrained=False,
    )
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def test_constraint_prefill_metadata_carries_step_zero() -> None:
    metadata = build_beam_request_metadata(
        _req(
            is_constraint_prefill=True,
            is_constrained=True,
            beam_width=2,
            prefix_len=0,
            max_decode_steps=2,
        ),
        num_scheduled_tokens=4,
    )

    assert metadata is not None
    assert metadata["is_beam_decode"] is False
    assert metadata["is_constraint_prefill"] is True
    assert metadata["is_constrained"] is True
    assert metadata["constraint_step"] == 0
    assert metadata["beam_width"] == 2


def test_unconstrained_prefill_metadata_carries_step_zero_without_constraint() -> None:
    """A full-vocabulary prefill keeps step 0 but is not marked constrained.

    ``is_constraint_prefill`` only says the EngineCore cache hook saw a beam
    prefill; ``is_constrained`` is what selects the Constraint Table, so the
    Worker never treats an unconstrained prefill as a constrained one.
    """

    metadata = build_beam_request_metadata(
        _req(
            is_constraint_prefill=True,
            beam_width=2,
            prefix_len=0,
            max_decode_steps=2,
        ),
        num_scheduled_tokens=4,
    )

    assert metadata is not None
    assert metadata["is_constraint_prefill"] is True
    assert metadata["is_constrained"] is False
    assert metadata["constraint_step"] == 0


def test_constrained_decode_metadata_carries_step() -> None:
    metadata = build_beam_request_metadata(
        _req(
            is_beam_decode=True,
            is_constrained=True,
            beam_width=2,
            constraint_step=1,
            beam_suffix=[[0], [1]],
        ),
        num_scheduled_tokens=1,
    )

    assert metadata["is_constrained"] is True
    assert metadata["constraint_step"] == 1
    assert metadata["beam_suffix"] == [[0], [1]]


@pytest.mark.parametrize("step", [None, 0, 3, -1, True])  # type: ignore[misc]
def test_constrained_decode_requires_valid_step(step: object) -> None:
    """Missing or invalid constraint_step fails fast (never defaults to 0).

    Decode steps are 1/2 only: step 0 belongs to Constraint Table prefill, and a
    decode request carrying 0 would make the Worker build a prefill-shaped
    sequence_group while bypassing the beam-suffix cross-check.
    """
    with pytest.raises(RuntimeError, match="constraint_step"):
        build_beam_request_metadata(
            _req(
                is_beam_decode=True,
                is_constrained=True,
                beam_width=2,
                constraint_step=step,
            ),
            num_scheduled_tokens=1,
        )


def test_unconstrained_decode_has_no_constrained_fields() -> None:
    metadata = build_beam_request_metadata(
        _req(
            is_beam_decode=True,
            beam_width=2,
        ),
        num_scheduled_tokens=1,
    )

    assert metadata is not None
    assert "is_constrained" not in metadata
    assert "constraint_step" not in metadata
