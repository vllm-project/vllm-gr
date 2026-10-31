# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small, device-free tests for the shared Beam execution contract."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from types import SimpleNamespace

import pytest

from vllm_gr.v1.beam.execution_batch import (
    BeamBatchExecutionView,
    BeamExecutionContractError,
)


class _StageKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


def _metadata(
    *,
    batch_size: int = 2,
    beam_width: int = 3,
    kind: str = "prefill",
) -> SimpleNamespace:
    prompt_counts = tuple(range(2, batch_size + 2))
    is_decode = kind == "decode"
    return SimpleNamespace(
        session_id="batch-0",
        dispatch_id=int(is_decode),
        stage_index=int(is_decode),
        kind=kind,
        decode_step=0 if is_decode else None,
        owner_generation=4,
        request_ids=tuple(f"request-{index}" for index in range(batch_size)),
        num_prompt_tokens_by_request=prompt_counts,
        num_prompt_tokens=sum(prompt_counts),
        beam_params=SimpleNamespace(beam_width=beam_width),
        execution_capacity=SimpleNamespace(beam_width=beam_width),
        produces_output=True,
    )


@pytest.mark.parametrize("batch_size", range(1, 9))  # type: ignore[misc]
def test_prefill_maps_one_row_per_request(batch_size: int) -> None:
    view = BeamBatchExecutionView.from_stage_metadata(_metadata(batch_size=batch_size))

    assert view.batch_size == batch_size
    assert view.num_execution_rows == batch_size
    assert view.row_request_indices == tuple(range(batch_size))
    assert view.row_beam_indices == (0,) * batch_size
    assert view.expand_per_request(tuple(range(batch_size))) == tuple(range(batch_size))


@pytest.mark.parametrize("batch_size", range(1, 9))  # type: ignore[misc]
def test_decode_maps_request_major_b_times_w_rows(batch_size: int) -> None:
    view = BeamBatchExecutionView.from_stage_metadata(
        _metadata(batch_size=batch_size, beam_width=3, kind="decode")
    )

    expected_requests = tuple(index for index in range(batch_size) for _ in range(3))
    assert view.num_execution_rows == batch_size * 3
    assert view.row_request_indices == expected_requests
    assert view.row_beam_indices == (0, 1, 2) * batch_size
    assert view.physical_row(batch_size - 1, 2) == batch_size * 3 - 1
    assert view.row_owner(batch_size * 3 - 1) == (batch_size - 1, 2)
    assert tuple(view.rows_for_request(batch_size - 1)) == tuple(
        range((batch_size - 1) * 3, batch_size * 3)
    )


def test_prompt_offsets_and_request_expansion_are_stable() -> None:
    metadata = _metadata(kind="decode")
    metadata.kind = _StageKind.DECODE
    view = BeamBatchExecutionView.from_stage_metadata(metadata)

    assert view.prompt_token_offsets == (0, 2, 5)
    assert view.row_request_ids == ("request-0",) * 3 + ("request-1",) * 3
    assert view.expand_per_request((10, 20), name="positions") == (10, 10, 10, 20, 20, 20)


def test_request_order_and_cross_stage_identity_fail_closed() -> None:
    prefill = BeamBatchExecutionView.from_stage_metadata(_metadata())
    decode = BeamBatchExecutionView.from_stage_metadata(_metadata(kind="decode"))

    prefill.assert_same_logical_batch(decode)
    with pytest.raises(BeamExecutionContractError, match="reordered"):
        decode.validate_request_order(("request-1", "request-0"))

    changed = _metadata(kind="decode")
    changed.owner_generation += 1
    with pytest.raises(BeamExecutionContractError, match="identity changed"):
        prefill.assert_same_logical_batch(BeamBatchExecutionView.from_stage_metadata(changed))


def test_decode_parent_rows_are_localized_per_request() -> None:
    view = BeamBatchExecutionView.from_stage_metadata(_metadata(kind="decode"))

    assert view.localize_global_parent_rows((2, 0, 1, 4, 3, 5)) == (2, 0, 1, 1, 0, 2)
    with pytest.raises(BeamExecutionContractError, match="crosses"):
        view.localize_global_parent_rows((2, 0, 3, 4, 3, 5))


@pytest.mark.parametrize(  # type: ignore[misc]
    ("beam_width", "candidate_width", "expected_top_k"),
    [
        (8, None, 8),  # unset K keeps the pre-decoupling K == W
        (8, 8, 8),
        (8, 4, 4),  # K < W (rectangular, narrower candidates)
        (8, 64, 64),  # K > W (rectangular, wider candidates)
        (1024, 256, 256),
    ],
)
def test_candidate_top_k_is_K_not_the_survivor_count(
    beam_width: int,
    candidate_width: int | None,
    expected_top_k: int,
) -> None:
    """The sampler fills ``[rows, K]`` candidate buffers.

    ``candidate_top_k`` is passed straight to the constrained backend and the
    unconstrained Top-K, so returning ``output_width`` (W) sized the buffers
    with K columns but sampled W per row: K < W wrote past the row, K > W left
    columns unwritten.
    """
    metadata = _metadata(kind="decode", beam_width=beam_width)
    if candidate_width is not None:
        metadata.beam_params.candidate_width = candidate_width
    view = BeamBatchExecutionView.from_stage_metadata(metadata)

    assert view.candidate_top_k == expected_top_k
    assert view.output_width == beam_width


def test_candidate_top_k_must_cover_the_widest_dynamic_stage() -> None:
    metadata = _metadata(kind="decode", beam_width=8)
    metadata.beam_params.candidate_width = 2
    metadata.beam_params.dynamic_beam_widths = (4, 4)
    metadata.stage_index = 1
    metadata.decode_step = 0

    with pytest.raises(BeamExecutionContractError, match="candidate_width 2 is below"):
        BeamBatchExecutionView.from_stage_metadata(metadata)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("beam_width", "candidate_width", "schedule"),
    [
        (4, 4, (2, 2, 2)),  # #525 review: uniform but narrower ran TopK=2
        (2, 2, (1, 1, 1)),  # #525 review: same defect at K=2
        (4, 4, (4, 2, 1)),
        (4, 4, (4, 4, 4)),
    ],
)
def test_explicit_candidate_width_is_the_top_k_of_every_dynamic_stage(
    beam_width: int,
    candidate_width: int,
    schedule: tuple[int, ...],
) -> None:
    """Every stage samples K candidates per parent and keeps its own width.

    The #525 review observed ``K=4`` with ``(2, 2, 2)`` executing TopK=2 at
    every stage, because ``candidate_top_k`` returned the stage width.  The
    sampler's Top-K is this property, so it must be K at every stage while the
    retained survivor count still follows the schedule.
    """
    for stage in range(len(schedule)):
        view = BeamBatchExecutionView(
            session_id="session",
            dispatch_id=stage,
            stage_index=stage,
            kind="prefill" if stage == 0 else "decode",
            decode_step=None if stage == 0 else stage - 1,
            owner_generation=0,
            request_ids=("only",),
            num_prompt_tokens_by_request=(8,),
            beam_width=beam_width,
            produces_output=True,
            dynamic_beam_widths=schedule,
            candidate_width=candidate_width,
        )
        assert view.candidate_top_k == candidate_width
        assert view.output_width == schedule[stage]


def test_per_parent_beam_widths_take_precedence_over_the_candidate_width() -> None:
    """Each stage samples its own scheduled K, not the configured scalar."""
    schedule = (4, 2, 1)
    for stage in range(len(schedule)):
        view = BeamBatchExecutionView(
            session_id="session",
            dispatch_id=stage,
            stage_index=stage,
            kind="prefill" if stage == 0 else "decode",
            decode_step=None if stage == 0 else stage - 1,
            owner_generation=0,
            request_ids=("only",),
            num_prompt_tokens_by_request=(8,),
            beam_width=4,
            produces_output=True,
            dynamic_beam_widths=schedule,
            candidate_width=2,
            per_parent_beam_widths=schedule,
        )
        assert view.candidate_top_k == schedule[stage]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("change", "message"),
    [
        (lambda m: setattr(m, "request_ids", ("same", "same")), "unique"),
        (lambda m: setattr(m, "num_prompt_tokens", 99), "per-request"),
        (lambda m: setattr(m.beam_params, "beam_width", 0), "positive"),
        (lambda m: setattr(m, "decode_step", 1), "stage_index-1"),
        (
            lambda m: setattr(m.execution_capacity, "beam_width", 4),
            "execution_capacity",
        ),
    ],
)
def test_invalid_stage_geometry_is_rejected(
    change: Callable[[SimpleNamespace], None], message: str
) -> None:
    metadata = _metadata(kind="decode")
    change(metadata)

    with pytest.raises(BeamExecutionContractError, match=message):
        BeamBatchExecutionView.from_stage_metadata(metadata)
