# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Columnar results preserve payload ownership, protocol checks and public replies."""

from types import SimpleNamespace

import msgspec
import numpy as np
import numpy.typing as npt
import pytest

from tests.v1.worker.test_beam_final_output_payload import bindings
from vllm_gr.ops.cuda.beam_state import FINISH_LENGTH, FINISH_STOP
from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchResult,
    BeamOutputOptions,
    BeamSequenceColumns,
    materialize_beam_result,
    sequence_columns,
    validate_complete_sequences,
)
from vllm_gr.v1.worker.cuda.beam_final_output import _build_batch_result, _PayloadLayout

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def payload(
    count: int = 64, *, stopped: bool = False
) -> tuple[_PayloadLayout, npt.NDArray[np.int32]]:
    layout = _PayloadLayout.create(count, 3, 2)
    values = np.zeros(layout.size, dtype=np.int32)
    for item in range(2):
        start = 3 + item * layout.member_stride
        values[start] = count
        rows = values[start + 1 : start + layout.member_stride].reshape(count, layout.row_stride)
        rows[:, :3] = np.arange(count * 3).reshape(count, 3) + 200 * item
        rows[:, 3:6] = np.arange(3)
        rows.view(np.float32)[:, 6] = -np.arange(1, count + 1, dtype=np.float32)
        rows[:, 7] = 3
        rows[:, 8] = FINISH_LENGTH
        rows[:, 9] = -1
        if stopped:
            rows[0, 7:] = (2, FINISH_STOP, int(rows[0, 1]))
    return layout, values


def build(
    layout: _PayloadLayout,
    values: npt.NDArray[np.int32],
    options: BeamOutputOptions,
    *,
    columnar: bool,
) -> BeamBatchResult:
    return _build_batch_result(
        values,
        layout,
        session_id="session",
        bindings=bindings(2),
        options=options,
        columnar=columnar,
    )


@pytest.mark.parametrize("scores", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("parents", [False, True])  # type: ignore[misc]
def test_columnar_wire_roundtrip_owns_only_requested_columns(scores: bool, parents: bool) -> None:
    layout, values = payload()
    options = BeamOutputOptions(64, True, scores, parents)
    expected = build(layout, values, options, columnar=False)
    result = build(layout, values, options, columnar=True)
    assert result.output is not None
    assert all(isinstance(item.sequences, BeamSequenceColumns) for item in result.output.items)
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(result), type=BeamBatchResult)
    values.fill(0)  # The Worker may immediately reuse its host output allocation.
    assert materialize_beam_result(decoded) == expected
    assert decoded.output is not None
    for item in decoded.output.items:
        columns = item.sequences
        assert isinstance(columns, BeamSequenceColumns)
        validate_complete_sequences(columns)
        assert len(columns.token_ids) == 64 * 3 * 4
        assert (columns.cumulative_logprobs is not None) == scores
        assert (columns.parent_beam_indices is not None) == parents


@pytest.mark.parametrize("count,stopped", [(1, False), (63, False), (64, True)])  # type: ignore[misc]
def test_small_and_stopped_results_keep_ordinary_representation(count: int, stopped: bool) -> None:
    layout, values = payload(count, stopped=stopped)
    options = BeamOutputOptions(count, True, True, True)
    actual = build(layout, values, options, columnar=True)
    assert actual == build(layout, values, options, columnar=False)
    assert actual.output is not None
    assert all(isinstance(item.sequences, tuple) for item in actual.output.items)


def test_malformed_columns_are_rejected() -> None:
    layout, values = payload()
    result = build(layout, values, BeamOutputOptions(64, True, True, True), columnar=True)
    assert result.output is not None
    columns = result.output.items[0].sequences
    assert isinstance(columns, BeamSequenceColumns)
    with pytest.raises(ValueError):
        validate_complete_sequences(
            msgspec.structs.replace(columns, token_ids=columns.token_ids[:-4])
        )
    with pytest.raises(ValueError, match="immutable bytes"):
        validate_complete_sequences(
            msgspec.structs.replace(columns, token_ids=bytearray(columns.token_ids))
        )


def test_grpc_default_reply_matches_ordinary_output_and_binary_preserves_columns() -> None:
    pytest.importorskip("grpc", minversion="1.84.0")
    pytest.importorskip("google.protobuf", minversion="7.35.1")
    pytest.importorskip("smg_grpc_servicer")
    from vllm_gr.entrypoints.grpc.serving import BeamVllmEngineServicer

    layout, values = payload()
    options = BeamOutputOptions(64, True, True, True)
    ordinary = build(layout, values, options, columnar=False)
    columns = build(layout, values, options, columnar=True)
    assert columns.output is not None
    request = SimpleNamespace(
        batch_id="session",
        items=[
            SimpleNamespace(request=SimpleNamespace(prompt_token_ids=[1, 2, 3])) for _ in range(2)
        ],
    )
    servicer = object.__new__(BeamVllmEngineServicer)
    identities = ("native-0", "native-1")
    expected = servicer.build_response(request, identities, ordinary)
    assert servicer.build_response(request, identities, columns) == expected
    binary = servicer.build_response(request, identities, columns, binary=True)
    for item, source in zip(binary.items, columns.output.items, strict=True):
        encoded = item.binary_beams
        assert isinstance(source.sequences, BeamSequenceColumns)
        tokens, scores, parents = sequence_columns(source.sequences)
        assert (encoded.count, encoded.steps, encoded.prompt_tokens) == (64, 3, 3)
        np.testing.assert_array_equal(
            np.frombuffer(encoded.output_ids_i32_le, dtype="<i4").reshape(64, 3), tokens
        )
        np.testing.assert_array_equal(
            np.frombuffer(encoded.cumulative_logprobs_f32_le, dtype="<f4"), scores
        )
        np.testing.assert_array_equal(
            np.frombuffer(encoded.parent_beam_indices_i32_le, dtype="<i4").reshape(64, 3),
            parents,
        )


def test_binary_reply_can_mix_columns_and_small_result_fallback() -> None:
    pytest.importorskip("grpc", minversion="1.84.0")
    pytest.importorskip("google.protobuf", minversion="7.35.1")
    pytest.importorskip("smg_grpc_servicer")
    from vllm_gr.entrypoints.grpc.serving import BeamVllmEngineServicer

    layout, values = payload()
    values[3 + layout.member_stride] = 1
    options = BeamOutputOptions(64, True, True, True)
    ordinary = build(layout, values, options, columnar=False)
    mixed = build(layout, values, options, columnar=True)
    assert mixed.output is not None
    assert isinstance(mixed.output.items[0].sequences, BeamSequenceColumns)
    assert isinstance(mixed.output.items[1].sequences, tuple)
    assert materialize_beam_result(mixed) == ordinary
    request = SimpleNamespace(
        batch_id="session",
        items=[
            SimpleNamespace(request=SimpleNamespace(prompt_token_ids=[1, 2, 3])) for _ in range(2)
        ],
    )
    servicer = object.__new__(BeamVllmEngineServicer)
    identities = ("native-0", "native-1")
    default = servicer.build_response(request, identities, mixed)
    assert default == servicer.build_response(request, identities, ordinary)
    binary = servicer.build_response(request, identities, mixed, binary=True)
    assert binary.items[0].HasField("binary_beams")
    assert not binary.items[1].HasField("binary_beams")
    assert binary.items[1].beams == default.items[1].beams
