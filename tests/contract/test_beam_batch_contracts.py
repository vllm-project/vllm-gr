# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

import vllm_gr.v1.beam as beam_package
import vllm_gr.v1.beam.batch_contracts as contracts
from vllm_gr.config import MAX_BEAM_WIDTH, MAX_CANDIDATE_WIDTH
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchItem,
    BeamBatchLimits,
    BeamBatchOutput,
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
    BeamContractError,
    BeamOutputOptions,
    BeamOutputSequence,
    BeamRequestOutput,
    validate_beam_batch_limits,
    validate_beam_batch_request,
    validate_beam_batch_request_limits,
    validate_beam_batch_result,
)
from vllm_gr.v1.engine.codec import build_engine_core_decoders
from vllm_gr.v1.engine.gr_session_client import _terminal_result


def _engine_request(
    request_id: str,
    prompt: list[int] | None,
    *,
    max_tokens: int = 99,
) -> EngineCoreRequest:
    values = dict(
        request_id=request_id,
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return EngineCoreRequest(**values)


def _params(**overrides: object) -> BeamBatchParams:
    values: dict[str, object] = {
        "beam_width": 2,
        "max_tokens": 3,
        "temperature": 1.0,
        "stop_token_ids": (2, 99),
        "ignore_stop_tokens": False,
        "constraint_resource_id": "default",
    }
    values.update(overrides)
    return BeamBatchParams(**values)


def _options(**overrides: object) -> BeamOutputOptions:
    values: dict[str, object] = {
        "num_return_sequences": 2,
        "include_stop_token": True,
        "include_cumulative_logprob": True,
        "include_parent_beam_indices": True,
    }
    values.update(overrides)
    return BeamOutputOptions(**values)


def _request(**overrides: object) -> BeamBatchRequest:
    values: dict[str, object] = {
        "protocol_version": BEAM_BATCH_PROTOCOL_VERSION,
        "batch_id": "batch-0",
        "items": (
            BeamBatchItem(0, _engine_request("request-0", [1, 2, 3])),
            BeamBatchItem(1, _engine_request("request-1", [4, 5])),
        ),
        "beam_params": _params(),
        "output_options": _options(),
    }
    values.update(overrides)
    return BeamBatchRequest(**values)


def _success(request: BeamBatchRequest | None = None) -> BeamBatchResult:
    request = request or _request()
    items = tuple(
        BeamRequestOutput(
            item_index=item.item_index,
            request_id=item.request.request_id,
            sequences=(
                BeamOutputSequence(
                    token_ids=(10 + item.item_index, 20 + item.item_index, 30),
                    cumulative_logprob=-0.5 - item.item_index,
                    parent_beam_indices=(0, 0, 1),
                    finish_reason="length",
                    stop_token_id=None,
                ),
            ),
        )
        for item in request.items
    )
    return BeamBatchResult(request.protocol_version, request.batch_id, BeamBatchOutput(items), None)


def _limits(**overrides: object) -> BeamBatchLimits:
    values: dict[str, object] = {
        "max_batch_size": 4,
        "max_beam_width": 256,
        "max_tokens": 3,
        "max_num_beam_rows": 512,
        "max_num_batched_tokens": 1024,
        "max_model_len": 4096,
        "supported_constraint_resource_ids": ("default",),
    }
    values.update(overrides)
    return BeamBatchLimits(**values)


def test_public_inventory_has_request_result_and_no_old_abi() -> None:
    assert {BeamBatchRequest.__name__, BeamBatchResult.__name__} == {
        "BeamBatchRequest",
        "BeamBatchResult",
    }
    forbidden = (
        "Child",
        "Lane",
        "Lease",
        "RunSlot",
        "ExecutionParameters",
        "RuntimeCapabilities",
        "Continuation",
        "Fragment",
        "Epoch",
    )
    assert not any(word in name for name in contracts.__all__ for word in forbidden)
    assert "BeamBatchRequest" not in beam_package.__all__
    assert "BeamItemCandidates" not in beam_package.__all__


def test_request_and_result_validate_at_terminal_boundary() -> None:
    request = _request()
    result = _success(request)
    validate_beam_batch_request(request)
    _terminal_result(request, SimpleNamespace(kv_transfer_params={"gr_batch_result": result}))


def test_request_uses_engine_request_identity_and_beam_params_authority() -> None:
    request = _request()
    assert request.items[0].request.sampling_params.max_tokens == 99
    assert request.beam_params.max_tokens == 3
    validate_beam_batch_request(request)

    duplicate = _request(
        items=(
            BeamBatchItem(0, _engine_request("same", [1])),
            BeamBatchItem(1, _engine_request("same", [2])),
        )
    )
    with pytest.raises(BeamContractError, match="unique"):
        validate_beam_batch_request(duplicate)


def test_request_keeps_engine_core_input_generic() -> None:
    engine_request = _engine_request("request-0", None)
    engine_request.prompt_embeds = torch.ones((2, 8), dtype=torch.float32)
    request = _request(items=(BeamBatchItem(0, engine_request),))
    validate_beam_batch_request(request)


@pytest.mark.parametrize("bad_value", [0, -1, True, 1025])  # type: ignore[misc]
def test_request_positive_geometry_is_fail_closed(bad_value: object) -> None:
    with pytest.raises(BeamContractError, match="beam_width"):
        validate_beam_batch_request(_request(beam_params=_params(beam_width=bad_value)))


def test_request_rejects_non_default_resource_duplicate_stops_and_wide_output() -> None:
    with pytest.raises(BeamContractError, match="unique"):
        validate_beam_batch_request(_request(beam_params=_params(stop_token_ids=(2, 2))))
    with pytest.raises(BeamContractError, match="only admits"):
        validate_beam_batch_request(
            _request(beam_params=_params(constraint_resource_id="catalog-a"))
        )
    with pytest.raises(BeamContractError, match="<="):
        validate_beam_batch_request(_request(output_options=_options(num_return_sequences=3)))


def test_request_beam_width_is_capped_by_the_candidate_ceiling() -> None:
    """A direct request cannot ask for a width the config layer rejects.

    ``MAX_BEAM_WIDTH`` (1024) is deliberately higher than
    ``MAX_CANDIDATE_WIDTH`` (256) so the wide-``W`` operators can be exercised,
    but this build still samples one candidate per beam row.  A request above
    the candidate ceiling therefore describes a K no prewarmed workspace can
    hold; accepting it only defers the failure to the first workspace lookup.
    """
    assert MAX_BEAM_WIDTH > MAX_CANDIDATE_WIDTH

    validate_beam_batch_request(_request(beam_params=_params(beam_width=MAX_CANDIDATE_WIDTH)))
    with pytest.raises(BeamContractError, match="must be <= 256"):
        validate_beam_batch_request(
            _request(beam_params=_params(beam_width=MAX_CANDIDATE_WIDTH + 1))
        )
    with pytest.raises(BeamContractError, match="must be <= 256"):
        validate_beam_batch_request(_request(beam_params=_params(beam_width=512)))


def test_explicit_candidate_width_must_match_the_beam_width() -> None:
    """An internal request cannot carry a K the build will not execute.

    ``GRConfig`` rejects ``candidate_width != beam_max_width``, but a caller
    that constructs ``BeamBatchRequest`` directly never passes through that
    validation.  The executed top-k comes from the active stage width, so an
    unchecked K here is accepted and then silently run as W.
    """
    with pytest.raises(BeamContractError, match="must equal beam_params.beam_width"):
        validate_beam_batch_request(_request(beam_params=_params(candidate_width=1)))
    with pytest.raises(BeamContractError, match="must equal beam_params.beam_width"):
        validate_beam_batch_request(_request(beam_params=_params(beam_width=8, candidate_width=4)))

    # Restating today's behaviour (K == W) and leaving K unset both pass.
    validate_beam_batch_request(_request(beam_params=_params(candidate_width=2)))
    validate_beam_batch_request(_request(beam_params=_params(candidate_width=None)))


def test_explicit_candidate_width_must_match_every_dynamic_stage_width() -> None:
    """A stated K is only honest when every stage retains exactly K beams.

    ``BeamBatchExecutionView.candidate_top_k`` still returns the active stage
    width in this build, so a *uniform but narrower* schedule silently samples
    fewer candidates per row than the request states: K=4 with ``(2, 2, 2)``
    runs TopK=2 at every stage and no error is raised.  Non-uniform schedules
    fail the same way, one stage at a time.
    """
    for schedule in ((2, 2, 2), (4, 2, 1)):
        with pytest.raises(BeamContractError, match="does not match"):
            validate_beam_batch_request(
                _request(
                    beam_params=_params(
                        beam_width=4,
                        candidate_width=4,
                        dynamic_beam_widths=schedule,
                    ),
                    output_options=_options(num_return_sequences=1),
                )
            )
    # A schedule whose every stage retains exactly K still works.
    validate_beam_batch_request(
        _request(
            beam_params=_params(beam_width=4, candidate_width=4, dynamic_beam_widths=(4, 4, 4)),
            output_options=_options(num_return_sequences=4),
        )
    )
    # And dynamic schedules keep their original behaviour with K unset.
    validate_beam_batch_request(
        _request(
            beam_params=_params(beam_width=4, dynamic_beam_widths=(4, 2, 1)),
            output_options=_options(num_return_sequences=1),
        )
    )


def test_limits_are_separate_from_protocol_and_check_batch_geometry() -> None:
    request = _request(beam_params=_params(beam_width=256, max_tokens=5))
    validate_beam_batch_request(request)
    with pytest.raises(BeamContractError, match="max_beam_width"):
        validate_beam_batch_request_limits(request, _limits(max_beam_width=128), max_num_seqs=8)
    with pytest.raises(BeamContractError, match="max_num_seqs"):
        validate_beam_batch_request_limits(_request(), _limits(), max_num_seqs=1)
    with pytest.raises(BeamContractError, match=r"B \* W"):
        validate_beam_batch_request_limits(_request(), _limits(max_num_beam_rows=3), max_num_seqs=8)


def test_limits_are_non_wire_values_and_fail_closed() -> None:
    limits = _limits()
    assert not isinstance(limits, msgspec.Struct)
    validate_beam_batch_limits(limits)
    with pytest.raises(BeamContractError, match=f"max_beam_width must be <= {MAX_BEAM_WIDTH}"):
        validate_beam_batch_limits(replace(limits, max_beam_width=MAX_BEAM_WIDTH + 1))
    with pytest.raises(BeamContractError, match="positive"):
        validate_beam_batch_limits(_limits(max_num_batched_tokens=0))


def test_result_is_output_xor_failure_and_compact_shortfall_is_normal() -> None:
    request = _request()
    empty_output = BeamBatchOutput(
        tuple(
            BeamRequestOutput(item.item_index, item.request.request_id, ())
            for item in request.items
        )
    )
    validate_beam_batch_result(BeamBatchResult(1, request.batch_id, empty_output, None))

    both = BeamBatchResult(
        1,
        request.batch_id,
        empty_output,
        BeamBatchFailure("MODEL_FAILURE", "failed", None),
    )
    with pytest.raises(BeamContractError, match="exactly one"):
        validate_beam_batch_result(both)


def test_terminal_result_honors_score_and_lineage_options() -> None:
    request = _request(
        beam_params=_params(max_tokens=3),
        output_options=_options(
            num_return_sequences=1,
            include_stop_token=False,
            include_cumulative_logprob=False,
            include_parent_beam_indices=True,
        ),
    )
    sequence = BeamOutputSequence((10, 20, 30), None, (0, 0, 0), "length", None)
    output = BeamBatchOutput(
        tuple(
            BeamRequestOutput(item.item_index, item.request.request_id, (sequence,))
            for item in request.items
        )
    )
    _terminal_result(
        request,
        SimpleNamespace(
            kv_transfer_params={
                "gr_batch_result": BeamBatchResult(1, request.batch_id, output, None)
            }
        ),
    )


@pytest.mark.parametrize("wire", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "widths,parents,accepted",
    [
        ((4, 2, 1), (0, 3, 1), True),
        ((1, 2, 4), (0, 0, 1), True),
        ((4, 2, 1), (1, 3, 1), False),
        ((1, 2, 4), (0, 1, 1), False),
        ((4, 2, 1), (0, 3, 2), False),
        ((1, 2, 4), (0, 0, 2), False),
        ((4,), (0,), True),
        ((4,), (1,), False),
        ((1, 4), (0, 0), True),
        ((1, 4), (0, 1), False),
        (None, (1, 3, 2), True),
        (None, (0, 3, 4), False),
    ],
)
def test_terminal_parent_indices_follow_retained_stage_widths(
    wire: bool,
    widths: tuple[int, ...] | None,
    parents: tuple[int, ...],
    accepted: bool,
) -> None:
    request = _request(
        beam_params=_params(
            beam_width=4,
            max_tokens=len(parents),
            dynamic_beam_widths=widths,
            ignore_stop_tokens=widths is not None,
        ),
        output_options=_options(num_return_sequences=1),
    )
    validate_beam_batch_request(request)
    result = BeamBatchResult(
        request.protocol_version,
        request.batch_id,
        BeamBatchOutput(
            tuple(
                BeamRequestOutput(
                    item.item_index,
                    item.request.request_id,
                    (
                        BeamOutputSequence(
                            tuple(range(10, 10 + len(parents))),
                            -0.5,
                            # A valid first item must not hide invalid lineage in
                            # another request in the same logical batch.
                            parents if item.item_index else (0,) * len(parents),
                            "length",
                            None,
                        ),
                    ),
                )
                for item in request.items
            )
        ),
        None,
    )
    payload = MsgpackDecoder().decode(MsgpackEncoder().encode(result)) if wire else result
    output = SimpleNamespace(kv_transfer_params={"gr_batch_result": payload})
    if accepted:
        assert _terminal_result(request, output) == result
    else:
        with pytest.raises(ValueError, match="invalid parent indices"):
            _terminal_result(request, output)


def test_result_rejects_invalid_finish_reason_and_bad_length_output() -> None:
    request = _request()
    invalid = BeamOutputSequence((1,), -1.0, (0,), "other", None)
    output = BeamBatchOutput(
        (BeamRequestOutput(0, "request-0", (invalid,)),) + _success(request).output.items[1:]
    )
    with pytest.raises(BeamContractError, match="finish_reason"):
        validate_beam_batch_result(BeamBatchResult(1, request.batch_id, output, None))

    short = BeamOutputSequence((1,), -1.0, (0,), "length", None)
    output = BeamBatchOutput(
        (BeamRequestOutput(0, "request-0", (short,)),) + _success(request).output.items[1:]
    )
    with pytest.raises(ValueError, match="effective budget"):
        _terminal_result(
            request,
            SimpleNamespace(
                kv_transfer_params={
                    "gr_batch_result": BeamBatchResult(1, request.batch_id, output, None)
                }
            ),
        )


@pytest.mark.parametrize("wire", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "batch_size,item_index,accepted",
    [
        (1, None, True),
        (1, 0, True),
        (1, -1, False),
        (1, 1, False),
        (2, None, True),
        (2, 0, True),
        (2, 1, True),
        (2, -1, False),
        (2, 2, False),
        (4, None, True),
        (4, 0, True),
        (4, 1, True),
        (4, 2, True),
        (4, 3, True),
        (4, -1, False),
        (4, 4, False),
    ],
)
def test_failure_item_index_is_context_checked(
    wire: bool, batch_size: int, item_index: int | None, accepted: bool
) -> None:
    request = _request(
        items=tuple(
            BeamBatchItem(index, _engine_request(f"request-{index}", [1, 2, 3]))
            for index in range(batch_size)
        )
    )
    result = BeamBatchResult(
        1,
        request.batch_id,
        None,
        BeamBatchFailure("no_valid_beams", "No valid final Beam candidates", item_index),
    )
    payload = MsgpackDecoder().decode(MsgpackEncoder().encode(result)) if wire else result
    output = SimpleNamespace(kv_transfer_params={"gr_batch_result": payload})
    if accepted:
        assert _terminal_result(request, output) == result
    else:
        with pytest.raises(ValueError, match="item identity"):
            _terminal_result(request, output)


def test_production_codec_round_trips_request_and_terminal_result() -> None:
    request = _request()
    result = _success(request)
    frames = MsgpackEncoder().encode(request)
    decoded = build_engine_core_decoders(None).submit_gr_session.decode(frames)
    assert decoded == request
    validate_beam_batch_request(decoded)
    wire_result = MsgpackDecoder().decode(MsgpackEncoder().encode(result))
    assert (
        _terminal_result(
            request, SimpleNamespace(kv_transfer_params={"gr_batch_result": wire_result})
        )
        == result
    )


def test_production_codec_preserves_nested_engine_request_tensor_frames() -> None:
    engine_request = _engine_request("request-0", None)
    engine_request.prompt_embeds = torch.arange(256, dtype=torch.float32).reshape(2, 128)
    request = _request(items=(BeamBatchItem(0, engine_request),))
    decoded = build_engine_core_decoders(None).submit_gr_session.decode(
        MsgpackEncoder().encode(request)
    )
    torch.testing.assert_close(decoded.items[0].request.prompt_embeds, engine_request.prompt_embeds)


def test_wire_decode_and_request_validation_have_distinct_boundaries() -> None:
    decoder = build_engine_core_decoders(None).submit_gr_session
    frames = MsgpackEncoder().encode(_request())
    with pytest.raises(msgspec.DecodeError):
        decoder.decode([bytes(frames[0])[:4]])
    raw = msgspec.msgpack.decode(frames[0])
    raw[0] = 99
    decoded = decoder.decode([msgspec.msgpack.encode(raw)])
    with pytest.raises(BeamContractError, match="protocol_version"):
        validate_beam_batch_request(decoded)
