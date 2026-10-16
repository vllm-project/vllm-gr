# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import replace

import msgspec
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.engine import EngineCoreRequest

import vllm_gr.v1.beam as beam_package
import vllm_gr.v1.beam.batch_contracts as contracts
from vllm_gr.config import BeamCandidateMode
from vllm_gr.v1.beam.batch_codec import BeamBatchCodec, BeamBatchCodecError
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchItem,
    BeamBatchLimits,
    BeamBatchOutput,
    BeamBatchParams,
    BeamBatchPlan,
    BeamBatchRequest,
    BeamBatchResult,
    BeamContractError,
    BeamOutputOptions,
    BeamOutputSequence,
    BeamPrefillData,
    BeamRequestOutput,
    beam_batch_protocol_version,
    validate_beam_batch_limits,
    validate_beam_batch_plan,
    validate_beam_batch_plan_limits,
    validate_beam_batch_plan_scheduler_output,
    validate_beam_batch_request,
    validate_beam_batch_request_limits,
    validate_beam_batch_result,
)


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
    # vLLM 0.22.1 has no eos_token_id field. Keep local collection useful when
    # an older developer environment still requires it.
    if "eos_token_id" in EngineCoreRequest.__struct_fields__:
        values["eos_token_id"] = None
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


def _plan(request: BeamBatchRequest | None = None, **overrides: object) -> BeamBatchPlan:
    request = request or _request()
    values: dict[str, object] = {
        "protocol_version": request.protocol_version,
        "batch_id": request.batch_id,
        "items": (
            BeamPrefillData(0, "request-0", 3, 1, 2),
            BeamPrefillData(1, "request-1", 2, 0, 2),
        ),
        "beam_params": request.beam_params,
        "output_options": request.output_options,
    }
    values.update(overrides)
    return BeamBatchPlan(**values)


def _success(plan: BeamBatchPlan | None = None) -> BeamBatchResult:
    plan = plan or _plan()
    items = tuple(
        BeamRequestOutput(
            item_index=item.item_index,
            request_id=item.request_id,
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
        for item in plan.items
    )
    return BeamBatchResult(plan.protocol_version, plan.batch_id, BeamBatchOutput(items), None)


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


def _scheduler_output(plan: BeamBatchPlan) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [
        NewRequestData(
            req_id=item.request_id,
            prompt_token_ids=[0] * item.num_prompt_tokens,
            mm_features=[],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_ids=([],),
            num_computed_tokens=item.num_computed_tokens,
            lora_request=None,
        )
        for item in plan.items
    ]
    output.num_scheduled_tokens = {
        item.request_id: item.num_scheduled_tokens for item in plan.items
    }
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    return output


def test_public_inventory_has_three_top_level_messages_and_no_old_abi() -> None:
    assert {BeamBatchRequest.__name__, BeamBatchPlan.__name__, BeamBatchResult.__name__} == {
        "BeamBatchRequest",
        "BeamBatchPlan",
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


def test_request_plan_and_result_validate_in_context() -> None:
    request = _request()
    plan = _plan(request)
    result = _success(plan)
    validate_beam_batch_request(request)
    validate_beam_batch_plan(plan, expected_request=request)
    validate_beam_batch_result(result, expected_plan=plan)


def test_full_vocab_wire_round_trip_preserves_mode_and_absent_resource() -> None:
    params = _params(
        temperature=0.0,
        candidate_mode=BeamCandidateMode.FULL_VOCAB,
        constraint_resource_id=None,
    )
    request = _request(protocol_version=beam_batch_protocol_version(params), beam_params=params)
    codec = BeamBatchCodec()
    decoded = codec.decode_request(codec.encode_request(request))
    assert decoded.protocol_version == 2
    assert decoded.beam_params == params
    assert decoded.beam_params.candidate_mode is BeamCandidateMode.FULL_VOCAB
    plan = _plan(decoded)
    decoded_plan = codec.decode_plan(codec.encode_plan(plan))
    validate_beam_batch_plan(decoded_plan, expected_request=decoded)
    result = _success(plan)
    validate_beam_batch_result(codec.decode_result(codec.encode_result(result)), expected_plan=plan)
    validate_beam_batch_request_limits(
        decoded, _limits(supported_constraint_resource_ids=(None,)), max_num_seqs=4
    )
    with pytest.raises(BeamContractError, match="not supported by this runtime"):
        validate_beam_batch_request_limits(decoded, _limits(), max_num_seqs=4)


def test_previous_beam_params_wire_defaults_to_constraint_table() -> None:
    # v1 messages from before candidate_mode was appended have eight fields.
    raw = msgspec.msgpack.encode([2, 3, 0.0, [2], True, "default", 1.0, None])
    params = msgspec.msgpack.decode(raw, type=BeamBatchParams)
    assert params.candidate_mode is BeamCandidateMode.CONSTRAINT_TABLE
    assert params.constraint_resource_id == "default"
    assert beam_batch_protocol_version(params) == BEAM_BATCH_PROTOCOL_VERSION


@pytest.mark.parametrize(  # type: ignore[misc]
    "mode,resource_id",
    [("full_vocab", "default"), ("constraint_table", None), ("unknown", None)],
)
def test_candidate_source_rejects_conflicting_mode_and_resource(
    mode: str, resource_id: str | None
) -> None:
    params = _params(candidate_mode=mode, constraint_resource_id=resource_id)
    with pytest.raises(BeamContractError, match="candidate_mode"):
        validate_beam_batch_request(_request(beam_params=params))


@pytest.mark.parametrize(  # type: ignore[misc]
    "mode,resource_id,version",
    [("full_vocab", None, 1), ("constraint_table", "default", 2)],
)
def test_candidate_source_requires_its_wire_version(
    mode: str, resource_id: str | None, version: int
) -> None:
    params = _params(candidate_mode=mode, constraint_resource_id=resource_id)
    with pytest.raises(BeamContractError, match="protocol_version"):
        validate_beam_batch_request(_request(protocol_version=version, beam_params=params))


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


@pytest.mark.parametrize("bad_value", [0, -1, True, 257])  # type: ignore[misc]
def test_request_positive_geometry_is_fail_closed(bad_value: object) -> None:
    with pytest.raises(BeamContractError, match="beam_width"):
        validate_beam_batch_request(_request(beam_params=_params(beam_width=bad_value)))


def test_request_rejects_non_default_resource_duplicate_stops_and_wide_output() -> None:
    with pytest.raises(BeamContractError, match="unique"):
        validate_beam_batch_request(_request(beam_params=_params(stop_token_ids=(2, 2))))
    with pytest.raises(BeamContractError, match="constraint_resource_id"):
        validate_beam_batch_request(
            _request(beam_params=_params(constraint_resource_id="catalog-a"))
        )
    with pytest.raises(BeamContractError, match="<="):
        validate_beam_batch_request(_request(output_options=_options(num_return_sequences=3)))


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
    with pytest.raises(BeamContractError, match="max_beam_width must be <= 256"):
        validate_beam_batch_limits(replace(limits, max_beam_width=257))
    with pytest.raises(BeamContractError, match="positive"):
        validate_beam_batch_limits(_limits(max_num_batched_tokens=0))


def test_plan_requires_full_prefill_and_exact_request_items() -> None:
    request = _request()
    incomplete = _plan(items=(BeamPrefillData(0, "request-0", 3, 1, 1),))
    with pytest.raises(BeamContractError, match="full Prefill"):
        validate_beam_batch_plan(incomplete)

    missing = _plan(items=(BeamPrefillData(0, "request-0", 3, 1, 2),))
    with pytest.raises(BeamContractError, match="exactly one item"):
        validate_beam_batch_plan(missing, expected_request=request)


def test_plan_prompt_count_is_not_derived_from_request_tokens() -> None:
    request = _request(
        items=(BeamBatchItem(0, _engine_request("request-0", [1])),),
        output_options=_options(num_return_sequences=1),
    )
    plan = _plan(
        request,
        items=(BeamPrefillData(0, "request-0", 7, 3, 4),),
        output_options=request.output_options,
    )
    validate_beam_batch_plan(plan, expected_request=request)


def test_plan_cross_checks_native_scheduler_output() -> None:
    plan = _plan()
    scheduler_output = _scheduler_output(plan)
    validate_beam_batch_plan_scheduler_output(plan, scheduler_output)

    scheduler_output.num_scheduled_tokens["request-0"] = 1
    with pytest.raises(BeamContractError, match="num_scheduled_tokens"):
        validate_beam_batch_plan_scheduler_output(plan, scheduler_output)


def test_plan_limits_use_scheduled_budget_and_authoritative_prompt_length() -> None:
    plan = _plan(
        items=(
            BeamPrefillData(0, "request-0", 100, 98, 2),
            BeamPrefillData(1, "request-1", 200, 198, 2),
        )
    )
    validate_beam_batch_plan_limits(plan, _limits(max_num_batched_tokens=4), max_num_seqs=8)
    with pytest.raises(BeamContractError, match="max_num_batched_tokens"):
        validate_beam_batch_plan_limits(plan, _limits(max_num_batched_tokens=3), max_num_seqs=8)
    with pytest.raises(BeamContractError, match="max_model_len"):
        validate_beam_batch_plan_limits(plan, _limits(max_model_len=202), max_num_seqs=8)


def test_result_is_output_xor_failure_and_compact_shortfall_is_normal() -> None:
    plan = _plan()
    empty_output = BeamBatchOutput(
        tuple(BeamRequestOutput(item.item_index, item.request_id, ()) for item in plan.items)
    )
    validate_beam_batch_result(
        BeamBatchResult(1, plan.batch_id, empty_output, None), expected_plan=plan
    )

    both = BeamBatchResult(
        1,
        plan.batch_id,
        empty_output,
        BeamBatchFailure("MODEL_FAILURE", "failed", None),
    )
    with pytest.raises(BeamContractError, match="exactly one"):
        validate_beam_batch_result(both)


def test_result_options_control_score_lineage_and_stop_materialization() -> None:
    request = _request(
        beam_params=_params(max_tokens=3),
        output_options=_options(
            num_return_sequences=1,
            include_stop_token=False,
            include_cumulative_logprob=False,
            include_parent_beam_indices=True,
        ),
    )
    plan = _plan(request)
    sequence = BeamOutputSequence((), None, (0,), "stop", 2)
    output = BeamBatchOutput(
        tuple(
            BeamRequestOutput(item.item_index, item.request_id, (sequence,)) for item in plan.items
        )
    )
    validate_beam_batch_result(BeamBatchResult(1, plan.batch_id, output, None), expected_plan=plan)


def test_result_rejects_invalid_finish_reason_and_bad_length_output() -> None:
    plan = _plan()
    invalid = BeamOutputSequence((1,), -1.0, (0,), "other", None)
    output = BeamBatchOutput(
        (BeamRequestOutput(0, "request-0", (invalid,)),) + _success(plan).output.items[1:]
    )
    with pytest.raises(BeamContractError, match="finish_reason"):
        validate_beam_batch_result(BeamBatchResult(1, plan.batch_id, output, None))

    short = BeamOutputSequence((1,), -1.0, (0,), "length", None)
    output = BeamBatchOutput(
        (BeamRequestOutput(0, "request-0", (short,)),) + _success(plan).output.items[1:]
    )
    with pytest.raises(BeamContractError, match="max_tokens"):
        validate_beam_batch_result(
            BeamBatchResult(1, plan.batch_id, output, None), expected_plan=plan
        )


def test_failure_item_index_is_context_checked() -> None:
    plan = _plan()
    result = BeamBatchResult(
        1,
        plan.batch_id,
        None,
        BeamBatchFailure("UNSUPPORTED_INPUT", "unsupported modality", 2),
    )
    with pytest.raises(BeamContractError, match="outside"):
        validate_beam_batch_result(result, expected_plan=plan)


def test_typed_codec_round_trips_all_top_level_messages() -> None:
    codec = BeamBatchCodec()
    request = _request()
    plan = _plan(request)
    result = _success(plan)
    assert codec.decode_request(codec.encode_request(request)) == request
    assert codec.decode_plan(codec.encode_plan(plan)) == plan
    assert codec.decode_result(codec.encode_result(result)) == result


def test_typed_codec_preserves_nested_engine_request_tensor_frames() -> None:
    codec = BeamBatchCodec()
    engine_request = _engine_request("request-0", None)
    engine_request.prompt_embeds = torch.arange(256, dtype=torch.float32).reshape(2, 128)
    request = _request(items=(BeamBatchItem(0, engine_request),))
    decoded = codec.decode_request(codec.encode_request(request))
    torch.testing.assert_close(decoded.items[0].request.prompt_embeds, engine_request.prompt_embeds)


def test_codec_error_and_contract_error_have_distinct_boundaries() -> None:
    codec = BeamBatchCodec()
    frames = codec.encode_request(_request())
    with pytest.raises(BeamBatchCodecError):
        codec.decode_request([bytes(frames[0])[:4]])

    raw = msgspec.msgpack.decode(frames[0])
    raw[0] = 99
    with pytest.raises(BeamContractError, match="protocol_version"):
        codec.decode_request([msgspec.msgpack.encode(raw)])
