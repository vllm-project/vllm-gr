# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the VLLM project

from __future__ import annotations

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchOutput,
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputOptions,
    BeamOutputSequence,
    BeamRequestOutput,
)
from vllm_gr.v1.engine.beam_batch import (
    BeamBatchDuplicateBatchError,
    BeamBatchIngressError,
    BeamBatchMixedModeError,
    BeamBatchRecord,
    EngineCoreBeamBatchIngress,
    build_beam_batch_request,
)


def _engine_request(request_id: str, prompt: list[int] | None) -> EngineCoreRequest:
    values: dict[str, object] = dict(
        request_id=request_id,
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=99, temperature=0.0),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    fields = EngineCoreRequest.__struct_fields__
    return EngineCoreRequest(**{k: v for k, v in values.items() if k in fields})


def _params() -> BeamBatchParams:
    return BeamBatchParams(
        beam_width=2,
        max_tokens=3,
        temperature=1.0,
        stop_token_ids=(2, 99),
        ignore_stop_tokens=False,
        constraint_resource_id="default",
    )


def _options() -> BeamOutputOptions:
    return BeamOutputOptions(
        num_return_sequences=2,
        include_stop_token=True,
        include_cumulative_logprob=True,
        include_parent_beam_indices=True,
    )


def _batch_request(batch_id: str = "batch-0") -> BeamBatchRequest:
    return build_beam_batch_request(
        [
            _engine_request("request-0", [1, 2, 3]),
            _engine_request("request-1", [4, 5]),
        ],
        beam_params=_params(),
        output_options=_options(),
        batch_id=batch_id,
    )


def _result(batch_id: str = "batch-0") -> BeamBatchResult:
    sequence = BeamOutputSequence(
        token_ids=(1, 2),
        cumulative_logprob=-1.0,
        parent_beam_indices=(0, 1),
        finish_reason="stop",
        stop_token_id=2,
    )
    return BeamBatchResult(
        protocol_version=1,
        batch_id=batch_id,
        output=BeamBatchOutput(
            items=(
                BeamRequestOutput(0, "request-0", (sequence,)),
                BeamRequestOutput(1, "request-1", ()),
            )
        ),
        failure=None,
    )


def _ingress(**kwargs: object) -> tuple[EngineCoreBeamBatchIngress, list[BeamBatchRecord]]:
    enqueued: list[BeamBatchRecord] = []
    enqueue = kwargs.pop("enqueue", lambda record: enqueued.append(record))
    ingress = EngineCoreBeamBatchIngress(enqueue=enqueue, **kwargs)
    return ingress, enqueued


def test_submit_batch_enqueues_once_and_routes_terminal() -> None:
    ingress, enqueued = _ingress(reject_mixed_requests=True)

    handle = ingress.submit_batch(_batch_request())
    assert len(enqueued) == 1
    assert ingress.active_count == 1

    result = _result()
    routed = ingress.route_terminal_result(result)
    assert routed.done()
    assert handle.result().batch_id == "batch-0"
    assert ingress.active_count == 0


def test_submit_single_builds_single_item_batch() -> None:
    ingress, enqueued = _ingress(batch_id_factory=lambda: "auto-batch-1")

    handle = ingress.submit_single(
        _engine_request("request-0", [1, 2, 3]),
        beam_params=_params(),
        output_options=_options(),
    )
    assert handle.batch_id == "auto-batch-1"
    assert len(enqueued) == 1
    assert len(ingress.get_record("auto-batch-1").request.items) == 1


def test_duplicate_batch_id_rejected() -> None:
    ingress, _ = _ingress()
    ingress.submit_batch(_batch_request())
    with pytest.raises(BeamBatchDuplicateBatchError):
        ingress.submit_batch(_batch_request())


def test_enqueue_failure_rolls_back_registry() -> None:
    def failing_enqueue(record: object) -> None:
        raise RuntimeError("enqueue failed")

    ingress, _ = _ingress(enqueue=failing_enqueue)
    with pytest.raises(BeamBatchIngressError):
        ingress.submit_batch(_batch_request())
    assert ingress.active_count == 0


def test_mixed_ordinary_request_is_rejected() -> None:
    ingress, _ = _ingress(reject_mixed_requests=True)
    ingress.set_beam_runtime_enabled(True)
    ingress.submit_batch(_batch_request())

    with pytest.raises(BeamBatchMixedModeError):
        ingress.note_ordinary_request("ordinary-0")


def test_duplicate_terminal_result_rejected() -> None:
    ingress, _ = _ingress()
    ingress.submit_batch(_batch_request())
    result = _result()

    ingress.route_terminal_result(result)
    with pytest.raises(BeamBatchDuplicateBatchError):
        ingress.route_terminal_result(result)


def test_terminal_record_is_retired_from_registry() -> None:
    ingress, _ = _ingress()
    ingress.submit_batch(_batch_request())

    ingress.route_terminal_result(_result())

    assert ingress.get_record("batch-0") is None
    assert ingress.get_batch("batch-0") is None
    assert ingress.active_count == 0


def test_completed_batch_id_cannot_be_reused() -> None:
    ingress, _ = _ingress()
    ingress.submit_batch(_batch_request("batch-0"))
    ingress.route_terminal_result(_result("batch-0"))

    with pytest.raises(BeamBatchDuplicateBatchError):
        ingress.submit_batch(_batch_request("batch-0"))


def test_cancel_retires_record_and_rejects_terminal() -> None:
    ingress, _ = _ingress()
    handle = ingress.submit_batch(_batch_request())

    assert ingress.cancel(handle) is True
    assert ingress.get_record("batch-0") is None
    assert handle.cancelled()
    with pytest.raises(BeamBatchIngressError):
        ingress.route_terminal_result(_result())


def test_fail_beam_batch_routes_failure_and_retires() -> None:
    ingress, _ = _ingress()
    ingress.submit_batch(_batch_request())

    handle = ingress.fail_beam_batch(
        "batch-0",
        code="ENGINE_ERROR",
        message="boom",
    )
    result = handle.result()
    assert result.failure is not None
    assert result.failure.code == "ENGINE_ERROR"
    assert ingress.get_record("batch-0") is None


def test_submit_batch_materializes_native_children() -> None:
    ingress, enqueued = _ingress()

    ingress.submit_batch(_batch_request())

    batch = ingress.get_batch("batch-0")
    assert batch is not None
    assert len(batch.children) == 2
    assert [child.request_id for child in batch.children] == [
        "request-0",
        "request-1",
    ]
    assert len(enqueued) == 1
    assert enqueued[0].batch is batch


def test_child_builder_failure_is_not_registered() -> None:
    def failing_builder(request: object) -> object:
        raise RuntimeError("child build failed")

    ingress = EngineCoreBeamBatchIngress(child_builder=failing_builder)

    with pytest.raises(BeamBatchIngressError):
        ingress.submit_batch(_batch_request())

    assert ingress.active_count == 0
    assert ingress.get_record("batch-0") is None


def test_beam_batch_wire_values_registered() -> None:
    from vllm.v1.engine import EngineCoreRequestType

    from vllm_gr.v1.engine.wire import register_engine_core_request_types

    register_engine_core_request_types()

    assert EngineCoreRequestType.BEAM_BATCH_SUBMIT.value == b"\x08"
    assert EngineCoreRequestType.BEAM_BATCH_ABORT.value == b"\x09"
