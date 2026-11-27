# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""A refused request has to reach its caller as the caller's own mistake.

The engine answers a request it will not run -- an ordinary one arriving while
a beam session owns it, say -- with a bare ``FinishReason.ERROR`` output, and
every serving loop turns that into an internal server error. The reason rides
along in ``kv_transfer_params``. These tests pin both halves of recovering it:
the frontend output processor parks the reason on its way past, and the shared
generation-error check raises it as a client error instead of a server fault.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections import defaultdict
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import pytest
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.exceptions import GenerationError, VLLMValidationError
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor

from vllm_gr.entrypoints.openai.beam_search_patch import (
    _validate_client_request_id_patch,
    patch_admission_rejection_reporting,
    patch_client_request_id,
)
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    GR_BATCH_RESULT_PARAM,
    BeamBatchFailure,
    BeamBatchResult,
    admission_rejected,
    admission_rejection,
)
from vllm_gr.v1.engine import output_processor
from vllm_gr.v1.engine.output_processor import (
    make_patched_process_outputs,
    note_admission_rejections,
    take_admission_rejection,
)

REASON = "Asynchronous scheduling requests cannot mix with ordinary or legacy Beam requests"
# A second refusal, to tell two parked reasons apart.
OTHER_REASON = "That is a different refusal"
# Any code but the refusal: a broken engine or Worker, which stays a 500.
ENGINE_FAULT = BeamBatchResult(
    BEAM_BATCH_PROTOCOL_VERSION,
    "cmpl-1",
    None,
    BeamBatchFailure("worker_protocol_error", "worker died", None),
)


def _record_outputs(self: Any, outputs: Any, *args: Any, **kwargs: Any) -> list[Any]:
    """Stand-in for upstream's unbound ``OutputProcessor.process_outputs``."""
    del args, kwargs
    self.forwarded.append(list(outputs))
    return []


class _Processor:
    """The little of ``OutputProcessor`` the parked reason lives on."""

    def __init__(self) -> None:
        self.request_states: dict[str, Any] = {}
        self.parent_requests: dict[str, Any] = {}
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
        self.tokenizer = None
        self.log_stats = False
        self.stream_interval = 1
        self.forwarded: list[list[Any]] = []
        # A plain function, because the patch reaches the original the way
        # upstream's unbound method lookup does: ``original(self, outputs)``.
        # A bound method here would receive the processor twice.
        self.process_outputs = _record_outputs

    def _update_admission_stats(self) -> None:
        pass


def _engine_request(request_id: str) -> Any:
    return SimpleNamespace(
        request_id=request_id,
        external_req_id=request_id,
        sampling_params=None,
        pooling_params=SimpleNamespace(output_kind=0),
        lora_request=None,
        prompt_token_ids=[1],
        prompt_embeds=None,
        arrival_time=0.0,
        resumable=False,
    )


def error_output(request_id: str, params: dict[str, Any] | None = None) -> EngineCoreOutput:
    """The terminal output a refused request ends on."""
    output = EngineCoreOutput(request_id, [], finish_reason=FinishReason.ERROR)
    output.kv_transfer_params = params
    return output


def carrying(payload: Any, request_id: str = "cmpl-1") -> EngineCoreOutput:
    return error_output(request_id, {GR_BATCH_RESULT_PARAM: payload})


def refusal(request_id: str = "cmpl-1") -> EngineCoreOutput:
    return carrying(admission_rejected(request_id, REASON), request_id)


def other_refusal(request_id: str) -> EngineCoreOutput:
    return carrying(admission_rejected(request_id, OTHER_REASON), request_id)


def client(processor: _Processor | None) -> Any:
    """A stand-in for the frontend engine client the reason is parked on."""
    return None if processor is None else SimpleNamespace(output_processor=processor)


def serving(processor: _Processor | None) -> Any:
    """A stand-in for ``GenerateBaseServing`` holding only what the patch reads."""
    return SimpleNamespace(engine_client=client(processor))


def finished_output(request_id: str) -> EngineCoreOutput:
    return EngineCoreOutput(request_id, [], finish_reason=FinishReason.STOP)


def submit_request(
    processor: _Processor, base_id: str, *engine_ids: str
) -> output_processor._EngineRequestIds:
    """Register engine sub-requests under a fresh caller request tracker."""
    patch_client_request_id()
    token = output_processor._ENGINE_REQUEST_IDS.set(None)
    try:
        BaseServing._base_request_id(None, base_id)
        for rid in engine_ids:
            OutputProcessor.add_request(processor, _engine_request(rid), "")
        ids = output_processor._ENGINE_REQUEST_IDS.get()
        assert ids is not None
        return ids
    finally:
        output_processor._ENGINE_REQUEST_IDS.reset(token)


def _tracker(*engine_ids: str) -> output_processor._EngineRequestIds:
    tracker = output_processor.start_client_request_ids()
    tracker.extend(engine_ids)
    return tracker


def raise_if_error(
    processor: _Processor | None,
    request_id: str = "cmpl-1",
    engine_request_ids: output_processor._EngineRequestIds | None = None,
) -> None:
    """Run the serving layer's shared error check, patch and all."""
    patch_admission_rejection_reporting()
    # The frontend sets this while it handles a request; each request is its
    # own task in production, so the test puts it back itself.
    token = output_processor._ENGINE_REQUEST_IDS.set(engine_request_ids)
    try:
        # Read through the class so the test exercises whatever patch is
        # installed.
        GenerateBaseServing._raise_if_error(serving(processor), "error", request_id)
    finally:
        output_processor._ENGINE_REQUEST_IDS.reset(token)


def test_ordinary_request_refusal_is_reported_as_a_client_error() -> None:
    processor = _Processor()
    note_admission_rejections(processor, [refusal()])

    with pytest.raises(VLLMValidationError) as caught:
        raise_if_error(processor)

    assert str(caught.value) == REASON


def test_a_refusal_is_reported_once() -> None:
    processor = _Processor()
    note_admission_rejections(processor, [refusal()])

    with pytest.raises(VLLMValidationError):
        raise_if_error(processor)

    # The reason described one request. A later error for the same id is an
    # engine fault again, not the same client mistake replayed.
    with pytest.raises(GenerationError):
        raise_if_error(processor)


def test_every_other_engine_error_stays_a_server_error() -> None:
    processor = _Processor()
    note_admission_rejections(processor, [error_output("cmpl-1")])

    with pytest.raises(GenerationError):
        raise_if_error(processor)


def test_an_engine_client_without_an_output_processor_is_unaffected() -> None:
    with pytest.raises(GenerationError):
        raise_if_error(None)


@pytest.mark.parametrize(  # type: ignore[misc]
    "output",
    [
        pytest.param(error_output("cmpl-1"), id="no-kv-transfer-params"),
        pytest.param(error_output("cmpl-1", {}), id="no-terminal-result"),
        pytest.param(carrying(None), id="null-terminal-result"),
        pytest.param(carrying(ENGINE_FAULT), id="another-failure-code"),
        pytest.param(
            carrying(msgspec.structs.astuple(ENGINE_FAULT)), id="another-code-on-the-wire"
        ),
        pytest.param(carrying("not-a-result"), id="not-a-terminal-result"),
        pytest.param(carrying([1, 2, 3]), id="unparsable-wire-shape"),
    ],
)
def test_only_a_refusal_carries_a_reason(output: EngineCoreOutput) -> None:
    assert admission_rejection(output) is None


def test_a_refusal_carries_its_reason_in_process() -> None:
    assert admission_rejection(refusal()) == REASON


def test_a_refusal_carries_its_reason_across_the_wire() -> None:
    """``kv_transfer_params`` is untyped IPC, so the Struct arrives as a list."""
    wire = msgspec.structs.astuple(admission_rejected("cmpl-1", REASON))

    assert admission_rejection(carrying(wire)) == REASON


def test_the_output_processor_parks_the_reason_and_forwards_the_output() -> None:
    processor = _Processor()
    wrapped = make_patched_process_outputs(processor.process_outputs)
    refused = refusal()
    other = error_output("cmpl-2")

    wrapped(processor, [refused, other])

    # Parking must not change what upstream processing sees.
    assert [id(out) for out in processor.forwarded[0]] == [id(refused), id(other)]
    assert take_admission_rejection(client(processor), "cmpl-1") == REASON
    assert take_admission_rejection(client(processor), "cmpl-2") is None


def test_parking_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(output_processor, "_ADMISSION_REJECTION_MAX", 2)
    processor = _Processor()

    note_admission_rejections(processor, [refusal(f"cmpl-{i}") for i in range(4)])

    parked = getattr(processor, output_processor._ADMISSION_REJECTIONS_ATTR)
    assert list(parked) == ["cmpl-2", "cmpl-3"]


def test_an_id_that_extends_another_does_not_claim_its_refusal() -> None:
    """A caller names its own request, so one id may extend another's."""
    processor = _Processor()
    note_admission_rejections(processor, [other_refusal("cmpl-1-child-0-884f2ecb")])

    with pytest.raises(GenerationError):
        raise_if_error(processor, "cmpl-1")


def test_a_refusal_a_request_never_claims_is_not_reused() -> None:
    """A refused V1 session reports off its own payload and claims nothing."""
    processor = _Processor()
    note_admission_rejections(processor, [refusal("cmpl-1-gr-beam-v1-9a1f0c")])

    with pytest.raises(GenerationError):
        raise_if_error(processor, "cmpl-1")


def test_a_finished_request_leaves_no_refusal_for_the_next_one() -> None:
    """An id the caller reuses must not inherit the first request's refusal."""
    processor = _Processor()
    note_admission_rejections(processor, [refusal("cmpl-1")])
    note_admission_rejections(processor, [error_output("cmpl-1")])

    with pytest.raises(GenerationError):
        raise_if_error(processor, "cmpl-1")


def test_a_mapped_refusal_is_consumed_once_under_either_identity() -> None:
    """The two exact identities refer to one consumable rejection."""
    processor = _Processor()
    ids = submit_request(processor, "cmpl-1", "cmpl-1-8b31d4")
    note_admission_rejections(processor, [refusal("cmpl-1-8b31d4")])

    with pytest.raises(VLLMValidationError) as caught:
        raise_if_error(processor, "cmpl-1", ids)

    assert str(caught.value) == REASON
    assert take_admission_rejection(client(processor), "cmpl-1-8b31d4") is None
    assert take_admission_rejection(client(processor), "cmpl-1", ids) is None


def test_an_engine_request_does_not_answer_for_another_request() -> None:
    """The engine's names are looked up exactly; none is inferred from another."""
    processor = _Processor()
    ids = submit_request(processor, "cmpl-1", "cmpl-1-0-884f2ecb")
    note_admission_rejections(processor, [refusal("cmpl-1-child-0-884f2ecb")])

    with pytest.raises(GenerationError):
        raise_if_error(processor, "cmpl-1", ids)


def test_a_missing_mapped_refusal_does_not_consume_another_identity() -> None:
    processor = _Processor()
    ids = submit_request(processor, "missing", "cmpl-1-8b31d4")
    note_admission_rejections(processor, [refusal("cmpl-1")])

    assert take_admission_rejection(client(processor), "missing", ids) is None
    # The reason filed under the caller's own id is left where it was.
    assert take_admission_rejection(client(processor), "cmpl-1") == REASON


def test_a_refusal_of_any_engine_request_answers_for_the_request() -> None:
    """A prompt list or ``n > 1`` is several engine requests; any may be refused."""
    processor = _Processor()
    ids = submit_request(processor, "cmpl-1", "cmpl-1-0-884f2ecb", "cmpl-1-1-1a2b3c4d")
    note_admission_rejections(
        processor, [refusal("cmpl-1-0-884f2ecb"), finished_output("cmpl-1-1-1a2b3c4d")]
    )

    with pytest.raises(VLLMValidationError) as caught:
        raise_if_error(processor, "cmpl-1", ids)

    assert str(caught.value) == REASON


def test_a_refused_sibling_does_not_answer_for_an_engine_fault() -> None:
    processor = _Processor()
    ids = submit_request(processor, "cmpl-1", "cmpl-1-0-884f2ecb", "cmpl-1-1-1a2b3c4d")
    note_admission_rejections(
        processor, [refusal("cmpl-1-0-884f2ecb"), error_output("cmpl-1-1-1a2b3c4d")]
    )

    with pytest.raises(GenerationError):
        raise_if_error(processor, "cmpl-1", ids)
    assert not getattr(processor, output_processor._ADMISSION_REJECTIONS_ATTR, None)


def test_sibling_eviction_does_not_mask_an_in_flight_engine_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(output_processor, "_ADMISSION_REJECTION_MAX", 2)
    patch_admission_rejection_reporting()

    # In-flight OpenAI requests keep their fault classification outside the bounded pool.
    in_flight_processor = _Processor()
    ctx = contextvars.Context()

    def run_in_flight() -> None:
        BaseServing._base_request_id(None, "cmpl-1")
        ids = ["worker", "refused-0", "refused-1", "refused-2"]
        for rid in ids:
            OutputProcessor.add_request(in_flight_processor, _engine_request(rid), "")
        note_admission_rejections(
            in_flight_processor,
            [error_output("worker"), *(refusal(f"refused-{i}") for i in range(3))],
        )
        GenerateBaseServing._raise_if_error(serving(in_flight_processor), "error", "cmpl-1")

    with pytest.raises(GenerationError):
        ctx.run(run_in_flight)


def test_other_callers_refusals_do_not_evict_an_engine_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(output_processor, "_ADMISSION_REJECTION_MAX", 2)
    patch_admission_rejection_reporting()

    in_flight_processor = _Processor()
    ctx = contextvars.Context()

    def submit_and_fail() -> None:
        BaseServing._base_request_id(None, "cmpl-1")
        for rid in ("worker", "refused"):
            OutputProcessor.add_request(in_flight_processor, _engine_request(rid), "")
        note_admission_rejections(in_flight_processor, [error_output("worker")])
        note_admission_rejections(in_flight_processor, [refusal(f"other-{i}") for i in range(3)])
        note_admission_rejections(in_flight_processor, [refusal("refused")])
        GenerateBaseServing._raise_if_error(serving(in_flight_processor), "error", "cmpl-1")

    with pytest.raises(GenerationError):
        ctx.run(submit_and_fail)


def test_a_stream_keeps_the_engine_requests_of_the_task_that_made_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stream is iterated outside the handler task that started the list."""

    async def stream(self: Any) -> AsyncIterator[output_processor._EngineRequestIds | None]:
        yield output_processor._ENGINE_REQUEST_IDS.get()

    monkeypatch.setattr(OpenAIServingCompletion, "completion_stream_generator", stream)
    patch_client_request_id()

    async def handler() -> AsyncIterator[output_processor._EngineRequestIds | None]:
        output_processor._ENGINE_REQUEST_IDS.set(_tracker("cmpl-1-0-884f2ecb"))
        return cast(
            AsyncIterator[output_processor._EngineRequestIds | None],
            OpenAIServingCompletion.completion_stream_generator(None),
        )

    async def serve() -> list[output_processor._EngineRequestIds | None]:
        response = await asyncio.create_task(handler())
        return [chunk async for chunk in response]

    assert asyncio.run(serve()) == [["cmpl-1-0-884f2ecb"]]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("owner", "method_name"),
    [
        (OpenAIServingCompletion, "completion_stream_generator"),
        (OpenAIServingChat, "chat_completion_stream_generator"),
    ],
)
@pytest.mark.parametrize("exit_kind", ["normal", "raise", "early-close", "cancel"])  # type: ignore[misc]
def test_a_stream_restores_the_callers_context_after_exit(
    monkeypatch: pytest.MonkeyPatch,
    owner: type[Any],
    method_name: str,
    exit_kind: str,
) -> None:
    async def run() -> None:
        seen: dict[str, Any] = {}
        ready = asyncio.Event()
        hold = asyncio.Event()

        async def inner(self: Any) -> AsyncIterator[str]:
            try:
                seen["inside"] = list(output_processor._ENGINE_REQUEST_IDS.get() or [])
                yield "chunk"
                if exit_kind == "raise":
                    raise RuntimeError("engine failed")
                if exit_kind == "cancel":
                    ready.set()
                    await hold.wait()
            finally:
                seen["closed"] = True

        monkeypatch.setattr(owner, method_name, inner)
        patch_client_request_id()

        token = output_processor._ENGINE_REQUEST_IDS.set(_tracker("engine"))
        stream = getattr(owner, method_name)(None)
        output_processor._ENGINE_REQUEST_IDS.reset(token)

        async def drive() -> None:
            outer_token = output_processor._ENGINE_REQUEST_IDS.set(_tracker("outer"))
            try:
                if exit_kind == "early-close":
                    assert await anext(stream) == "chunk"
                    seen["between"] = list(output_processor._ENGINE_REQUEST_IDS.get() or [])
                    await stream.aclose()
                else:
                    try:
                        async for _ in stream:
                            seen["between"] = list(output_processor._ENGINE_REQUEST_IDS.get() or [])
                    except (RuntimeError, asyncio.CancelledError):
                        pass
                seen["after"] = list(output_processor._ENGINE_REQUEST_IDS.get() or [])
            finally:
                output_processor._ENGINE_REQUEST_IDS.reset(outer_token)

        task = asyncio.create_task(drive(), context=contextvars.Context())
        if exit_kind == "cancel":
            await ready.wait()
            task.cancel()
        await task

        assert seen["closed"] is True
        assert seen["inside"] == ["engine"]
        assert seen["between"] == ["outer"]
        assert seen["after"] == ["outer"]

    asyncio.run(run())


def test_client_request_id_validation_checks_stream_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_client_request_id()
    assert _validate_client_request_id_patch() is None

    for owner, method_name in (
        (OpenAIServingCompletion, "completion_stream_generator"),
        (OpenAIServingChat, "chat_completion_stream_generator"),
    ):
        with monkeypatch.context() as changes:
            changes.setattr(owner, method_name, lambda *args, **kwargs: None)
            detail = _validate_client_request_id_patch()
            assert detail is not None and f"{owner.__name__}.{method_name}" in detail


def test_a_refused_ordinary_request_reaches_the_caller_as_a_client_error() -> None:
    """The whole path: engine output in, the caller's own status class out."""
    processor = _Processor()
    wrapped = make_patched_process_outputs(processor.process_outputs)

    wrapped(processor, [refusal()])

    with pytest.raises(VLLMValidationError) as caught:
        raise_if_error(processor)
    assert str(caught.value) == REASON


pytestmark = pytest.mark.cpu_test
