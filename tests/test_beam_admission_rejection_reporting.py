# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A refused request has to reach its caller as the caller's own mistake.

The engine answers a request it will not run -- an ordinary one arriving while
a beam session owns it, say -- with a bare ``FinishReason.ERROR`` output, and
every serving loop turns that into an internal server error. The reason rides
along in ``kv_transfer_params``. These tests pin both halves of recovering it:
the frontend output processor parks the reason on its way past, and the shared
generation-error check raises it as a client error instead of a server fault.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
from vllm.entrypoints.openai.engine.protocol import GenerationError
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine import EngineCoreOutput, FinishReason

import vllm_gr.v1.engine.async_llm as async_llm
from vllm_gr.entrypoints.openai.beam_search_patch import (
    patch_admission_rejection_reporting,
)
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    GR_BATCH_RESULT_PARAM,
    BeamBatchFailure,
    BeamBatchResult,
    admission_rejected,
    admission_rejection,
)
from vllm_gr.v1.engine.async_llm import (
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
        self.forwarded: list[list[Any]] = []
        # A plain function, because the patch reaches the original the way
        # upstream's unbound method lookup does: ``original(self, outputs)``.
        # A bound method here would receive the processor twice.
        self.process_outputs = _record_outputs


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
    """A stand-in for ``OpenAIServing`` holding only what the patch reads."""
    return SimpleNamespace(engine_client=client(processor))


def raise_if_error(
    processor: _Processor | None,
    request_id: str = "cmpl-1",
    client_request_id: str | None = None,
) -> None:
    """Run the serving layer's shared error check, patch and all."""
    patch_admission_rejection_reporting()
    # The frontend sets this while it handles a request; each request is its
    # own task in production, so the test puts it back itself.
    token = async_llm._CLIENT_REQUEST_ID.set(client_request_id)
    try:
        # Read through the class so the test exercises whatever patch is
        # installed.
        getattr(OpenAIServing, "_raise_if_error")(serving(processor), "error", request_id)
    finally:
        async_llm._CLIENT_REQUEST_ID.reset(token)


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
    monkeypatch.setattr(async_llm, "_ADMISSION_REJECTION_MAX", 2)
    processor = _Processor()

    note_admission_rejections(processor, [refusal(f"cmpl-{i}") for i in range(4)])

    parked = getattr(processor, async_llm._ADMISSION_REJECTIONS_ATTR)
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
    note_admission_rejections(processor, [refusal("cmpl-1-8b31d4")])

    async_llm.remember_client_request_id(processor, "caller-name", "cmpl-1-8b31d4")

    with pytest.raises(VLLMValidationError) as caught:
        raise_if_error(processor, "cmpl-1", "caller-name")

    assert str(caught.value) == REASON
    assert take_admission_rejection(client(processor), "cmpl-1-8b31d4") is None
    assert take_admission_rejection(client(processor), "cmpl-1", "caller-name") is None


def test_a_caller_name_finds_the_request_it_was_submitted_as() -> None:
    """The engine extends the id it is handed, so the two names are paired."""
    processor = _Processor()
    note_admission_rejections(processor, [refusal("cmpl-1-0-884f2ecb")])
    async_llm.remember_client_request_id(processor, "caller-name", "cmpl-1-0-884f2ecb")

    with pytest.raises(VLLMValidationError) as caught:
        raise_if_error(processor, "cmpl-1", "caller-name")

    assert str(caught.value) == REASON


def test_a_caller_name_does_not_answer_for_another_request() -> None:
    """Both names are looked up exactly; neither is inferred from the other."""
    processor = _Processor()
    note_admission_rejections(processor, [refusal("cmpl-1-child-0-884f2ecb")])
    async_llm.remember_client_request_id(processor, "caller-name", "cmpl-1-0-884f2ecb")

    with pytest.raises(GenerationError):
        raise_if_error(processor, "cmpl-1", "caller-name")


def test_a_reused_caller_name_finds_the_latest_request_only() -> None:
    """A name the caller reuses belongs to the request it was last given to."""
    processor = _Processor()
    note_admission_rejections(processor, [refusal("cmpl-1-0-884f2ecb")])
    async_llm.remember_client_request_id(processor, "caller-name", "cmpl-1-0-884f2ecb")
    async_llm.remember_client_request_id(processor, "caller-name", "cmpl-1-0-1a2b3c4d")

    with pytest.raises(GenerationError):
        raise_if_error(processor, "cmpl-1", "caller-name")


def test_a_missing_mapped_refusal_does_not_consume_another_identity() -> None:
    processor = _Processor()
    note_admission_rejections(processor, [refusal("cmpl-1")])

    async_llm.remember_client_request_id(processor, "caller-name", "cmpl-1-8b31d4")
    assert take_admission_rejection(client(processor), "missing", "caller-name") is None
    # The reason filed under the caller's own id is left where it was.
    assert take_admission_rejection(client(processor), "cmpl-1") == REASON


def test_a_refused_ordinary_request_reaches_the_caller_as_a_client_error() -> None:
    """The whole path: engine output in, the caller's own status class out."""
    processor = _Processor()
    wrapped = make_patched_process_outputs(processor.process_outputs)

    wrapped(processor, [refusal()])

    with pytest.raises(VLLMValidationError) as caught:
        raise_if_error(processor)
    assert str(caught.value) == REASON
