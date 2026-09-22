# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public output contracts when V1 creates lazy beams without dataclass init."""

from __future__ import annotations

import copy
from dataclasses import asdict, fields, replace
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.lora.request import LoRARequest
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchEngineError
from vllm_gr.entrypoints.beam_search_v1.output_processor import (
    BeamSearchOutputProcessor,
)
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.entrypoints.gr import _LazyMaterializedBeamSequence
from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchFailure,
    BeamBatchItem,
    BeamBatchOutput,
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputOptions,
    BeamOutputSequence,
    BeamRequestOutput,
)


class _Tokenizer:
    unk_token_id = None

    def __init__(self) -> None:
        self.decode_calls = 0

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return {"begin": 10, "end": 20}.get(token)

    def decode(self, ids: Any) -> str:
        self.decode_calls += 1
        return "|".join(map(str, ids))

    def batch_decode(self, rows: Any) -> list[str]:
        return [self.decode(row) for row in rows]


def _case(
    begin: str | None = "begin", end: str | None = "end", include_stop: bool = True
) -> tuple[
    BeamSearchOutputProcessor,
    OfflineBeamSearchRequest,
    BeamBatchRequest,
    BeamBatchResult,
    _Tokenizer,
]:
    tokenizer = _Tokenizer()
    params = SimpleNamespace(
        begin_token=begin, end_token=end, include_stop_str_in_output=include_stop
    )
    call = OfflineBeamSearchRequest([{"prompt_token_ids": [1, 2], "prompt": "source"}], params)
    lora = LoRARequest("test-lora", 1, "/tmp/test-lora")
    native = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2] + ([10] if begin else []),
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=lora,
        cache_salt="salt",
        data_parallel_rank=None,
    )
    request = BeamBatchRequest(
        1,
        "request-1",
        (BeamBatchItem(0, native),),
        BeamBatchParams(2, 3, 0.0, (20,), False),
        BeamOutputOptions(2, include_stop, True, True),
    )
    sequences = (
        BeamOutputSequence((3, 4), -1.25, (), "length", None),
        BeamOutputSequence((5, 20) if include_stop else (5,), -2.5, (), "stop", 20),
    )
    result = BeamBatchResult(
        1, "request-1", BeamBatchOutput((BeamRequestOutput(0, "request-1", sequences),)), None
    )
    return BeamSearchOutputProcessor(tokenizer), call, request, result, tokenizer


@pytest.mark.parametrize("begin", [None, "begin"])  # type: ignore[misc]
@pytest.mark.parametrize("end", [None, "end"])  # type: ignore[misc]
@pytest.mark.parametrize("include_stop", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("first_field", ["tokens", "text", "logprobs"])  # type: ignore[misc]
def test_output_boundaries_and_lazy_materialization(
    begin: str | None, end: str | None, include_stop: bool, first_field: str
) -> None:
    processor, call, request, result, tokenizer = _case(begin, end, include_stop)
    beams = processor.process_outputs(call, request, result)[0].sequences
    assert tokenizer.decode_calls == 0
    assert all(type(beam) is _LazyMaterializedBeamSequence for beam in beams)
    assert all("_v1_pending_fields" not in beam.__dict__ for beam in beams)
    assert [beam.cum_logprob for beam in beams] == [-1.25, -2.5]
    assert tokenizer.decode_calls == 0

    prefix = [1, 2] + ([10] if begin else [])
    expected = [
        prefix + [3, 4] + ([20] if end else []),
        prefix + [5] + ([20] if include_stop else []),
    ]
    value = getattr(beams[0], first_field)
    if first_field == "tokens":
        assert value == expected[0]
        assert tokenizer.decode_calls == 0
        assert beams[0].__dict__["text"] is None
    elif first_field == "text":
        assert value == "|".join(map(str, expected[0]))
        assert tokenizer.decode_calls == 1
        assert beams[0].__dict__["tokens"] == []
    else:
        assert value == []
        assert tokenizer.decode_calls == 0
        assert beams[0].__dict__["tokens"] == []
        assert beams[0].__dict__["text"] is None
    assert type(beams[0]) is _LazyMaterializedBeamSequence
    assert type(beams[1]) is _LazyMaterializedBeamSequence
    for i, beam in enumerate(beams):
        assert beam.tokens == expected[i]
        assert beam.logprobs == []
        assert beam.text == "|".join(map(str, expected[i]))
        assert beam.orig_prompt == tokens_input([1, 2], prompt="source", cache_salt="salt")
        assert beam.lora_request is request.items[0].request.lora_request
        assert beam.finish_reason == ("length" if i == 0 else "stop")
        assert beam.stop_reason == (None if i == 0 else 20)
    assert tokenizer.decode_calls == 2
    assert request.items[0].request.prompt_token_ids == prefix


def test_shared_text_and_tokens_materialize_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.entrypoints.beam_search_v1 import output_processor

    monkeypatch.setattr(output_processor, "_can_share_prompt_decode", lambda *_: True)
    processor, call, request, result, tokenizer = _case()
    beams = processor.process_outputs(call, request, result)[0].sequences

    assert beams[0].tokens == [1, 2, 10, 3, 4, 20]
    assert tokenizer.decode_calls == 0
    assert beams[0].__dict__["text"] is None

    assert beams[1].text == "1|2|10" + "5|20"
    assert tokenizer.decode_calls == 2
    assert beams[1].__dict__["tokens"] == []

    assert beams[0].text == "1|2|10" + "3|4|20"
    assert tokenizer.decode_calls == 3
    assert type(beams[0]) is BeamSearchSequence
    assert type(beams[1]) is _LazyMaterializedBeamSequence


def test_user_assignment_is_not_overwritten_by_later_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.entrypoints.beam_search_v1 import output_processor

    monkeypatch.setattr(output_processor, "_can_share_prompt_decode", lambda *_: True)
    processor, call, request, result, tokenizer = _case()
    beam = processor.process_outputs(call, request, result)[0].sequences[0]

    beam.tokens = [99]
    assert beam.text == "1|2|10" + "3|4|20"
    assert beam.tokens == [99]
    assert tokenizer.decode_calls == 2
    assert type(beam) is BeamSearchSequence


@pytest.mark.parametrize("operation", [copy.copy, replace, asdict])  # type: ignore[misc]
def test_public_dataclass_contract(operation: Any) -> None:
    processor, call, request, result, _ = _case()
    beam = processor.process_outputs(call, request, result)[0].sequences[0]
    reference = BeamSearchSequence(
        orig_prompt=tokens_input([1, 2], prompt="source", cache_salt="salt"),
        tokens=[],
        logprobs=[],
        lora_request=request.items[0].request.lora_request,
        cum_logprob=-1.25,
        text=None,
        finish_reason="length",
        stop_reason=None,
    )
    # Catch new upstream dataclass fields/defaults or initialization side effects.
    public_state = {key: value for key, value in vars(beam).items() if not key.startswith("_")}
    assert public_state == vars(reference)
    assert {field.name for field in fields(BeamSearchSequence)} <= public_state.keys()
    reference.tokens = [1, 2, 10, 3, 4, 20]
    reference.text = "1|2|10|3|4|20"
    value = operation(beam)
    assert (value if isinstance(value, dict) else asdict(value)) == asdict(reference)


def test_retained_outputs_survive_new_request_and_mutation() -> None:
    processor, call, request, result, _ = _case()
    old = processor.process_outputs(call, request, result)[0].sequences
    _, call2, request2, result2, _ = _case(None, None, False)
    new = processor.process_outputs(call2, request2, result2)[0].sequences
    new[0].tokens.append(99)
    old[0].tokens.append(88)
    old[0].logprobs.append({})
    assert old[1].tokens == [1, 2, 10, 5, 20]
    assert old[1].logprobs == []
    assert new[1].tokens == [1, 2, 5]
    assert new[1].logprobs == []
    assert old[0].tokens == [1, 2, 10, 3, 4, 20, 88]
    assert new[0].tokens == [1, 2, 3, 4, 99]


def test_item_order_and_empty_outputs() -> None:
    processor, call, request, result, _ = _case()
    assert result.output is not None
    first = result.output.items[0]
    second_native = copy.copy(request.items[0].request)
    second_native.prompt_token_ids = [7, 8, 10]
    second_native.cache_salt = "second"
    request2 = BeamBatchRequest(
        1,
        "batch",
        (*request.items, BeamBatchItem(1, second_native)),
        request.beam_params,
        request.output_options,
    )
    call2 = OfflineBeamSearchRequest([*call.prompts, {"prompt_token_ids": [7, 8]}], call.params)
    output = BeamBatchOutput((BeamRequestOutput(1, "second", first.sequences), first))
    converted = processor.process_outputs(
        call2, request2, BeamBatchResult(1, "batch", output, None)
    )
    assert converted[0].sequences[0].tokens == [7, 8, 10, 3, 4, 20]
    assert converted[0].sequences[0].orig_prompt["cache_salt"] == "second"
    assert converted[1].sequences[0].tokens == [1, 2, 10, 3, 4, 20]
    empty = BeamBatchResult(1, "batch", BeamBatchOutput(()), None)
    assert processor.process_outputs(call2, request2, empty) == []


def test_failure_and_invalid_boundary_preserve_error_contract() -> None:
    processor, call, request, result, _ = _case()
    failure = BeamBatchFailure("expected_failure", "expected", None)
    with pytest.raises(BeamSearchEngineError, match="expected"):
        processor.process_outputs(call, request, BeamBatchResult(1, "batch", None, failure))
    call.params.end_token = "invalid"
    with pytest.raises(BeamSearchEngineError, match="end_token must be a valid") as exc:
        processor.process_outputs(call, request, result)
    assert exc.value.failure.code == "output_conversion_failed"
    # The same processor remains usable after a failed conversion.
    call.params.end_token = "end"
    assert processor.process_outputs(call, request, result)[0].sequences[0].tokens[-1] == 20


@pytest.mark.parametrize("include_stop", [False, True])  # type: ignore[misc]
def test_online_output_boundary_is_unchanged(include_stop: bool) -> None:
    processor, call, request, result, _ = _case(include_stop=include_stop)
    online = OnlineBeamSearchRequest(call.prompts[0], "external-id", call.params)
    output = processor.process_online_outputs(online, request, result)[0]
    assert output.request_id == "external-id"
    assert output.prompt_token_ids == [1, 2]
    assert output.outputs[0].token_ids == [10, 3, 4, 20]
    assert output.outputs[1].token_ids == [10, 5] + ([20] if include_stop else [])
    assert output.outputs[0].text == "10|3|4|20"
