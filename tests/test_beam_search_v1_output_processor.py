# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public output contracts when V1 creates lazy beams without dataclass init."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import asdict, fields, replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.lora.request import LoRARequest
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.entrypoints.beam_output import _LazyMaterializedBeamSequence
from vllm_gr.entrypoints.beam_search_v1 import output_processor as output_module
from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchEngineError
from vllm_gr.entrypoints.beam_search_v1.output_processor import (
    BeamSearchOutputProcessor,
    _NPUBeamTailMaterializer,
    _V1BeamTailMaterializer,
)
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
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


class _CountingPrompt(np.ndarray[Any, np.dtype[np.int64]]):  # type: ignore[misc]
    """Share the conversion counter with the original-prompt slice."""

    def __new__(cls, values: Sequence[int]) -> _CountingPrompt:
        prompt = np.asarray(values, dtype=np.int64).view(cls)
        prompt.tolist_calls = []
        return cast(_CountingPrompt, prompt)

    def __array_finalize__(self, source: Any) -> None:
        self.tolist_calls = getattr(source, "tolist_calls", [])

    def tolist(self) -> list[int]:
        self.tolist_calls.append(1)
        return cast(list[int], super().tolist())


def _joined_tokens(tokenizer: Any, ids: Sequence[int]) -> str:
    # A concatenative decoder exercises the shared-prefix path byte for byte.
    return "".join(f"<{token_id}>" for token_id in ids)


@pytest.fixture  # type: ignore[misc]
def npu_output_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(output_module, "current_platform", SimpleNamespace(device_type="npu"))


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
    call = OfflineBeamSearchRequest(
        [{"prompt_token_ids": [1, 2], "prompt": "source"}], params, output_mode="full"
    )
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
        BEAM_BATCH_PROTOCOL_VERSION,
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
        BEAM_BATCH_PROTOCOL_VERSION,
        "request-1",
        BeamBatchOutput((BeamRequestOutput(0, "request-1", sequences),)),
        None,
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
    assert [beam.cum_logprob for beam in beams] == [-1.25, -2.5]
    assert tokenizer.decode_calls == 0

    prefix = [1, 2] + ([10] if begin else [])
    expected = [
        prefix + [3, 4] + ([20] if end else []),
        prefix + [5] + ([20] if include_stop else []),
    ]
    getattr(beams[0], first_field)
    assert tokenizer.decode_calls == 1
    assert type(beams[0]) is BeamSearchSequence
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
    assert request.items[0].request.prompt_token_ids.tolist() == prefix


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


@pytest.mark.parametrize("first_field", ["tokens", "text", "logprobs"])  # type: ignore[misc]
@pytest.mark.parametrize("shared_prompt", [False, True])  # type: ignore[misc]
def test_npu_full_output_caches_prompt_once_and_preserves_owned_beams(
    monkeypatch: pytest.MonkeyPatch,
    npu_output_platform: None,
    first_field: str,
    shared_prompt: bool,
) -> None:
    processor, call, request, _, _ = _case()
    prompt = _CountingPrompt([1, 2, 10])
    request.items[0].request.prompt_token_ids = prompt
    sequences = tuple(BeamOutputSequence((3, i), -float(i), (), "length", None) for i in range(128))
    result = BeamBatchResult(
        1, "request-1", BeamBatchOutput((BeamRequestOutput(0, "request-1", sequences),)), None
    )
    share_checks: list[int] = []
    decode_inputs: list[list[int]] = []

    def can_share(tokenizer: Any, ids: Sequence[int]) -> bool:
        share_checks.append(1)
        return shared_prompt

    def decode(tokenizer: Any, ids: Sequence[int]) -> str:
        decode_inputs.append(list(ids))
        return _joined_tokens(tokenizer, ids)

    monkeypatch.setattr(output_module, "_can_share_prompt_decode", can_share)
    monkeypatch.setattr(output_module, "fast_decode", decode)
    beams = processor.process_outputs(call, request, result)[0].sequences
    # Only the original prompt is converted during output construction.
    assert len(prompt.tolist_calls) == 1
    materializer = beams[0].__dict__["_lazy_materializer"]
    assert type(materializer) is _NPUBeamTailMaterializer
    assert all(beam.__dict__["_lazy_materializer"] is materializer for beam in beams)
    assert materializer._prepared_list is None
    assert beams[0].cum_logprob == 0.0
    assert len(prompt.tolist_calls) == 1
    assert decode_inputs == []

    getattr(beams[0], first_field)
    assert len(prompt.tolist_calls) == 2
    assert beams[0].tokens == [1, 2, 10, 3, 0, 20]
    assert beams[0].text == "<1><2><10><3><0><20>"
    assert beams[0].tokens is not materializer._prepared_list
    # A caller may mutate the first beam before any siblings materialize.
    beams[0].tokens[0] = 999
    beams[0].tokens.append(88)
    for i, beam in enumerate(beams[1:], start=1):
        assert beam.tokens == [1, 2, 10, 3, i, 20]
        assert beam.text == f"<1><2><10><3><{i}><20>"
        assert beam.tokens is not materializer._prepared_list
    assert len({id(beam.tokens) for beam in beams}) == len(beams)
    assert list(prompt) == [1, 2, 10]
    assert materializer._prepared_list == [1, 2, 10]
    assert len(prompt.tolist_calls) == 2
    assert share_checks == [1]
    if shared_prompt:
        assert decode_inputs[0] == [1, 2, 10]
        assert decode_inputs[1:] == [[3, i, 20] for i in range(128)]
    else:
        assert decode_inputs == [[1, 2, 10, 3, i, 20] for i in range(128)]


@pytest.mark.parametrize("shared_prompt", [False, True])  # type: ignore[misc]
def test_npu_full_prompt_snapshot_is_deferred_until_first_access(
    monkeypatch: pytest.MonkeyPatch, npu_output_platform: None, shared_prompt: bool
) -> None:
    processor, call, request, result, _ = _case()
    prompt = _CountingPrompt([1, 2, 10])
    request.items[0].request.prompt_token_ids = prompt
    monkeypatch.setattr(output_module, "_can_share_prompt_decode", lambda *_: shared_prompt)
    monkeypatch.setattr(output_module, "fast_decode", _joined_tokens)
    beams = processor.process_outputs(call, request, result)[0].sequences
    prompt[0] = 6
    assert len(prompt.tolist_calls) == 1
    assert beams[0].text == "<6><2><10><3><4><20>"
    assert beams[0].tokens == [6, 2, 10, 3, 4, 20]
    prompt[0] = 9
    assert beams[1].tokens == [6, 2, 10, 5, 20]
    assert beams[1].text == "<6><2><10><5><20>"
    assert len(prompt.tolist_calls) == 2
    assert beams[0].orig_prompt["prompt_token_ids"] == [1, 2]


@pytest.mark.parametrize("shared_prompt", [False, True])  # type: ignore[misc]
def test_npu_full_prompt_cache_is_per_request_and_per_conversion(
    monkeypatch: pytest.MonkeyPatch, npu_output_platform: None, shared_prompt: bool
) -> None:
    processor, call, request, result, _ = _case()
    first_prompt = _CountingPrompt([1, 2, 10])
    second_prompt = _CountingPrompt([7, 8, 10])
    request.items[0].request.prompt_token_ids = first_prompt
    second = copy.copy(request.items[0].request)
    second.prompt_token_ids = second_prompt
    request = BeamBatchRequest(
        request.protocol_version,
        request.batch_id,
        (*request.items, BeamBatchItem(1, second)),
        request.beam_params,
        request.output_options,
    )
    call = replace(call, prompts=[*call.prompts, {"prompt_token_ids": [7, 8]}])
    assert result.output is not None
    first_output = result.output.items[0]
    result = BeamBatchResult(
        result.protocol_version,
        result.batch_id,
        BeamBatchOutput((BeamRequestOutput(1, "second", first_output.sequences), first_output)),
        None,
    )
    monkeypatch.setattr(output_module, "_can_share_prompt_decode", lambda *_: shared_prompt)
    monkeypatch.setattr(output_module, "fast_decode", _joined_tokens)
    old = processor.process_outputs(call, request, result)
    assert len(first_prompt.tolist_calls) == len(second_prompt.tolist_calls) == 1
    assert old[0].sequences[0].tokens == [7, 8, 10, 3, 4, 20]
    assert len(first_prompt.tolist_calls) == 1
    assert len(second_prompt.tolist_calls) == 2
    # A later conversion must create fresh request caches, even for the same
    # accepted request object; retained outputs keep their own snapshots.
    second_prompt[0] = 9
    new = processor.process_outputs(call, request, result)
    assert new[0].sequences[0].tokens == [9, 8, 10, 3, 4, 20]
    assert old[0].sequences[1].tokens == [7, 8, 10, 5, 20]
    assert new[0].sequences[1].tokens == [9, 8, 10, 5, 20]
    assert old[1].sequences[0].tokens == [1, 2, 10, 3, 4, 20]
    assert new[1].sequences[0].tokens == [1, 2, 10, 3, 4, 20]
    assert len(first_prompt.tolist_calls) == len(second_prompt.tolist_calls) == 4
    assert old[0].sequences[1].text == "<7><8><10><5><20>"
    assert new[0].sequences[1].text == "<9><8><10><5><20>"


@pytest.mark.parametrize("prepared", [[1, 2], (1, 2), ()])  # type: ignore[misc]
@pytest.mark.parametrize("shared_prompt", [False, True])  # type: ignore[misc]
def test_npu_tail_materializer_supports_sequences_without_tolist(
    monkeypatch: pytest.MonkeyPatch, prepared: Sequence[int], shared_prompt: bool
) -> None:
    processor, call, request, result, _ = _case(None, None)
    beams = processor.process_outputs(call, request, result)[0].sequences
    materializer = _NPUBeamTailMaterializer(processor.tokenizer, prepared)
    monkeypatch.setattr(output_module, "_can_share_prompt_decode", lambda *_: shared_prompt)
    monkeypatch.setattr(output_module, "fast_decode", _joined_tokens)
    for beam in beams:
        beam.__dict__["_lazy_materializer"] = materializer
    assert materializer._prepared_list is None
    assert beams[0].tokens == list(prepared) + [3, 4]
    beams[0].tokens.append(99)
    assert beams[1].tokens == list(prepared) + [5, 20]
    assert beams[1].text == _joined_tokens(processor.tokenizer, beams[1].tokens)
    assert materializer._prepared_list == list(prepared)
    assert beams[0].tokens is not beams[1].tokens
    if isinstance(prepared, list):
        assert materializer._prepared_list is not prepared


@pytest.mark.parametrize("device_type", ["cuda", "cpu"])  # type: ignore[misc]
@pytest.mark.parametrize("shared_prompt", [False, True])  # type: ignore[misc]
def test_non_npu_full_output_retains_original_materializer(
    monkeypatch: pytest.MonkeyPatch, device_type: str, shared_prompt: bool
) -> None:
    monkeypatch.setattr(output_module, "current_platform", SimpleNamespace(device_type=device_type))
    monkeypatch.setattr(output_module, "_can_share_prompt_decode", lambda *_: shared_prompt)
    monkeypatch.setattr(output_module, "fast_decode", _joined_tokens)
    processor, call, request, _, _ = _case()
    prompt = _CountingPrompt([1, 2, 10])
    request.items[0].request.prompt_token_ids = prompt
    sequences = tuple(BeamOutputSequence((3, i), -float(i), (), "length", None) for i in range(128))
    result = BeamBatchResult(
        1, "request-1", BeamBatchOutput((BeamRequestOutput(0, "request-1", sequences),)), None
    )
    beams = processor.process_outputs(call, request, result)[0].sequences
    materializer = beams[0].__dict__["_lazy_materializer"]
    assert type(materializer) is _V1BeamTailMaterializer
    assert all(beam.__dict__["_lazy_materializer"] is materializer for beam in beams)
    assert len(prompt.tolist_calls) == 1
    assert beams[0].tokens == [1, 2, 10, 3, 0, 20]
    beams[0].tokens[0] = 999
    for i, beam in enumerate(beams[1:], start=1):
        assert beam.tokens == [1, 2, 10, 3, i, 20]
        assert beam.text == f"<1><2><10><3><{i}><20>"
    # The legacy class converts once per Beam, plus the original prompt and
    # the first shared-prefix text decode. NPU snapshot caching must not leak.
    assert len(prompt.tolist_calls) == 129 + int(shared_prompt)
    assert len({id(beam.tokens) for beam in beams}) == len(beams)


@pytest.mark.parametrize("device_type", ["cuda", "cpu"])  # type: ignore[misc]
def test_non_npu_later_beams_still_read_current_prepared_ids(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    monkeypatch.setattr(output_module, "current_platform", SimpleNamespace(device_type=device_type))
    monkeypatch.setattr(output_module, "_can_share_prompt_decode", lambda *_: False)
    monkeypatch.setattr(output_module, "fast_decode", _joined_tokens)
    processor, call, request, result, _ = _case()
    prompt = _CountingPrompt([1, 2, 10])
    request.items[0].request.prompt_token_ids = prompt
    beams = processor.process_outputs(call, request, result)[0].sequences
    assert type(beams[0].__dict__["_lazy_materializer"]) is _V1BeamTailMaterializer
    prompt[0] = 6
    assert beams[0].tokens == [6, 2, 10, 3, 4, 20]
    prompt[0] = 9
    assert beams[1].tokens == [9, 2, 10, 5, 20]
    assert beams[1].text == "<9><2><10><5><20>"
    assert beams[0].orig_prompt["prompt_token_ids"] == [1, 2]
    assert len(prompt.tolist_calls) == 3


@pytest.mark.parametrize("output_mode", ["compact", "online"])  # type: ignore[misc]
def test_npu_other_output_modes_do_not_construct_cached_materializer(
    monkeypatch: pytest.MonkeyPatch, npu_output_platform: None, output_mode: str
) -> None:
    def reject_materializer(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Only full NPU output may construct the cached materializer")

    monkeypatch.setattr(output_module, "_NPUBeamTailMaterializer", reject_materializer)
    processor, call, request, result, tokenizer = _case()
    if output_mode == "compact":
        compact = processor.process_outputs(replace(call, output_mode="compact"), request, result)
        assert compact[0].prompt_token_ids == (1, 2)
        assert compact[0].sequences[0].token_ids == (10, 3, 4, 20)
        assert tokenizer.decode_calls == 0
    else:
        online = OnlineBeamSearchRequest((call.prompts[0],), "external-id", call.params)
        outputs = processor.process_online_outputs(online, request, result)
        assert outputs[0].request_id == "external-id"
        assert outputs[0].prompt_token_ids == [1, 2]
        assert outputs[0].outputs[0].token_ids == [10, 3, 4, 20]
        assert outputs[0].outputs[0].text == "10|3|4|20"


def test_item_order_and_empty_outputs() -> None:
    processor, call, request, result, _ = _case()
    assert result.output is not None
    first = result.output.items[0]
    second_native = copy.copy(request.items[0].request)
    second_native.prompt_token_ids = np.array([7, 8, 10], dtype=np.int64)
    second_native.cache_salt = "second"
    request2 = BeamBatchRequest(
        BEAM_BATCH_PROTOCOL_VERSION,
        "batch",
        (*request.items, BeamBatchItem(1, second_native)),
        request.beam_params,
        request.output_options,
    )
    call2 = OfflineBeamSearchRequest(
        [*call.prompts, {"prompt_token_ids": [7, 8]}], call.params, output_mode="full"
    )
    output = BeamBatchOutput((BeamRequestOutput(1, "second", first.sequences), first))
    converted = processor.process_outputs(
        call2, request2, BeamBatchResult(BEAM_BATCH_PROTOCOL_VERSION, "batch", output, None)
    )
    assert converted[0].sequences[0].tokens == [7, 8, 10, 3, 4, 20]
    assert converted[0].sequences[0].orig_prompt["cache_salt"] == "second"
    assert converted[1].sequences[0].tokens == [1, 2, 10, 3, 4, 20]
    empty = BeamBatchResult(BEAM_BATCH_PROTOCOL_VERSION, "batch", BeamBatchOutput(()), None)
    assert processor.process_outputs(call2, request2, empty) == []


def test_failure_and_invalid_boundary_preserve_error_contract() -> None:
    processor, call, request, result, _ = _case()
    failure = BeamBatchFailure("expected_failure", "expected", None)
    with pytest.raises(BeamSearchEngineError, match="expected"):
        processor.process_outputs(
            call, request, BeamBatchResult(BEAM_BATCH_PROTOCOL_VERSION, "batch", None, failure)
        )
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
    online = OnlineBeamSearchRequest((call.prompts[0],), "external-id", call.params)
    output = processor.process_online_outputs(online, request, result)[0]
    assert output.request_id == "external-id"
    assert output.prompt_token_ids == [1, 2]
    assert output.outputs[0].token_ids == [10, 3, 4, 20]
    assert output.outputs[1].token_ids == [10, 5] + ([20] if include_stop else [])
    assert output.outputs[0].text == "10|3|4|20"


@pytest.mark.parametrize("begin", [None, "begin"])  # type: ignore[misc]
@pytest.mark.parametrize("end", [None, "end"])  # type: ignore[misc]
@pytest.mark.parametrize("include_stop", [False, True])  # type: ignore[misc]
def test_compact_matches_full_tail_scores_and_stop_metadata(
    begin: str | None, end: str | None, include_stop: bool
) -> None:
    from vllm_gr.outputs import CompactBeamSearchOutput

    processor, call, request, result, tokenizer = _case(begin, end, include_stop)
    compact = processor.process_outputs(replace(call, output_mode="compact"), request, result)[0]
    assert isinstance(compact, CompactBeamSearchOutput)
    assert compact.prompt_token_ids == (1, 2)
    assert tokenizer.decode_calls == 0
    full = processor.process_outputs(call, request, result)[0]
    for short, native in zip(compact.sequences, full.sequences, strict=True):
        assert short.token_ids == tuple(native.tokens[2:])
        assert short.cumulative_logprob == native.cum_logprob
        assert short.finish_reason == native.finish_reason
        assert short.stop_reason == native.stop_reason
    assert compact.sequences[1].token_ids == (
        ((10,) if begin else ()) + (5,) + ((20,) if include_stop else ())
    )
    # Compact values are owned, immutable CPU data, independent of accepted input mutation.
    request.items[0].request.prompt_token_ids[0] = 999
    assert compact.prompt_token_ids == (1, 2)


def test_compact_does_not_copy_prompt_or_decode_text_per_beam() -> None:
    class CountedPrompt(np.ndarray[Any, np.dtype[np.int64]]):  # type: ignore[misc]
        reads = 0

        def tolist(self) -> list[int]:
            self.reads += 1
            return cast(list[int], super().tolist())

    processor, call, request, result, tokenizer = _case(None, None)
    prompt = np.arange(1024, dtype=np.int64).view(CountedPrompt)
    request.items[0].request.prompt_token_ids = prompt
    sequences = tuple(
        BeamOutputSequence((3, 4, i), -float(i), (), "length", None) for i in range(256)
    )
    result = BeamBatchResult(
        1, "request-1", BeamBatchOutput((BeamRequestOutput(0, "request-1", sequences),)), None
    )
    compact = processor.process_outputs(replace(call, output_mode="compact"), request, result)[0]
    assert prompt.reads == 1
    assert compact.prompt_token_ids == tuple(range(1024))
    assert tokenizer.decode_calls == 0
    assert len(compact.sequences) == 256
    assert all(len(seq.token_ids) == 3 for seq in compact.sequences)
    assert compact.sequences[255].cumulative_logprob == -255.0


def test_compact_empty_candidates_and_request_order() -> None:
    processor, call, request, result, tokenizer = _case(None, None)
    second = copy.copy(request.items[0].request)
    second.prompt_token_ids = np.array([7, 8], dtype=np.int64)
    request = BeamBatchRequest(
        request.protocol_version,
        request.batch_id,
        (*request.items, BeamBatchItem(1, second)),
        request.beam_params,
        request.output_options,
    )
    call = replace(
        call, prompts=[*call.prompts, {"prompt_token_ids": [7, 8]}], output_mode="compact"
    )
    result = BeamBatchResult(
        1,
        "batch",
        BeamBatchOutput((BeamRequestOutput(1, "second", ()), result.output.items[0])),
        None,
    )
    out = processor.process_outputs(call, request, result)
    assert [x.prompt_token_ids for x in out] == [(7, 8), (1, 2)]
    assert out[0].sequences == ()
    assert tokenizer.decode_calls == 0


def test_invalid_output_mode_is_rejected_before_execution() -> None:
    with pytest.raises(ValueError, match="output_mode"):
        OfflineBeamSearchRequest([], object(), output_mode="invalid")


def test_default_output_is_compact_without_prompt_materialization() -> None:
    processor, call, request, result, tokenizer = _case()
    default_call = OfflineBeamSearchRequest(call.prompts, call.params)
    assert default_call.output_mode == "compact"
    output = processor.process_outputs(default_call, request, result)[0]
    assert output.prompt_token_ids == (1, 2)
    assert output.sequences[0].token_ids == (10, 3, 4, 20)
    assert output.sequences[0].cumulative_logprob == -1.25
    assert tokenizer.decode_calls == 0
