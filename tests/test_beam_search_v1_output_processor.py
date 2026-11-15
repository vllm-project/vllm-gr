# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Public output contracts when V1 creates lazy beams without dataclass init."""

from __future__ import annotations

import copy
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
from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchEngineError
from vllm_gr.entrypoints.beam_search_v1.output_processor import (
    BeamSearchOutputProcessor,
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


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("first_field", ["tokens", "text", "logprobs"])  # type: ignore[misc]
@pytest.mark.parametrize("failure_point", ["full", "prefix", "suffix"])  # type: ignore[misc]
@pytest.mark.parametrize("failures", [1, 2])  # type: ignore[misc]
def test_lazy_decode_failure_preserves_beam_until_retry(
    monkeypatch: pytest.MonkeyPatch, first_field: str, failure_point: str, failures: int
) -> None:
    import vllm_gr.entrypoints.beam_search_v1.output_processor as output_processor

    processor, call, request, result, tokenizer = _case()
    beams = processor.process_outputs(call, request, result)[0].sequences
    beam = beams[0]
    materializer = vars(beam)["_lazy_materializer"]
    suffix = vars(beam)["_v1_suffix"]
    prefix = (1, 2, 10)
    full = (*prefix, 3, 4, 20)
    failing_ids = {"full": full, "prefix": prefix, "suffix": suffix}[failure_point]
    attempts = []
    remaining = failures

    def decode(ids: Any) -> str:
        nonlocal remaining
        ids = tuple(ids)
        attempts.append(ids)
        if ids == failing_ids and remaining:
            remaining -= 1
            raise ValueError("transient decode failure")
        # Concatenative text makes shared-prefix and full decoding equivalent.
        return "".join(map(str, ids))

    monkeypatch.setattr(tokenizer, "decode", decode)
    monkeypatch.setattr(
        output_processor, "_can_share_prompt_decode", lambda *_: failure_point != "full"
    )
    for _ in range(failures):
        with pytest.raises(ValueError, match="transient decode failure"):
            getattr(beam, first_field)
        assert type(beam) is _LazyMaterializedBeamSequence
        assert vars(beam)["_v1_suffix"] is suffix
        assert vars(beam)["_lazy_materializer"] is materializer
        assert vars(beam)["tokens"] == []
        assert vars(beam)["logprobs"] == []
        assert vars(beam)["text"] is None
        assert "_lazy_materializing" not in vars(beam)
        assert beam.cum_logprob == -1.25
        assert type(beams[1]) is _LazyMaterializedBeamSequence

    getattr(beam, first_field)
    assert type(beam) is BeamSearchSequence
    assert beam.tokens == list(full)
    assert beam.logprobs == []
    assert beam.text == "12103420"
    assert beam.cum_logprob == -1.25
    assert beam.finish_reason == "length" and beam.stop_reason is None
    assert "_v1_suffix" not in vars(beam)
    assert vars(beam)["_lazy_materializer"] is None
    assert "_lazy_materializing" not in vars(beam)
    attempts_after_success = len(attempts)
    assert (beam.tokens, beam.text, beam.logprobs) == (list(full), "12103420", [])
    assert len(attempts) == attempts_after_success
    if failure_point == "suffix":
        assert attempts.count(prefix) == 1


def _fast_tokenizer_with_decode_failure(
    monkeypatch: pytest.MonkeyPatch, decoder_path: str, failing_ids: tuple[int, ...]
) -> tuple[Any, list[tuple[int, ...]], list[tuple[int, ...]]]:
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    tokens = [f"id_{index}" for index in range(21)]
    tokens[10], tokens[20] = "begin", "end"
    backend = Tokenizer(models.WordLevel(dict(zip(tokens, range(21), strict=True))))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, clean_up_tokenization_spaces=decoder_path == "_decode"
    )
    # Exercise the real added-token parity probe before the native fast path.
    tokenizer.add_tokens(["added"])
    attempts: list[tuple[int, ...]] = []
    shortcut_calls: list[tuple[int, ...]] = []
    failed = False

    def check_failure(ids: Any) -> None:
        nonlocal failed
        ids = tuple(ids)
        attempts.append(ids)
        if ids == failing_ids and not failed:
            failed = True
            raise ValueError("optimized decode failure")

    original_decode = tokenizer._decode

    def shortcut(token_ids: Any, **kwargs: Any) -> str:
        shortcut_calls.append(tuple(token_ids))
        if decoder_path == "_decode":
            check_failure(token_ids)
        return cast(str, original_decode(token_ids, **kwargs))

    monkeypatch.setattr(tokenizer, "_decode", shortcut)
    if decoder_path == "native":
        native_backend = tokenizer._tokenizer

        class FailingBackend:
            def __getattr__(self, name: str) -> Any:
                return getattr(native_backend, name)

            def decode(self, ids: Any, **kwargs: Any) -> str:
                check_failure(ids)
                return cast(str, native_backend.decode(ids, **kwargs))

        monkeypatch.setattr(tokenizer, "_tokenizer", FailingBackend())
    return tokenizer, attempts, shortcut_calls


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("decoder_path", ["native", "_decode"])  # type: ignore[misc]
@pytest.mark.parametrize("first_field", ["tokens", "text", "logprobs"])  # type: ignore[misc]
def test_lazy_output_retries_optimized_decode_branch(
    monkeypatch: pytest.MonkeyPatch, decoder_path: str, first_field: str
) -> None:
    _, call, request, result, _ = _case()
    full = (1, 2, 10, 3, 4, 20)
    tokenizer, attempts, shortcut_calls = _fast_tokenizer_with_decode_failure(
        monkeypatch, decoder_path, full
    )
    beam = (
        BeamSearchOutputProcessor(tokenizer).process_outputs(call, request, result)[0].sequences[0]
    )
    materializer = vars(beam)["_lazy_materializer"]
    suffix = vars(beam)["_v1_suffix"]
    assert attempts == []

    with pytest.raises(ValueError, match="optimized decode failure"):
        getattr(beam, first_field)
    assert type(beam) is _LazyMaterializedBeamSequence
    assert vars(beam)["_lazy_materializer"] is materializer
    assert vars(beam)["_v1_suffix"] is suffix
    assert vars(beam)["tokens"] == [] and vars(beam)["text"] is None
    assert "_lazy_materializing" not in vars(beam)

    getattr(beam, first_field)
    assert type(beam) is BeamSearchSequence
    assert beam.tokens == list(full)
    assert beam.text == "id_1id_2beginid_3id_4end"
    assert beam.logprobs == [] and beam.cum_logprob == -1.25
    assert beam.finish_reason == "length" and beam.stop_reason is None
    assert attempts.count(full) == 2
    assert (full in shortcut_calls) is (decoder_path == "_decode")
    completed_attempts = len(attempts)
    assert (beam.tokens, beam.text, beam.logprobs) == (list(full), "id_1id_2beginid_3id_4end", [])
    assert len(attempts) == completed_attempts


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("decoder_path", ["native", "_decode"])  # type: ignore[misc]
def test_shared_prompt_decode_retry_keeps_materialized_sibling(
    monkeypatch: pytest.MonkeyPatch, decoder_path: str
) -> None:
    import vllm_gr.entrypoints.beam_search_v1.output_processor as output_processor

    _, call, request, result, _ = _case()
    prefix, first_suffix, second_suffix = (1, 2, 10), (3, 4, 20), (5, 20)
    tokenizer, attempts, shortcut_calls = _fast_tokenizer_with_decode_failure(
        monkeypatch, decoder_path, second_suffix
    )
    monkeypatch.setattr(output_processor, "_can_share_prompt_decode", lambda *_: True)
    first, second = (
        BeamSearchOutputProcessor(tokenizer).process_outputs(call, request, result)[0].sequences
    )
    materializer = vars(second)["_lazy_materializer"]
    suffix = vars(second)["_v1_suffix"]
    assert vars(first)["_lazy_materializer"] is materializer
    assert first.text == "id_1id_2beginid_3id_4end"
    completed_first = asdict(first)

    with pytest.raises(ValueError, match="optimized decode failure"):
        _ = second.text
    assert type(first) is BeamSearchSequence
    assert asdict(first) == completed_first
    assert type(second) is _LazyMaterializedBeamSequence
    assert vars(second)["_lazy_materializer"] is materializer
    assert vars(second)["_v1_suffix"] is suffix
    assert "_lazy_materializing" not in vars(second)

    assert second.text == "id_1id_2beginid_5end"
    assert second.tokens == [1, 2, 10, 5, 20]
    assert second.logprobs == [] and second.cum_logprob == -2.5
    assert second.finish_reason == "stop" and second.stop_reason == 20
    assert asdict(first) == completed_first
    assert attempts.count(prefix) == attempts.count(first_suffix) == 1
    assert attempts.count(second_suffix) == 2
    assert (second_suffix in shortcut_calls) is (decoder_path == "_decode")


@pytest.mark.l1  # type: ignore[misc]
def test_recursive_lazy_field_access_does_not_publish_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, call, request, result, tokenizer = _case()
    beam = processor.process_outputs(call, request, result)[0].sequences[0]
    materializer = vars(beam)["_lazy_materializer"]
    suffix = vars(beam)["_v1_suffix"]
    decode = tokenizer.decode

    def reentrant_decode(ids: Any) -> str:
        return cast(str, beam.text)

    monkeypatch.setattr(tokenizer, "decode", reentrant_decode)
    with pytest.raises(RuntimeError, match="recursive Beam output materialization"):
        _ = beam.tokens
    assert type(beam) is _LazyMaterializedBeamSequence
    assert vars(beam)["_lazy_materializer"] is materializer
    assert vars(beam)["_v1_suffix"] is suffix
    assert "_lazy_materializing" not in vars(beam)
    monkeypatch.setattr(tokenizer, "decode", decode)
    assert beam.tokens == [1, 2, 10, 3, 4, 20]
    assert beam.text == "1|2|10|3|4|20"


@pytest.mark.l1  # type: ignore[misc]
def test_lazy_class_transition_failure_keeps_the_accepted_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.entrypoints.generate.beam_search.utils as beam_utils

    class IncompatibleSequence:
        __slots__ = ("different_layout",)

    processor, call, request, result, _ = _case()
    beam = processor.process_outputs(call, request, result)[0].sequences[0]
    materializer = vars(beam)["_lazy_materializer"]
    suffix = vars(beam)["_v1_suffix"]
    with monkeypatch.context() as patch:
        patch.setattr(beam_utils, "BeamSearchSequence", IncompatibleSequence)
        with pytest.raises(TypeError, match="__class__ assignment"):
            _ = beam.text
    assert type(beam) is _LazyMaterializedBeamSequence
    assert vars(beam)["_lazy_materializer"] is materializer
    assert vars(beam)["_v1_suffix"] is suffix
    assert vars(beam)["tokens"] == [] and vars(beam)["text"] is None
    assert "_lazy_materializing" not in vars(beam)
    assert beam.tokens == [1, 2, 10, 3, 4, 20]
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
    # One materialization per request: 256 Beams must not re-read the prompt.
    assert prompt.reads == 1
    assert compact.prompt_token_ids == tuple(range(1024))
    assert tokenizer.decode_calls == 0
    assert len(compact.sequences) == 256
    assert all(len(seq.token_ids) == 3 for seq in compact.sequences)
    assert compact.sequences[255].cumulative_logprob == -255.0


@pytest.mark.parametrize("begin", [None, "begin"])  # type: ignore[misc]
def test_compact_reuses_accepted_prompt_snapshot_without_array_materialization(
    begin: str | None,
) -> None:
    class NoMaterialization(np.ndarray[Any, np.dtype[np.int64]]):  # type: ignore[misc]
        def tolist(self) -> list[int]:
            raise AssertionError("compact output must reuse the accepted prompt snapshot")

    _, call, request, result, tokenizer = _case(begin=begin)
    prompt = request.items[0].request.prompt_token_ids.view(NoMaterialization)
    request.items[0].request.prompt_token_ids = prompt
    snapshot = ((1, 2),)
    processor = BeamSearchOutputProcessor(tokenizer, original_prompt_token_ids=snapshot)
    prompt[0] = 999
    call.prompts[0]["prompt_token_ids"][0] = 888
    compact = processor.process_outputs(replace(call, output_mode="compact"), request, result)[0]
    assert compact.prompt_token_ids is snapshot[0]
    assert compact.prompt_token_ids == (1, 2)
    assert compact.sequences[0].token_ids == (() if begin is None else (10,)) + (3, 4, 20)
    assert tokenizer.decode_calls == 0


def test_compact_output_keeps_prompt_token_ids_constructor_compatibility() -> None:
    from vllm_gr.outputs import CompactBeamSearchOutput, CompactBeamSearchSequence

    sequence = CompactBeamSearchSequence((3, 4), -1.25)
    output = CompactBeamSearchOutput((sequence,), prompt_token_ids=(1, 2))
    assert output.prompt_token_ids == (1, 2)
    assert CompactBeamSearchOutput((sequence,)).prompt_token_ids is None


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
    assert out[1].sequences
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


pytestmark = pytest.mark.cpu_test
