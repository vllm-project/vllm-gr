# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""Public output fidelity, including legal EOS and upstream compact results."""

import asyncio
import copy
from dataclasses import asdict, fields, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import msgspec
import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.lora.request import LoRARequest
from vllm.v1.engine import EngineCoreRequest

from tests.test_beam_search_v1_input_processor import _make_engine
from vllm_gr.entrypoints.beam_search_v1.client import (
    AsyncBeamSearchClient,
    BeamSearchClient,
)
from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchEngineError
from vllm_gr.entrypoints.beam_search_v1.output_processor import (
    BeamSearchOutputProcessor,
)
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.entrypoints.gr import _LazyMaterializedBeamSequence
from vllm_gr.sampling_params import BeamSearchParams
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


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize(
    "n,begin,end",
    [
        (1, None, None),
        (2, "<begin>", None),
        (3, None, "<end>"),
        (1, "<begin>", "<end>"),
        (2, "<begin>", "<end>"),
        (3, "<begin>", "<end>"),
    ],
)
def test_public_conversion_preserves_sequences_scores_and_boundary_budget(
    online: bool, n: int, begin: str | None, end: str | None
) -> None:
    engine = _make_engine()
    tokenizer = SimpleNamespace(
        eos_token_id=2,
        convert_tokens_to_ids=lambda token: {"<begin>": 7, "<end>": 8}[token],
        decode=Mock(side_effect=lambda ids: "".join(f"[{i}]" for i in ids)),
    )
    params = BeamSearchParams(
        beam_width=2,
        max_tokens=n + int(begin is not None) + int(end is not None),
        ignore_eos=True,
        begin_token=begin,
        end_token=end,
        length_penalty=2.0,
    )
    prompt = [10, 11]
    if online:
        serving = SimpleNamespace(
            engine_client=engine, input_processor=SimpleNamespace(tokenizer=tokenizer)
        )
        client = AsyncBeamSearchClient(serving)
        call = OnlineBeamSearchRequest(prompt, "caller-id", params)
        request = asyncio.run(client.prepare_request(call))
    else:
        engine.get_supported_tasks = lambda: ("generate",)
        llm = SimpleNamespace(
            llm_engine=engine,
            get_tokenizer=lambda: tokenizer,
            _preprocess_cmpl=lambda prompts: prompts,
            _lora_request_to_seq=lambda lora, size: [lora],
        )
        client = BeamSearchClient(llm)
        call = OfflineBeamSearchRequest([prompt], params)
        request = client.prepare_request(call)
    assert request.beam_params.max_tokens == n
    assert request.items[0].request.sampling_params.max_tokens == n
    assert request.items[0].request.prompt_token_ids == prompt + ([7] if begin else [])
    sequences = []
    for beam in range(2):
        # Legal EOS is an ordinary token, including when it is the final one.
        tokens = ([2, 3, 2] if beam == 0 else [4, 5, 2])[:n]
        sequences.append(
            BeamOutputSequence(tuple(tokens), -n * (0.25 + beam), (0,) * n, "length", None)
        )
    result = BeamBatchResult(
        1,
        request.batch_id,
        BeamBatchOutput((BeamRequestOutput(0, request.batch_id, tuple(sequences)),)),
        None,
    )
    result = msgspec.msgpack.decode(msgspec.msgpack.encode(result), type=BeamBatchResult)
    output = client.build_output(call, result)[0]
    if online:
        assert output.request_id == "caller-id" and output.finished
        assert output.prompt_token_ids == prompt
        returned = output.outputs
    else:
        returned = output.sequences
        assert returned[0].cum_logprob == sequences[0].cumulative_logprob
        tokenizer.decode.assert_not_called()
    for index, (actual, seq) in enumerate(zip(returned, sequences, strict=True)):
        tail = ([7] if begin else []) + list(seq.token_ids) + ([8] if end else [])
        assert (actual.token_ids if online else actual.tokens) == (
            tail if online else prompt + tail
        )
        assert actual.text == tokenizer.decode(tail if online else prompt + tail)
        assert actual.finish_reason == "length" and actual.stop_reason is None
        assert (
            actual.cumulative_logprob if online else actual.cum_logprob
        ) == seq.cumulative_logprob
        assert actual.logprobs is None if online else actual.logprobs == []
        if online:
            assert actual.index == index
    assert prompt == [10, 11]
    assert params.max_tokens == n + bool(begin) + bool(end)
    # A second conversion cannot mutate the snapshot or accumulate boundaries.
    repeated = client.build_output(call, result)[0]
    if online:
        assert repr(repeated) == repr(output)
    else:
        for actual, previous in zip(repeated.sequences, returned, strict=True):
            assert actual.tokens == previous.tokens
            assert actual.text == previous.text
            assert actual.cum_logprob == previous.cum_logprob
            assert actual.orig_prompt["prompt_token_ids"] == prompt
    with pytest.raises(BeamSearchEngineError, match="different session"):
        client.build_output(call, msgspec.structs.replace(result, batch_id="stale"))
    wrong = msgspec.structs.replace(result.output.items[0], request_id="wrong-item")
    with pytest.raises(BeamSearchEngineError, match="identity"):
        client.build_output(call, msgspec.structs.replace(result, output=BeamBatchOutput((wrong,))))
    failure = BeamBatchFailure("device_recoverable", "empty constraint", None)
    with pytest.raises(BeamSearchEngineError, match="empty constraint") as exc:
        client.build_output(call, msgspec.structs.replace(result, output=None, failure=failure))
    assert exc.value.failure == failure


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
    begin: str | None = "begin", end: str | None = "end"
) -> tuple[
    BeamSearchOutputProcessor,
    OfflineBeamSearchRequest,
    BeamBatchRequest,
    BeamBatchResult,
    _Tokenizer,
]:
    tokenizer = _Tokenizer()
    params = BeamSearchParams(
        beam_width=2,
        max_tokens=2 + int(begin is not None) + int(end is not None),
        ignore_eos=True,
        begin_token=begin,
        end_token=end,
    )
    lora = LoRARequest("test-lora", 1, "/tmp/test-lora")
    call = OfflineBeamSearchRequest(
        [{"prompt_token_ids": [1, 2], "prompt": "source"}], params, lora_request=lora
    )
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
        BeamBatchParams(2, 2, 0.0, (), True),
        BeamOutputOptions(2, True, True, True),
    )
    sequences = (
        BeamOutputSequence((3, 4), -1.25, (0, 0), "length", None),
        BeamOutputSequence((5, 6), -2.5, (0, 1), "length", None),
    )
    result = BeamBatchResult(
        1, "request-1", BeamBatchOutput((BeamRequestOutput(0, "request-1", sequences),)), None
    )
    processor = BeamSearchOutputProcessor(
        request, tokenizer, 10 if begin else None, 20 if end else None, "source"
    )
    return processor, call, request, result, tokenizer


@pytest.mark.parametrize("begin", [None, "begin"])
@pytest.mark.parametrize("end", [None, "end"])
@pytest.mark.parametrize("first_field", ["tokens", "text", "logprobs"])
def test_output_boundaries_and_lazy_materialization(
    begin: str | None, end: str | None, first_field: str
) -> None:
    processor, call, request, result, tokenizer = _case(begin, end)
    beams = processor.process_outputs(call, result)[0].sequences
    assert tokenizer.decode_calls == 0
    assert all(type(beam) is _LazyMaterializedBeamSequence for beam in beams)
    assert [beam.cum_logprob for beam in beams] == [-1.25, -2.5]
    assert tokenizer.decode_calls == 0

    prefix = [1, 2] + ([10] if begin else [])
    expected = [prefix + tail + ([20] if end else []) for tail in ([3, 4], [5, 6])]
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
        assert beam.finish_reason == "length"
        assert beam.stop_reason is None
    assert tokenizer.decode_calls == 2
    assert request.items[0].request.prompt_token_ids == prefix


@pytest.mark.parametrize("operation", [copy.copy, replace, asdict])
def test_public_dataclass_contract(operation: Any) -> None:
    processor, call, request, result, _ = _case()
    beam = processor.process_outputs(call, result)[0].sequences[0]
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
    processor, call, _, result, _ = _case()
    old = processor.process_outputs(call, result)[0].sequences
    processor2, call2, _, result2, _ = _case(None, None)
    new = processor2.process_outputs(call2, result2)[0].sequences
    new[0].tokens.append(99)
    old[0].tokens.append(88)
    old[0].logprobs.append({})
    assert old[1].tokens == [1, 2, 10, 5, 6, 20]
    assert old[1].logprobs == []
    assert new[1].tokens == [1, 2, 5, 6]
    assert new[1].logprobs == []
    assert old[0].tokens == [1, 2, 10, 3, 4, 20, 88]
    assert new[0].tokens == [1, 2, 3, 4, 99]


@pytest.mark.parametrize("end", [None, "end"])
def test_online_output_boundary_is_unchanged(end: str | None) -> None:
    processor, call, _, result, _ = _case(end=end)
    online = OnlineBeamSearchRequest(call.prompts[0], "external-id", call.params)
    output = processor.process_online_outputs(online, result)[0]
    assert output.request_id == "external-id"
    assert output.prompt_token_ids == [1, 2]
    assert output.outputs[0].token_ids == [10, 3, 4] + ([20] if end else [])
    assert output.outputs[1].token_ids == [10, 5, 6] + ([20] if end else [])
    assert output.outputs[0].text == ("10|3|4|20" if end else "10|3|4")
