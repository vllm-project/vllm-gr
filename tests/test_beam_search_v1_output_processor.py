# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
from vllm.inputs import tokens_input
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchEngineError
from vllm_gr.entrypoints.beam_search_v1.output_processor import (
    BeamSearchOutputProcessor,
)
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.sampling_params import BeamSearchParams
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
    eos_token_id = 2

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<begin>": 7, "<end>": 8}[token]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(f"<{token_id}>" for token_id in token_ids)

    def batch_decode(self, sequences: list[list[int]]) -> list[str]:
        return [self.decode(sequence) for sequence in sequences]


def _engine_request() -> EngineCoreRequest:
    values = dict(
        request_id="batch-0",
        prompt_token_ids=[10, 11, 7],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=2, temperature=1.0),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt="salt",
        data_parallel_rank=None,
    )
    if "eos_token_id" in EngineCoreRequest.__struct_fields__:
        values["eos_token_id"] = None
    return EngineCoreRequest(**values)


def _request_and_result() -> tuple[BeamBatchRequest, BeamBatchResult]:
    request = BeamBatchRequest(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id="batch-0",
        items=(BeamBatchItem(0, _engine_request()),),
        beam_params=BeamBatchParams(2, 2, 1.0, (8,), False),
        output_options=BeamOutputOptions(2, False, True, True),
    )
    result = BeamBatchResult(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id="batch-0",
        output=BeamBatchOutput(
            (
                BeamRequestOutput(
                    item_index=0,
                    request_id="batch-0",
                    sequences=(
                        BeamOutputSequence((20, 21), -0.1, (0, 0), "length", None),
                        BeamOutputSequence((30,), -0.2, (0, 1), "stop", 8),
                    ),
                ),
            )
        ),
        failure=None,
    )
    return request, result


def _params() -> BeamSearchParams:
    return BeamSearchParams(
        beam_width=2,
        max_tokens=4,
        begin_token="<begin>",
        end_token="<end>",
        include_stop_str_in_output=False,
    )


def test_converts_validated_terminal_result_for_offline_and_online_frontends() -> None:
    request, result = _request_and_result()
    processor = BeamSearchOutputProcessor(_Tokenizer())

    offline = processor.process_outputs(
        OfflineBeamSearchRequest(
            [tokens_input([10, 11], prompt="prompt", cache_salt="salt")], _params()
        ),
        request,
        result,
    )[0]
    assert all(
        sequence.__class__.__name__ == "_LazyMaterializedBeamSequence"
        for sequence in offline.sequences
    )
    assert [sequence.tokens for sequence in offline.sequences] == [
        [10, 11, 7, 20, 21, 8],
        [10, 11, 7, 30],
    ]
    assert [sequence.text for sequence in offline.sequences] == [
        "<10><11><7><20><21><8>",
        "<10><11><7><30>",
    ]
    assert offline.sequences[1].stop_reason == 8
    assert offline.sequences[0].orig_prompt["prompt_token_ids"] == [10, 11]

    online = processor.process_online_outputs(
        OnlineBeamSearchRequest(
            prompt=tokens_input([10, 11], prompt="prompt"),
            request_id="external-0",
            params=_params(),
        ),
        request,
        result,
    )[0]
    assert online.request_id == "external-0"
    assert online.prompt == "prompt"
    assert online.prompt_token_ids == [10, 11]
    assert [completion.token_ids for completion in online.outputs] == [
        [7, 20, 21, 8],
        [7, 30],
    ]
    assert [completion.text for completion in online.outputs] == [
        "<7><20><21><8>",
        "<7><30>",
    ]
    assert all(completion.logprobs is None for completion in online.outputs)


def test_preserves_worker_failures() -> None:
    request, _ = _request_and_result()
    processor = BeamSearchOutputProcessor(_Tokenizer())
    call = OnlineBeamSearchRequest([10, 11], "external-0", _params())

    failure = BeamBatchFailure("beam_device_error", "device failure", 0)
    with pytest.raises(BeamSearchEngineError) as propagated:
        processor.process_online_outputs(
            call,
            request,
            BeamBatchResult(BEAM_BATCH_PROTOCOL_VERSION, "batch-0", None, failure),
        )
    assert propagated.value.failure is failure
