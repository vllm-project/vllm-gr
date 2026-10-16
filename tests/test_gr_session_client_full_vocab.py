# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the common online/offline terminal transport with real EOS output."""

import asyncio
from types import SimpleNamespace

import msgspec
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.config import BeamCandidateMode
from vllm_gr.v1.beam.batch_contracts import (
    FULL_VOCAB_BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchItem,
    BeamBatchOutput,
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
    BeamContractError,
    BeamOutputOptions,
    BeamOutputSequence,
    BeamRequestOutput,
)
from vllm_gr.v1.engine.gr_session_client import (
    AsyncGRSessionHandle,
    SyncGRSessionHandle,
    _terminal_result,
    wait_gr_session_async,
    wait_gr_session_sync,
)


def _request(*, include_stop=True, scores=True, parents=True):
    items = tuple(
        BeamBatchItem(
            item_index=index,
            request=EngineCoreRequest(
                request_id=f"session-{index}",
                prompt_token_ids=[1, 2, 3],
                mm_features=None,
                sampling_params=SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True),
                pooling_params=None,
                arrival_time=1.0,
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            ),
        )
        for index in range(4)
    )
    return BeamBatchRequest(
        protocol_version=FULL_VOCAB_BEAM_BATCH_PROTOCOL_VERSION,
        batch_id="session-0",
        items=items,
        beam_params=BeamBatchParams(
            beam_width=4,
            max_tokens=3,
            temperature=0.0,
            stop_token_ids=(0,),
            ignore_stop_tokens=False,
            constraint_resource_id=None,
            candidate_mode=BeamCandidateMode.FULL_VOCAB,
        ),
        output_options=BeamOutputOptions(
            num_return_sequences=4,
            include_stop_token=include_stop,
            include_cumulative_logprob=scores,
            include_parent_beam_indices=parents,
        ),
    )


def _result(request):
    options = request.output_options
    items = []
    # The first three prompts stop after 1, 2, and 3 generation steps; the
    # fourth exhausts the budget. EOS=0 must not be mistaken for missing EOS.
    for source, generated in zip(request.items, ((0,), (7, 0), (7, 8, 0), (7, 8, 9))):
        stopped = generated[-1] == 0
        tokens = generated[:-1] if stopped and not options.include_stop_token else generated
        sequence = BeamOutputSequence(
            token_ids=tokens,
            cumulative_logprob=-1.0 if options.include_cumulative_logprob else None,
            parent_beam_indices=(0,) * len(generated)
            if options.include_parent_beam_indices
            else (),
            finish_reason="stop" if stopped else "length",
            stop_token_id=0 if stopped else None,
        )
        items.append(BeamRequestOutput(source.item_index, source.request.request_id, (sequence,)))
    return BeamBatchResult(
        request.protocol_version, request.batch_id, BeamBatchOutput(tuple(items)), None
    )


def _output(request, result, wire):
    # Round trip through the actual untyped msgpack carrier used by IPC.
    payload = msgspec.msgpack.decode(msgspec.msgpack.encode(result)) if wire else result
    return SimpleNamespace(
        request_id=request.batch_id,
        finished=True,
        kv_transfer_params={"gr_batch_result": payload},
    )


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("include_stop", [False, True])
@pytest.mark.parametrize(
    "scores,parents", [(False, False), (False, True), (True, False), (True, True)]
)
def test_full_vocab_shared_transport_accepts_mixed_eos_lengths(
    wire, online, include_stop, scores, parents
):
    request = _request(include_stop=include_stop, scores=scores, parents=parents)
    expected = _result(request)
    terminal = _output(request, expected, wire)
    progress = SimpleNamespace(request_id=request.batch_id, finished=False)
    if online:

        async def wait():
            collector = asyncio.Queue()
            collector.put_nowait(progress)
            collector.put_nowait(terminal)
            return await wait_gr_session_async(
                None, request.batch_id, AsyncGRSessionHandle(request, collector)
            )

        actual = asyncio.run(wait())
    else:
        steps = iter(([progress], [terminal]))
        actual = wait_gr_session_sync(
            SimpleNamespace(step=lambda: next(steps)),
            request.batch_id,
            SyncGRSessionHandle(request),
        )
    assert actual == expected


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "unknown_stop",
        "missing_stop",
        "wrong_last_token",
        "empty_tokens",
        "over_budget",
        "short_length",
        "parent_length",
        "parent_width",
        "hidden_stop_parent_missing",
        "hidden_stop_over_budget",
        "ignored_stop",
        "constrained_mode",
    ],
)
def test_full_vocab_terminal_transport_rejects_malformed_completion(wire, fault):
    request = _request()
    result = _result(request)
    sequence = result.output.items[0].sequences[0]
    if fault == "unknown_stop":
        sequence = msgspec.structs.replace(sequence, stop_token_id=99)
    elif fault == "missing_stop":
        sequence = msgspec.structs.replace(sequence, stop_token_id=None)
    elif fault == "wrong_last_token":
        sequence = msgspec.structs.replace(sequence, token_ids=(99,))
    elif fault == "empty_tokens":
        sequence = msgspec.structs.replace(sequence, token_ids=())
    elif fault == "over_budget":
        sequence = msgspec.structs.replace(sequence, token_ids=(7, 8, 9, 0))
    elif fault == "short_length":
        sequence = msgspec.structs.replace(sequence, finish_reason="length", stop_token_id=None)
    elif fault == "parent_length":
        sequence = msgspec.structs.replace(sequence, parent_beam_indices=(0, 0))
    elif fault == "parent_width":
        sequence = msgspec.structs.replace(sequence, parent_beam_indices=(4,))
    elif fault in ("hidden_stop_parent_missing", "hidden_stop_over_budget"):
        request = msgspec.structs.replace(
            request,
            output_options=msgspec.structs.replace(
                request.output_options, include_stop_token=False
            ),
        )
        sequence = msgspec.structs.replace(
            sequence,
            token_ids=() if fault == "hidden_stop_parent_missing" else (7, 8, 9),
            parent_beam_indices=() if fault == "hidden_stop_parent_missing" else (0, 0, 0, 0),
        )
    elif fault == "ignored_stop":
        request = msgspec.structs.replace(
            request,
            beam_params=msgspec.structs.replace(request.beam_params, ignore_stop_tokens=True),
        )
    elif fault == "constrained_mode":
        request = msgspec.structs.replace(
            request,
            protocol_version=1,
            beam_params=msgspec.structs.replace(
                request.beam_params,
                candidate_mode=BeamCandidateMode.CONSTRAINT_TABLE,
                constraint_resource_id="default",
            ),
        )
        result = msgspec.structs.replace(result, protocol_version=1)
    items = (
        msgspec.structs.replace(result.output.items[0], sequences=(sequence,)),
    ) + result.output.items[1:]
    result = msgspec.structs.replace(result, output=BeamBatchOutput(items))
    with pytest.raises(ValueError):
        _terminal_result(request, _output(request, result, wire))


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize(
    "tokens,score,parents", [((True,), -1.0, (0,)), ((0,), float("nan"), (0,)), ((0,), -1.0, (-1,))]
)
def test_terminal_transport_wraps_typed_validation_error_with_cause(wire, tokens, score, parents):
    request = _request()
    result = _result(request)
    bad = BeamOutputSequence(tokens, score, parents, "stop", 0)
    items = (
        msgspec.structs.replace(result.output.items[0], sequences=(bad,)),
    ) + result.output.items[1:]
    result = msgspec.structs.replace(result, output=BeamBatchOutput(items))
    with pytest.raises(BeamContractError, match="malformed terminal Beam result") as exc_info:
        _terminal_result(request, _output(request, result, wire))
    assert isinstance(exc_info.value.__cause__, msgspec.ValidationError)


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("failure_index", [None, 0, 1, 3, 4])
def test_terminal_transport_failure_index_matches_submitted_batch(wire, batch_size, failure_index):
    request = _request()
    request = msgspec.structs.replace(request, items=request.items[:batch_size])
    result = BeamBatchResult(
        request.protocol_version,
        request.batch_id,
        output=None,
        failure=BeamBatchFailure("no_candidates", "no finite candidate", failure_index),
    )
    output = _output(request, result, wire)
    if failure_index is None or failure_index < batch_size:
        assert _terminal_result(request, output) == result
    else:
        with pytest.raises(ValueError, match="invalid item identity"):
            _terminal_result(request, output)
