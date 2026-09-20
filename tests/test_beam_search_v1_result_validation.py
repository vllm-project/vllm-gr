# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""Light public result checks and optional internal diagnostics on both transports."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import msgspec
import pytest

from tests.test_gr_async_scheduler import _batch_request
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
    BeamBatchFailure,
    BeamBatchOutput,
    BeamBatchResult,
    BeamOutputSequence,
    BeamRequestOutput,
)
from vllm_gr.v1.engine.gr_result_validation import has_validated_sequences
from vllm_gr.v1.engine.gr_session_client import _terminal_result


def _result() -> BeamBatchResult:
    sequences = (
        BeamOutputSequence((0, 2, 3), -1.0, (0, 0, 1), "length", None),
        BeamOutputSequence((4, 5, 6), -2.0, (0, 1, 1), "length", None),
    )
    return BeamBatchResult(
        1, "session-0", BeamBatchOutput((BeamRequestOutput(0, "session-0", sequences),)), None
    )


def _replace_sequence(result: BeamBatchResult, **changes: Any) -> BeamBatchResult:
    assert result.output is not None
    item = result.output.items[0]
    sequence = msgspec.structs.replace(item.sequences[0], **changes)
    item = msgspec.structs.replace(item, sequences=(sequence,) + item.sequences[1:])
    return msgspec.structs.replace(result, output=BeamBatchOutput((item,)))


def _receive(result: BeamBatchResult, wire: bool) -> BeamBatchResult:
    payload = msgspec.msgpack.decode(msgspec.msgpack.encode(result)) if wire else result
    return _terminal_result(
        _batch_request(), SimpleNamespace(kv_transfer_params={"gr_batch_result": payload})
    )


def _processor(monkeypatch: pytest.MonkeyPatch, strict: bool) -> BeamSearchOutputProcessor:
    monkeypatch.setenv("VLLM_GR_BEAM_STRICT_RESULT_VALIDATION", "1" if strict else "0")
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids))
    return BeamSearchOutputProcessor(_batch_request(), tokenizer, None, None, None)


def _convert(processor: BeamSearchOutputProcessor, result: BeamBatchResult, online: bool) -> Any:
    params = BeamSearchParams(beam_width=4, max_tokens=3, ignore_eos=True)
    if online:
        return processor.process_online_outputs(
            OnlineBeamSearchRequest([1, 2, 3], "public-id", params), result
        )[0]
    return processor.process_outputs(OfflineBeamSearchRequest([[1, 2, 3]], params), result)[0]


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize(
    "changes",
    [
        {"token_ids": (True, 2, 3)},
        {"token_ids": (1.0, 2, 3)},
        {"token_ids": ("1", 2, 3)},
        {"parent_beam_indices": (0, False, 1)},
        {"parent_beam_indices": (0, "1", 1)},
        {"cumulative_logprob": True},
        {"cumulative_logprob": "-1"},
    ],
)
def test_receive_rejects_malformed_nested_types(wire: bool, changes: dict[str, Any]) -> None:
    with pytest.raises(msgspec.ValidationError):
        _receive(_replace_sequence(_result(), **changes), wire)


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize(
    "changes,match",
    [
        ({"token_ids": (-1, 2, 3)}, "non-negative"),
        ({"token_ids": (1, 2)}, "budget"),
        ({"token_ids": (1, 2, 3, 4)}, "budget"),
        ({"finish_reason": "stop", "stop_token_id": 2}, "budget"),
        ({"stop_token_id": 2}, "ignored EOS"),
        ({"cumulative_logprob": None}, "finite"),
        ({"cumulative_logprob": float("nan")}, "finite"),
        ({"cumulative_logprob": float("inf")}, "finite"),
        ({"cumulative_logprob": -float("inf")}, "finite"),
    ],
)
def test_public_value_checks_always_apply(
    monkeypatch: pytest.MonkeyPatch,
    wire: bool,
    strict: bool,
    online: bool,
    changes: dict[str, Any],
    match: str,
) -> None:
    processor = _processor(monkeypatch, strict)
    result = _receive(_replace_sequence(_result(), **changes), wire)
    with pytest.raises(BeamSearchEngineError, match=match) as exc:
        _convert(processor, result, online)
    assert exc.value.failure.code == "invalid_result"


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize(
    "changes,match",
    [
        ({"parent_beam_indices": (0, 1)}, "parent chain"),
        ({"parent_beam_indices": (0, 4, 1)}, "parent chain"),
        ({"parent_beam_indices": (0, -1, 1)}, "non-negative"),
        ({"cumulative_logprob": -3.0}, "sorted"),
    ],
)
def test_internal_lineage_and_ranking_checks_are_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    wire: bool,
    strict: bool,
    online: bool,
    changes: dict[str, Any],
    match: str,
) -> None:
    processor = _processor(monkeypatch, strict)
    result = _receive(_replace_sequence(_result(), **changes), wire)
    if strict:
        with pytest.raises(BeamSearchEngineError, match=match):
            _convert(processor, result, online)
    else:
        assert result.output is not None
        output = _convert(processor, result, online)
        assert len(output.outputs if online else output.sequences) == len(
            result.output.items[0].sequences
        )


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "version",
        "both",
        "neither",
        "empty",
        "multiple",
        "item",
        "request",
        "zero",
        "too_many",
        "failure_code",
        "failure_message",
        "failure_index",
    ],
)
def test_result_envelope_and_identity_checks_always_apply(
    monkeypatch: pytest.MonkeyPatch, wire: bool, strict: bool, online: bool, case: str
) -> None:
    result = _result()
    assert result.output is not None
    item = result.output.items[0]
    if case == "version":
        result = msgspec.structs.replace(result, protocol_version=2)
    elif case in ("both", "neither"):
        result = msgspec.structs.replace(
            result,
            output=result.output if case == "both" else None,
            failure=BeamBatchFailure("error", "failed", None) if case == "both" else None,
        )
    elif case.startswith("failure_"):
        failure = BeamBatchFailure("error", "failed", None)
        field, value = {
            "failure_code": ("code", " "),
            "failure_message": ("message", ""),
            "failure_index": ("item_index", -1),
        }[case]
        result = msgspec.structs.replace(
            result, output=None, failure=msgspec.structs.replace(failure, **{field: value})
        )
    else:
        items: tuple[BeamRequestOutput, ...] = (item,)
        if case == "empty":
            items = ()
        elif case == "multiple":
            items = (item, item)
        elif case == "item":
            items = (msgspec.structs.replace(item, item_index=1),)
        elif case == "request":
            items = (msgspec.structs.replace(item, request_id="wrong"),)
        elif case in ("zero", "too_many"):
            items = (
                msgspec.structs.replace(
                    item, sequences=() if case == "zero" else item.sequences * 3
                ),
            )
        result = msgspec.structs.replace(result, output=BeamBatchOutput(items))
    with pytest.raises(BeamSearchEngineError) as exc:
        _convert(_processor(monkeypatch, strict), _receive(result, wire), online)
    assert exc.value.failure.code == "invalid_result"


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("online", [False, True])
def test_strict_mode_preserves_valid_public_outputs(
    monkeypatch: pytest.MonkeyPatch, wire: bool, online: bool
) -> None:
    result = _receive(_result(), wire)
    outputs = []
    for strict in (False, True):
        output = _convert(_processor(monkeypatch, strict), result, online)
        beams = output.outputs if online else output.sequences
        outputs.append(
            [
                (
                    beam.token_ids if online else beam.tokens,
                    beam.text,
                    beam.cumulative_logprob if online else beam.cum_logprob,
                )
                for beam in beams
            ]
        )
    assert outputs[0] == outputs[1]


def test_strict_diagnostics_are_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_GR_BEAM_STRICT_RESULT_VALIDATION", raising=False)
    processor = BeamSearchOutputProcessor(_batch_request(), None, None, None, None)
    assert not processor.strict_result_validation


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "changes,match",
    [
        ({"token_ids": (1, -1, 3)}, "non-negative"),
        ({"token_ids": (1, 2, -1)}, "non-negative"),
        ({"token_ids": ()}, "budget"),
        ({"finish_reason": "stop"}, "budget"),
        ({"stop_token_id": 0}, "ignored EOS"),
        ({"cumulative_logprob": None}, "finite"),
        ({"cumulative_logprob": float("nan")}, "finite"),
    ],
)
def test_public_checks_reject_invalid_last_beam_in_wide_result(
    monkeypatch: pytest.MonkeyPatch, online: bool, strict: bool, changes: dict[str, Any], match: str
) -> None:
    processor = _processor(monkeypatch, strict=strict)
    processor.tokenizer.decode = Mock(side_effect=AssertionError("invalid result was decoded"))
    processor.request = msgspec.structs.replace(
        processor.request,
        beam_params=msgspec.structs.replace(processor.request.beam_params, beam_width=128),
        output_options=msgspec.structs.replace(
            processor.request.output_options, num_return_sequences=128
        ),
    )
    result = _result()
    assert result.output is not None
    item = result.output.items[0]
    good = item.sequences[0]
    bad = msgspec.structs.replace(good, **changes)
    item = msgspec.structs.replace(item, sequences=(good,) * 127 + (bad,))
    result = msgspec.structs.replace(result, output=BeamBatchOutput((item,)))
    with pytest.raises(BeamSearchEngineError, match=match):
        _convert(processor, result, online)
    processor.tokenizer.decode.assert_not_called()


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_typed_value_validation_preserves_all_budgets(
    monkeypatch: pytest.MonkeyPatch, wire: bool, online: bool, length: int
) -> None:
    processor = _processor(monkeypatch, False)
    processor.request = msgspec.structs.replace(
        processor.request,
        beam_params=msgspec.structs.replace(processor.request.beam_params, max_tokens=length),
    )
    tokens = tuple(range(length))
    result = _result()
    assert result.output is not None
    item = msgspec.structs.replace(
        result.output.items[0],
        sequences=(BeamOutputSequence(tokens, -1.0, (0,) * length, "length", None),),
    )
    result = msgspec.structs.replace(result, output=BeamBatchOutput((item,)))
    payload = msgspec.to_builtins(result) if wire else result
    received = _terminal_result(
        processor.request, SimpleNamespace(kv_transfer_params={"gr_batch_result": payload})
    )
    assert has_validated_sequences(received, length) == (length <= 3)
    assert received == result and result == received
    assert not (received != result or result != received)
    assert hash(received) == hash(result)
    assert received.output == result.output
    assert received.output is not None
    assert received.output.items[0] == item
    assert received.output.items[0].sequences == item.sequences
    assert msgspec.to_builtins(received) == msgspec.to_builtins(result)
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(received), type=BeamBatchResult) == result
    output = _convert(processor, received, online)
    beam = output.outputs[0] if online else output.sequences[0]
    assert (beam.token_ids if online else beam.tokens)[-length:] == list(tokens)
    assert (beam.cumulative_logprob if online else beam.cum_logprob) == -1.0


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("change", ["output", "budget", "unvalidated_constructor"])
def test_semantic_validation_receipt_cannot_cover_replaced_values(
    monkeypatch: pytest.MonkeyPatch, online: bool, change: str
) -> None:
    processor = _processor(monkeypatch, False)
    result = _receive(_result(), True)
    assert has_validated_sequences(result, 3)
    if change == "budget":
        processor.request = msgspec.structs.replace(
            processor.request,
            beam_params=msgspec.structs.replace(processor.request.beam_params, max_tokens=2),
        )
    else:
        result = _replace_sequence(result, token_ids=(1, -1, 3))
        if change == "unvalidated_constructor":
            result = type(result)(
                result.protocol_version, result.batch_id, result.output, result.failure
            )
    assert not has_validated_sequences(result, processor.request.beam_params.max_tokens)
    with pytest.raises(BeamSearchEngineError) as exc:
        _convert(processor, result, online)
    assert exc.value.failure.code == "invalid_result"


def test_typed_validation_detaches_mutable_native_containers() -> None:
    tokens = [1, 2, 3]
    native = _replace_sequence(_result(), token_ids=tokens)
    received = _receive(native, False)
    assert has_validated_sequences(received, 3)
    tokens[2] = -1
    assert received.output is not None
    assert received.output.items[0].sequences[0].token_ids == (1, 2, 3)
    # Re-entry through an in-process transport must not serialize the private
    # receipt as an extra field of the four-field Worker wire contract.
    repeated = _receive(received, False)
    assert has_validated_sequences(repeated, 3)
    assert repeated.output == received.output


@pytest.mark.parametrize("wire", [False, True])
def test_failed_terminal_result_preserves_value_contract(wire: bool) -> None:
    original = BeamBatchResult(1, "session-0", None, BeamBatchFailure("failed", "failed", None))
    received = _receive(original, wire)
    assert received == original and original == received
    assert not (received != original or original != received)
    assert hash(received) == hash(original)
    assert msgspec.to_builtins(received) == msgspec.to_builtins(original)
    assert not has_validated_sequences(received, 3)
