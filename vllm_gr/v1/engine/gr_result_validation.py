# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate fixed-budget Beam values during the existing typed IPC conversion."""

import sys
from dataclasses import dataclass
from typing import Annotated, ClassVar, Generic, Literal, TypeVar

import msgspec

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchOutput,
    BeamBatchResult,
    BeamOutputSequence,
    BeamRequestOutput,
)

_Token = Annotated[int, msgspec.Meta(ge=0)]
_FiniteScore = Annotated[float, msgspec.Meta(ge=-sys.float_info.max, le=sys.float_info.max)]
_Tokens = TypeVar("_Tokens", bound=tuple[int, ...])


class _ContractValue:
    """Keep value equality and hashing compatible with the original wire types."""

    __slots__ = ()
    _value_type: ClassVar[type[msgspec.Struct]]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self._value_type):
            return NotImplemented
        return all(
            getattr(self, name) == getattr(other, name)
            for name in self._value_type.__struct_fields__
        )

    def __ne__(self, other: object) -> bool:
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __hash__(self) -> int:
        return hash(
            self._value_type(*(getattr(self, name) for name in self._value_type.__struct_fields__))
        )


class _SemanticSequence(_ContractValue, BeamOutputSequence, Generic[_Tokens]):
    _value_type = BeamOutputSequence
    token_ids: _Tokens
    cumulative_logprob: _FiniteScore
    finish_reason: Literal["length"]
    stop_token_id: None


class _SemanticRequestOutput(_ContractValue, BeamRequestOutput, Generic[_Tokens]):
    _value_type = BeamRequestOutput
    sequences: tuple[_SemanticSequence[_Tokens], ...]


class _SemanticOutput(_ContractValue, BeamBatchOutput, Generic[_Tokens]):
    _value_type = BeamBatchOutput
    items: tuple[_SemanticRequestOutput[_Tokens], ...]


# Keep validation metadata outside the wire fields. msgspec requires gc=True
# for structs with an instance dictionary.
class _SemanticResult(_ContractValue, BeamBatchResult, Generic[_Tokens], dict=True, gc=True):
    _value_type = BeamBatchResult
    output: _SemanticOutput[_Tokens] | None


# V1 currently supports these fixed generation budgets. Other lengths retain
# the ordinary typed boundary and the public processor's full value checks.
_RESULT_TYPES = {
    1: _SemanticResult[tuple[_Token]],
    2: _SemanticResult[tuple[_Token, _Token]],
    3: _SemanticResult[tuple[_Token, _Token, _Token]],
}


@dataclass(frozen=True, slots=True)
class _SequenceValidation:
    output: BeamBatchOutput
    token_count: int


def decode_terminal_result(payload: list | tuple, max_tokens: int) -> BeamBatchResult:
    result_type = _RESULT_TYPES.get(max_tokens)
    if result_type is None:
        return msgspec.convert(payload, type=BeamBatchResult)
    try:
        result = msgspec.convert(payload, type=result_type)
    except msgspec.ValidationError:
        # Keep malformed types as codec errors, but let the public processor
        # report invalid values with its existing invalid_result semantics.
        return msgspec.convert(payload, type=BeamBatchResult)
    if result.output is not None:
        # Non-field metadata is local to this frontend. Encoding still emits
        # exactly the original four fields, and structs.replace drops the receipt.
        result.__dict__["_validation"] = _SequenceValidation(result.output, max_tokens)
    return result


def has_validated_sequences(result: BeamBatchResult, max_tokens: int) -> bool:
    if type(result) is not _SemanticResult:
        return False
    validation = result.__dict__.get("_validation")
    # Typed conversion made the entire output tree immutable. A replacement
    # output or another budget must not inherit the original validation.
    return (
        type(validation) is _SequenceValidation
        and result.output is validation.output
        and max_tokens == validation.token_count
    )
