# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Validated public-output conversion for persistent Beam Search V1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, cast

import numpy as np
from vllm.inputs import tokens_input

from vllm_gr.entrypoints.beam_search.boundary_tokens import boundary_token_id
from vllm_gr.entrypoints.beam_search.exceptions import BeamSearchEngineError
from vllm_gr.entrypoints.beam_search.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.entrypoints.common.detokenization import fast_batch_decode, fast_decode
from vllm_gr.outputs import (
    CompactBeamSearchOutput,
    CompactBeamSearchSequence,
    FlatBeamSearchOutput,
    FlatBeamSearchSequences,
)
from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchFailure,
    BeamBatchOutput,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputSequence,
    BeamRequestOutput,
    BeamSequenceColumns,
    materialize_beam_result,
    sequence_columns,
)


def _raise_engine_error(code: str, message: str, item_index: int | None = None) -> NoReturn:
    raise BeamSearchEngineError(BeamBatchFailure(code, message, item_index))


def _result_output(result: BeamBatchResult) -> BeamBatchOutput:
    if result.failure is not None:
        raise BeamSearchEngineError(result.failure)
    return result.output  # type: ignore[return-value]


def _materialized_sequences(item: BeamRequestOutput) -> tuple[BeamOutputSequence, ...]:
    sequences = item.sequences
    if not isinstance(sequences, tuple):
        raise TypeError("public Beam output conversion requires materialized sequences")
    return sequences


def _token_id(tokenizer: Any, token: str | None, name: str) -> int | None:
    return boundary_token_id(tokenizer, token, name)


def _prompt_text(prompt: Any) -> str | None:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, Mapping):
        text = prompt.get("prompt")
        if isinstance(text, str):
            return text
    return None


def _prompt_boundaries(
    tokenizer: Any,
    params: Any,
    boundary_token_ids: tuple[int | None, int | None] | None = None,
) -> tuple[int | None, int | None]:
    if boundary_token_ids is None:
        begin_token_id = _token_id(tokenizer, getattr(params, "begin_token", None), "begin_token")
        end_token_id = _token_id(tokenizer, getattr(params, "end_token", None), "end_token")
    else:
        begin_token_id, end_token_id = boundary_token_ids
    return begin_token_id, end_token_id


def _as_prompt_list(tokens: Any) -> list[int]:
    return tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)


def _prepared_prompt_lists(request: BeamBatchRequest) -> list[list[int]]:
    """Materialize accepted prompts once per request, never per Beam."""
    return [_as_prompt_list(item.request.prompt_token_ids) for item in request.items]


def _original_prompt_lists(
    prepared_prompts: list[list[int]],
    begin_token_id: int | None,
) -> list[list[int]]:
    if begin_token_id is None:
        return prepared_prompts
    return [prepared[:-1] for prepared in prepared_prompts]


def _output_suffix(sequence: BeamOutputSequence, end_token_id: int | None) -> list[int]:
    suffix = list(sequence.token_ids)
    # end_token reserves one public-output position but is not sampled when a
    # fixed-length Beam survives. Stop completions already own their boundary
    # and must respect include_stop_token without an artificial end marker.
    if sequence.finish_reason == "length" and end_token_id is not None:
        suffix.append(end_token_id)
    return suffix


def _compact_sequences(
    sequences: tuple[BeamOutputSequence, ...],
    begin: tuple[int, ...],
    end: tuple[int, ...],
) -> tuple[CompactBeamSearchSequence, ...]:
    return tuple(
        CompactBeamSearchSequence(
            token_ids=begin
            + sequence.token_ids
            + (end if sequence.finish_reason == "length" else ()),
            cumulative_logprob=float(cast(float, sequence.cumulative_logprob)),
            finish_reason=sequence.finish_reason,
            stop_reason=sequence.stop_token_id,
        )
        for sequence in sequences
    )


def _flat_sequences(
    sequences: BeamSequenceColumns,
    begin: tuple[int, ...],
    end: tuple[int, ...],
) -> FlatBeamSearchSequences:
    tokens, scores, _ = sequence_columns(sequences)
    if scores is None:
        raise ValueError("flat Beam output requires cumulative logprobs")
    if not begin and not end:
        return FlatBeamSearchSequences(tokens, scores)

    # Boundary IDs can exceed the int32 storage used for sampled token IDs.
    token_end = len(begin) + tokens.shape[1]
    rows = np.empty((sequences.count, token_end + len(end)), dtype=np.int64)
    rows[:, : len(begin)] = begin
    rows[:, len(begin) : token_end] = tokens
    rows[:, token_end:] = end
    tokens_with_boundaries = np.frombuffer(rows.tobytes(), dtype=rows.dtype).reshape(rows.shape)
    return FlatBeamSearchSequences(tokens_with_boundaries, scores)


def _can_share_prompt_decode(tokenizer: Any, prepared_prompt: Sequence[int]) -> bool:
    try:
        from tokenizers.decoders import ByteLevel
        from transformers import Qwen2TokenizerFast

        if (
            isinstance(tokenizer, Qwen2TokenizerFast)
            and isinstance(tokenizer.backend_tokenizer.decoder, ByteLevel)
            and not tokenizer.clean_up_tokenization_spaces
            and len(prepared_prompt) > 0
        ):
            final_prompt_token = tokenizer.convert_ids_to_tokens(int(prepared_prompt[-1]))
            return bool(final_prompt_token) and final_prompt_token.isascii()
    except (AttributeError, ImportError):
        pass
    return False


class _V1BeamTailMaterializer:
    """Populate one V1 Beam on first public access using PR380's lazy class."""

    __slots__ = ("_tokenizer", "_prepared", "_shared_prompt", "_prompt_text")

    def __init__(self, tokenizer: Any, prepared_prompt: list[int]) -> None:
        self._tokenizer = tokenizer
        # Materialized once per request; Beams only copy the reference list.
        self._prepared = prepared_prompt
        self._shared_prompt: bool | None = None
        self._prompt_text: str | None = None

    def __call__(self, beam: Any) -> None:
        from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence

        suffix = beam.__dict__.pop("_v1_suffix")
        full = self._prepared.copy()
        full.extend(suffix)
        if self._shared_prompt is None:
            self._shared_prompt = _can_share_prompt_decode(self._tokenizer, self._prepared)
        if self._shared_prompt:
            if self._prompt_text is None:
                self._prompt_text = fast_decode(self._tokenizer, self._prepared)
            text = self._prompt_text + fast_decode(self._tokenizer, suffix)
        else:
            text = fast_decode(self._tokenizer, full)
        beam.__dict__["tokens"] = full
        beam.__dict__["logprobs"] = []
        beam.__dict__["text"] = text
        beam.__dict__["_lazy_materializer"] = None
        beam.__class__ = BeamSearchSequence


class BeamSearchOutputProcessor:
    """Convert one trusted terminal result without retaining session state."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        boundary_token_ids: tuple[int | None, int | None] | None = None,
        original_prompt_token_ids: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        if tokenizer is None:
            raise ValueError("beam_search output conversion requires a tokenizer")
        self.tokenizer = tokenizer
        # Prepared IDs describe the accepted request, even if caller params
        # change while it runs. Direct output-converter callers still resolve.
        self._boundary_token_ids = boundary_token_ids
        self._original_prompt_token_ids = original_prompt_token_ids

    def _accepted_prompt(
        self,
        request: BeamBatchRequest,
        item_index: int,
        begin_token_id: int | None,
    ) -> tuple[int, ...]:
        """Accepted prompt IDs, reusing the caller-side snapshot when present."""
        if self._original_prompt_token_ids is not None:
            return self._original_prompt_token_ids[item_index]
        prepared = _as_prompt_list(request.items[item_index].request.prompt_token_ids)
        return tuple(prepared[:-1] if begin_token_id is not None else prepared)

    def process_outputs(
        self,
        call: OfflineBeamSearchRequest,
        request: BeamBatchRequest,
        result: BeamBatchResult,
    ) -> list[Any]:
        output = _result_output(result)
        try:
            begin_token_id, end_token_id = _prompt_boundaries(
                self.tokenizer, call.params, self._boundary_token_ids
            )

            if call.output_mode != "full":
                begin = () if begin_token_id is None else (begin_token_id,)
                end = () if end_token_id is None else (end_token_id,)
                if call.output_mode == "flat":
                    if any(not -(1 << 63) <= token < (1 << 63) for token in (*begin, *end)):
                        # Keep Python integer token IDs outside the array's range.
                        output = _result_output(materialize_beam_result(result))
                    return [
                        FlatBeamSearchOutput(
                            sequences=(
                                _flat_sequences(item.sequences, begin, end)
                                if isinstance(item.sequences, BeamSequenceColumns)
                                else _compact_sequences(item.sequences, begin, end)
                            ),
                            prompt_token_ids=self._accepted_prompt(
                                request, item.item_index, begin_token_id
                            ),
                        )
                        for item in output.items
                    ]
                return [
                    CompactBeamSearchOutput(
                        sequences=_compact_sequences(_materialized_sequences(item), begin, end),
                        prompt_token_ids=self._accepted_prompt(
                            request, item.item_index, begin_token_id
                        ),
                    )
                    for item in output.items
                ]

            from vllm.entrypoints.generate.beam_search.utils import BeamSearchOutput

            from vllm_gr.entrypoints.beam_search.lazy_output import (
                _LazyMaterializedBeamSequence,
            )

            prepared_prompts = _prepared_prompt_lists(request)
            original_prompts = _original_prompt_lists(prepared_prompts, begin_token_id)
            end_suffix = () if end_token_id is None else (end_token_id,)
            converted = []
            for item in output.items:
                source_prompt = call.prompts[item.item_index]
                prompt = tokens_input(
                    original_prompts[item.item_index],
                    prompt=_prompt_text(source_prompt),
                    cache_salt=request.items[item.item_index].request.cache_salt,
                )
                materializer = _V1BeamTailMaterializer(
                    self.tokenizer, prepared_prompts[item.item_index]
                )
                lora_request = request.items[item.item_index].request.lora_request
                sequences = []
                for sequence in _materialized_sequences(item):
                    # Initialize the final lazy type in one pass. Its generated
                    # __init__ would invoke three Python descriptor setters per
                    # beam; constructing the base first also needs a class swap.
                    # Keep every public BeamSearchSequence field initialized.
                    beam = object.__new__(_LazyMaterializedBeamSequence)
                    beam.__dict__ = {
                        "orig_prompt": prompt,
                        "tokens": [],
                        "logprobs": [],
                        "lora_request": lora_request,
                        "cum_logprob": float(cast(float, sequence.cumulative_logprob)),
                        "text": None,
                        "finish_reason": sequence.finish_reason,
                        "stop_reason": sequence.stop_token_id,
                        # Stop completions already carry their requested boundary.
                        "_v1_suffix": sequence.token_ids + end_suffix
                        if sequence.finish_reason == "length"
                        else sequence.token_ids,
                        "_lazy_materializer": materializer,
                    }
                    sequences.append(beam)
                converted.append(BeamSearchOutput(sequences=sequences))
            return converted
        except BeamSearchEngineError:
            raise
        except Exception as exc:
            _raise_engine_error("output_conversion_failed", str(exc))

    def process_online_outputs(
        self,
        call: OnlineBeamSearchRequest,
        request: BeamBatchRequest,
        result: BeamBatchResult,
    ) -> list[Any]:
        output = _result_output(result)
        try:
            begin_token_id, end_token_id = _prompt_boundaries(
                self.tokenizer, call.params, self._boundary_token_ids
            )
            original_prompts = _original_prompt_lists(
                _prepared_prompt_lists(request), begin_token_id
            )

            from vllm.outputs import CompletionOutput, RequestOutput

            results = []
            for item in output.items:
                item_index = item.item_index
                sequences = _materialized_sequences(item)
                completion_token_ids = [
                    ([] if begin_token_id is None else [begin_token_id])
                    + _output_suffix(sequence, end_token_id)
                    for sequence in sequences
                ]
                texts = fast_batch_decode(self.tokenizer, completion_token_ids)
                lora_request = request.items[item_index].request.lora_request
                completions = [
                    CompletionOutput(
                        index=choice_index,
                        text=text,
                        token_ids=token_ids,
                        cumulative_logprob=sequence.cumulative_logprob,
                        logprobs=None,
                        finish_reason=sequence.finish_reason,
                        stop_reason=sequence.stop_token_id,
                        lora_request=lora_request,
                    )
                    for choice_index, (sequence, token_ids, text) in enumerate(
                        zip(sequences, completion_token_ids, texts, strict=True)
                    )
                ]
                prompt_token_ids = original_prompts[item_index]
                prompt_text = _prompt_text(call.prompts[item_index])
                if prompt_text is None:
                    prompt_text = fast_decode(self.tokenizer, prompt_token_ids)
                results.append(
                    RequestOutput(
                        request_id=call.request_id,
                        prompt=prompt_text,
                        prompt_token_ids=prompt_token_ids,
                        prompt_logprobs=None,
                        outputs=completions,
                        finished=True,
                        lora_request=lora_request,
                    )
                )
            return results
        except BeamSearchEngineError:
            raise
        except Exception as exc:
            _raise_engine_error("output_conversion_failed", str(exc))
