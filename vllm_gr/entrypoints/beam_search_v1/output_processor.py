# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert the shared terminal result without reprocessing the request budget."""

import math
import os
from collections.abc import Sequence
from typing import Any, cast

from vllm.entrypoints.generate.beam_search.utils import BeamSearchOutput
from vllm.inputs import tokens_input
from vllm.outputs import CompletionOutput, RequestOutput

from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchEngineError
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.tokenizer_fast import fast_batch_decode, fast_decode
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputSequence,
    validate_beam_batch_result,
)
from vllm_gr.v1.engine.gr_result_validation import has_validated_sequences


def _can_share_prompt_decode(tokenizer: Any, prepared_prompt: Sequence[int]) -> bool:
    try:
        from tokenizers.decoders import ByteLevel
        from transformers import Qwen2TokenizerFast

        if (
            isinstance(tokenizer, Qwen2TokenizerFast)
            and isinstance(tokenizer.backend_tokenizer.decoder, ByteLevel)
            and not tokenizer.clean_up_tokenization_spaces
            and prepared_prompt
        ):
            final_prompt_token = tokenizer.convert_ids_to_tokens(prepared_prompt[-1])
            return bool(final_prompt_token) and final_prompt_token.isascii()
    except (AttributeError, ImportError):
        pass
    return False


class _V1BeamTailMaterializer:
    """Populate one V1 Beam on first access to its lazy public fields."""

    __slots__ = ("_tokenizer", "_prepared", "_shared_prompt", "_prompt_text")

    def __init__(self, tokenizer: Any, prepared_prompt: Sequence[int]) -> None:
        self._tokenizer = tokenizer
        self._prepared = prepared_prompt
        self._shared_prompt: bool | None = None
        self._prompt_text: str | None = None

    def __call__(self, beam: Any) -> None:
        from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence

        suffix = beam.__dict__.pop("_v1_suffix")
        full = list(self._prepared)
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
    """Keep normalized prompt identity and the already resolved boundary tokens."""

    def __init__(
        self,
        request: BeamBatchRequest,
        tokenizer: Any,
        begin_token_id: int | None,
        end_token_id: int | None,
        prompt_text: str | None,
    ) -> None:
        self.request = request
        self.tokenizer = tokenizer
        self.begin = [] if begin_token_id is None else [begin_token_id]
        self.end = [] if end_token_id is None else [end_token_id]
        self.native_prompt = list(request.items[0].request.prompt_token_ids or ())
        self.prompt_tokens = self.native_prompt[:-1] if self.begin else self.native_prompt[:]
        self.prompt_text = prompt_text
        self.strict_result_validation = (
            os.environ.get("VLLM_GR_BEAM_STRICT_RESULT_VALIDATION") == "1"
        )

    def _convert_sequences(
        self, result: BeamBatchResult, call: OfflineBeamSearchRequest | None
    ) -> list[Any]:
        """Validate and construct each output once, before exposing any result."""
        try:
            if result.protocol_version != BEAM_BATCH_PROTOCOL_VERSION:
                raise ValueError("terminal result has an unsupported protocol_version")
            if result.batch_id != self.request.batch_id:
                raise ValueError("terminal result belongs to a different session")
            if (result.output is None) == (result.failure is None):
                raise ValueError("exactly one of output and failure must be present")
            if result.failure is not None:
                failure = result.failure
                if not failure.code.strip() or not failure.message.strip():
                    raise ValueError("terminal failure code and message must be non-empty")
                if failure.item_index is not None and not 0 <= failure.item_index < len(
                    self.request.items
                ):
                    raise ValueError("terminal failure item_index is outside the submitted batch")
                raise BeamSearchEngineError(failure)
            assert result.output is not None
            if len(result.output.items) != 1:
                raise ValueError("terminal result must contain one request")
            item = result.output.items[0]
            if item.item_index != 0 or item.request_id != self.request.items[0].request.request_id:
                raise ValueError("terminal result request identity differs from submission")
            sequences: tuple[BeamOutputSequence, ...] = item.sequences
            if not 0 < len(sequences) <= self.request.output_options.num_return_sequences:
                raise ValueError("terminal result has an invalid sequence count")
            self._validate_sequences(result, sequences)
            if self.strict_result_validation:
                self._validate_diagnostics(result, sequences)
            if call is not None:
                from vllm_gr.entrypoints.gr import _LazyMaterializedBeamSequence

                materializer = _V1BeamTailMaterializer(self.tokenizer, self.native_prompt)
                prompt = tokens_input(
                    self.prompt_tokens[:],
                    prompt=self.prompt_text,
                    cache_salt=self.request.items[0].request.cache_salt,
                )
                end_suffix = tuple(self.end)
                # Copy the request-shared fields in C, then fill only the
                # per-beam fields. Mutable lists remain private to each beam.
                copy_state = {
                    "orig_prompt": prompt,
                    "tokens": None,
                    "logprobs": None,
                    "cum_logprob": None,
                    "text": None,
                    "finish_reason": "length",
                    "stop_reason": None,
                    "lora_request": call.lora_request,
                    "_v1_suffix": None,
                    "_lazy_materializer": materializer,
                }.copy
            outputs: list[Any] = []
            append = outputs.append
            new = object.__new__
            for seq in sequences:
                tokens = seq.token_ids
                score = seq.cumulative_logprob
                if call is not None:
                    beam = new(_LazyMaterializedBeamSequence)
                    state = copy_state()
                    state["tokens"] = []
                    state["logprobs"] = []
                    state["cum_logprob"] = score
                    state["_v1_suffix"] = tokens + end_suffix
                    beam.__dict__ = state
                    append(beam)
                else:
                    append(
                        CompletionOutput(
                            index=len(outputs),
                            text="",
                            token_ids=self.begin + list(tokens) + self.end,
                            cumulative_logprob=score,
                            logprobs=None,
                            finish_reason="length",
                            stop_reason=None,
                        )
                    )
            return outputs
        except ValueError as exc:
            raise BeamSearchEngineError(BeamBatchFailure("invalid_result", str(exc), None)) from exc

    def _validate_sequences(
        self, result: BeamBatchResult, sequences: tuple[BeamOutputSequence, ...]
    ) -> None:
        expected_length = self.request.beam_params.max_tokens
        if has_validated_sequences(result, expected_length):
            return
        isfinite = math.isfinite
        for seq in sequences:
            tokens = seq.token_ids
            score = seq.cumulative_logprob
            if len(tokens) != expected_length or not tokens or seq.finish_reason != "length":
                raise ValueError("terminal result does not match the generation budget")
            if seq.stop_token_id is not None:
                raise ValueError("terminal result must preserve ignored EOS")
            if (
                tokens[0] < 0
                or (expected_length > 1 and tokens[1] < 0)
                or (expected_length > 2 and tokens[2] < 0)
                or (expected_length > 3 and min(tokens[3:]) < 0)
            ):
                raise ValueError("terminal token_ids must be non-negative")
            if score is None or not isfinite(score):
                raise ValueError("terminal scores must be present and finite")

    def _validate_diagnostics(
        self, result: BeamBatchResult, sequences: tuple[BeamOutputSequence, ...]
    ) -> None:
        """Audit internal lineage and ranking when explicitly requested."""
        validate_beam_batch_result(result)
        params = self.request.beam_params
        previous_score = float("inf")
        for seq in sequences:
            if len(seq.parent_beam_indices) != params.max_tokens or any(
                parent >= params.beam_width for parent in seq.parent_beam_indices
            ):
                raise ValueError("terminal result has an invalid parent chain")
            score = cast(float, seq.cumulative_logprob)
            if score > previous_score:
                raise ValueError("terminal scores must be sorted")
            previous_score = score

    def process_outputs(
        self, call: OfflineBeamSearchRequest, result: BeamBatchResult
    ) -> list[BeamSearchOutput]:
        return [BeamSearchOutput(sequences=self._convert_sequences(result, call))]

    def process_online_outputs(
        self, call: OnlineBeamSearchRequest, result: BeamBatchResult
    ) -> list[RequestOutput]:
        outputs = self._convert_sequences(result, None)
        texts = fast_batch_decode(self.tokenizer, [output.token_ids for output in outputs])
        for output, text in zip(outputs, texts, strict=True):
            output.text = text
        return [
            RequestOutput(
                request_id=call.request_id,
                prompt=(
                    self.prompt_text
                    if self.prompt_text is not None
                    else fast_decode(self.tokenizer, self.prompt_tokens)
                ),
                prompt_token_ids=self.prompt_tokens[:],
                prompt_logprobs=None,
                outputs=outputs,
                finished=True,
            )
        ]
