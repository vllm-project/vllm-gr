# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio

try:
    import minheap_cpp as minheap
except ImportError:
    minheap = None
import time
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import numpy as np

from vllm.beam_search import BeamSearchSequence
from vllm.entrypoints.openai.protocol import VLLMValidationError
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.inputs.parse import is_explicit_encoder_decoder_prompt
from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalDataDict
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.utils import random_uuid
from vllm.utils.async_utils import (
    collect_from_async_generator,
)

from vllm_gr.v1.engine.types import BeamSearchRequest

async def beam_search(
    self,
    prompt: PromptType,
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    beam_width = params.beam_width
    max_tokens = params.max_tokens
    ignore_eos = params.ignore_eos
    temperature = params.temperature
    length_penalty = params.length_penalty
    include_stop_str_in_output = params.include_stop_str_in_output

    input_processor = self.input_processor
    tokenizer = input_processor.tokenizer

    # Engine-internal beam search fast path
    if hasattr(self.engine_client, 'engine_beam_search'):
        if tokenizer is None:
            raise VLLMValidationError(
                "You cannot use beam search when "
                "`skip_tokenizer_init=True`",
                parameter="skip_tokenizer_init",
                value=True,
            )
        eos_token_id: int = tokenizer.eos_token_id  # type: ignore

        # todo: remove. should get them from client (online or offline)
        params.begin_token = "<|sid_begin|>"
        params.end_token = "<|sid_end|>"
        print(f"OpenAiServing::beam_search begin_token:{params.begin_token} end_token: {params.begin_token}")

        # Resolve begin_token / end_token to token IDs
        sid_begin_token_id: int | None = None
        if params.begin_token is not None:
            sid_begin_token_id = tokenizer.convert_tokens_to_ids(
                params.begin_token)
            if sid_begin_token_id is None or sid_begin_token_id == -1:
                raise VLLMValidationError(
                    "begin_token must be a valid token in the "
                    "tokenizer vocabulary",
                    parameter="begin_token",
                    value=params.begin_token,
                )
        sid_end_token_id: int | None = None
        if params.end_token is not None:
            sid_end_token_id = tokenizer.convert_tokens_to_ids(
                params.end_token)
            if sid_end_token_id is None or sid_end_token_id == -1:
                raise VLLMValidationError(
                    "end_token must be a valid token in the "
                    "tokenizer vocabulary",
                    parameter="end_token",
                    value=params.end_token,
                )

        # Get prompt_token_ids from the prompt
        if isinstance(prompt, str):
            prompt_text = prompt
            prompt_token_ids = tokenizer.encode(prompt)
        else:
            prompt_text = prompt.get("prompt")  # type: ignore
            prompt_token_ids = prompt.get(  # type: ignore
                "prompt_token_ids", [])
            if not prompt_token_ids and prompt_text:
                prompt_token_ids = tokenizer.encode(prompt_text)

        bs_request = BeamSearchRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            eos_token_id=eos_token_id,
            beam_width=beam_width,
            max_tokens=max_tokens,
            n_best=beam_width,
            temperature=temperature,
            length_penalty=length_penalty,
            ignore_eos=ignore_eos,
            include_stop_str_in_output=include_stop_str_in_output,
            sid_begin_token_id=sid_begin_token_id,
            sid_end_token_id=sid_end_token_id,
        )
        bs_result = await self.engine_client.engine_beam_search(
            bs_request)

        tokenized_length = len(prompt_token_ids)
        best_beams = []
        for seq in bs_result.sequences:
            tokens = seq.token_ids
            if (seq.finish_reason == "stop" and tokens
                    and tokens[-1] == eos_token_id and not ignore_eos):
                text = tokenizer.decode(tokens[:-1])
            else:
                text = tokenizer.decode(tokens)
            # Build per-token logprobs matching token_ids length.
            # We only have cumulative logprob from engine-internal
            # beam search, so distribute it evenly as placeholder.
            n_tokens = len(tokens)
            if n_tokens > 0 and seq.cumulative_logprob != 0.0:
                per_tok_lp = seq.cumulative_logprob / n_tokens
            else:
                per_tok_lp = 0.0
            token_logprobs = [
                {tid: Logprob(logprob=per_tok_lp)}
                for tid in tokens
            ]
            best_beams.append(
                CompletionOutput(
                    text=text,
                    cumulative_logprob=seq.cumulative_logprob,
                    token_ids=tokens,
                    index=len(best_beams),
                    logprobs=token_logprobs,
                    finish_reason=seq.finish_reason,
                    stop_reason=seq.stop_reason,
                )
            )

        yield RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=best_beams,
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
        )

