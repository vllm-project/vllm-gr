# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tokenizer-free RECIF beam search on the asynchronous vLLM engine."""

from typing import Any

from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.utils import random_uuid

from vllm_gr.entrypoints.openai.serving_engine import run_beam_search_steps_async
from vllm_gr.models.recif import (
    NUM_SID_LEVELS,
    build_beam_search_steps,
)


async def recif_beam_search(
    engine_client: Any,
    prompt_token_ids: list[int],
    beam_width: int,
    request_id: str | None = None,
    *,
    lora_request: Any = None,
    trace_headers: Any = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
) -> RequestOutput:
    """Run RECIF's model-owned hierarchy through the async engine."""

    request_id = request_id or f"cmpl-{random_uuid()}"
    steps = build_beam_search_steps(NUM_SID_LEVELS, beam_width)
    original_prompt = tokens_input(prompt_token_ids)
    initial_logprobs = FlatLogprobs()
    active = [
        BeamSearchSequence(
            orig_prompt=original_prompt,
            tokens=prompt_token_ids,
            cum_logprob=0.0,
            logprobs=initial_logprobs,
            lora_request=lora_request,
        )
    ]
    active[0]._lp_parent = None
    active[0]._lp_step_data = None

    active = await run_beam_search_steps_async(
        engine_client,
        active,
        steps,
        initial_logprobs,
        request_id,
        trace_headers=trace_headers,
        priority=priority,
        data_parallel_rank=data_parallel_rank,
    )
    if not active:
        raise RuntimeError("RECIF beam search produced no candidates")

    active.sort(key=lambda beam: beam.cum_logprob, reverse=True)
    completion_outputs: list[CompletionOutput] = []
    for rank, beam in enumerate(active[:beam_width]):
        generated = beam.tokens[len(prompt_token_ids) :]
        if len(generated) != NUM_SID_LEVELS:
            raise RuntimeError("RECIF beam search must generate three SID levels")
        completion_outputs.append(
            CompletionOutput(
                index=rank,
                text="",
                token_ids=generated,
                cumulative_logprob=float(beam.cum_logprob),
                logprobs=None,
                finish_reason="length",
            )
        )
    return RequestOutput(
        request_id=request_id,
        prompt="",
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        outputs=completion_outputs,
        finished=True,
        lora_request=lora_request,
    )
