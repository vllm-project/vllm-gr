# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Async RECIF beam search built on the shared beam transition."""

from types import SimpleNamespace
from typing import Any

from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from vllm_gr.beam_search_step import apply_beam_search_step
from vllm_gr.entrypoints.openai.serving_engine import _add_batch_step
from vllm_gr.entrypoints.recif.protocol import (
    RecifBeamResult,
    RecifBeamSearchRequest,
    RecifBeamSearchResponse,
)
from vllm_gr.models.recif import (
    NUM_SID_LEVELS,
    VOCAB,
    build_beam_search_steps,
    build_prefix,
    bytes_to_sid,
)


async def recif_beam_search(
    engine_client: Any,
    request: RecifBeamSearchRequest,
    request_id: str | None = None,
) -> RecifBeamSearchResponse:
    """Run the three RECIF SID levels through the async vLLM engine."""

    request_id = request_id or f"recif-{random_uuid()}"
    steps = build_beam_search_steps(
        SimpleNamespace(
            beam_width=request.beam_width,
            branch_factors=request.branch_factors,
        )
    )
    prompt_token_ids = build_prefix(request.history_sids, "cpu")[0].tolist()
    original_prompt = tokens_input(prompt_token_ids)
    initial_logprobs = FlatLogprobs()
    active = [
        BeamSearchSequence(
            orig_prompt=original_prompt,
            tokens=prompt_token_ids,
            cum_logprob=0.0,
            logprobs=initial_logprobs,
            lora_request=None,
        )
    ]
    active[0]._lp_parent = None
    active[0]._lp_step_data = None

    for level, step in enumerate(steps):
        if not active:
            break
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=step.expand_width,
            allowed_token_ids=step.allowed_token_ids,
            detokenize=False,
            flat_logprobs=True,
        )
        outputs, _ = await _add_batch_step(
            engine_client,
            tuple(beam.get_prompt() for beam in active),
            tuple(None for _ in active),
            f"{request_id}-level-{level}",
            sampling_params,
            False,
            None,
        )
        active = apply_beam_search_step(
            active,
            outputs,
            step,
            initial_logprobs,
        ).active

    if not active:
        raise RuntimeError("RECIF beam search produced no candidates")

    active.sort(key=lambda beam: beam.cum_logprob, reverse=True)
    results: list[RecifBeamResult] = []
    for rank, beam in enumerate(active[: request.beam_width]):
        generated = beam.tokens[len(prompt_token_ids) :]
        if len(generated) != NUM_SID_LEVELS:
            raise RuntimeError("RECIF beam search must generate three SID levels")
        sid = bytes_to_sid(
            generated[0],
            generated[1] - VOCAB,
            generated[2] - 2 * VOCAB,
        )
        results.append(
            RecifBeamResult(
                rank=rank,
                sid=sid,
                cumulative_logprob=float(beam.cum_logprob),
            )
        )
    return RecifBeamSearchResponse(id=request_id, results=results)


__all__ = ["recif_beam_search"]
