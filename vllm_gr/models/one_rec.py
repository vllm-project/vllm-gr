# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OneRec model-specific input and output handling for shared beam search."""

import copy
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.generate.beam_search.utils import BeamSearchOutput

from vllm_gr.entrypoints.beam_search_config import (
    BeamSearchStep,
    PreparedBeamInputs,
    select_best_beams,
)
from vllm_gr.logger import init_logger
from vllm_gr.logprobs import reconstruct_beam_logprobs

if TYPE_CHECKING:
    from vllm.logprobs import FlatLogprobs
    from vllm.sampling_params import BeamSearchParams

    from vllm_gr.entrypoints.gr import GRLLM

logger = init_logger(__name__)


def _resolve_special_token(
    tokenizer: Any,
    token: str | None,
    name: str,
) -> int | None:
    if token is None:
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == -1:
        logger.warning(
            "%s must be a valid token in the tokenizer vocabulary, got %s",
            name,
            token,
        )
        return None
    return int(token_id)


def prepare_beam_search_inputs(
    llm: "GRLLM",
    prompts: Any,
    params: "BeamSearchParams",
) -> PreparedBeamInputs:
    """Tokenize prompts and apply optional OneRec SID boundary tokens."""

    tokenizer = llm.get_tokenizer()
    begin_token_id = _resolve_special_token(
        tokenizer,
        getattr(params, "begin_token", None),
        "begin_token",
    )
    end_token_id = _resolve_special_token(
        tokenizer,
        getattr(params, "end_token", None),
        "end_token",
    )

    engine_inputs = []
    generated_starts = []
    output_starts = []
    prompt_texts = []
    for prompt in llm._preprocess_cmpl(prompts):
        if prompt["type"] == "embeds":
            raise NotImplementedError(
                "Embedding prompt not supported for beam search"
            )
        engine_prompt = copy.deepcopy(prompt)
        decoder_prompt = (
            engine_prompt
            if engine_prompt["type"] != "enc_dec"
            else engine_prompt["decoder_prompt"]
        )
        prompt_tokens = decoder_prompt["prompt_token_ids"]
        if begin_token_id is not None:
            prompt_tokens = list(prompt_tokens)
            prompt_tokens.append(begin_token_id)
            decoder_prompt["prompt_token_ids"] = prompt_tokens

        engine_inputs.append(engine_prompt)
        generated_starts.append(
            len(prompt_tokens) - 1
            if begin_token_id is not None
            else len(prompt_tokens)
        )
        output_starts.append(len(prompt_tokens))
        prompt_texts.append(tokenizer.decode(prompt_tokens))

    return PreparedBeamInputs(
        engine_inputs=engine_inputs,
        generated_starts=generated_starts,
        output_starts=output_starts,
        prompt_texts=prompt_texts,
        tokenizer=tokenizer,
        begin_token_id=begin_token_id,
        end_token_id=end_token_id,
    )


def build_beam_search_steps(
    num_steps: int,
    beam_width: int,
) -> list[BeamSearchStep]:
    """Use the existing fixed-width OneRec search behavior."""

    return [
        BeamSearchStep(expand_width=beam_width, keep_width=beam_width)
        for _ in range(num_steps)
    ]


def finalize_beam_search_outputs(
    instances: list[Any],
    prepared: PreparedBeamInputs,
    initial_logprobs: "FlatLogprobs",
    beam_width: int,
) -> list[BeamSearchOutput]:
    """Append the SID end token and decode OneRec token sequences as text."""

    assert prepared.tokenizer is not None
    outputs = []
    for index, instance in enumerate(instances):
        if prepared.end_token_id is not None:
            for beam in instance.beams:
                beam.tokens.append(prepared.end_token_id)
        best_beams = select_best_beams(instance, beam_width)
        prompt_text = prepared.prompt_texts[index]
        prompt_len = prepared.output_starts[index]
        for beam in best_beams:
            reconstruct_beam_logprobs(
                beam,
                initial_logprobs,
                prepared.end_token_id,
            )
            generated_tokens = beam.tokens[prompt_len:]
            beam.text = prompt_text + prepared.tokenizer.decode(generated_tokens)
        outputs.append(BeamSearchOutput(sequences=best_beams))
    return outputs
