# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs

from vllm_gr.constraints import BeamConstraintCoordinator
from vllm_gr.entrypoints.gr import GRLLM


class _Tokenizer:
    eos_token_id = 2

    def __init__(self) -> None:
        self.decoded: list[list[int]] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|sid_begin|>": 1, "<|sid_end|>": 3}[token]

    def decode(self, tokens: list[int]) -> str:
        self.decoded.append(list(tokens))
        return "".join({1: "<|sid_begin|>", 3: "<|sid_end|>"}.get(t, f"[{t}]") for t in tokens)


class _BatchTokenizer(_Tokenizer):
    def batch_decode(self, tokens: list[list[int]]) -> list[str]:
        return [self.decode(row) for row in tokens]


@pytest.mark.parametrize("batch_decode", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("include_prompt", [None, False, True])  # type: ignore[misc]
@pytest.mark.parametrize("begin", [None, "<|sid_begin|>"])  # type: ignore[misc]
@pytest.mark.parametrize("end", [None, "<|sid_end|>"])  # type: ignore[misc]
@pytest.mark.parametrize("stop_at_eos", [False, True])  # type: ignore[misc]
def test_offline_output_uses_original_prompt_boundary(
    batch_decode: bool,
    include_prompt: bool | None,
    begin: str | None,
    end: str | None,
    stop_at_eos: bool,
) -> None:
    tokenizer = _BatchTokenizer() if batch_decode else _Tokenizer()
    original_tokens = [10, 11]
    prepared = tokens_input(original_tokens)
    next_token = tokenizer.eos_token_id if stop_at_eos else 20

    def generate(prompts: Any, **kwargs: Any) -> list[SimpleNamespace]:
        flat = FlatLogprobs(
            start_indices=[0],
            end_indices=[2],
            token_ids=[next_token, next_token],
            logprobs=[-0.2, -0.2],
            ranks=[1, 1],
            decoded_tokens=[f"[{next_token}]", f"[{next_token}]"],
        )
        return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=flat)]) for _ in prompts]

    runner: Any = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        _preprocess_cmpl=lambda prompts: [prepared],
        _lora_request_to_seq=lambda request, count: [None] * count,
        constraint_strategy=None,
        constraint_coordinator=BeamConstraintCoordinator(None),
        llm_engine=SimpleNamespace(vllm_config=None),
        generate=generate,
    )
    params: Any = SimpleNamespace(
        beam_width=1,
        max_tokens=1 + int(begin is not None) + int(end is not None),
        temperature=1.0,
        ignore_eos=False,
        begin_token=begin,
        end_token=end,
    )
    kwargs = {} if include_prompt is None else {"decode_prompt_in_text": include_prompt}
    outputs = GRLLM.beam_search(runner, [{"prompt_token_ids": original_tokens}], params, **kwargs)
    beam = outputs[0].sequences[0]
    generated = ([1] if begin else []) + [next_token] + ([3] if end and not stop_at_eos else [])
    expected_tail = (begin or "") + f"[{next_token}]" + ((end or "") if not stop_at_eos else "")
    assert beam.tokens == original_tokens + generated
    assert beam.text == ("[10][11]" if include_prompt is not False else "") + expected_tail
    assert generated in tokenizer.decoded
    assert (original_tokens in tokenizer.decoded) == (include_prompt is not False)
    assert list(beam.logprobs.token_ids) == generated
    assert beam.cum_logprob == pytest.approx(-0.2)
    assert prepared["prompt_token_ids"] == [10, 11]
