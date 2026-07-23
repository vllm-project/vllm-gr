# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration tests for the shared OneRec/RECIF beam-search trunk."""

import inspect
from types import SimpleNamespace

import pytest

from vllm_gr.entrypoints.beam_search_config import BeamSearchModel
from vllm_gr.entrypoints.gr import GRLLM
from vllm_gr.models.one_rec import (
    build_beam_search_steps as build_onerec_beam_search_steps,
    prepare_beam_search_inputs as prepare_onerec_beam_search_inputs,
)
from vllm_gr.models.recif import (
    build_beam_search_steps as build_recif_beam_search_steps,
)
from vllm_gr.sampling_params import BeamSearchParams


def test_existing_onerec_params_keep_fixed_width_steps() -> None:
    # This is the existing OneRec construction style: branch_factors is not
    # supplied and therefore does not change its behavior.
    params = BeamSearchParams(beam_width=32, max_tokens=5)

    assert params.branch_factors is None
    steps = build_onerec_beam_search_steps(
        num_steps=3,
        beam_width=params.beam_width,
    )
    assert [step.expand_width for step in steps] == [32, 32, 32]
    assert [step.keep_width for step in steps] == [32, 32, 32]
    assert all(step.allowed_token_ids is None for step in steps)


def test_recif_steps_have_per_level_widths_and_token_ranges() -> None:
    params = BeamSearchParams(
        beam_width=64,
        max_tokens=3,
        branch_factors=(4, 8, 16),
    )

    steps = build_recif_beam_search_steps(params)
    allowed_ranges = [step.allowed_token_ids for step in steps]
    assert all(token_ids is not None for token_ids in allowed_ranges)
    assert [step.expand_width for step in steps] == [4, 8, 16]
    assert [step.keep_width for step in steps] == [4, 64, 64]
    assert [
        (token_ids[0], token_ids[-1])
        for token_ids in allowed_ranges
        if token_ids is not None
    ] == [
        (0, 8191),
        (8192, 16383),
        (16384, 24575),
    ]


def test_onerec_keeps_distinct_catalog_and_output_offsets() -> None:
    class _Tokenizer:
        eos_token_id = 2

        def convert_tokens_to_ids(self, token: str) -> int:
            return {"<begin>": 100, "<end>": 101}[token]

        def decode(self, token_ids: list[int]) -> str:
            return ",".join(str(token_id) for token_id in token_ids)

    llm = SimpleNamespace(
        get_tokenizer=lambda: _Tokenizer(),
        _preprocess_cmpl=lambda prompts: [
            {"type": "token", "prompt_token_ids": [10, 20]}
        ],
    )
    params = SimpleNamespace(
        begin_token="<begin>",
        end_token="<end>",
    )

    prepared = prepare_onerec_beam_search_inputs(
        llm,
        [{}],
        params,
    )  # type: ignore[arg-type]

    assert prepared.generated_starts == [2]
    assert prepared.output_starts == [3]
    assert prepared.prompt_texts == ["10,20,100"]


@pytest.mark.parametrize(
    "branch_factors",
    [(8, 8), (0, 8, 8), (8, 8, 8193)],
)
def test_recif_rejects_invalid_branch_factors(branch_factors: tuple[int, ...]) -> None:
    params = SimpleNamespace(
        beam_width=64,
        branch_factors=branch_factors,
    )
    with pytest.raises(ValueError):
        build_recif_beam_search_steps(params)  # type: ignore[arg-type]


def test_grllm_defaults_to_onerec_beam_search_model() -> None:
    parameter = inspect.signature(GRLLM.__init__).parameters[
        "beam_search_model"
    ]

    assert parameter.default is BeamSearchModel.ONE_REC
    assert BeamSearchModel.RECIF.value == "recif"
