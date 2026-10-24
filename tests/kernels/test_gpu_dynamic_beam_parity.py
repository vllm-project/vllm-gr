# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in parity between GPU dynamic widths and an independent CPU oracle.

The oracle replays the same fixed-depth successor tree with the tiny model in
float32. Constraint Table candidates are scored over the shared logits window
that covers every legal token (token IDs 0..5 here), matching the CUDA scoring
contract. Run in a fresh process with VLLM_GR_RUN_MODEL_SMOKE=1 and
VLLM_ENABLE_V1_MULTIPROCESSING=0.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("VLLM_GR_RUN_MODEL_SMOKE") != "1" or not torch.cuda.is_available(),
    reason="opt-in CUDA model smoke",
)

SUCCESSORS = {
    (): [1, 3],
    (1,): [2, 4],
    (3,): [5],
    (1, 2): [0, 1],
    (1, 4): [3],
    (3, 5): [1, 4],
}
# One shared window covers the table's legal token IDs 0..5.
SCORING_WINDOW = 6
# Auxiliary cross-dtype check: the engine is FP16 and the oracle FP32.
# This allowance does not establish graph parity; the separate same-dtype
# eager/graph regression uses abs=1e-4, rel=1e-5 and exact ordered tokens.
SCORE_TOLERANCE = 2e-2
# (dynamic_beam_widths, result_width, length_penalty). None keeps the capacity
# width. The capacity-uniform case with result_width == beam_width uses the
# native fixed-width path, so both execution paths are checked against the same
# oracle. The last case checks that a length penalty stays inert at the fixed
# horizon (the native ranking ratio is exactly 1).
CASES = [
    (None, 2, 1.0),
    ((4, 4, 4), 2, 1.0),
    ((4, 4, 4), 4, 1.0),
    ((4, 2, 1), 1, 1.0),
    ((1, 3, 4), 2, 1.0),
    ((2, 4, 2), 2, 1.0),
    ((4, 2, 1), 1, 2.0),
]


def oracle(model, prompt, widths, result_width):
    """Exact fixed-depth search over full-recompute float32 logits."""
    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    with torch.no_grad():
        for width in widths:
            candidates: list[tuple[tuple[int, ...], float]] = []
            for prefix, score in beams:
                ids = torch.tensor([list(prompt) + list(prefix)])
                logits = model(ids).logits[0, -1].float()
                logprobs = logits[:SCORING_WINDOW].log_softmax(-1)
                for token in SUCCESSORS.get(prefix, []):
                    candidates.append((prefix + (token,), score + float(logprobs[token])))
            candidates.sort(key=lambda item: -item[1])
            beams = candidates[:width]
    return beams[:result_width]


def test_dynamic_matches_cpu_oracle(tmp_path):
    from transformers import LlamaForCausalLM
    from vllm.inputs import tokens_input

    from tests.kernels.test_gpu_dynamic_beam_model import build_tiny_engine
    from vllm_gr.sampling_params import BeamSearchParams

    llm, model_path = build_tiny_engine(tmp_path)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).eval()
    prompts = ([1, 4, 3], [1, 4, 3, 3])
    try:
        for widths, result_width, length_penalty in CASES:
            schedule = (4, 4, 4) if widths is None else widths
            for batch in (1, 2):
                params = BeamSearchParams(
                    beam_width=4,
                    dynamic_beam_widths=None if widths is None else list(widths),
                    result_width=result_width,
                    max_tokens=3,
                    ignore_eos=True,
                    temperature=0.0,
                    length_penalty=length_penalty,
                )
                outputs = llm.beam_search_v1([tokens_input(p) for p in prompts[:batch]], params)
                assert len(outputs) == batch
                for prompt, output in zip(prompts[:batch], outputs):
                    expected = oracle(model, prompt, schedule, result_width)
                    # Engine sequences include the prompt prefix.
                    got = [
                        (tuple(sequence.tokens)[len(prompt) :], sequence.cum_logprob)
                        for sequence in output.sequences
                    ]
                    context = (
                        f"widths={widths} result_width={result_width} "
                        f"length_penalty={length_penalty} batch={batch} prompt={prompt}"
                    )
                    assert [tokens for tokens, _ in got] == [tokens for tokens, _ in expected], (
                        f"{context} got={got} expected={expected}"
                    )
                    for (_, got_score), (_, expected_score) in zip(got, expected):
                        assert got_score == pytest.approx(expected_score, abs=SCORE_TOLERANCE), (
                            f"{context} got={got} expected={expected}"
                        )
    finally:
        llm.__exit__(None, None, None)
