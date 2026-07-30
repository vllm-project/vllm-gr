# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest

from vllm_gr.beam_search_decision_utils import (
    complete_eos_candidates,
    materialize_selected_beams,
    select_top_indices,
)


def _parents() -> list[BeamSearchSequence]:
    lora_request = LoRARequest("adapter", 1, "/tmp/adapter")
    parents = [
        BeamSearchSequence(
            orig_prompt=tokens_input([10]),
            tokens=[10],
            logprobs=FlatLogprobs(),
            cum_logprob=-0.1,
            lora_request=lora_request,
        ),
        BeamSearchSequence(
            orig_prompt=tokens_input([20]),
            tokens=[20],
            logprobs=FlatLogprobs(),
            cum_logprob=-0.2,
        ),
    ]
    for parent in parents:
        parent._lp_parent = None
        parent._lp_step_data = None
    return parents


def _run_shared_decision(
    *,
    include_stop_token: bool,
    inherit_stop_lora: bool,
) -> tuple[list[BeamSearchSequence], list[BeamSearchSequence], list[tuple[int, int]]]:
    parents = _parents()
    token_ids = np.asarray([2, 11, 21, 22])
    scores = np.asarray([-0.2, -0.3, -0.4, -0.5])
    beam_idx = np.asarray([0, 0, 1, 1], dtype=np.intp)
    cache = [
        ([2, 11], [-0.1, -0.2], [1, 2], ["2", "11"]),
        ([21, 22], [-0.2, -0.3], [1, 2], ["21", "22"]),
    ]

    completed, selection_scores = complete_eos_candidates(
        parents=parents,
        token_ids=token_ids,
        scores=scores,
        beam_idx=beam_idx,
        beam_flat_cache=cache,
        stop_token_ids={2},
        initial_logprobs=FlatLogprobs(),
        include_eos_token=include_stop_token,
        inherit_lora_request=inherit_stop_lora,
    )
    selected = select_top_indices(selection_scores, 2, "fill")
    active, fork_info = materialize_selected_beams(
        parents=parents,
        selected_indices=selected,
        token_ids=token_ids,
        scores=selection_scores,
        beam_idx=beam_idx,
        beam_flat_cache=cache,
        initial_logprobs=FlatLogprobs(),
    )
    return completed, active, fork_info


def test_offline_and_openai_share_decisions_but_keep_output_policy() -> None:
    offline_completed, offline_active, offline_forks = _run_shared_decision(
        include_stop_token=True,
        inherit_stop_lora=True,
    )
    openai_completed, openai_active, openai_forks = _run_shared_decision(
        include_stop_token=False,
        inherit_stop_lora=False,
    )

    assert [beam.tokens[-1] for beam in offline_active] == [
        beam.tokens[-1] for beam in openai_active
    ]
    assert [beam.cum_logprob for beam in offline_active] == [
        beam.cum_logprob for beam in openai_active
    ]
    assert offline_forks == openai_forks
    assert offline_completed[0].tokens == [10, 2]
    assert openai_completed[0].tokens == [10]
    assert offline_completed[0].lora_request is not None
    assert openai_completed[0].lora_request is None
