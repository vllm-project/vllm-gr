# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest

from vllm_gr.entrypoints.gr import _finalize_beam_search_instance, _parse_step_logprobs
from vllm_gr.utils.catalog import Catalog


def _beam(tokens: list[int], score: float = 0.0) -> BeamSearchSequence:
    beam = BeamSearchSequence(
        orig_prompt=tokens_input(tokens),
        tokens=tokens,
        logprobs=FlatLogprobs(),
        cum_logprob=score,
    )
    beam._lp_parent = None
    beam._lp_step_data = None
    return beam


def _result_with_flat(flat: object) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[SimpleNamespace(finish_reason=None, logprobs=flat)],
    )


def test_offline_grouped_candidates_keep_explicit_parent_indices() -> None:
    instance = SimpleNamespace(
        beams=[_beam([10], -0.5), _beam([20], -0.2)],
        prompt_len=1,
    )
    flat = SimpleNamespace(
        token_ids=[7, 3, 7, 8, 4, 8, 999],
        logprobs=[-0.01, -0.1, -0.2, -0.02, -0.3, -0.4, -9.0],
        ranks=None,
        decoded_tokens=None,
    )

    is_error, token_ids, scores, beam_idx, cache, terminal = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=1,
        logprobs_num=2,
        fork_info_j=[(0, 7), (1, 8)],
        catalog=None,
    )

    assert not is_error
    assert token_ids == [3, 7, 4, 8]
    assert scores == pytest.approx([-0.6, -0.7, -0.5, -0.6])
    assert beam_idx == [0, 0, 1, 1]
    assert [entry[0] for entry in cache] == [[3, 7], [4, 8]]
    assert 999 not in token_ids
    assert terminal == []


def test_offline_terminal_catalog_beam_is_reported() -> None:
    instance = SimpleNamespace(
        beams=[_beam([10, 11], -0.5)],
        prompt_len=2,
    )
    flat = SimpleNamespace(
        token_ids=[11, 11, 12],
        logprobs=[-0.1, -0.1, -0.2],
        ranks=None,
        decoded_tokens=None,
        start_indices=[0],
        end_indices=[3],
    )

    catalog = MagicMock()
    catalog.valid.return_value = set()
    catalog.is_terminal.return_value = True

    is_error, token_ids, scores, beam_idx, cache, terminal = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=0,
        logprobs_num=2,
        fork_info_j=None,
        catalog=catalog,
    )

    assert not is_error
    assert terminal == [0]
    assert cache == [None]
    assert token_ids == []
    assert scores == []
    assert beam_idx == []

    # A non-terminal dead-end (catalog prefix with no valid continuation but
    # not a complete item) must still be masked and kept in the candidate pool.
    catalog.is_terminal.return_value = False
    is_error, token_ids, scores, beam_idx, cache, terminal = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=0,
        logprobs_num=2,
        fork_info_j=None,
        catalog=catalog,
    )
    assert not is_error
    assert terminal == []
    assert len(cache) == 1
    assert cache[0][0] == [11, 12]
    assert cache[0][1] == [-float("inf"), -float("inf")]


def _mock_catalog_from_sequences(sequences: list[list[str]]) -> Catalog:
    tokenizer = MagicMock()

    def convert_tokens_to_ids(seq: str | list[str]) -> list[int] | int:
        if isinstance(seq, list):
            return [int(x) for x in seq]
        return int(seq)

    tokenizer.convert_tokens_to_ids.side_effect = convert_tokens_to_ids
    return _catalog_loaded(tokenizer, sequences)


def _catalog_loaded(tokenizer: MagicMock, sequences: list[list[str]]) -> Catalog:
    import json
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(sequences, f)
        catalog_path = f.name
    catalog = Catalog(tokenizer)
    try:
        catalog.load(catalog_path)
    finally:
        os.remove(catalog_path)
    return catalog


def test_finalize_error_instance_with_catalog_does_not_crash() -> None:
    """Regression: an engine error empties instance.beams mid-search; final
    assembly must not dereference beams[0] to recover the LoRA handle."""
    catalog = _mock_catalog_from_sequences([["1", "2"], ["3", "4"]])
    lora_request = LoRARequest("adapter", 1, "/tmp/adapter")
    instance = SimpleNamespace(
        beams=[],
        completed=[],
    )
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda token_ids: "".join(str(t) for t in token_ids)

    output = _finalize_beam_search_instance(
        instance=instance,
        beam_width=2,
        catalog=catalog,
        prompt_cache={"text": "prompt", "len": 1, "tokens": [0]},
        tokenizer=tokenizer,
        sid_begin_token_id=None,
        sid_end_token_id=None,
        initial_logprobs=FlatLogprobs(),
        lora_request=lora_request,
        processed_tokens=7,
        max_tokens=5,
    )

    assert len(output.sequences) == 2
    assert output.processed_tokens == 7
    for beam in output.sequences:
        assert beam.lora_request is lora_request
        assert beam.cum_logprob == -10000.0
        assert beam.finish_reason == "stop"
