# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import json
import os
import random
import tempfile
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from vllm.logprobs import FlatLogprobs

from vllm_gr.utils.catalog import (
    Catalog,
    complete_catalog_beam,
    fill_catalog_beams,
)


@pytest.fixture  # type: ignore
def mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()

    def convert_tokens_to_ids(seq: list[str]) -> list[int] | int:
        # Mock behavior: convert string tokens to integers
        # Assuming seq is a list of strings like ["1", "2"]
        if isinstance(seq, list):
            return [int(x) for x in seq]
        # Fallback if seq is a single string
        return int(seq)

    tokenizer.convert_tokens_to_ids.side_effect = convert_tokens_to_ids
    return tokenizer


def test_catalog_functionality(mock_tokenizer: MagicMock) -> None:
    # Define catalog data
    # 3 sequences:
    # [1, 2, 3, 4, 5]
    # [1, 2, 3, 6, 7]
    # [8, 9, 10, 11, 12]
    catalog_data = [
        ["1", "2", "3", "4", "5"],
        ["1", "2", "3", "6", "7"],
        ["8", "9", "10", "11", "12"],
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(catalog_data, f)
        catalog_path = f.name

    try:
        catalog = Catalog(mock_tokenizer)
        catalog.load(catalog_path)

        # Case 1: Empty prefix -> returns start tokens of all sequences
        # Expected: {1, 8}
        assert set(catalog.valid([])) == {1, 8}

        # Case 2: Prefix [1, 2, 3] -> returns {4, 6}
        assert set(catalog.valid([1, 2, 3])) == {4, 6}

        # Case 3: Prefix [1, 2, 3, 4] -> returns {5}
        assert set(catalog.valid([1, 2, 3, 4])) == {5}

        # Case 4: Full sequence [1, 2, 3, 4, 5] -> returns []
        assert catalog.valid([1, 2, 3, 4, 5]) == set()

        # Case 5: Invalid start token
        assert catalog.valid([99]) == set()
        # Case 6: Valid start, invalid continuation
        assert catalog.valid([1, 99]) == set()

    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def test_catalog_size_smaller_than_beam_width(mock_tokenizer: MagicMock) -> None:
    """Test catalog where the number of items is much smaller than typical beam widths."""
    # Catalog with only 3 items
    catalog_data = [
        ["1", "2", "3"],
        ["1", "2", "4"],
        ["5", "6", "7"],
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(catalog_data, f)
        catalog_path = f.name

    try:
        catalog = Catalog(mock_tokenizer)
        catalog.load(catalog_path)

        # Empty prefix should return all start tokens
        assert set(catalog.valid([])) == {1, 5}

        # Prefix [1, 2] should return next valid tokens {3, 4}
        assert set(catalog.valid([1, 2])) == {3, 4}

        # Each full path should be exhausted
        assert catalog.valid([1, 2, 3]) == set()
        assert catalog.valid([1, 2, 4]) == set()
        assert catalog.valid([5, 6, 7]) == set()

    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def test_catalog_single_item(mock_tokenizer: MagicMock) -> None:
    """Test catalog containing only one item."""
    catalog_data = [
        ["1", "2", "3", "4"],
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(catalog_data, f)
        catalog_path = f.name

    try:
        catalog = Catalog(mock_tokenizer)
        catalog.load(catalog_path)

        # Empty prefix should return the single start token
        assert set(catalog.valid([])) == {1}

        # Following the only path
        assert set(catalog.valid([1])) == {2}
        assert set(catalog.valid([1, 2])) == {3}
        assert set(catalog.valid([1, 2, 3])) == {4}
        assert catalog.valid([1, 2, 3, 4]) == set()

        # Any deviation from the single path should return empty
        assert catalog.valid([2]) == set()
        assert catalog.valid([1, 99]) == set()

    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def test_catalog_multiple_items_long_shared_prefix(mock_tokenizer: MagicMock) -> None:
    """Test catalog with multiple items sharing a long common prefix."""
    # All 5 sequences share a 6-token prefix, then diverge
    catalog_data = [
        ["10", "20", "30", "40", "50", "60", "100"],
        ["10", "20", "30", "40", "50", "60", "101"],
        ["10", "20", "30", "40", "50", "60", "102"],
        ["10", "20", "30", "40", "50", "60", "103"],
        ["10", "20", "30", "40", "50", "60", "104"],
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(catalog_data, f)
        catalog_path = f.name

    try:
        catalog = Catalog(mock_tokenizer)
        catalog.load(catalog_path)

        # Empty prefix should return start tokens (only 10 in this case)
        assert set(catalog.valid([])) == {10}

        # Following the shared prefix
        assert set(catalog.valid([10])) == {20}
        assert set(catalog.valid([10, 20])) == {30}
        assert set(catalog.valid([10, 20, 30])) == {40}
        assert set(catalog.valid([10, 20, 30, 40])) == {50}
        assert set(catalog.valid([10, 20, 30, 40, 50])) == {60}

        # After shared prefix, all 5 divergence tokens are valid
        assert set(catalog.valid([10, 20, 30, 40, 50, 60])) == {100, 101, 102, 103, 104}

        # Each divergence path should be terminal
        assert catalog.valid([10, 20, 30, 40, 50, 60, 100]) == set()
        assert catalog.valid([10, 20, 30, 40, 50, 60, 104]) == set()

    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def _load_catalog(mock_tokenizer: MagicMock, catalog_data: list[list[str]]) -> tuple[Catalog, str]:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(catalog_data, f)
        catalog_path = f.name
    catalog = Catalog(mock_tokenizer)
    catalog.load(catalog_path)
    return catalog, catalog_path


def test_catalog_stores_sequences_compactly(mock_tokenizer: MagicMock) -> None:
    catalog_data = [
        ["1", "2", "3"],
        ["4", "5"],
    ]
    catalog, catalog_path = _load_catalog(mock_tokenizer, catalog_data)
    try:
        assert len(catalog) == 2
        assert catalog.sequence_at(0) == [1, 2, 3]
        assert catalog.sequence_at(1) == [4, 5]
        # Sequences must be stored in a single flat packed array, not as a
        # Python list-of-lists second representation.
        assert isinstance(catalog._seq_data, np.ndarray)
        assert catalog._seq_data.dtype == np.uint32
        assert catalog._seq_data.tolist() == [1, 2, 3, 4, 5]
        assert catalog._seq_offsets.tolist() == [0, 3, 5]
        assert not hasattr(catalog, "sequences")
    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def test_apply_catalog_filter_masks_invalid_tokens(mock_tokenizer: MagicMock) -> None:
    catalog = Catalog(mock_tokenizer)

    logprobs = catalog.apply_catalog_filter(
        [7, 3, 5, 9],
        [-0.1, -0.2, -0.3, -0.4],
        {3, 9},
    )

    assert logprobs == [-float("inf"), -0.2, -float("inf"), -0.4]


def test_apply_catalog_filter_empty_valid_set_masks_all(mock_tokenizer: MagicMock) -> None:
    catalog = Catalog(mock_tokenizer)

    logprobs = catalog.apply_catalog_filter([1, 2], [-0.1, -0.2], set())

    assert logprobs == [-float("inf"), -float("inf")]


def test_sample_sequences_prefers_unique_picks(mock_tokenizer: MagicMock) -> None:
    catalog_data = [
        ["1", "2"],
        ["3", "4"],
        ["5", "6"],
    ]
    catalog, catalog_path = _load_catalog(mock_tokenizer, catalog_data)
    try:
        stored = [catalog.sequence_at(i) for i in range(len(catalog))]
        rng = random.Random(0)
        sampled = catalog.sample_sequences(2, rng=rng)
        assert len(sampled) == 2
        assert len({tuple(seq) for seq in sampled}) == 2
        assert all(seq in stored for seq in sampled)

        # More picks than catalog entries: falls back to duplicates.
        sampled = catalog.sample_sequences(5, rng=rng)
        assert len(sampled) == 5
        assert all(seq in stored for seq in sampled)

        assert catalog.sample_sequences(0, rng=rng) == []
    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def test_fill_catalog_beams_tops_up_to_beam_width(mock_tokenizer: MagicMock) -> None:
    catalog_data = [
        ["1", "2"],
        ["3", "4"],
    ]
    catalog, catalog_path = _load_catalog(mock_tokenizer, catalog_data)
    try:
        rng = random.Random(0)
        best_beams = fill_catalog_beams(
            [],
            2,
            catalog,
            prompt_token_ids=[0],
            prompt_text="prompt",
            sid_begin_token_id=None,
            sid_end_token_id=None,
            initial_logprobs=FlatLogprobs(),
            max_tokens=5,
            rng=rng,
        )

        assert len(best_beams) == 2
        stored = [catalog.sequence_at(i) for i in range(len(catalog))]
        for beam in best_beams:
            assert beam.cum_logprob == -10000.0
            assert beam.finish_reason == "stop"
            assert beam._lp_parent is None
            assert beam._lp_step_data is None
            assert beam.tokens[0] == 0
            assert beam.tokens[1:] in stored
    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def test_fill_catalog_beams_strips_begin_end_tokens(mock_tokenizer: MagicMock) -> None:
    catalog_data = [
        ["1", "2", "3"],
    ]
    catalog, catalog_path = _load_catalog(mock_tokenizer, catalog_data)
    try:
        rng = random.Random(0)
        best_beams = fill_catalog_beams(
            [],
            1,
            catalog,
            prompt_token_ids=[0, 1],
            prompt_text="prompt",
            sid_begin_token_id=1,
            sid_end_token_id=3,
            initial_logprobs=FlatLogprobs(),
            max_tokens=3,
            rng=rng,
        )

        assert len(best_beams) == 1
        assert best_beams[0].tokens == [0, 1, 2, 3]
        assert best_beams[0].stop_reason == 3
    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def test_fill_catalog_beams_no_op_when_full(mock_tokenizer: MagicMock) -> None:
    catalog_data = [
        ["1", "2"],
    ]
    catalog, catalog_path = _load_catalog(mock_tokenizer, catalog_data)
    try:
        best_beams = [object(), object()]
        result = fill_catalog_beams(
            best_beams,
            2,
            catalog,
            prompt_token_ids=[0],
            prompt_text="prompt",
            sid_begin_token_id=None,
            sid_end_token_id=None,
            initial_logprobs=FlatLogprobs(),
            max_tokens=5,
            rng=random.Random(0),
        )
        assert result is best_beams
    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def test_fill_catalog_beams_enforces_max_tokens_budget(mock_tokenizer: MagicMock) -> None:
    catalog_data = [
        ["1", "2", "3", "4", "5"],
        ["9", "8"],
    ]
    catalog, catalog_path = _load_catalog(mock_tokenizer, catalog_data)
    try:
        rng = random.Random(0)
        # No begin/end tokens: budget == max_tokens.  The 5-token entry must
        # be skipped; the 2-token entry fits and may be repeated by the
        # duplicate fallback to preserve the requested width.
        best_beams = fill_catalog_beams(
            [],
            2,
            catalog,
            prompt_token_ids=[0],
            prompt_text="prompt",
            sid_begin_token_id=None,
            sid_end_token_id=None,
            initial_logprobs=FlatLogprobs(),
            max_tokens=3,
            rng=rng,
        )

        assert len(best_beams) == 2
        assert all(beam.tokens == [0, 9, 8] for beam in best_beams)

        # When nothing fits, no beams are added.
        best_beams = fill_catalog_beams(
            [],
            2,
            catalog,
            prompt_token_ids=[0],
            prompt_text="prompt",
            sid_begin_token_id=None,
            sid_end_token_id=None,
            initial_logprobs=FlatLogprobs(),
            max_tokens=1,
            rng=rng,
        )
        assert len(best_beams) == 0
    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def _real_beam(tokens: list[int], score: float) -> Any:
    from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
    from vllm.inputs import tokens_input

    beam = BeamSearchSequence(
        orig_prompt=tokens_input(tokens),
        tokens=tokens,
        logprobs=FlatLogprobs(),
        cum_logprob=score,
    )
    beam._lp_parent = None
    beam._lp_step_data = None
    return beam


def test_fill_catalog_beams_avoids_duplicating_real_beams(mock_tokenizer: MagicMock) -> None:
    catalog_data = [
        ["1", "2"],
        ["3", "4"],
        ["5", "6"],
    ]
    catalog, catalog_path = _load_catalog(mock_tokenizer, catalog_data)
    try:
        rng = random.Random(0)
        # The real beam already produced catalog item [1, 2].
        real_beam = _real_beam([0, 1, 2], -0.5)
        best_beams = fill_catalog_beams(
            [real_beam],
            3,
            catalog,
            prompt_token_ids=[0],
            prompt_text="prompt",
            sid_begin_token_id=None,
            sid_end_token_id=None,
            initial_logprobs=FlatLogprobs(),
            max_tokens=5,
            rng=rng,
        )

        assert len(best_beams) == 3
        generated = {tuple(beam.tokens[1:]) for beam in best_beams}
        assert generated == {(1, 2), (3, 4), (5, 6)}
    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def test_fill_catalog_beams_allows_duplicates_when_catalog_exhausted(
    mock_tokenizer: MagicMock,
) -> None:
    catalog_data = [
        ["1", "2"],
    ]
    catalog, catalog_path = _load_catalog(mock_tokenizer, catalog_data)
    try:
        rng = random.Random(0)
        best_beams = fill_catalog_beams(
            [],
            3,
            catalog,
            prompt_token_ids=[0],
            prompt_text="prompt",
            sid_begin_token_id=None,
            sid_end_token_id=None,
            initial_logprobs=FlatLogprobs(),
            max_tokens=5,
            rng=rng,
        )

        # The catalog has a single distinct entry; duplicates are allowed to
        # preserve the requested beam width.
        assert len(best_beams) == 3
        assert all(beam.tokens == [0, 1, 2] for beam in best_beams)
    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def test_fill_catalog_beams_duplicates_fill_above_catalog_size(
    mock_tokenizer: MagicMock,
) -> None:
    catalog_data = [
        ["1", "2"],
    ]
    catalog, catalog_path = _load_catalog(mock_tokenizer, catalog_data)
    try:
        rng = random.Random(0)
        best_beams = fill_catalog_beams(
            [],
            4,
            catalog,
            prompt_token_ids=[0],
            prompt_text="prompt",
            sid_begin_token_id=None,
            sid_end_token_id=None,
            initial_logprobs=FlatLogprobs(),
            max_tokens=5,
            rng=rng,
        )

        # One distinct entry but four requested beams: the duplicate fallback
        # must retry enough times to fill all four (regression test for the
        # serving dead-end case where catalog size < beam_width).
        assert len(best_beams) == 4
        assert all(beam.tokens == [0, 1, 2] for beam in best_beams)
    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def test_complete_catalog_beam_preserves_score_and_finishes() -> None:
    beam = _real_beam([0, 1, 2], -3.5)
    beam._lp_parent = _real_beam([0, 1], -3.0)
    beam._lp_step_data = ([2], [-0.5], [None], [None])

    completed = complete_catalog_beam(beam)

    assert completed is not beam
    assert completed.tokens == [0, 1, 2]
    assert completed.tokens is not beam.tokens
    assert completed.cum_logprob == -3.5
    assert completed.finish_reason == "stop"
    assert completed.stop_reason is None
    assert completed.lora_request is beam.lora_request
    assert completed._lp_parent is beam._lp_parent
    assert completed._lp_step_data is beam._lp_step_data
