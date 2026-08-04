# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from vllm_gr.utils.catalog import Catalog


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


def _apply(node, catalog: Catalog, strategy: str) -> tuple:
    """Thin wrapper around apply_catalog_filter for the parity checks below."""
    return catalog.apply_catalog_filter(
        token_ids_pos=[1, 2, 3, 4],
        logprobs_pos=[-0.1, -0.2, -0.3, -0.4],
        ranks_pos=[1, 2, 3, 4],
        decoded_pos=["a", "b", "c", "d"],
        valid_tokens_node=node,
        catalog_strategy=strategy,
    )


def test_apply_catalog_filter_dict_matches_set_null_strategy(mock_tokenizer: MagicMock) -> None:
    """dict node (with marker) behaves identically to the equivalent set under "null"."""
    catalog = Catalog(mock_tokenizer)
    node = {2: {}, 4: {}, Catalog._TERMINAL_MARKER: True}

    dict_result = _apply(node, catalog, "null")
    set_result = _apply({2, 4}, catalog, "null")

    assert dict_result == set_result
    assert dict_result[0] == [1, 2, 3, 4]
    assert dict_result[1] == [-float("inf"), -0.2, -float("inf"), -0.4]


def test_apply_catalog_filter_dict_matches_set_fill_strategy(mock_tokenizer: MagicMock) -> None:
    """dict node (with marker) behaves identically to the equivalent set under "fill"."""
    catalog = Catalog(mock_tokenizer)
    node = {2: {}, 4: {}, Catalog._TERMINAL_MARKER: True}

    dict_result = _apply(node, catalog, "fill")
    set_result = _apply({2, 4}, catalog, "fill")

    assert dict_result == set_result
    # Invalid slots 1 and 3 are filled with the sorted valid tokens 2, 4.
    assert dict_result[0] == [2, 2, 4, 4]
    assert all(lp < 0 for lp in dict_result[1])


def test_apply_catalog_filter_ignores_terminal_marker_in_dict(mock_tokenizer: MagicMock) -> None:
    """The _TERMINAL_MARKER string key must never surface as a fillable token."""
    catalog = Catalog(mock_tokenizer)

    # Node whose only child is the terminal marker (a completed sequence).
    terminal_only = {Catalog._TERMINAL_MARKER: True}
    result = _apply(terminal_only, catalog, "fill")

    # No valid next token -> beam is a dead end, eliminated entirely.
    assert result[0] == []
    assert result[1] == []

    # Mixed node: marker present alongside valid int keys -> marker is excluded.
    mixed = {2: {}, Catalog._TERMINAL_MARKER: True}
    result = _apply(mixed, catalog, "fill")
    assert result[0] == [2, 2, 2, 2]
    assert result[2] == [None, None, None, None]  # ranks reset for fill-injected tokens


def test_apply_catalog_filter_empty_node_matches_empty_set(mock_tokenizer: MagicMock) -> None:
    """An empty dict node (dead-end beam) is eliminated exactly like an empty set."""
    catalog = Catalog(mock_tokenizer)

    dict_result = _apply({}, catalog, "null")
    set_result = _apply(set(), catalog, "null")

    assert dict_result == set_result
    assert dict_result[0] == []
    assert dict_result[1] == []


def test_fill_injects_catalog_tokens_when_valid_set_is_smaller_than_window(
    mock_tokenizer: MagicMock,
) -> None:
    """'fill' semantics: valid tokens not present in the top-K window are injected.

    This is the exact offline-path parity guarantee: under "fill" a beam whose
    candidate window has fewer valid catalog tokens than slots keeps generating
    with catalog-valid tokens rather than dead-ending (as plain masking would).
    """
    catalog = Catalog(mock_tokenizer)

    # One valid continuation (5); the window holds 4 slots.
    result = catalog.apply_catalog_filter(
        token_ids_pos=[1, 2, 3, 4],
        logprobs_pos=[-0.1, -0.2, -0.3, -0.4],
        ranks_pos=[1, 2, 3, 4],
        decoded_pos=["a", "b", "c", "d"],
        valid_tokens_node={5: {}, Catalog._TERMINAL_MARKER: True},
        catalog_strategy="fill",
    )

    # Every slot is replaced by the single valid token 5 (with fill scores).
    assert result[0] == [5, 5, 5, 5]
    # With a single valid token the injected score is identical across slots
    # (fill_used % n_valid == 0); descending scores only apply with n_valid > 1.
    assert result[1] == [-10000.0, -10000.0, -10000.0, -10000.0]
    assert result[2] == [None, None, None, None]  # fill-injected ranks reset


def test_fill_preserves_geometry_when_valid_token_in_window(
    mock_tokenizer: MagicMock,
) -> None:
    """'fill' keeps an in-window valid token at its original logprob."""
    catalog = Catalog(mock_tokenizer)

    result = catalog.apply_catalog_filter(
        token_ids_pos=[1, 2, 3, 4],
        logprobs_pos=[-0.1, -0.2, -0.3, -0.4],
        ranks_pos=[1, 2, 3, 4],
        decoded_pos=["a", "b", "c", "d"],
        valid_tokens_node={2: {}, 4: {}},
        catalog_strategy="fill",
    )

    assert result[0] == [2, 2, 4, 4]
    assert result[1][1] == -0.2  # token 2 kept its original logprob
    assert result[1][3] == -0.4  # token 4 kept its original logprob
