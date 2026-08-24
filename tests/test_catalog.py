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


def test_catalog_transfer_payload_is_scoped_per_rank() -> None:
    """The sequences transfer is tracked per target engine rank: each rank
    receives the sequences once; later requests carry the hash only."""
    from typing import Any

    from vllm_gr.entrypoints.openai.serving_models import (
        _catalog_transfer_payload,
        _engine_catalog_sent,
    )

    _engine_catalog_sent.clear()
    try:
        seqs = [[1, 2, 3], [1, 2, 4]]

        class FakeCatalog:
            _sequences_hash = 42

            @staticmethod
            def to_sequences() -> list[list[int]]:
                return seqs

        catalog: Any = FakeCatalog()
        h0, s0 = _catalog_transfer_payload(catalog, 0)
        h1, s1 = _catalog_transfer_payload(catalog, 1)
        h0b, s0b = _catalog_transfer_payload(catalog, 0)
        assert (h0, s0) == (42, seqs)
        assert (h1, s1) == (42, seqs)
        assert (h0b, s0b) == (42, None)

        # The single-engine (non-DP) rank key is tracked independently.
        hn, sn = _catalog_transfer_payload(catalog, None)
        assert (hn, sn) == (42, seqs)
        hn2, sn2 = _catalog_transfer_payload(catalog, None)
        assert (hn2, sn2) == (42, None)
    finally:
        _engine_catalog_sent.clear()
