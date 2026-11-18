# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Tests for building Constraint Tables from tokenizer s_a/s_b/s_c groups."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_gr.constraints.tokenizer_builder import build_constraint_tables_from_tokenizer


class _Tokenizer(SimpleNamespace):
    def get_added_vocab(self) -> dict[str, int]:
        return dict(self.added_vocab)


def _make_tokenizer(na: int, nb: int, nc: int, start: int = 100) -> _Tokenizer:
    added: dict[str, int] = {}
    for i in range(na):
        added[f"<s_a_{i}>"] = start + i
    for i in range(nb):
        added[f"<s_b_{i}>"] = start + 10000 + i
    for i in range(nc):
        added[f"<s_c_{i}>"] = start + 20000 + i
    return _Tokenizer(added_vocab=added)


def _sb_id(i: int, start: int = 100) -> int:
    return start + 10000 + i


def _sc_id(i: int, start: int = 100) -> int:
    return start + 20000 + i


def test_no_token_groups_returns_none() -> None:
    assert build_constraint_tables_from_tokenizer(_Tokenizer(added_vocab={}), 1000) is None


def test_small_groups_build_fully_connected() -> None:
    resource = build_constraint_tables_from_tokenizer(_make_tokenizer(3, 2, 4), vocab_size=100000)
    assert resource is not None
    # All 3 s_a tokens are candidates at step 0.
    assert resource.first_token_ids.tolist() == [100, 101, 102]
    # Every s_a maps to every s_b: degree == nb.
    assert resource.max_prefix1_degree == 2
    # Every (s_a, s_b) pair maps to every s_c: degree == nc.
    assert resource.max_prefix2_degree == 4
    assert len(resource.prefix2_values) == 3 * 2 * 4


def test_truncation_fails_fast() -> None:
    """Item groups beyond max_valid_items must error, never silently truncate.

    Silent truncation produced candidates disconnected from real prompts
    (cross-sample duplicate low-numbered SIDs, pid=0 lookups).
    """
    with pytest.raises(ValueError, match="truncated"):
        build_constraint_tables_from_tokenizer(_make_tokenizer(8192, 8192, 8192), vocab_size=100000)


def test_truncation_respects_explicit_cap() -> None:
    with pytest.raises(ValueError, match="max_valid_items=10"):
        build_constraint_tables_from_tokenizer(
            _make_tokenizer(11, 5, 5), vocab_size=100000, max_valid_items=10
        )


def test_exact_boundary_builds() -> None:
    """Groups exactly at the cap build without error (cap is inclusive)."""
    resource = build_constraint_tables_from_tokenizer(
        _make_tokenizer(10, 5, 5), vocab_size=100000, max_valid_items=10
    )
    assert resource is not None
    assert len(resource.first_token_ids) == 10


def test_valid_item_ids_subset_still_builds() -> None:
    """An explicit valid_item_ids subset is an operator choice, not truncation."""
    resource = build_constraint_tables_from_tokenizer(
        _make_tokenizer(8192, 8192, 8192), vocab_size=100000, valid_item_ids=[1, 2, 3]
    )
    assert resource is not None
    assert resource.first_token_ids.tolist() == [101, 102, 103]
    assert resource.max_prefix1_degree == 3
    assert resource.max_prefix2_degree == 3


def test_sparse_files_bypass_truncation_check(tmp_path: Path) -> None:
    prefix1 = tmp_path / "prefix1.json"
    prefix2 = tmp_path / "prefix2.json"
    prefix1.write_text('{"0": [1, 2]}')
    prefix2.write_text('{"0,1": [3], "0,2": [4, 5]}')
    resource = build_constraint_tables_from_tokenizer(
        _make_tokenizer(8192, 8192, 8192),
        vocab_size=100000,
        prefix1_file=str(prefix1),
        prefix2_file=str(prefix2),
    )
    assert resource is not None
    assert resource.first_token_ids.tolist() == [100]
    assert resource.max_prefix1_degree == 2
    assert resource.max_prefix2_degree == 2
    # pair keys use vocab_size multiplier: (100 * 100000 + sb_id(1)) etc.
    assert resource.prefix1_pair_keys.tolist() == [
        100 * 100000 + _sb_id(1),
        100 * 100000 + _sb_id(2),
    ]
    assert resource.prefix2_values.tolist() == [_sc_id(3), _sc_id(4), _sc_id(5)]


def test_fully_connected_uses_whole_universe() -> None:
    """With no truncation, the built universe covers every item group member."""
    resource = build_constraint_tables_from_tokenizer(_make_tokenizer(4, 4, 4), vocab_size=100000)
    assert resource is not None
    # prefix1_values is the tiled s_b list: each s_a row covers all 4 s_b.
    values = np.asarray(resource.prefix1_values)
    assert values.shape == (16,)
    assert set(values.tolist()) == {_sb_id(i) for i in range(4)}


pytestmark = pytest.mark.cpu_test
