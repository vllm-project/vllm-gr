# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_gr.constraints.compiler import ConstraintTableCompiler
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.constraints.table import ConstraintTable, validate_constraint_table

TOKENIZER_DIGEST = "sha256:" + "a" * 64


@pytest.fixture
def tokenizer() -> MagicMock:
    result = MagicMock()
    vocab = {str(token): token for token in range(32)}
    result.get_vocab.return_value = vocab
    result.convert_tokens_to_ids.side_effect = lambda values: [
        vocab.get(value, -1) for value in values
    ]
    return result


def _document() -> dict:
    return compile_relations_json([["1", "3", "5"], ["1", "4", "6"], ["2", "4", "7"]])


def test_compiles_exact_six_table_layout(tokenizer: MagicMock) -> None:
    table = ConstraintTableCompiler(
        tokenizer,
        8,
        tokenizer_digest=TOKENIZER_DIGEST,
    ).compile(_document())

    assert isinstance(table, ConstraintTable)
    assert table.first_token_ids.tolist() == [1, 2]
    assert table.prefix1_offsets.tolist() == [0, 0, 2, 3, 3, 3, 3, 3, 3]
    assert table.prefix1_values.tolist() == [3, 4, 4]
    assert table.prefix1_pair_keys.tolist() == [11, 12, 20]
    assert table.prefix2_value_offsets.tolist() == [0, 1, 2, 3]
    assert table.prefix2_values.tolist() == [5, 6, 7]
    assert table.info.artifact_digest is None
    assert table.info.max_prefix1_degree == 2
    assert table.info.max_prefix2_degree == 1
    validate_constraint_table(table)


def test_compiler_output_is_immutable(tokenizer: MagicMock) -> None:
    table = ConstraintTableCompiler(tokenizer, 8, tokenizer_digest=TOKENIZER_DIGEST).compile(
        _document()
    )
    with pytest.raises(ValueError):
        table.first_token_ids[0] = 7


def test_rejects_unknown_token(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "3", "unknown"]])
    with pytest.raises(ValueError, match="Failed to map token"):
        ConstraintTableCompiler(tokenizer, 8, tokenizer_digest=TOKENIZER_DIGEST).compile(document)


def test_rejects_out_of_range_token(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "3", "9"]])
    with pytest.raises(ValueError, match="out of range"):
        ConstraintTableCompiler(tokenizer, 8, tokenizer_digest=TOKENIZER_DIGEST).compile(document)


def test_rejects_wrong_document_version(tokenizer: MagicMock) -> None:
    document = _document()
    document["format_version"] = 2
    with pytest.raises(ValueError, match="format_version"):
        ConstraintTableCompiler(tokenizer, 8, tokenizer_digest=TOKENIZER_DIGEST).compile(document)


def test_rejects_inconsistent_derived_relations(tokenizer: MagicMock) -> None:
    document = copy.deepcopy(_document())
    document["relations"]["a_to_b"]["1"] = ["3"]
    with pytest.raises(ValueError, match="a_to_b"):
        ConstraintTableCompiler(tokenizer, 8, tokenizer_digest=TOKENIZER_DIGEST).compile(document)


def test_pair_keys_are_unambiguous_for_tokens_containing_commas() -> None:
    tokenizer = MagicMock()
    vocab = {"a,b": 1, "c": 2, "a": 3, "b,c": 4, "x": 5, "y": 6}
    tokenizer.get_vocab.return_value = vocab
    tokenizer.convert_tokens_to_ids.side_effect = lambda values: [vocab[value] for value in values]
    document = compile_relations_json([["a,b", "c", "x"], ["a", "b,c", "y"]])
    table = ConstraintTableCompiler(tokenizer, 8, tokenizer_digest=TOKENIZER_DIGEST).compile(
        document
    )
    assert table.prefix1_pair_keys.tolist() == sorted([1 * 8 + 2, 3 * 8 + 4])


def test_compiler_does_not_read_files_or_load_tokenizer(
    tokenizer: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("compiler crossed its in-memory offline boundary")

    monkeypatch.setattr("builtins.open", forbidden)
    table = ConstraintTableCompiler(tokenizer, 8, tokenizer_digest=TOKENIZER_DIGEST).compile(
        _document()
    )
    assert np.array_equal(table.first_token_ids, np.asarray([1, 2], dtype=np.int32))


def test_relations_pair_keys_use_json_encoding() -> None:
    document = compile_relations_json([["a,b", "c", "x"]])
    assert json.dumps(["a,b", "c"]) in document["relations"]["ab_to_c"]
