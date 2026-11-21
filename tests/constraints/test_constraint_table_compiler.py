# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Tests for the ConstraintTableCompiler (task 3-A)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_gr.constraints.compiler import (
    MAX_TOP_K,
    ConstraintTableCompiler,
    ConstraintTableResource,
    validate_resource_tables,
)
from vllm_gr.constraints.json_compiler import compile_relations_json


def _make_tokenizer() -> MagicMock:
    """A tokenizer whose vocab maps ``str(i) -> i`` for the string-token tests.

    The vocabulary covers 0..999 so every token the tests use (including the
    degree test, which goes up to ``MAX_TOP_K + 1``) is a member; a wide enough
    vocab keeps the ``get_vocab`` unknown-token check from firing before the
    degree check.
    """
    result = MagicMock()
    result.convert_tokens_to_ids.side_effect = lambda values: [int(value) for value in values]
    result.get_vocab.return_value = {str(i): i for i in range(1000)}
    return result


@pytest.fixture  # type: ignore[misc]
def tokenizer() -> MagicMock:
    return _make_tokenizer()


def _compile(
    catalog: list[list[str]],
    vocab_size: int = 10,
    tokenizer: MagicMock | None = None,
) -> ConstraintTableResource:
    if tokenizer is None:
        tokenizer = _make_tokenizer()
    document = compile_relations_json(catalog)
    return ConstraintTableCompiler(tokenizer, vocab_size).compile(document)


def test_compiles_basic_catalog(tokenizer: MagicMock) -> None:
    resource = _compile([["1", "2", "3"], ["1", "4", "5"]], vocab_size=10, tokenizer=tokenizer)

    assert isinstance(resource, ConstraintTableResource)
    assert resource.first_token_ids.tolist() == [1]
    assert resource.prefix1_values.tolist() == [2, 4]
    assert resource.prefix1_pair_keys.tolist() == [12, 14]
    assert resource.prefix2_values.tolist() == [3, 5]
    assert resource.prefix2_value_offsets.tolist() == [0, 1, 2]
    assert resource.prefix1_offsets[1] == 0
    assert resource.prefix1_offsets[2] == 2
    assert resource.prefix1_offsets[-1] == 2
    assert resource.max_prefix1_degree == 2
    assert resource.max_prefix2_degree == 1
    assert resource.vocab_size == 10


def test_dtypes_follow_operator_contract(tokenizer: MagicMock) -> None:
    resource = _compile([["1", "2", "3"]], vocab_size=10, tokenizer=tokenizer)

    assert resource.prefix1_pair_keys.dtype == np.int64
    assert resource.first_token_ids.dtype == np.int32
    assert resource.prefix1_offsets.dtype == np.int32
    assert resource.prefix1_values.dtype == np.int32
    assert resource.prefix2_value_offsets.dtype == np.int32
    assert resource.prefix2_values.dtype == np.int32


def test_multiple_first_tokens_sorted(tokenizer: MagicMock) -> None:
    resource = _compile(
        [["7", "8", "9"], ["1", "2", "3"], ["3", "4", "5"]],
        vocab_size=20,
        tokenizer=tokenizer,
    )

    # first_token_ids must be sorted ascending by token ID.
    assert resource.first_token_ids.tolist() == [1, 3, 7]
    # pair_keys must be strictly increasing.
    pair_keys = resource.prefix1_pair_keys.tolist()
    assert pair_keys == sorted(pair_keys) and len(set(pair_keys)) == len(pair_keys)


def test_pair_keys_strictly_increasing_across_different_a(tokenizer: MagicMock) -> None:
    # s_a tokens with large IDs: ensure a_id * vocab_size + b_id stays increasing
    # even when a second a has a smaller b_id.
    resource = _compile(
        [["3", "5", "6"], ["7", "1", "2"]],
        vocab_size=20,
        tokenizer=tokenizer,
    )
    pair_keys = resource.prefix1_pair_keys.tolist()
    assert pair_keys == [3 * 20 + 5, 7 * 20 + 1]
    assert pair_keys == sorted(pair_keys) and len(set(pair_keys)) == len(pair_keys)


def test_digest_is_stable_and_order_independent(tokenizer: MagicMock) -> None:
    catalog = [["1", "2", "3"], ["1", "4", "5"], ["7", "8", "9"]]
    r1 = _compile(catalog, vocab_size=20, tokenizer=tokenizer)
    r2 = _compile(list(reversed(catalog)), vocab_size=20, tokenizer=tokenizer)

    assert r1.digest == r2.digest
    assert r1.digest
    assert r1.digest != r2.first_token_ids.tobytes().hex()


def test_rejects_wrong_format_version(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    document["format_version"] = 999
    with pytest.raises(ValueError, match="Unsupported constraint triples format_version"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_unknown_token(tokenizer: MagicMock) -> None:
    bad_tokenizer = MagicMock()
    bad_tokenizer.convert_tokens_to_ids.side_effect = lambda values: [
        int(value) if int(value) <= 5 else -1 for value in values
    ]
    document = compile_relations_json([["1", "2", "9"]])
    with pytest.raises(ValueError, match="Failed to map token"):
        ConstraintTableCompiler(bad_tokenizer, 10).compile(document)


def test_rejects_out_of_range_token(tokenizer: MagicMock) -> None:
    with pytest.raises(ValueError, match="out of range for vocab_size"):
        _compile([["1", "2", "3"]], vocab_size=3, tokenizer=tokenizer)


def test_rejects_empty_triples(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    document["triples"] = []
    with pytest.raises(ValueError, match="non-empty"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_compiles_wide_degree_over_top_k(tokenizer: MagicMock) -> None:
    """A prefix with more valid continuations than MAX_TOP_K must compile.

    top_k (<= 256) is a runtime output limit; the candidate degree is not.  The
    operator's bucketed top-k handles wide candidate sets.
    """
    degree = 512
    wide = [["1", str(i), str(i)] for i in range(degree)]
    document = compile_relations_json(wide)
    resource = ConstraintTableCompiler(tokenizer, degree + 10).compile(document)

    assert resource.max_prefix1_degree == degree
    assert resource.max_prefix2_degree == 1
    assert len(resource.prefix1_values) == degree
    assert resource.max_prefix1_degree > MAX_TOP_K


def test_digest_differs_when_vocab_size_changes(tokenizer: MagicMock) -> None:
    r_small = _compile([["1", "2", "3"]], vocab_size=10, tokenizer=tokenizer)
    r_large = _compile([["1", "2", "3"]], vocab_size=100, tokenizer=tokenizer)

    assert r_small.digest != r_large.digest


# --- P1a: triples/relations consistency ---


def test_rejects_emptied_ab_to_c(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    document["relations"]["ab_to_c"] = {}
    with pytest.raises(ValueError, match="ab_to_c"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_orphaned_ab_to_c_pair(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    document["relations"]["ab_to_c"]['["9", "8"]'] = ["7"]
    with pytest.raises(ValueError, match="ab_to_c"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_inconsistent_first(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    document["relations"]["first"] = ["9"]
    with pytest.raises(ValueError, match="first"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_missing_a_to_b_pair(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"], ["1", "4", "5"]])
    document["relations"]["a_to_b"] = {"1": ["2"]}
    with pytest.raises(ValueError, match="a_to_b"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


# --- P1b: frozen arrays ---


def test_arrays_are_read_only(tokenizer: MagicMock) -> None:
    resource = _compile([["1", "2", "3"], ["1", "4", "5"]], vocab_size=10, tokenizer=tokenizer)

    assert not resource.prefix1_values.flags.writeable
    assert not resource.prefix1_offsets.flags.writeable
    assert not resource.prefix1_pair_keys.flags.writeable

    old_digest = resource.digest
    with pytest.raises(ValueError):
        resource.prefix1_values[0] = 999
    assert resource.digest == old_digest


def test_arrays_cannot_be_reopened_for_write(tokenizer: MagicMock) -> None:
    """setflags(write=True) must fail because arrays are bytes-backed."""
    resource = _compile([["1", "2", "3"]], vocab_size=10, tokenizer=tokenizer)

    with pytest.raises(ValueError):
        resource.prefix1_values.setflags(write=True)
    with pytest.raises(ValueError):
        resource.prefix1_values[0] = 999


# --- P2a: tokenizer result length ---


def test_rejects_short_tokenizer_result() -> None:
    short_tokenizer = MagicMock()
    short_tokenizer.convert_tokens_to_ids.side_effect = lambda values: [
        int(value) for value in values[:1]
    ]
    document = compile_relations_json([["1", "2", "3"]])
    with pytest.raises(ValueError, match="expected one ID per token"):
        ConstraintTableCompiler(short_tokenizer, 10).compile(document)


# --- P2-3: malformed document -> ValueError, not KeyError/AttributeError ---


def test_rejects_triple_missing_field(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    del document["triples"][0]["a"]
    with pytest.raises(ValueError, match="missing required field"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_non_dict_triple(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    document["triples"][0] = [1, 2, 3]
    with pytest.raises(ValueError, match="must be an object"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_missing_relations(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    del document["relations"]
    with pytest.raises(ValueError, match="relations"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_relations_not_dict(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    document["relations"] = [1, 2]
    with pytest.raises(ValueError, match="relations"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_first_not_list(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    document["relations"]["first"] = "abc"
    with pytest.raises(ValueError, match="first must be a list"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_a_to_b_not_dict(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    document["relations"]["a_to_b"] = [1]
    with pytest.raises(ValueError, match="a_to_b must be a dict"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_non_string_token(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    document["triples"][0]["c"] = 3
    with pytest.raises(ValueError, match="must be a str"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_duplicate_triples(tokenizer: MagicMock) -> None:
    document = compile_relations_json([["1", "2", "3"]])
    document["triples"].append({"a": "1", "b": "2", "c": "3"})
    with pytest.raises(ValueError, match="Duplicate triple"):
        ConstraintTableCompiler(tokenizer, 10).compile(document)


def test_rejects_token_mapped_to_unk_via_vocab() -> None:
    """A tokenizer that maps an unknown token to a valid unk_id is caught via get_vocab."""
    unk_tokenizer = MagicMock()
    vocab = {"1": 1, "2": 2, "3": 3, "0": 0}
    unk_tokenizer.get_vocab.return_value = vocab
    unk_tokenizer.convert_tokens_to_ids.side_effect = lambda values: [
        vocab.get(value, 0) for value in values
    ]
    document = compile_relations_json([["1", "2", "9"]])
    with pytest.raises(ValueError, match="not present in the tokenizer vocabulary"):
        ConstraintTableCompiler(unk_tokenizer, 10).compile(document)


def test_fallback_without_get_vocab() -> None:
    """A tokenizer without get_vocab falls back to the -1/None marker check."""
    plain_tokenizer = MagicMock()
    plain_tokenizer.convert_tokens_to_ids.side_effect = lambda values: [
        int(value) for value in values
    ]
    del plain_tokenizer.get_vocab  # ensure no get_vocab attribute exists

    document = compile_relations_json([["1", "2", "3"]])
    resource = ConstraintTableCompiler(plain_tokenizer, 10).compile(document)

    assert resource.prefix1_values.tolist() == [2]


def _build_multi_degree(tokenizer: MagicMock, vocab_size: int = 1000) -> ConstraintTableResource:
    """A catalog with real per-level degree (s_a 1 has two s_b continuations)."""
    catalog = [["1", "2", "10"], ["1", "3", "11"], ["4", "5", "12"]]
    document = compile_relations_json(catalog)
    return ConstraintTableCompiler(tokenizer, vocab_size).compile(document)


def test_digest_covers_max_degrees(tokenizer: MagicMock) -> None:
    """Changing only a degree must change the digest (operator truncates on it)."""
    resource = _build_multi_degree(tokenizer)
    tampered = ConstraintTableResource(
        first_token_ids=resource.first_token_ids,
        prefix1_offsets=resource.prefix1_offsets,
        prefix1_values=resource.prefix1_values,
        prefix1_pair_keys=resource.prefix1_pair_keys,
        prefix2_value_offsets=resource.prefix2_value_offsets,
        prefix2_values=resource.prefix2_values,
        max_prefix1_degree=resource.max_prefix1_degree + 1,
        max_prefix2_degree=resource.max_prefix2_degree,
        vocab_size=resource.vocab_size,
    )
    assert tampered.digest != resource.digest


def test_validate_resource_tables_rejects_derived_degree_mismatch(tokenizer: MagicMock) -> None:
    resource = _build_multi_degree(tokenizer)
    tampered = ConstraintTableResource(
        first_token_ids=resource.first_token_ids,
        prefix1_offsets=resource.prefix1_offsets,
        prefix1_values=resource.prefix1_values,
        prefix1_pair_keys=resource.prefix1_pair_keys,
        prefix2_value_offsets=resource.prefix2_value_offsets,
        prefix2_values=resource.prefix2_values,
        max_prefix1_degree=1,
        max_prefix2_degree=resource.max_prefix2_degree,
        vocab_size=resource.vocab_size,
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_resource_tables(tampered)


def test_validate_resource_tables_rejects_dtype_corruption(tokenizer: MagicMock) -> None:
    resource = _build_multi_degree(tokenizer)
    bad = ConstraintTableResource(
        first_token_ids=resource.first_token_ids,
        prefix1_offsets=resource.prefix1_offsets.astype(np.int64),
        prefix1_values=resource.prefix1_values,
        prefix1_pair_keys=resource.prefix1_pair_keys,
        prefix2_value_offsets=resource.prefix2_value_offsets,
        prefix2_values=resource.prefix2_values,
        max_prefix1_degree=resource.max_prefix1_degree,
        max_prefix2_degree=resource.max_prefix2_degree,
        vocab_size=resource.vocab_size,
    )
    with pytest.raises(ValueError, match="dtype must be"):
        validate_resource_tables(bad)


def test_validate_resource_tables_passes_valid_resource(tokenizer: MagicMock) -> None:
    resource = _build_multi_degree(tokenizer)
    validate_resource_tables(resource)  # must not raise


def test_validate_rejects_empty_first_token_ids_directly(tokenizer: MagicMock) -> None:
    resource = _build_multi_degree(tokenizer)
    tampered = ConstraintTableResource(
        first_token_ids=np.array([], dtype=np.int32),
        prefix1_offsets=resource.prefix1_offsets,
        prefix1_values=resource.prefix1_values,
        prefix1_pair_keys=resource.prefix1_pair_keys,
        prefix2_value_offsets=resource.prefix2_value_offsets,
        prefix2_values=resource.prefix2_values,
        max_prefix1_degree=resource.max_prefix1_degree,
        max_prefix2_degree=resource.max_prefix2_degree,
        vocab_size=resource.vocab_size,
    )
    with pytest.raises(ValueError, match="exactly match"):
        validate_resource_tables(tampered)


def test_validate_rejects_2d_prefix2_values(tokenizer: MagicMock) -> None:
    resource = _build_multi_degree(tokenizer)
    reshaped = replace(resource, prefix2_values=resource.prefix2_values.reshape(-1, 1))
    with pytest.raises(ValueError, match="prefix2_values must be 1-D"):
        validate_resource_tables(reshaped)


def test_digest_changes_when_shape_changes(tokenizer: MagicMock) -> None:
    """Reshaping a table must change the digest even though bytes are identical."""
    resource = _build_multi_degree(tokenizer)
    reshaped = ConstraintTableResource(
        first_token_ids=resource.first_token_ids,
        prefix1_offsets=resource.prefix1_offsets,
        prefix1_values=resource.prefix1_values,
        prefix1_pair_keys=resource.prefix1_pair_keys,
        prefix2_value_offsets=resource.prefix2_value_offsets,
        prefix2_values=resource.prefix2_values.reshape(-1, 1),
        max_prefix1_degree=resource.max_prefix1_degree,
        max_prefix2_degree=resource.max_prefix2_degree,
        vocab_size=resource.vocab_size,
    )
    assert reshaped.digest != resource.digest


def _colliding_tokenizer(
    collide_a: bool = False, collide_b: bool = False, collide_c: bool = False
) -> MagicMock:
    """A tokenizer whose get_vocab contains all tokens, but two distinct strings
    share a token ID depending on which position is flagged.

    All catalog tokens are members of get_vocab, so any rejection must come
    from the ID-collision check, not the unknown-token check.
    """
    result = MagicMock()

    # vocab: 1->1, 2->2, 3->3, 4->4, 5->5, plus collision aliases.
    def convert(values: list[str]) -> list[int]:
        out: list[int] = []
        for v in values:
            if collide_a and v == "4":
                out.append(1)  # '4' collides with '1' (both s_a in this test)
            elif collide_b and v == "3":
                out.append(2)  # '3' collides with '2' (both s_b)
            elif collide_c and v == "11":
                out.append(10)  # '11' collides with '10' (both s_c)
            else:
                out.append(int(v))
        return out

    result.convert_tokens_to_ids.side_effect = convert
    result.get_vocab.return_value = {str(i): i for i in range(1000)}
    return result


def test_rejects_colliding_s_a_token_ids() -> None:
    """Two distinct s_a strings mapping to the same ID must be rejected."""
    tokenizer = _colliding_tokenizer(collide_a=True)
    document = compile_relations_json([["1", "2", "10"], ["4", "5", "12"]])
    with pytest.raises(ValueError, match="maps distinct constraint tokens to ID 1"):
        ConstraintTableCompiler(tokenizer, 100).compile(document)


def test_rejects_colliding_s_b_token_ids() -> None:
    tokenizer = _colliding_tokenizer(collide_b=True)
    document = compile_relations_json([["1", "2", "10"], ["1", "3", "11"]])
    with pytest.raises(ValueError, match="maps distinct constraint tokens to ID 2"):
        ConstraintTableCompiler(tokenizer, 100).compile(document)


def test_rejects_colliding_s_c_token_ids() -> None:
    tokenizer = _colliding_tokenizer(collide_c=True)
    document = compile_relations_json([["1", "2", "10"], ["1", "2", "11"]])
    with pytest.raises(ValueError, match="maps distinct constraint tokens to ID 10"):
        ConstraintTableCompiler(tokenizer, 100).compile(document)


pytestmark = pytest.mark.cpu_test
