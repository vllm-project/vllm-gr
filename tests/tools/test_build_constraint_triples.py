# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Tests for the Constraint Tables triples compiler and its CLI tool."""

from __future__ import annotations

import json
import os
from pathlib import Path
from stat import S_IMODE
from typing import TextIO

import pytest

from tools.constraints.build_constraint_triples import process_catalog
from vllm_gr.constraints.json_compiler import (
    compile_relations_json,
    compile_triples_json,
    compile_triples_json_string,
)


def _write_catalog(tmp_path: Path, sequences: list[list[str]]) -> str:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(sequences), encoding="utf-8")
    return str(path)


def test_compiles_three_token_catalog_without_sentinels() -> None:
    catalog = [["1", "2", "3"], ["1", "4", "5"]]
    document = compile_triples_json(catalog)

    assert document["triples"] == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "1", "b": "4", "c": "5"},
    ]
    assert document["sentinels"] == {"begin": None, "end": None}
    assert document["format_version"] == 1


def test_strips_consistent_begin_end_sentinels() -> None:
    catalog = [
        ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>", "<|sid_end|>"],
        ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_4>", "<|sid_end|>"],
    ]
    document = compile_triples_json(catalog, begin_token="<|sid_begin|>", end_token="<|sid_end|>")

    assert document["triples"] == [
        {"a": "<s_a_1>", "b": "<s_b_2>", "c": "<s_c_3>"},
        {"a": "<s_a_1>", "b": "<s_b_2>", "c": "<s_c_4>"},
    ]


def test_compiles_relations_document() -> None:
    catalog = [["1", "2", "3"], ["1", "4", "5"], ["7", "8", "9"]]
    document = compile_relations_json(catalog)

    assert document["format_version"] == 1
    assert document["relations"] == {
        "first": ["1", "7"],
        "a_to_b": {
            "1": ["2", "4"],
            "7": ["8"],
        },
        "ab_to_c": {
            '["1", "2"]': ["3"],
            '["1", "4"]': ["5"],
            '["7", "8"]': ["9"],
        },
    }


def test_ab_to_c_key_does_not_collide_on_comma_in_tokens() -> None:
    """('a,b', 'c') and ('a', 'b,c') must map to different ab_to_c keys."""
    catalog = [
        ["a,b", "c", "x"],
        ["a", "b,c", "y"],
    ]
    document = compile_relations_json(catalog)

    ab_to_c = document["relations"]["ab_to_c"]
    assert len(ab_to_c) == 2
    assert ab_to_c['["a,b", "c"]'] == ["x"]
    assert ab_to_c['["a", "b,c"]'] == ["y"]


def test_triples_and_relations_are_order_independent() -> None:
    """Same logical catalog in different input order must serialize identically."""
    forward = [["1", "2", "3"], ["1", "4", "5"], ["7", "8", "9"]]
    backward = list(reversed(forward))

    doc_fwd = compile_relations_json(forward)
    doc_bwd = compile_relations_json(backward)

    # Compare full serialized documents so key *order* in the JSON is checked,
    # not just dict equality.
    assert json.dumps(doc_fwd, ensure_ascii=False) == json.dumps(doc_bwd, ensure_ascii=False)


def test_rejects_wrong_semantic_depth() -> None:
    with pytest.raises(ValueError, match="exactly three semantic tokens"):
        compile_triples_json([["1", "2", "3", "4"]])


def test_rejects_missing_begin_sentinel_when_configured() -> None:
    catalog = [["<s_a_1>", "<s_b_2>", "<s_c_3>", "<|sid_end|>"]]
    with pytest.raises(ValueError, match="begin sentinel"):
        compile_triples_json(catalog, begin_token="<|sid_begin|>", end_token="<|sid_end|>")


def test_rejects_duplicate_triples() -> None:
    catalog = [["1", "2", "3"], ["1", "2", "3"]]
    with pytest.raises(ValueError, match="Duplicate triple"):
        compile_triples_json(catalog)


def test_rejects_empty_catalog() -> None:
    with pytest.raises(ValueError, match="Catalog is empty"):
        compile_triples_json([])


def test_rejects_non_sequence_with_sentinels() -> None:
    """A non-sequence entry (None) must report its container type."""
    with pytest.raises(ValueError, match="got NoneType"):
        compile_triples_json([None], begin_token="<|sid_begin|>", end_token="<|sid_end|>")


def test_rejects_non_string_elements_with_sentinels() -> None:
    """A list containing non-string elements must report the element types."""
    with pytest.raises(ValueError, match="found element types.*int"):
        compile_triples_json([[1, 2, 3]], begin_token="<|sid_begin|>", end_token="<|sid_end|>")
    with pytest.raises(ValueError, match="found element types.*list"):
        compile_triples_json(
            [[["1"], ["2"], ["3"]]],
            begin_token="<|sid_begin|>",
            end_token="<|sid_end|>",
        )


def test_accepts_tuple_sequences() -> None:
    """Catalog sequences may be tuples (not only lists) of token strings."""
    document = compile_triples_json([("1", "2", "3"), ("1", "4", "5")])
    assert document["triples"] == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "1", "b": "4", "c": "5"},
    ]


def test_rejects_malformed_without_sentinels() -> None:
    """Malformed entries must be rejected even when no sentinel is configured.

    Regression: shape validation must run on every sequence regardless of
    sentinel presence, not only when a sentinel is configured.
    """
    with pytest.raises(ValueError, match="found element types.*int"):
        compile_triples_json([["1", 2, "3"]])
    with pytest.raises(ValueError, match="got NoneType"):
        compile_triples_json([None])


def test_compile_string_is_valid_json() -> None:
    text = compile_triples_json_string([["1", "2", "3"]])
    assert json.loads(text)["triples"] == [{"a": "1", "b": "2", "c": "3"}]


def test_process_catalog_writes_output_file(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path, [["1", "2", "3"], ["1", "4", "5"]])
    output_path = str(tmp_path / "triples.json")
    try:
        count = process_catalog(catalog_path, output_path)
        assert count == 2

        with open(output_path) as file:
            document = json.load(file)
        assert document["triples"] == [
            {"a": "1", "b": "2", "c": "3"},
            {"a": "1", "b": "4", "c": "5"},
        ]
        assert document["format_version"] == 1
        assert document["relations"] == {
            "first": ["1"],
            "a_to_b": {"1": ["2", "4"]},
            "ab_to_c": {'["1", "2"]': ["3"], '["1", "4"]': ["5"]},
        }
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_process_catalog_reports_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        process_catalog(str(tmp_path / "missing.json"), str(tmp_path / "out.json"))


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.skipif(os.name != "posix", reason="requires POSIX file modes")  # type: ignore[misc]
def test_process_catalog_honors_umask_and_preserves_output_mode(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path, [["1", "2", "3"]])
    output_path = tmp_path / "triples.json"
    previous_umask = os.umask(0o027)
    try:
        process_catalog(catalog_path, str(output_path))
        assert S_IMODE(output_path.stat().st_mode) == 0o640
        output_path.chmod(0o664)
        process_catalog(catalog_path, str(output_path))
        assert S_IMODE(output_path.stat().st_mode) == 0o664
    finally:
        os.umask(previous_umask)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.skipif(os.name != "posix", reason="requires POSIX symlinks")  # type: ignore[misc]
@pytest.mark.parametrize("target_exists", [True, False], ids=["existing", "dangling"])  # type: ignore[misc]
def test_process_catalog_preserves_relative_output_symlink(
    tmp_path: Path, target_exists: bool
) -> None:
    catalog_path = _write_catalog(tmp_path, [["1", "2", "3"]])
    relative_target = Path("generated/triples.json")
    target = tmp_path / relative_target
    target.parent.mkdir()
    if target_exists:
        target.write_text("old", encoding="utf-8")
    output_path = tmp_path / "triples.json"
    output_path.symlink_to(relative_target)

    assert process_catalog(catalog_path, str(output_path)) == 1
    assert output_path.is_symlink()
    assert output_path.readlink() == relative_target
    assert json.loads(target.read_text(encoding="utf-8"))["triples"] == [
        {"a": "1", "b": "2", "c": "3"}
    ]


@pytest.mark.l1  # type: ignore[misc]
def test_failed_output_write_preserves_previous_triples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = _write_catalog(tmp_path, [["1", "2", "3"]])
    output_path = tmp_path / "triples.json"
    process_catalog(catalog_path, str(output_path))
    original = output_path.read_bytes()
    files = set(tmp_path.iterdir())

    def interrupted_dump(_document: object, output: TextIO, **_kwargs: object) -> None:
        output.write("{")
        raise OSError("disk full")

    monkeypatch.setattr(json, "dump", interrupted_dump)
    with pytest.raises(OSError, match="disk full"):
        process_catalog(catalog_path, str(output_path))

    assert output_path.read_bytes() == original
    assert set(tmp_path.iterdir()) == files


def test_process_catalog_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="list of sequences"):
        process_catalog(str(path), str(tmp_path / "out.json"))


def test_process_catalog_with_sentinels(tmp_path: Path) -> None:
    catalog = [
        ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>", "<|sid_end|>"],
        ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_4>", "<|sid_end|>"],
    ]
    catalog_path = _write_catalog(tmp_path, catalog)
    output_path = str(tmp_path / "triples.json")
    try:
        count = process_catalog(
            catalog_path,
            output_path,
            begin_token="<|sid_begin|>",
            end_token="<|sid_end|>",
        )
        assert count == 2
        with open(output_path) as file:
            document = json.load(file)
        assert document["triples"][0] == {"a": "<s_a_1>", "b": "<s_b_2>", "c": "<s_c_3>"}
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


pytestmark = pytest.mark.cpu_test
