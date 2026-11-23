# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import tools.constraints.build_constraint_table as build_constraint_table_module
from tools.constraints.build_constraint_table import (
    _resolve_model_revision,
    build_constraint_table,
)
from vllm_gr.constraints.artifact import load_constraint_table_artifact
from vllm_gr.constraints.artifact_builder import (
    build_single_trie_artifact,
    load_triples,
)
from vllm_gr.constraints.json_compiler import compile_relations_json


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {str(i): i for i in range(20)}
    tokenizer.get_added_vocab.return_value = {}
    tokenizer.convert_tokens_to_ids.side_effect = lambda values: [int(value) for value in values]
    for name, value in {
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "unk_token_id": 3,
        "sep_token_id": None,
        "cls_token_id": None,
        "mask_token_id": None,
        "additional_special_tokens_ids": [],
        "all_special_ids": [0, 1, 2, 3],
    }.items():
        setattr(tokenizer, name, value)
    return tokenizer


@pytest.mark.l1  # type: ignore[misc]
def test_legacy_triples_build_matches_strict_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    tokenizer = _tokenizer()
    transformers = ModuleType("transformers")
    auto_tokenizer = MagicMock(return_value=tokenizer)
    auto_config = MagicMock(return_value=SimpleNamespace(vocab_size=20))
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=auto_tokenizer)  # type: ignore[attr-defined]
    transformers.AutoConfig = SimpleNamespace(from_pretrained=auto_config)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"triples": [{"a": 1, "b": 4, "c": 5}, {"a": 1, "b": 2, "c": 3}]}),
        encoding="utf-8",
    )
    rows = [["1", "2", "3"], ["1", "4", "5"]]
    assert load_triples(legacy) == rows
    output = tmp_path / "legacy.safetensors"
    artifact_digest, tokenizer_digest = build_single_trie_artifact(model, legacy, output)
    auto_tokenizer.assert_called_once_with(model, trust_remote_code=True)
    auto_config.assert_called_once_with(model, trust_remote_code=True)

    canonical = tmp_path / "canonical.json"
    document = compile_relations_json(rows)
    canonical.write_text(json.dumps(document), encoding="utf-8")
    reference = tmp_path / "canonical.safetensors"
    build_strict = partial(
        build_constraint_table, model=str(model), revision=None, output_path=reference
    )
    assert build_strict(triples_path=canonical) == (
        artifact_digest,
        tokenizer_digest,
        20,
    )
    assert output.read_bytes() == reference.read_bytes()
    loaded = load_constraint_table_artifact(
        output,
        expected_digest=artifact_digest,
        expected_tokenizer_digest=tokenizer_digest,
        expected_vocab_size=20,
    )
    assert loaded.first_token_ids.tolist() == [1]
    assert loaded.prefix1_values.tolist() == [2, 4]
    assert loaded.prefix2_values.tolist() == [3, 5]

    with pytest.raises(ValueError, match="Unsupported constraint triples format_version"):
        build_strict(triples_path=legacy)
    inconsistent = tmp_path / "inconsistent.json"
    inconsistent.write_text(
        json.dumps({**document, "relations": {**document["relations"], "first": []}}),
        encoding="utf-8",
    )
    rebuilt = tmp_path / "rebuilt.safetensors"
    assert build_single_trie_artifact(model, inconsistent, rebuilt) == (
        artifact_digest,
        tokenizer_digest,
    )
    assert rebuilt.read_bytes() == output.read_bytes()
    with pytest.raises(ValueError, match="relations.first is inconsistent"):
        build_strict(triples_path=inconsistent)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("triples", "error"),
    [([], "not a nonempty Constraint Table"), ([{"a": 1, "b": 2}], "invalid triples row 0")],
)
def test_legacy_triples_rejects_invalid_rows(
    tmp_path: Path, triples: list[dict[str, int]], error: str
) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"triples": triples}), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_triples(path)


def test_build_tool_uses_one_revision_and_writes_loadable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _tokenizer()
    auto_tokenizer = MagicMock()
    auto_tokenizer.from_pretrained.return_value = tokenizer
    auto_config = MagicMock()
    auto_config.from_pretrained.return_value = SimpleNamespace(vocab_size=20)
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = auto_tokenizer  # type: ignore[attr-defined]
    transformers.AutoConfig = auto_config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    commit_sha = "a" * 40
    resolve_revision = MagicMock(return_value=commit_sha)
    monkeypatch.setattr(
        build_constraint_table_module,
        "_resolve_model_revision",
        resolve_revision,
    )

    triples_path = tmp_path / "triples.json"
    triples_path.write_text(
        json.dumps(compile_relations_json([["1", "2", "3"], ["1", "4", "5"]])),
        encoding="utf-8",
    )
    output = tmp_path / "table.safetensors"

    artifact_digest, tokenizer_digest, vocab_size = build_constraint_table(
        triples_path=triples_path,
        model="model-id",
        revision="revision-id",
        output_path=output,
    )

    expected_kwargs = {
        "revision": commit_sha,
        "trust_remote_code": False,
    }
    resolve_revision.assert_called_once_with("model-id", "revision-id")
    auto_tokenizer.from_pretrained.assert_called_once_with("model-id", **expected_kwargs)
    auto_config.from_pretrained.assert_called_once_with("model-id", **expected_kwargs)
    assert artifact_digest.startswith("sha256:")
    assert tokenizer_digest.startswith("sha256:")
    assert vocab_size == 20
    loaded = load_constraint_table_artifact(
        output,
        expected_digest=artifact_digest,
        expected_tokenizer_digest=tokenizer_digest,
        expected_vocab_size=20,
    )
    assert np.array_equal(loaded.first_token_ids, np.asarray([1], dtype=np.int32))


def test_build_tool_is_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _tokenizer()
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=MagicMock(return_value=tokenizer)
    )
    transformers.AutoConfig = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=MagicMock(return_value=SimpleNamespace(vocab_size=20))
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        build_constraint_table_module,
        "_resolve_model_revision",
        MagicMock(return_value="a" * 40),
    )
    triples = tmp_path / "triples.json"
    triples.write_text(
        json.dumps(compile_relations_json([["1", "2", "3"]])),
        encoding="utf-8",
    )
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"

    first_result = build_constraint_table(
        triples_path=triples,
        model="model-id",
        revision="revision-id",
        output_path=first,
    )
    second_result = build_constraint_table(
        triples_path=triples,
        model="model-id",
        revision="revision-id",
        output_path=second,
    )

    assert first_result == second_result
    assert first.read_bytes() == second.read_bytes()


def test_remote_model_requires_revision_before_loading() -> None:
    with pytest.raises(ValueError, match="requires --revision"):
        build_constraint_table(
            triples_path="unused.json",
            model="organization/remote-model",
            revision=None,
            output_path="unused.safetensors",
        )


def test_build_tries_manifest_uses_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: _tokenizer())  # type: ignore[attr-defined]
    transformers.AutoConfig = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *a, **kw: SimpleNamespace(vocab_size=20)
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    for index, c in enumerate((3, 5)):
        (tmp_path / f"trie{index}.json").write_text(
            json.dumps(compile_relations_json([["1", "2", str(c)]]))
        )
    manifest = tmp_path / "tries.json"
    manifest.write_text(
        json.dumps({"tries": [{"trie_id": i, "triples_path": f"trie{i}.json"} for i in range(2)]})
    )
    output = tmp_path / "tries.safetensors"
    digest, tokenizer_digest, vocab = build_constraint_table(
        triples_path=None,
        multi_trie_manifest_path=manifest,
        model=str(tmp_path),
        revision=None,
        output_path=output,
    )
    loaded = load_constraint_table_artifact(
        output,
        expected_digest=digest,
        expected_tokenizer_digest=tokenizer_digest,
        expected_vocab_size=vocab,
        expected_format="constraint_table_v2",
    )
    assert loaded.root_offsets.tolist() == [0, 1, 2]
    assert loaded.prefix1_pair_keys.tolist() == [22, 422]
    assert loaded.prefix2_values.tolist() == [3, 5]


def test_symbolic_remote_revision_is_resolved_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    commit_sha = "b" * 40
    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(sha=commit_sha)
    api_type = MagicMock(return_value=api)
    monkeypatch.setattr(huggingface_hub, "HfApi", api_type)

    assert _resolve_model_revision("organization/model", "release") == commit_sha
    api_type.assert_called_once_with()
    api.model_info.assert_called_once_with("organization/model", revision="release")


def test_commit_revision_does_not_require_remote_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    api_type = MagicMock()
    monkeypatch.setattr(huggingface_hub, "HfApi", api_type)

    commit_sha = "C" * 40
    assert _resolve_model_revision("organization/model", commit_sha) == commit_sha.lower()
    api_type.assert_not_called()


def test_local_model_directory_allows_omitted_revision(tmp_path: Path) -> None:
    model_path = tmp_path / "immutable-model-snapshot"
    model_path.mkdir()

    assert _resolve_model_revision(str(model_path), None) is None


pytestmark = pytest.mark.cpu_test
