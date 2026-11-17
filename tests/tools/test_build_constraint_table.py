# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import tools.build_constraint_table as build_constraint_table_module
from tools.build_constraint_table import _resolve_model_revision, build_constraint_table
from vllm_gr.constraints.artifact import load_constraint_table_artifact
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


def test_remote_model_requires_revision_before_loading(tmp_path: Path) -> None:
    triples = tmp_path / "triples.json"
    triples.write_text(json.dumps(compile_relations_json([["1", "2", "3"]])))
    with pytest.raises(ValueError, match="requires --revision"):
        build_constraint_table(
            triples_path=triples,
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


def _forbid_model_access(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    resolve_revision = MagicMock(side_effect=AssertionError("must validate local inputs first"))
    monkeypatch.setattr(
        build_constraint_table_module,
        "_resolve_model_revision",
        resolve_revision,
    )
    return resolve_revision


@pytest.mark.l1
@pytest.mark.parametrize("source_count", [0, 2])
def test_rejects_invalid_sources_before_model_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_count: int
) -> None:
    resolve_revision = _forbid_model_access(monkeypatch)
    source = tmp_path / "unused.json" if source_count else None
    with pytest.raises(ValueError, match="provide exactly one"):
        build_constraint_table(
            triples_path=source,
            multi_trie_manifest_path=source,
            model="organization/remote-model",
            revision="release",
            output_path=tmp_path / "unused.safetensors",
        )
    resolve_revision.assert_not_called()


@pytest.mark.l1
@pytest.mark.parametrize("source_kind", ["triples", "manifest"])
@pytest.mark.parametrize("contents", [None, b"{", b"\xff", b"[]"])
def test_local_json_failures_precede_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    contents: bytes | None,
) -> None:
    resolve_revision = _forbid_model_access(monkeypatch)
    source = tmp_path / "source.json"
    if contents is not None:
        source.write_bytes(contents)
    output = tmp_path / "unused.safetensors"
    with pytest.raises(ValueError) as error:
        build_constraint_table(
            triples_path=source if source_kind == "triples" else None,
            multi_trie_manifest_path=source if source_kind == "manifest" else None,
            model="organization/remote-model",
            revision="release",
            output_path=output,
        )
    assert repr(str(source)) in str(error.value)
    context = "constraint triples document" if source_kind == "triples" else "tries manifest"
    assert context in str(error.value)
    resolve_revision.assert_not_called()
    assert not output.exists()


@pytest.mark.l1
@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {"tries": []},
        {"tries": {}},
        {"tries": [None]},
        {"tries": [{"trie_id": 1, "triples_path": "trie.json"}]},
        {"tries": [{"trie_id": False, "triples_path": "trie.json"}]},
        {"tries": [{"trie_id": 0, "triples_path": 1}]},
        {"tries": [{"trie_id": 0, "triples_path": ""}]},
    ],
)
def test_manifest_structure_precedes_model_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: dict[str, object]
) -> None:
    resolve_revision = _forbid_model_access(monkeypatch)
    source = tmp_path / "tries.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="tries manifest") as error:
        build_constraint_table(
            triples_path=None,
            multi_trie_manifest_path=source,
            model="organization/remote-model",
            revision="release",
            output_path=tmp_path / "unused.safetensors",
        )
    assert repr(str(source)) in str(error.value)
    resolve_revision.assert_not_called()


@pytest.mark.l1
@pytest.mark.parametrize("contents", [None, b"{", b"\xff", b"[]"])
def test_child_json_failures_include_manifest_and_trie_before_model_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: bytes | None
) -> None:
    resolve_revision = _forbid_model_access(monkeypatch)
    (tmp_path / "trie0.json").write_text("{}", encoding="utf-8")
    source = tmp_path / "tries.json"
    source.write_text(
        json.dumps({"tries": [{"trie_id": i, "triples_path": f"trie{i}.json"} for i in range(2)]}),
        encoding="utf-8",
    )
    child = tmp_path / "trie1.json"
    if contents is not None:
        child.write_bytes(contents)
    with pytest.raises(ValueError) as error:
        build_constraint_table(
            triples_path=None,
            multi_trie_manifest_path=source,
            model="organization/remote-model",
            revision="release",
            output_path=tmp_path / "unused.safetensors",
        )
    assert "Trie 1" in str(error.value)
    assert repr(str(source)) in str(error.value)
    assert repr(str(child)) in str(error.value)
    resolve_revision.assert_not_called()


pytestmark = pytest.mark.cpu_test
