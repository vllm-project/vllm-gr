# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tools.build_constraint_table import build_constraint_table
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
        "revision": "revision-id",
        "trust_remote_code": False,
    }
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
